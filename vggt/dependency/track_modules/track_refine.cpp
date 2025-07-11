/**
 * @file track_refine.cpp
 * @brief Implementation of functions for refining tracks using fine-grained features
 */

#include "track_refine.h"
#include <torch/torch.h>
#include <stdexcept>
#include <cmath>

namespace vggt {

namespace {
    // Helper function to rearrange tensor dimensions
    torch::Tensor rearrange(torch::Tensor tensor, const std::string& pattern) {
        if (pattern == "b s n c p q -> (b n) s c p q") {
            auto sizes = tensor.sizes();
            int b = sizes[0];
            int s = sizes[1];
            int n = sizes[2];
            int c = sizes[3];
            int p = sizes[4];
            int q = sizes[5];
            return tensor.permute({0, 2, 1, 3, 4, 5}).reshape({b * n, s, c, p, q});
        } else if (pattern == "(b n) s u v -> b s n u v") {
            auto sizes = tensor.sizes();
            int bn = sizes[0];
            int s = sizes[1];
            int u = sizes[2];
            int v = sizes[3];
            int b = bn / n; // n is passed as a parameter
            return tensor.reshape({b, n, s, u, v}).permute({0, 2, 1, 3, 4});
        } else {
            throw std::runtime_error("Unsupported rearrange pattern: " + pattern);
        }
    }
}

std::tuple<torch::Tensor, torch::Tensor> refine_track(
    torch::Tensor images,
    torch::jit::Module fine_fnet,
    torch::jit::Module fine_tracker,
    torch::Tensor coarse_pred,
    bool compute_score,
    int pradius,
    int sradius,
    int fine_iters,
    int chunk
) {
    // coarse_pred shape: BxSxNx2,
    // where B is the batch, S is the video/images length, and N is the number of tracks
    auto sizes = coarse_pred.sizes();
    int B = sizes[0];
    int S = sizes[1];
    int N = sizes[2];

    auto image_sizes = images.sizes();
    int H = image_sizes[3];
    int W = image_sizes[4];

    // Given the radius of a patch, compute the patch size
    int psize = pradius * 2 + 1;

    // Note that we assume the first frame is the query frame
    // so the 2D locations of the first frame are the query points
    auto query_points = coarse_pred.index({torch::indexing::Slice(), 0});

    torch::Tensor content_to_extract;
    int C_in;

    {
        torch::NoGradGuard no_grad;
        content_to_extract = images.reshape({B * S, 3, H, W});
        C_in = content_to_extract.sizes()[1];

        // Unfold operation to build patches
        content_to_extract = content_to_extract.unfold(2, psize, 1).unfold(3, psize, 1);
    }

    // Floor the coarse predictions to get integers and save the fractional/decimal
    auto track_int = coarse_pred.floor().toType(torch::kInt);
    auto track_frac = coarse_pred - track_int.toType(coarse_pred.scalar_type());

    // Note the points represent the center of patches
    // now we get the location of the top left corner of patches
    auto topleft = track_int - pradius;
    auto topleft_BSN = topleft.clone();

    // clamp the values so that we will not go out of indexes
    // NOTE: (VERY IMPORTANT: This operation ASSUMES H=W).
    // You need to separately clamp x and y if H!=W
    topleft = topleft.clamp(0, H - psize);

    // Reshape from BxSxNx2 -> (B*S)xNx2
    topleft = topleft.reshape({B * S, N, 2});

    // Prepare batches for indexing, shape: (B*S)xN
    auto batch_indices = torch::arange(B * S, topleft.options().dtype(torch::kLong))
                            .unsqueeze(1)
                            .expand({-1, N});

    // Extract image patches based on top left corners
    // extracted_patches: (B*S) x N x C_in x Psize x Psize
    auto extracted_patches = content_to_extract.index({
        batch_indices,
        torch::indexing::Slice(),
        topleft.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}),
        topleft.index({torch::indexing::Slice(), torch::indexing::Slice(), 0})
    });

    torch::Tensor patch_feat;

    if (chunk < 0) {
        // Feed patches to fine fnet for features
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(extracted_patches.reshape({B * S * N, C_in, psize, psize}));
        patch_feat = fine_fnet.forward(inputs).toTensor();
    } else {
        auto patches = extracted_patches.reshape({B * S * N, C_in, psize, psize});

        std::vector<torch::Tensor> patch_feat_list;
        for (int i = 0; i < patches.sizes()[0]; i += chunk) {
            int end_idx = std::min(i + chunk, static_cast<int>(patches.sizes()[0]));
            auto p = patches.slice(0, i, end_idx);

            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(p);
            patch_feat_list.push_back(fine_fnet.forward(inputs).toTensor());
        }
        patch_feat = torch::cat(patch_feat_list, 0);
    }

    int C_out = patch_feat.sizes()[1];

    // Refine the coarse tracks by fine_tracker
    // reshape back to B x S x N x C_out x Psize x Psize
    patch_feat = patch_feat.reshape({B, S, N, C_out, psize, psize});

    // Rearrange: "b s n c p q -> (b n) s c p q"
    patch_feat = rearrange(patch_feat, "b s n c p q -> (b n) s c p q");

    // Prepare for the query points for fine tracker
    // They are relative to the patch left top corner,
    // instead of the image top left corner now
    auto patch_query_points = track_frac.index({torch::indexing::Slice(), 0}) + pradius;
    patch_query_points = patch_query_points.reshape({B * N, 2}).unsqueeze(1);

    // Feed the PATCH query points and tracks into fine tracker
    std::vector<torch::jit::IValue> tracker_inputs;
    tracker_inputs.push_back(patch_query_points);
    tracker_inputs.push_back(patch_feat);
    tracker_inputs.push_back(fine_iters);
    tracker_inputs.push_back(true); // return_feat=True

    auto tracker_outputs = fine_tracker.forward(tracker_inputs).toTuple();
    auto fine_pred_track_lists = tracker_outputs->elements()[0].toList().vec();
    auto query_point_feat = tracker_outputs->elements()[3].toTensor();

    // relative the patch top left
    auto fine_pred_track = fine_pred_track_lists.back().toTensor().clone();

    // From (relative to the patch top left) to (relative to the image top left)
    std::vector<torch::Tensor> fine_pred_track_tensors;
    for (const auto& track_list_item : fine_pred_track_lists) {
        auto fine_level = track_list_item.toTensor();
        // Rearrange: "(b n) s u v -> b s n u v"
        fine_level = rearrange(fine_level, "(b n) s u v -> b s n u v");
        fine_level = fine_level.squeeze(-2);
        fine_level = fine_level + topleft_BSN;
        fine_pred_track_tensors.push_back(fine_level);
    }

    // relative to the image top left
    auto refined_tracks = fine_pred_track_tensors.back().clone();
    refined_tracks.index_put_({torch::indexing::Slice(), 0}, query_points);

    torch::Tensor score = torch::Tensor();

    if (compute_score) {
        score = compute_score_fn(
            query_point_feat, patch_feat, fine_pred_track,
            sradius, psize, B, N, S, C_out
        );
    }

    return std::make_tuple(refined_tracks, score);
}

std::tuple<torch::Tensor, torch::Tensor> refine_track_v0(
    torch::Tensor images,
    torch::jit::Module fine_fnet,
    torch::jit::Module fine_tracker,
    torch::Tensor coarse_pred,
    bool compute_score,
    int pradius,
    int sradius,
    int fine_iters
) {
    // This is essentially the same as refine_track but without chunking
    return refine_track(
        images, fine_fnet, fine_tracker, coarse_pred,
        compute_score, pradius, sradius, fine_iters, -1
    );
}

torch::Tensor compute_score_fn(
    torch::Tensor query_point_feat,
    torch::Tensor patch_feat,
    torch::Tensor fine_pred_track,
    int sradius,
    int psize,
    int B,
    int N,
    int S,
    int C_out
) {
    // Compute the scores, i.e., the standard deviation of the 2D similarity heatmaps

    // query_point_feat initial shape: B x N x C_out
    query_point_feat = query_point_feat.reshape({B, N, C_out});
    // reshape and expand to B x (S-1) x N x C_out
    query_point_feat = query_point_feat.unsqueeze(1).expand({-1, S - 1, -1, -1});
    // and reshape to (B*(S-1)*N) x C_out
    query_point_feat = query_point_feat.reshape({B * (S - 1) * N, C_out});

    // Radius and size for computing the score
    int ssize = sradius * 2 + 1;

    // Reshape patch_feat
    // Rearrange: "(b n) s c p q -> b s n c p q"
    auto sizes = patch_feat.sizes();
    int bn = sizes[0];
    patch_feat = patch_feat.reshape({B, N, S, C_out, psize, psize}).permute({0, 2, 1, 3, 4, 5});

    // Again, we unfold the patches to smaller patches
    // so that we can then focus on smaller patches
    auto patch_feat_unfold = patch_feat.unfold(4, ssize, 1).unfold(5, ssize, 1);

    // Do the same stuffs above, i.e., the same as extracting patches
    auto fine_prediction_floor = fine_pred_track.floor().toType(torch::kInt);
    auto fine_level_floor_topleft = fine_prediction_floor - sradius;

    // Clamp to ensure the smaller patch is valid
    fine_level_floor_topleft = fine_level_floor_topleft.clamp(0, psize - ssize);
    fine_level_floor_topleft = fine_level_floor_topleft.squeeze(2);

    // Prepare the batch indices and xy locations
    auto batch_indices_score = torch::arange(B, fine_level_floor_topleft.options().dtype(torch::kLong))
                                .unsqueeze(1)
                                .unsqueeze(1)
                                .expand({-1, S, N})
                                .reshape({-1});

    auto y_indices = fine_level_floor_topleft.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}).flatten();
    auto x_indices = fine_level_floor_topleft.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}).flatten();

    auto reference_frame_feat = patch_feat_unfold.reshape({
        B * S * N, C_out, psize - sradius * 2, psize - sradius * 2, ssize, ssize
    });

    // Note again, according to pytorch convention
    // x_indices corresponds to [..., 1] and y_indices corresponds to [..., 0]
    reference_frame_feat = reference_frame_feat.index({
        batch_indices_score, torch::indexing::Slice(), x_indices, y_indices
    });

    reference_frame_feat = reference_frame_feat.reshape({B, S, N, C_out, ssize, ssize});
    // pick the frames other than the first one, so we have S-1 frames here
    reference_frame_feat = reference_frame_feat.index({
        torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)
    }).reshape({B * (S - 1) * N, C_out, ssize * ssize});

    // Compute similarity
    auto sim_matrix = torch::einsum("mc,mcr->mr", {query_point_feat, reference_frame_feat});
    float softmax_temp = 1.0f / std::sqrt(static_cast<float>(C_out));
    auto heatmap = torch::softmax(softmax_temp * sim_matrix, 1);
    // 2D heatmaps
    heatmap = heatmap.reshape({B * (S - 1) * N, ssize, ssize});

    // Create normalized grid
    auto y = torch::linspace(-1.0f, 1.0f, ssize, heatmap.options());
    auto x = torch::linspace(-1.0f, 1.0f, ssize, heatmap.options());
    auto grid_y, grid_x = torch::meshgrid({y, x}, "ij");
    auto grid = torch::stack({grid_x, grid_y}, -1).reshape({1, ssize * ssize, 2});

    // Compute spatial expectation (centroid)
    auto coords_normalized = torch::zeros({B * (S - 1) * N, 2}, heatmap.options());
    for (int i = 0; i < B * (S - 1) * N; ++i) {
        auto heatmap_flat = heatmap[i].reshape({-1});
        coords_normalized[i] = (grid * heatmap_flat.unsqueeze(-1)).sum(1);
    }

    // Compute spatial variance
    auto var_normalized = torch::zeros({B * (S - 1) * N, 2}, heatmap.options());
    for (int i = 0; i < B * (S - 1) * N; ++i) {
        auto heatmap_flat = heatmap[i].reshape({-1});
        auto diff = grid - coords_normalized[i].unsqueeze(0);
        var_normalized[i] = (diff * diff * heatmap_flat.unsqueeze(-1)).sum(1);
    }

    // Compute standard deviation
    auto std_normalized = torch::sqrt(var_normalized);
    auto std_score = std_normalized.reshape({B, S - 1, N, 2}).mean({1, 3});

    return std_score;
}

torch::Tensor extract_glimpse(
    torch::Tensor tensor,
    std::tuple<int, int> size,
    torch::Tensor offsets,
    const std::string& mode,
    const std::string& padding_mode,
    bool debug,
    int orib
) {
    auto tensor_size = tensor.sizes();
    int B = tensor_size[0];
    int C = tensor_size[1];
    int H = tensor_size[2];
    int W = tensor_size[3];

    auto offsets_size = offsets.sizes();
    int N = offsets_size[1];

    int h = std::get<0>(size);
    int w = std::get<1>(size);

    // Create normalized grid
    auto y = torch::linspace(-1.0f, 1.0f, h, tensor.options());
    auto x = torch::linspace(-1.0f, 1.0f, w, tensor.options());
    auto grid_y, grid_x = torch::meshgrid({y, x}, "ij");
    auto grid = torch::stack({grid_x, grid_y}, -1);

    // Normalize offsets to [-1, 1]
    auto offsets_normalized = offsets.clone();
    offsets_normalized.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}) =
        2.0f * offsets_normalized.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}) / (W - 1) - 1.0f;
    offsets_normalized.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}) =
        2.0f * offsets_normalized.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}) / (H - 1) - 1.0f;

    // Expand grid for each offset
    auto grid_expanded = grid.unsqueeze(0).unsqueeze(0).expand({B, N, -1, -1, -1});
    auto offsets_expanded = offsets_normalized.unsqueeze(2).unsqueeze(2).expand({-1, -1, h, w, -1});

    // Apply offsets to grid
    auto grid_with_offsets = grid_expanded + offsets_expanded;

    // Reshape for grid_sample
    grid_with_offsets = grid_with_offsets.reshape({B * N, h, w, 2});

    // Sample from tensor
    auto sampled = torch::grid_sample(
        tensor.reshape({B, 1, C, H, W}).expand({-1, N, -1, -1, -1}).reshape({B * N, C, H, W}),
        grid_with_offsets,
        torch::nn::functional::GridSampleFuncOptions().mode(mode).padding_mode(padding_mode).align_corners(true)
    );

    // Reshape back
    sampled = sampled.reshape({B, N, C, h, w});

    if (debug) {
        if (orib > 0) {
            sampled = sampled.reshape({orib, -1, N, C, h, w});
        }
    }

    return sampled;
}

} // namespace vggt
