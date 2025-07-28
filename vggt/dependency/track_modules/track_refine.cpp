#include "track_refine.h"

namespace vggt {
namespace dependency {
namespace track_modules {

std::tuple<torch::Tensor, torch::Tensor> refine_track(
    torch::Tensor images,
    torch::nn::AnyModule& fine_fnet,
    torch::nn::AnyModule& fine_tracker,
    torch::Tensor coarse_pred,
    bool compute_score,
    int64_t pradius,
    int64_t sradius,
    int64_t fine_iters,
    int64_t chunk
) {
    // coarse_pred shape: BxSxNx2,
    // where B is the batch, S is the video/images length, and N is the number of tracks
    // now we are going to extract patches with the center at coarse_pred
    // Please note that the last dimension indicates x and y, and hence has a dim number of 2
    auto B = coarse_pred.size(0);
    auto S = coarse_pred.size(1);
    auto N = coarse_pred.size(2);
    auto H = images.size(3);
    auto W = images.size(4);

    // Given the raidus of a patch, compute the patch size
    auto psize = pradius * 2 + 1;

    // Note that we assume the first frame is the query frame
    // so the 2D locations of the first frame are the query points
    auto query_points = coarse_pred.index({torch::indexing::Slice(), 0});

    // Given 2D positions, we can use grid_sample to extract patches
    // but it takes too much memory.
    // Instead, we use the floored track xy to sample patches.

    torch::Tensor content_to_extract;
    int64_t C_in;
    {
        torch::NoGradGuard no_grad;
        content_to_extract = images.reshape({B * S, 3, H, W});
        C_in = content_to_extract.size(1);

        // Please refer to https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
        // for the detailed explanation of unfold()
        // Here it runs sliding windows (psize x psize) to build patches
        // The shape changes from
        // (B*S)x C_in x H x W to (B*S)x C_in x H_new x W_new x Psize x Psize
        // where Psize is the size of patch
        content_to_extract = content_to_extract.unfold(2, psize, 1).unfold(3, psize, 1);
    }

    // Floor the coarse predictions to get integers and save the fractional/decimal
    auto track_int = coarse_pred.floor().to(torch::kInt);
    auto track_frac = coarse_pred - track_int;

    // Note the points represent the center of patches
    // now we get the location of the top left corner of patches
    // because the ouput of pytorch unfold are indexed by top left corner
    auto topleft = track_int - pradius;
    auto topleft_BSN = topleft.clone();

    // clamp the values so that we will not go out of indexes
    // NOTE: (VERY IMPORTANT: This operation ASSUMES H=W).
    // You need to seperately clamp x and y if H!=W
    topleft = topleft.clamp(0, H - psize);

    // Reshape from BxSxNx2 -> (B*S)xNx2
    topleft = topleft.reshape({B * S, N, 2});

    // Prepare batches for indexing, shape: (B*S)xN
    auto batch_indices = torch::arange(B * S, topleft.options()).unsqueeze(1).expand({-1, N});

    // extracted_patches: (B*S) x N x C_in x Psize x Psize
    auto extracted_patches = content_to_extract.index({batch_indices, torch::indexing::Slice(),
                                                     topleft.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}),
                                                     topleft.index({torch::indexing::Slice(), torch::indexing::Slice(), 0})});

    torch::Tensor patch_feat;
    if (chunk < 0) {
        // Extract image patches based on top left corners
        // Feed patches to fine fent for features
        patch_feat = fine_fnet.forward<torch::Tensor>(extracted_patches.reshape({B * S * N, C_in, psize, psize}));
    } else {
        auto patches = extracted_patches.reshape({B * S * N, C_in, psize, psize});

        std::vector<torch::Tensor> patch_feat_list;
        for (auto p : torch::split(patches, chunk)) {
            patch_feat_list.push_back(fine_fnet.forward<torch::Tensor>(p));
        }
        patch_feat = torch::cat(patch_feat_list, 0);
    }

    auto C_out = patch_feat.size(1);

    // Refine the coarse tracks by fine_tracker
    // reshape back to B x S x N x C_out x Psize x Psize
    patch_feat = patch_feat.reshape({B, S, N, C_out, psize, psize});
    patch_feat = einops::rearrange(patch_feat, "b s n c p q -> (b n) s c p q");

    // Prepare for the query points for fine tracker
    // They are relative to the patch left top corner,
    // instead of the image top left corner now
    // patch_query_points: N x 1 x 2
    // only 1 here because for each patch we only have 1 query point
    auto patch_query_points = track_frac.index({torch::indexing::Slice(), 0}) + pradius;
    patch_query_points = patch_query_points.reshape({B * N, 2}).unsqueeze(1);

    // Feed the PATCH query points and tracks into fine tracker
    auto result = fine_tracker.forward<std::tuple<std::vector<torch::Tensor>, torch::Tensor>>(patch_query_points, patch_feat, fine_iters, true);

    std::vector<torch::Tensor> fine_pred_track_lists;
    torch::Tensor vis_e, track_feats, query_point_feat;

    // Extract the results from the tuple
    std::tie(fine_pred_track_lists, vis_e) = result;
    // Note: In the Python version, there might be additional elements in the tuple
    // that we're not using in this C++ implementation

    // relative the patch top left
    auto fine_pred_track = fine_pred_track_lists.back().clone();

    // From (relative to the patch top left) to (relative to the image top left)
    for (size_t idx = 0; idx < fine_pred_track_lists.size(); ++idx) {
        auto fine_level = einops::rearrange(fine_pred_track_lists[idx], "(b n) s u v -> b s n u v", einops::axis("b", B));
        fine_level = fine_level.squeeze(-2);
        fine_level = fine_level + topleft_BSN;
        fine_pred_track_lists[idx] = fine_level;
    }

    // relative to the image top left
    auto refined_tracks = fine_pred_track_lists.back().clone();
    refined_tracks.index_put_({torch::indexing::Slice(), 0}, query_points);

    torch::Tensor score;

    if (compute_score) {
        score = compute_score_fn(query_point_feat, patch_feat, fine_pred_track, sradius, psize, B, N, S, C_out);
    }

    return {refined_tracks, score};
}

std::tuple<torch::Tensor, torch::Tensor> refine_track_v0(
    torch::Tensor images,
    torch::nn::AnyModule& fine_fnet,
    torch::nn::AnyModule& fine_tracker,
    torch::Tensor coarse_pred,
    bool compute_score,
    int64_t pradius,
    int64_t sradius,
    int64_t fine_iters
) {
    // coarse_pred shape: BxSxNx2,
    // where B is the batch, S is the video/images length, and N is the number of tracks
    // now we are going to extract patches with the center at coarse_pred
    // Please note that the last dimension indicates x and y, and hence has a dim number of 2
    auto B = coarse_pred.size(0);
    auto S = coarse_pred.size(1);
    auto N = coarse_pred.size(2);
    auto H = images.size(3);
    auto W = images.size(4);

    // Given the raidus of a patch, compute the patch size
    auto psize = pradius * 2 + 1;

    // Note that we assume the first frame is the query frame
    // so the 2D locations of the first frame are the query points
    auto query_points = coarse_pred.index({torch::indexing::Slice(), 0});

    // Given 2D positions, we can use grid_sample to extract patches
    // but it takes too much memory.
    // Instead, we use the floored track xy to sample patches.

    torch::Tensor content_to_extract;
    int64_t C_in;
    {
        torch::NoGradGuard no_grad;
        content_to_extract = images.reshape({B * S, 3, H, W});
        C_in = content_to_extract.size(1);

        // Please refer to https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
        // for the detailed explanation of unfold()
        // Here it runs sliding windows (psize x psize) to build patches
        // The shape changes from
        // (B*S)x C_in x H x W to (B*S)x C_in x H_new x W_new x Psize x Psize
        // where Psize is the size of patch
        content_to_extract = content_to_extract.unfold(2, psize, 1).unfold(3, psize, 1);
    }

    // Floor the coarse predictions to get integers and save the fractional/decimal
    auto track_int = coarse_pred.floor().to(torch::kInt);
    auto track_frac = coarse_pred - track_int;

    // Note the points represent the center of patches
    // now we get the location of the top left corner of patches
    // because the ouput of pytorch unfold are indexed by top left corner
    auto topleft = track_int - pradius;
    auto topleft_BSN = topleft.clone();

    // clamp the values so that we will not go out of indexes
    // NOTE: (VERY IMPORTANT: This operation ASSUMES H=W).
    // You need to seperately clamp x and y if H!=W
    topleft = topleft.clamp(0, H - psize);

    // Reshape from BxSxNx2 -> (B*S)xNx2
    topleft = topleft.reshape({B * S, N, 2});

    // Prepare batches for indexing, shape: (B*S)xN
    auto batch_indices = torch::arange(B * S, topleft.options()).unsqueeze(1).expand({-1, N});

    // Extract image patches based on top left corners
    // extracted_patches: (B*S) x N x C_in x Psize x Psize
    auto extracted_patches = content_to_extract.index({batch_indices, torch::indexing::Slice(),
                                                     topleft.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}),
                                                     topleft.index({torch::indexing::Slice(), torch::indexing::Slice(), 0})});

    // Feed patches to fine fent for features
    auto patch_feat = fine_fnet.forward<torch::Tensor>(extracted_patches.reshape({B * S * N, C_in, psize, psize}));

    auto C_out = patch_feat.size(1);

    // Refine the coarse tracks by fine_tracker

    // reshape back to B x S x N x C_out x Psize x Psize
    patch_feat = patch_feat.reshape({B, S, N, C_out, psize, psize});
    patch_feat = einops::rearrange(patch_feat, "b s n c p q -> (b n) s c p q");

    // Prepare for the query points for fine tracker
    // They are relative to the patch left top corner,
    // instead of the image top left corner now
    // patch_query_points: N x 1 x 2
    // only 1 here because for each patch we only have 1 query point
    auto patch_query_points = track_frac.index({torch::indexing::Slice(), 0}) + pradius;
    patch_query_points = patch_query_points.reshape({B * N, 2}).unsqueeze(1);

    // Feed the PATCH query points and tracks into fine tracker
    auto result = fine_tracker.forward<std::tuple<std::vector<torch::Tensor>, torch::Tensor>>(patch_query_points, patch_feat, fine_iters, true);

    std::vector<torch::Tensor> fine_pred_track_lists;
    torch::Tensor vis_e, track_feats, query_point_feat;

    // Extract the results from the tuple
    std::tie(fine_pred_track_lists, vis_e) = result;
    // Note: In the Python version, there might be additional elements in the tuple
    // that we're not using in this C++ implementation

    // relative the patch top left
    auto fine_pred_track = fine_pred_track_lists.back().clone();

    // From (relative to the patch top left) to (relative to the image top left)
    for (size_t idx = 0; idx < fine_pred_track_lists.size(); ++idx) {
        auto fine_level = einops::rearrange(fine_pred_track_lists[idx], "(b n) s u v -> b s n u v", einops::axis("b", B));
        fine_level = fine_level.squeeze(-2);
        fine_level = fine_level + topleft_BSN;
        fine_pred_track_lists[idx] = fine_level;
    }

    // relative to the image top left
    auto refined_tracks = fine_pred_track_lists.back().clone();
    refined_tracks.index_put_({torch::indexing::Slice(), 0}, query_points);

    torch::Tensor score;

    if (compute_score) {
        score = compute_score_fn(query_point_feat, patch_feat, fine_pred_track, sradius, psize, B, N, S, C_out);
    }

    return {refined_tracks, score};
}

torch::Tensor compute_score_fn(
    torch::Tensor query_point_feat,
    torch::Tensor patch_feat,
    torch::Tensor fine_pred_track,
    int64_t sradius,
    int64_t psize,
    int64_t B,
    int64_t N,
    int64_t S,
    int64_t C_out
) {
    // query_point_feat initial shape: B x N x C_out,
    // query_point_feat indicates the feat at the coorponsing query points
    // Therefore we don't have S dimension here
    query_point_feat = query_point_feat.reshape({B, N, C_out});
    // reshape and expand to B x (S-1) x N x C_out
    query_point_feat = query_point_feat.unsqueeze(1).expand({-1, S - 1, -1, -1});
    // and reshape to (B*(S-1)*N) x C_out
    query_point_feat = query_point_feat.reshape({B * (S - 1) * N, C_out});

    // Radius and size for computing the score
    auto ssize = sradius * 2 + 1;

    // Reshape, you know it, so many reshaping operations
    patch_feat = einops::rearrange(patch_feat, "(b n) s c p q -> b s n c p q", einops::axis("b", B));

    // Again, we unfold the patches to smaller patches
    // so that we can then focus on smaller patches
    // patch_feat_unfold shape:
    // B x S x N x C_out x (psize - 2*sradius) x (psize - 2*sradius) x ssize x ssize
    // well a bit scary, but actually not
    auto patch_feat_unfold = patch_feat.unfold(4, ssize, 1).unfold(5, ssize, 1);

    // Do the same stuffs above, i.e., the same as extracting patches
    auto fine_prediction_floor = fine_pred_track.floor().to(torch::kInt);
    auto fine_level_floor_topleft = fine_prediction_floor - sradius;

    // Clamp to ensure the smaller patch is valid
    fine_level_floor_topleft = fine_level_floor_topleft.clamp(0, psize - ssize);
    fine_level_floor_topleft = fine_level_floor_topleft.squeeze(2);

    // Prepare the batch indices and xy locations
    auto batch_indices_score = torch::arange(B, fine_level_floor_topleft.options()).unsqueeze(1).unsqueeze(1).expand({-1, S, N});  // BxSxN
    batch_indices_score = batch_indices_score.reshape({-1});  // B*S*N
    auto y_indices = fine_level_floor_topleft.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}).flatten();  // Flatten H indices
    auto x_indices = fine_level_floor_topleft.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}).flatten();  // Flatten W indices

    auto reference_frame_feat = patch_feat_unfold.reshape(
        {B * S * N, C_out, psize - sradius * 2, psize - sradius * 2, ssize, ssize}
    );

    // Note again, according to pytorch convention
    // x_indices cooresponds to [..., 1] and y_indices cooresponds to [..., 0]
    reference_frame_feat = reference_frame_feat.index({batch_indices_score, torch::indexing::Slice(), x_indices, y_indices});
    reference_frame_feat = reference_frame_feat.reshape({B, S, N, C_out, ssize, ssize});
    // pick the frames other than the first one, so we have S-1 frames here
    reference_frame_feat = reference_frame_feat.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}).reshape({B * (S - 1) * N, C_out, ssize * ssize});

    // Compute similarity
    auto sim_matrix = torch::einsum("mc,mcr->mr", {query_point_feat, reference_frame_feat});
    auto softmax_temp = 1.0 / std::sqrt(static_cast<double>(C_out));
    auto heatmap = torch::softmax(softmax_temp * sim_matrix, 1);
    // 2D heatmaps
    heatmap = heatmap.reshape({B * (S - 1) * N, ssize, ssize});  // * x ssize x ssize

    // Import kornia functionality
    // Since we don't have direct access to kornia in C++, we'll implement the functionality directly

    // Create normalized grid
    auto y_grid = torch::linspace(-1.0, 1.0, ssize, heatmap.options());
    auto x_grid = torch::linspace(-1.0, 1.0, ssize, heatmap.options());
    auto meshgrid = torch::meshgrid({y_grid, x_grid});
    auto grid_x = meshgrid[0];
    auto grid_y = meshgrid[1];
    std::vector<torch::Tensor> grid_tensors = {grid_x, grid_y};
    auto grid = torch::stack(grid_tensors, -1).reshape({1, -1, 2});

    // Compute spatial expectation (similar to kornia's dsnt.spatial_expectation2d)
    auto normalized_heatmap = heatmap.reshape({-1, ssize * ssize});
    normalized_heatmap = normalized_heatmap / normalized_heatmap.sum(1, true);
    auto coords_normalized = torch::matmul(normalized_heatmap, grid.reshape({ssize * ssize, 2}));

    // Compute variance and standard deviation
    auto var = torch::sum(grid.pow(2) * heatmap.reshape({-1, ssize * ssize, 1}), 1) - coords_normalized.pow(2);
    auto std = torch::sum(torch::sqrt(torch::clamp(var, 1e-10)), -1);  // clamp needed for numerical stability

    auto score = std.reshape({B, S - 1, N});
    // set score as 1 for the query frame
    score = torch::cat({torch::ones_like(score.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)})), score}, 1);

    return score;
}

} // namespace track_modules
} // namespace dependency
} // namespace vggt
