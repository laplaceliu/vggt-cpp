/**
 * @file base_track_predictor.cpp
 * @brief Implementation of base tracker predictor
 */

#include "base_track_predictor.h"
#include "blocks.h"
#include "utils.h"
#include <torch/torch.h>
#include <stdexcept>
#include <vector>

namespace vggt {

BaseTrackerPredictor::BaseTrackerPredictor(
    int stride,
    int corr_levels,
    int corr_radius,
    int latent_dim,
    int hidden_size,
    bool use_spaceatt,
    int depth,
    bool fine
) : stride(stride), latent_dim(latent_dim), corr_levels(corr_levels),
    corr_radius(corr_radius), hidden_size(hidden_size), fine(fine) {
    // Initialize dimensions
    flows_emb_dim = latent_dim / 2;
    transformer_dim = corr_levels * (corr_radius * 2 + 1) * (corr_radius * 2 + 1) + latent_dim * 2;

    // Adjust transformer dimension to make it divisible by 4
    if (fine) {
        // TODO: This is the old dummy code, will remove this when we train next model
        transformer_dim += (transformer_dim % 2 == 0) ? 4 : 5;
    } else {
        transformer_dim += (4 - transformer_dim % 4) % 4;
    }

    // Initialize modules
    int space_depth = use_spaceatt ? depth : 0;
    int time_depth = depth;

    updateformer = register_module("updateformer",
        EfficientUpdateFormer(
            space_depth,
            time_depth,
            transformer_dim,
            hidden_size,
            latent_dim + 2,
            4.0,
            use_spaceatt
        ));

    norm = register_module("norm",
        torch::nn::GroupNorm(1, latent_dim));

    // Feature updater
    torch::nn::Sequential ffeat_seq;
    ffeat_seq->push_back(torch::nn::Linear(latent_dim, latent_dim));
    ffeat_seq->push_back(torch::nn::GELU());
    ffeat_updater = register_module("ffeat_updater", ffeat_seq);

    // Visibility predictor (only for coarse prediction)
    if (!fine) {
        torch::nn::Sequential vis_seq;
        vis_seq->push_back(torch::nn::Linear(latent_dim, 1));
        vis_predictor = register_module("vis_predictor", vis_seq);
    }
}

std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor>
BaseTrackerPredictor::forward(
    torch::Tensor query_points,
    torch::Tensor fmaps,
    int iters,
    bool return_feat,
    int down_ratio
) {
    // Validate input tensors
    if (query_points.dim() != 3 || query_points.size(2) != 2) {
        throw std::invalid_argument("query_points must have shape [B, N, 2]");
    }
    if (fmaps.dim() != 5) {
        throw std::invalid_argument("fmaps must have shape [B, S, C, HH, WW]");
    }

    int B = query_points.size(0);
    int N = query_points.size(1);
    int S = fmaps.size(1);
    int C = fmaps.size(2);
    int HH = fmaps.size(3);
    int WW = fmaps.size(4);

    // Scale the input query_points
    if (down_ratio > 1) {
        query_points = query_points / static_cast<float>(down_ratio);
    }
    query_points = query_points / static_cast<float>(stride);

    // Initialize coordinates with query points
    auto coords = query_points.clone().reshape({B, 1, N, 2}).repeat({1, S, 1, 1});

    // Sample features at query points
    auto query_track_feat = sample_features4d(fmaps.index({torch::indexing::Slice(), 0}),
                                            coords.index({torch::indexing::Slice(), 0}));

    // Initialize track features
    auto track_feats = query_track_feat.unsqueeze(1).repeat({1, S, 1, 1});  // B, S, N, C
    auto coords_backup = coords.clone();

    // Create correlation block
    auto fcorr_fn = CorrBlock(fmaps, corr_levels, corr_radius);

    std::vector<torch::Tensor> coord_preds;

    // Iterative refinement
    for (int itr = 0; itr < iters; ++itr) {
        coords = coords.detach();

        // Compute correlation
        fcorr_fn.corr(track_feats);
        auto fcorrs = fcorr_fn.sample(coords);  // B, S, N, corrdim
        int corrdim = fcorrs.size(3);

        // Reshape for transformer
        auto fcorrs_ = fcorrs.permute({0, 2, 1, 3}).reshape({B * N, S, corrdim});

        // Compute flows (movement relative to query points)
        auto flows = (coords - coords.index(torch::indexing::Slice()).index({0, 1}))
                    .permute({0, 2, 1, 3}).reshape({B * N, S, 2});

        // Get flow embeddings
        auto flows_emb = get_2d_embedding(flows, flows_emb_dim, false);
        flows_emb = torch::cat(std::vector<at::Tensor>{flows_emb, flows}, 2);

        // Reshape track features
        auto track_feats_ = track_feats.permute({0, 2, 1, 3}).reshape({B * N, S, latent_dim});

        // Concatenate inputs for transformer
        auto transformer_input = torch::cat({flows_emb, fcorrs_, track_feats_}, 2);

        // Pad if necessary
        if (transformer_input.size(2) < transformer_dim) {
            int pad_dim = transformer_dim - transformer_input.size(2);
            auto pad = torch::zeros_like(flows_emb.index({torch::indexing::Slice(),
                                                         torch::indexing::Slice(),
                                                         torch::indexing::Slice(0, pad_dim)}));
            transformer_input = torch::cat({transformer_input, pad}, 2);
        }

        // Get 2D positional embedding
        auto pos_embed = get_2d_sincos_pos_embed(transformer_dim, {HH, WW}).to(query_points.device());
        auto sampled_pos_emb = sample_features4d(pos_embed.expand({B, -1, -1, -1}),
                                               coords.index({torch::indexing::Slice(), 0}));
        sampled_pos_emb = sampled_pos_emb.reshape({B * N, -1}).unsqueeze(1);

        // Add positional embedding
        auto x = transformer_input + sampled_pos_emb;
        x = x.reshape({B, N, S, transformer_dim});

        // Compute delta coordinates and features
        auto delta = updateformer->forward<EfficientUpdateFormer>(x);
        delta = delta.reshape({B * N * S, latent_dim + 2});
        auto delta_coords_ = delta.index({torch::indexing::Slice(), torch::indexing::Slice(0, 2)})
                            .reshape({B, N, S, 2}).permute({0, 2, 1, 3});
        auto delta_feats_ = delta.index({torch::indexing::Slice(), torch::indexing::Slice(2, torch::indexing::None)})
                           .reshape({B * N * S, latent_dim});

        // Update track features
        auto track_feats_ = track_feats.reshape({B * N * S, latent_dim});
        track_feats_ = ffeat_updater->forward<torch::nn::Sequential>(norm(delta_feats_)) + track_feats_;
        track_feats = track_feats_.reshape({B, N, S, latent_dim}).permute({0, 2, 1, 3});

        // Update coordinates
        coords = coords + delta_coords_;

        // Force coord0 as query (query points should not change)
        coords.index_put_({torch::indexing::Slice(), 0}, coords_backup.index({torch::indexing::Slice(), 0}));

        // Store predicted coordinates (in original image scale)
        if (down_ratio > 1) {
            coord_preds.push_back(coords * stride * down_ratio);
        } else {
            coord_preds.push_back(coords * stride);
        }
    }

    // Compute visibility scores (only for coarse prediction)
    torch::Tensor vis_e;
    if (!fine) {
        vis_e = vis_predictor->forward<torch::nn::Sequential>(
            track_feats.reshape({B * S * N, latent_dim})).reshape({B, S, N});
        vis_e = torch::sigmoid(vis_e);
    }

    if (return_feat) {
        return std::make_tuple(coord_preds, vis_e, track_feats, query_track_feat);
    } else {
        return std::make_tuple(coord_preds, vis_e, torch::Tensor(), torch::Tensor());
    }
}

} // namespace vggt
