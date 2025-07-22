

#include "base_track_predictor.h"
#include <iostream>
#include <stdexcept>

namespace vggt {
namespace track_modules {

BaseTrackerPredictor::BaseTrackerPredictor(
    int stride,
    int corr_levels,
    int corr_radius,
    int latent_dim,
    int hidden_size,
    bool use_spaceatt,
    int depth,
    bool fine
) : stride(stride),
    corr_levels(corr_levels),
    corr_radius(corr_radius),
    latent_dim(latent_dim),
    hidden_size(hidden_size),
    use_spaceatt(use_spaceatt),
    depth(depth),
    fine(fine) {

    // Initialize derived parameters
    flows_emb_dim = latent_dim / 2;
    transformer_dim = corr_levels * (corr_radius * 2 + 1) * (corr_radius * 2 + 1) + latent_dim * 2;

    if (fine) {
        // TODO: Adjust transformer_dim for fine predictor
        transformer_dim += 4;
    } else {
        transformer_dim += (4 - transformer_dim % 4) % 4;
    }

    // Initialize transformer
    register_module("updateformer",
        EfficientUpdateFormer(
            depth,
            depth,
            transformer_dim,
            hidden_size,
            latent_dim + 2,
            4.0,
            use_spaceatt
        )
    );

    // Initialize normalization layer
    register_module("norm", torch::nn::GroupNorm(1, latent_dim));

    // Initialize feature updater
    register_module("ffeat_updater",
        torch::nn::Sequential(
            torch::nn::Linear(latent_dim, latent_dim),
            torch::nn::GELU()
        )
    );

    if (!fine) {
        // Initialize visibility predictor
        register_module("vis_predictor",
            torch::nn::Sequential(
                torch::nn::Linear(latent_dim, 1)
            )
        );
    }
}

std::tuple<std::vector<torch::Tensor>, torch::Tensor> BaseTrackerPredictor::forward(
    const torch::Tensor& query_points,
    const torch::Tensor& fmaps,
    int iters,
    int down_ratio
) {
    // Validate input dimensions
    TORCH_CHECK(query_points.dim() == 3, "query_points must be 3D tensor [B,N,2]");
    TORCH_CHECK(fmaps.dim() == 5, "fmaps must be 5D tensor [B,S,C,HH,WW]");

    auto B = query_points.size(0);
    auto N = query_points.size(1);
    auto S = fmaps.size(1);
    auto HH = fmaps.size(3);
    auto WW = fmaps.size(4);

    // Scale the input query_points
    if (down_ratio > 1) {
        query_points = query_points / float(down_ratio);
    }
    query_points = query_points / float(stride);

    // Initialize coordinates
    auto coords = query_points.view({B, 1, N, 2}).repeat({1, S, 1, 1});
    auto coords_backup = coords.clone();

    // Sample features for query points
    auto query_track_feat = sample_features4d(fmaps.select(1, 0), coords.select(1, 0));
    auto track_feats = query_track_feat.unsqueeze(1).repeat({1, S, 1, 1});

    // Initialize correlation block
    auto fcorr_fn = CorrBlock(fmaps, corr_levels, corr_radius);

    std::vector<torch::Tensor> coord_preds;

    // Iterative refinement
    for (int itr = 0; itr < iters; ++itr) {
        coords = coords.detach();

        // Compute correlation
        fcorr_fn.corr(track_feats);
        auto fcorrs = fcorr_fn.sample(coords);

        // Process features
        auto fcorrs_ = fcorrs.permute({0, 2, 1, 3}).reshape({B * N, S, -1});
        auto flows = (coords - coords.slice(1, 0, 1)).permute({0, 2, 1, 3}).reshape({B * N, S, 2});
        auto flows_emb = get_2d_embedding(flows, flows_emb_dim, false);
        flows_emb = torch::cat({flows_emb, flows}, -1);

        auto track_feats_ = track_feats.permute({0, 2, 1, 3}).reshape({B * N, S, latent_dim});
        auto transformer_input = torch::cat({flows_emb, fcorrs_, track_feats_}, -1);

        // Apply transformer
        auto delta = updateformer->forward(transformer_input);
        auto delta_coords_ = delta.slice(-1, 0, 2);
        auto delta_feats_ = delta.slice(-1, 2);

        // Update features and coordinates
        track_feats_ = ffeat_updater->forward(norm->forward(delta_feats_)) + track_feats_;
        track_feats = track_feats_.view({B, N, S, latent_dim}).permute({0, 2, 1, 3});
        coords = coords + delta_coords_.view({B, N, S, 2}).permute({0, 2, 1, 3});

        // Reset first frame coordinates
        coords.slice(1, 0, 1) = coords_backup.slice(1, 0, 1);

        // Store predictions
        if (down_ratio > 1) {
            coord_preds.push_back(coords * stride * down_ratio);
        } else {
            coord_preds.push_back(coords * stride);
        }
    }

    // Predict visibility
    torch::Tensor vis_e;
    if (!fine) {
        vis_e = vis_predictor->forward(track_feats.reshape({B * S * N, latent_dim})).reshape({B, S, N});
        vis_e = torch::sigmoid(vis_e);
    }

    return {coord_preds, vis_e};
}

} // namespace track_modules
} // namespace vggt
