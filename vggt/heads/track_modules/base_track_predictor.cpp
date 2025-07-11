/**
 * @file base_track_predictor.cpp
 * @brief Implementation of base tracker predictor
 */

#include "base_track_predictor.h"
#include <stdexcept>
#include <cmath>

namespace vggt {

BaseTrackerPredictor::BaseTrackerPredictor(
    int stride,
    int corr_levels,
    int corr_radius,
    int latent_dim,
    int hidden_size,
    bool use_spaceatt,
    int depth,
    int max_scale,
    bool predict_conf
) : stride(stride),
    latent_dim(latent_dim),
    corr_levels(corr_levels),
    corr_radius(corr_radius),
    hidden_size(hidden_size),
    max_scale(max_scale),
    predict_conf(predict_conf) {

    // Initialize dimensions
    flows_emb_dim = latent_dim / 2;
    transformer_dim = latent_dim + latent_dim + latent_dim + 4;

    // Register modules
    fmap_norm = register_module("fmap_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({latent_dim})));
    ffeat_norm = register_module("ffeat_norm",
        torch::nn::GroupNorm(1, latent_dim));

    // Correlation MLP
    corr_mlp = register_module("corr_mlp", torch::nn::Sequential(
        torch::nn::Linear(
            corr_levels * (2 * corr_radius + 1) * (2 * corr_radius + 1),
            hidden_size
        ),
        torch::nn::GELU(),
        torch::nn::Linear(hidden_size, latent_dim)
    ));

    // Feature updater
    ffeat_updater = register_module("ffeat_updater", torch::nn::Sequential(
        torch::nn::Linear(latent_dim, latent_dim),
        torch::nn::GELU()
    ));

    // Visibility predictor
    vis_predictor = register_module("vis_predictor", torch::nn::Sequential(
        torch::nn::Linear(latent_dim, 1)
    ));

    // Confidence predictor (if enabled)
    if (predict_conf) {
        conf_predictor = register_module("conf_predictor", torch::nn::Sequential(
            torch::nn::Linear(latent_dim, 1)
        ));
    }

    // Update former
    updateformer = register_module("updateformer",
        EfficientUpdateFormer(
            space_depth = use_spaceatt ? depth : 0,
            time_depth = depth,
            input_dim = transformer_dim,
            hidden_size = hidden_size,
            output_dim = latent_dim + 2,
            mlp_ratio = 4.0,
            add_space_attn = use_spaceatt
        )
    );

    // Query reference token
    query_ref_token = register_parameter("query_ref_token",
        torch::randn({1, 2, transformer_dim}));

    // Initialize weights
    for (auto& module : modules()) {
        if (auto linear = dynamic_cast<torch::nn::Linear*>(module.get())) {
            torch::nn::init::xavier_uniform_(linear->weight);
            if (linear->bias.defined()) {
                torch::nn::init::constant_(linear->bias, 0);
            }
        }
    }
}

std::tuple<
    std::vector<torch::Tensor>,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
> BaseTrackerPredictor::forward(
    torch::Tensor query_points,
    torch::Tensor fmaps,
    int iters,
    bool return_feat,
    int down_ratio,
    bool apply_sigmoid
) {
    auto B = query_points.size(0);
    auto N = query_points.size(1);
    auto D = query_points.size(2);
    auto S = fmaps.size(1);
    auto C = fmaps.size(2);
    auto HH = fmaps.size(3);
    auto WW = fmaps.size(4);

    if (D != 2) {
        throw std::invalid_argument("Input points must be 2D coordinates");
    }

    // Apply layer norm to feature maps
    fmaps = fmap_norm(fmaps.permute({0, 1, 3, 4, 2}));
    fmaps = fmaps.permute({0, 1, 4, 2, 3});

    // Scale query points
    if (down_ratio > 1) {
        query_points = query_points / static_cast<float>(down_ratio);
    }
    query_points = query_points / static_cast<float>(stride);

    // Initialize coordinates
    auto coords = query_points.reshape({B, 1, N, 2}).repeat({1, S, 1, 1});
    auto coords_backup = coords.clone();

    // Sample query track features
    auto query_track_feat = sample_features4d(fmaps.index({torch::indexing::Slice(), 0}),
                                            coords.index({torch::indexing::Slice(), 0}));
    auto track_feats = query_track_feat.unsqueeze(1).repeat({1, S, 1, 1});

    // Create correlation block
    auto fcorr_fn = CorrBlock(fmaps, corr_levels, corr_radius);

    std::vector<torch::Tensor> coord_preds;

    // Iterative refinement
    for (int iter = 0; iter < iters; ++iter) {
        coords = coords.detach();

        // Sample correlations
        auto fcorrs = fcorr_fn.corr_sample(track_feats, coords);
        auto corr_dim = fcorrs.size(3);
        auto fcorrs_ = fcorrs.permute({0, 2, 1, 3}).reshape({B * N, S, corr_dim});
        fcorrs_ = corr_mlp->forward(fcorrs_);

        // Compute flow embeddings
        auto flows = (coords - coords.index({torch::indexing::Slice(), 0, torch::indexing::Slice(), torch::indexing::Slice()}))
                    .permute({0, 2, 1, 3}).reshape({B * N, S, 2});
        auto flows_emb = get_2d_embedding(flows, flows_emb_dim, false);
        flows_emb = torch::cat({flows_emb, flows / max_scale, flows / max_scale}, 2);

        // Prepare track features
        auto track_feats_ = track_feats.permute({0, 2, 1, 3}).reshape({B * N, S, latent_dim});

        // Concatenate inputs for transformer
        auto transformer_input = torch::cat({flows_emb, fcorrs_, track_feats_}, 2);

        // Add positional embeddings
        auto pos_embed = get_2d_sincos_pos_embed(transformer_dim, {HH, WW}).to(query_points.device());
        auto sampled_pos_emb = sample_features4d(pos_embed.expand({B, -1, -1, -1}),
                                                coords.index({torch::indexing::Slice(), 0}));
        sampled_pos_emb = sampled_pos_emb.reshape({B * N, -1}).unsqueeze(1);
        auto x = transformer_input + sampled_pos_emb;

        // Add query reference token
        auto query_ref_token_ = torch::cat({
            query_ref_token.index({torch::indexing::Slice(), 0, torch::indexing::Slice()}).unsqueeze(1),
            query_ref_token.index({torch::indexing::Slice(), 1, torch::indexing::Slice()})
                .expand({-1, S - 1, -1})
        }, 1);
        x = x + query_ref_token_.to(x.device()).to(x.dtype());

        // Reshape for transformer
        x = x.reshape({B, N, S, transformer_dim});

        // Apply transformer
        auto delta = updateformer->forward(x);
        auto delta_coords_ = delta.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(),
                                        torch::indexing::Slice(0, 2)});
        auto delta_feats_ = delta.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(),
                                       torch::indexing::Slice(2, delta.size(3))});

        // Update track features
        delta_feats_ = delta_feats_.reshape({B * N * S, -1});
        track_feats_ = track_feats_.reshape({B * N * S, -1});
        track_feats_ = ffeat_updater->forward(ffeat_norm->forward(delta_feats_)) + track_feats_;
        track_feats = track_feats_.reshape({B, N, S, latent_dim}).permute({0, 2, 1, 3});

        // Update coordinates
        coords = coords + delta_coords_.reshape({B, N, S, 2}).permute({0, 2, 1, 3});
        coords.index_put_({torch::indexing::Slice(), 0}, coords_backup.index({torch::indexing::Slice(), 0}));

        // Store predictions
        if (down_ratio > 1) {
            coord_preds.push_back(coords * stride * down_ratio);
        } else {
            coord_preds.push_back(coords * stride);
        }
    }

    // Predict visibility
    auto vis_e = vis_predictor->forward(track_feats.reshape({B * S * N, latent_dim}))
                .reshape({B, S, N});
    if (apply_sigmoid) {
        vis_e = torch::sigmoid(vis_e);
    }

    // Predict confidence (if enabled)
    torch::Tensor conf_e;
    if (predict_conf) {
        conf_e = conf_predictor->forward(track_feats.reshape({B * S * N, latent_dim}))
                .reshape({B, S, N});
        if (apply_sigmoid) {
            conf_e = torch::sigmoid(conf_e);
        }
    }

    if (return_feat) {
        return {coord_preds, vis_e, track_feats, query_track_feat, conf_e};
    } else {
        return {coord_preds, vis_e, torch::Tensor(), torch::Tensor(), conf_e};
    }
}

} // namespace vggt
