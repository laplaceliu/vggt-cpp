/**
 * @file base_track_predictor.h
 * @brief Base tracker predictor implementation
 */

#pragma once

#include <torch/torch.h>
#include <torch/nn/modules/container/any.h>
#include <vector>
#include <tuple>

namespace vggt {

class BaseTrackerPredictor : public torch::nn::Module {
public:
    /**
     * @brief Construct a new Base Tracker Predictor object
     *
     * @param stride Stride of the feature maps (default: 4)
     * @param corr_levels Number of correlation pyramid levels (default: 5)
     * @param corr_radius Correlation radius (default: 4)
     * @param latent_dim Latent dimension (default: 128)
     * @param hidden_size Hidden size (default: 384)
     * @param use_spaceatt Whether to use spatial attention (default: true)
     * @param depth Number of transformer layers (default: 6)
     * @param fine Whether this is a fine predictor (default: false)
     */
    BaseTrackerPredictor(
        int stride = 4,
        int corr_levels = 5,
        int corr_radius = 4,
        int latent_dim = 128,
        int hidden_size = 384,
        bool use_spaceatt = true,
        int depth = 6,
        bool fine = false
    );

    /**
     * @brief Forward pass for track prediction
     *
     * @param query_points Query points [B, N, 2] (xy coordinates)
     * @param fmaps Feature maps [B, S, C, HH, WW]
     * @param iters Number of iterations (default: 4)
     * @param return_feat Whether to return features (default: false)
     * @param down_ratio Downsampling ratio (default: 1)
     * @return std::tuple containing:
     *   - coord_preds: List of predicted coordinates at each iteration
     *   - vis_e: Visibility scores [B, S, N] (nullptr if fine=true)
     *   - track_feats: Track features (only if return_feat=true)
     *   - query_track_feat: Query track features (only if return_feat=true)
     */
    std::tuple<
        std::vector<torch::Tensor>,
        torch::Tensor,
        torch::Tensor,
        torch::Tensor
    > forward(
        torch::Tensor query_points,
        torch::Tensor fmaps,
        int iters = 4,
        bool return_feat = false,
        int down_ratio = 1
    );

private:
    int stride;
    int latent_dim;
    int corr_levels;
    int corr_radius;
    int hidden_size;
    bool fine;
    int flows_emb_dim;
    int transformer_dim;

    // Modules
    torch::nn::AnyModule updateformer;
    torch::nn::GroupNorm norm;
    torch::nn::AnyModule ffeat_updater;
    torch::nn::AnyModule vis_predictor;
};

} // namespace vggt
