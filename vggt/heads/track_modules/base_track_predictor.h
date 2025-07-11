/**
 * @file base_track_predictor.h
 * @brief Base tracker predictor implementation
 */

#pragma once

#include <torch/torch.h>
#include <vector>
#include "blocks.h"
#include "modules.h"
#include "utils.h"

namespace vggt {

/**
 * @brief Base tracker predictor class
 */
class BaseTrackerPredictor : public torch::nn::Module {
public:
    /**
     * @brief Construct a new BaseTrackerPredictor object
     *
     * @param stride Feature map stride
     * @param corr_levels Number of correlation levels
     * @param corr_radius Correlation radius
     * @param latent_dim Latent dimension size
     * @param hidden_size Hidden size
     * @param use_spaceatt Whether to use space attention
     * @param depth Number of transformer blocks
     * @param max_scale Maximum scale for flow embedding
     * @param predict_conf Whether to predict confidence
     */
    BaseTrackerPredictor(
        int stride = 1,
        int corr_levels = 5,
        int corr_radius = 4,
        int latent_dim = 128,
        int hidden_size = 384,
        bool use_spaceatt = true,
        int depth = 6,
        int max_scale = 518,
        bool predict_conf = true
    );

    /**
     * @brief Forward pass
     *
     * @param query_points Query points tensor [B, N, 2]
     * @param fmaps Feature maps tensor [B, S, C, HH, WW]
     * @param iters Number of refinement iterations
     * @param return_feat Whether to return features
     * @param down_ratio Downsampling ratio
     * @param apply_sigmoid Whether to apply sigmoid to outputs
     * @return std::tuple containing:
     *         - coord_preds: List of coordinate predictions
     *         - vis_e: Visibility predictions
     *         - track_feats: Track features (if return_feat=true)
     *         - query_track_feat: Query track features (if return_feat=true)
     *         - conf_e: Confidence predictions (if predict_conf=true)
     */
    std::tuple<
        std::vector<torch::Tensor>,  // coord_preds
        torch::Tensor,               // vis_e
        torch::Tensor,               // track_feats (optional)
        torch::Tensor,               // query_track_feat (optional)
        torch::Tensor                // conf_e (optional)
    > forward(
        torch::Tensor query_points,
        torch::Tensor fmaps,
        int iters = 6,
        bool return_feat = false,
        int down_ratio = 1,
        bool apply_sigmoid = true
    );

private:
    int stride;
    int latent_dim;
    int corr_levels;
    int corr_radius;
    int hidden_size;
    int max_scale;
    bool predict_conf;
    int flows_emb_dim;
    int transformer_dim;

    torch::nn::LayerNorm fmap_norm{nullptr};
    torch::nn::GroupNorm ffeat_norm{nullptr};
    torch::nn::Sequential corr_mlp{nullptr};
    torch::nn::Sequential ffeat_updater{nullptr};
    torch::nn::Sequential vis_predictor{nullptr};
    torch::nn::Sequential conf_predictor{nullptr};
    EfficientUpdateFormer updateformer{nullptr};
    torch::Tensor query_ref_token;
};

} // namespace vggt
