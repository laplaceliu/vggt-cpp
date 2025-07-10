/**
 * @file base_track_predictor.h
 * @brief Base tracker predictor for point tracking
 */

#pragma once

#include <torch/torch.h>
#include "corr_block.h"
#include "efficient_update_former.h"

namespace vggt {
namespace track_modules {

class BaseTrackerPredictorImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new BaseTrackerPredictorImpl object
     *
     * @param hidden_dim Hidden dimension size
     * @param corr_levels Number of correlation pyramid levels
     * @param corr_radius Correlation search radius
     * @param iters Number of refinement iterations
     * @param multiple_track_feats Whether to split target features per level
     * @param padding_mode Padding mode for sampling
     */
    BaseTrackerPredictorImpl(
        int64_t hidden_dim = 128,
        int64_t corr_levels = 4,
        int64_t corr_radius = 4,
        int64_t iters = 3,
        bool multiple_track_feats = false,
        const std::string& padding_mode = "zeros");

    /**
     * @brief Forward pass for point tracking
     *
     * @param fmaps Feature maps (B, S, C, H, W)
     * @param targets Target features (B, S, N, C)
     * @param coords Initial coordinates (B, S, N, 2)
     * @param visibility Initial visibility (B, S, N, 1)
     * @param confidence Initial confidence (B, S, N, 1)
     * @param mask Mask for valid points (B, S, N)
     * @return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
     *         Predicted coordinates, visibility, and confidence
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& fmaps,
        const torch::Tensor& targets,
        const torch::Tensor& coords,
        const torch::Tensor& visibility = {},
        const torch::Tensor& confidence = {},
        const torch::Tensor& mask = {});

private:
    int64_t hidden_dim_;
    int64_t corr_levels_;
    int64_t corr_radius_;
    int64_t iters_;
    bool multiple_track_feats_;
    std::string padding_mode_;

    // Network components
    EfficientUpdateFormer update_block_{nullptr};
    torch::nn::Linear corr_embed_{nullptr};
    torch::nn::Linear flow_embed_{nullptr};
    torch::nn::Linear vis_embed_{nullptr};
    torch::nn::Linear conf_embed_{nullptr};
    torch::nn::Linear coord_predictor_{nullptr};
    torch::nn::Linear vis_predictor_{nullptr};
    torch::nn::Linear conf_predictor_{nullptr};

    /**
     * @brief Generate 2D position embeddings
     *
     * @param coords Coordinates (B, S, N, 2)
     * @param H Feature height
     * @param W Feature width
     * @return torch::Tensor Position embeddings (B, S, N, hidden_dim)
     */
    torch::Tensor get_2d_embedding(
        const torch::Tensor& coords,
        int64_t H,
        int64_t W);
};

TORCH_MODULE(BaseTrackerPredictor);

} // namespace track_modules
} // namespace vggt
