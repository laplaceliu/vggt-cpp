/**
 * @file track_head.h
 * @brief Track head for point tracking
 */

#pragma once

#include <torch/torch.h>
#include "track_modules/base_track_predictor.h"
#include "dpt_head.h"

namespace vggt {

class TrackHeadImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new TrackHeadImpl object
     *
     * @param hidden_dim Hidden dimension size
     * @param corr_levels Number of correlation pyramid levels
     * @param corr_radius Correlation search radius
     * @param iters Number of refinement iterations
     * @param multiple_track_feats Whether to split target features per level
     * @param padding_mode Padding mode for sampling
     */
    TrackHeadImpl(
        int64_t hidden_dim = 128,
        int64_t corr_levels = 4,
        int64_t corr_radius = 4,
        int64_t iters = 3,
        bool multiple_track_feats = false,
        const std::string& padding_mode = "zeros");

    /**
     * @brief Forward pass for point tracking
     *
     * @param x Input features from backbone
     * @param targets Target features (B, S, N, C)
     * @param coords Initial coordinates (B, S, N, 2)
     * @param visibility Initial visibility (B, S, N, 1)
     * @param confidence Initial confidence (B, S, N, 1)
     * @param mask Mask for valid points (B, S, N)
     * @return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
     *         Predicted coordinates, visibility, and confidence
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        const std::vector<torch::Tensor>& x,
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
    DPTHead feature_extractor_{nullptr};
    track_modules::BaseTrackerPredictor tracker_predictor_{nullptr};
};

TORCH_MODULE(TrackHead);

} // namespace vggt
