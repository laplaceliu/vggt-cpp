#pragma once

#include <torch/torch.h>
#include <vector>

#include "../dependency/track_modules/base_track_predictor.h"

namespace vggt {
namespace heads {

/**
 * Track head that uses DPT head to process tokens and BaseTrackerPredictor for tracking.
 * The tracking is performed iteratively, refining predictions over multiple iterations.
 */
class TrackHeadImpl : public torch::nn::Module {
public:
    TrackHeadImpl(
        int64_t dim_in = 2048,
        int64_t patch_size = 14,
        int64_t features = 128,
        int64_t iters = 4,
        bool predict_conf = true,
        int64_t stride = 2,
        int64_t corr_levels = 7,
        int64_t corr_radius = 4,
        int64_t hidden_size = 384
    );

    /**
     * Forward pass of the TrackHead.
     *
     * @param aggregated_tokens_list List of aggregated tokens from the backbone
     * @param images Input images of shape (B, S, C, H, W)
     * @param patch_start_idx Starting index for patch tokens
     * @param query_points Initial query points to track, shape [B, N, 2]
     * @param iters Number of refinement iterations. If None, uses self.iters
     * @return Tuple of (coord_preds, vis_scores, conf_scores)
     *         - coord_preds: Vector of predicted coordinates for each iteration
     *         - vis_scores: Visibility scores for tracked points
     *         - conf_scores: Confidence scores for tracked points
     */
    std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor> forward(
        const std::vector<torch::Tensor>& aggregated_tokens_list,
        torch::Tensor images,
        int64_t patch_start_idx,
        torch::Tensor query_points,
        int64_t iters = -1
    );

private:
    int64_t patch_size_;
    int64_t iters_;

    // Feature extractor (DPT-based)
    // Note: DPTHead needs to be fully implemented for feature extraction
    // For now, we use a placeholder that will be replaced when DPTHead is complete

    // Tracker module
    dependency::track_modules::BaseTrackerPredictor tracker_{nullptr};
};

TORCH_MODULE(TrackHead);

} // namespace heads
} // namespace vggt
