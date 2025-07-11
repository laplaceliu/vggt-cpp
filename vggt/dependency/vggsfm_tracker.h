/**
 * @file vggsfm_tracker.h
 * @brief VGG-SFM tracker implementation
 */

#pragma once

#include <torch/torch.h>
#include <memory>

namespace vggt {

/**
 * @brief VGG-SFM tracker predictor
 */
class TrackerPredictor : public torch::nn::Module {
public:
    /**
     * @brief Construct a new Tracker Predictor object
     */
    TrackerPredictor();

    /**
     * @brief Forward pass for track prediction
     *
     * @param images Input images [B, S, 3, H, W] in range [0, 1]
     * @param query_points Query points [B, N, 2] (xy coordinates relative to top-left)
     * @param fmaps Optional precomputed feature maps
     * @param coarse_iters Number of iterations for coarse prediction (default: 6)
     * @param inference Whether to perform inference (default: true)
     * @param fine_tracking Whether to perform fine tracking (default: true)
     * @param fine_chunk Chunk size for fine tracking (default: 40960)
     * @return std::tuple containing:
     *   - fine_pred_track: Fine predicted tracks [B, S, N, 2]
     *   - coarse_pred_track: Coarse predicted tracks [B, S, N, 2]
     *   - pred_vis: Visibility scores [B, S, N]
     *   - pred_score: Confidence scores [B, S, N]
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
        torch::Tensor images,
        torch::Tensor query_points,
        torch::Tensor fmaps = {},
        int coarse_iters = 6,
        bool inference = true,
        bool fine_tracking = true,
        int fine_chunk = 40960
    );

    /**
     * @brief Process images to feature maps
     *
     * @param images Input images [S, 3, H, W]
     * @return torch::Tensor Processed feature maps
     */
    torch::Tensor process_images_to_fmaps(torch::Tensor images);

private:
    // Coarse feature network
    torch::nn::AnyModule coarse_fnet;
    // Coarse predictor
    torch::nn::AnyModule coarse_predictor;
    // Fine feature network
    torch::nn::AnyModule fine_fnet;
    // Fine predictor
    torch::nn::AnyModule fine_predictor;

    // Coarse down ratio
    int coarse_down_ratio = 2;
};

} // namespace vggt
