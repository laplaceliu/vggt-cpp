#pragma once

#include <torch/torch.h>
#include <vector>
#include <unordered_map>

#include "aggregator.h"
#include "../heads/camera_head.h"
// #include "../heads/dpt_head.h"  // TODO: DPTHead not fully implemented yet
#include "../heads/track_head.h"

namespace vggt {
namespace models {

/**
 * VGGT (Visual Geometry Grounded Transformer) model
 *
 * This model estimates camera poses, depth maps, and point tracking
 * from image sequences. It combines multiple specialized heads to handle
 * different visual geometry tasks.
 */
class VGGTImpl : public torch::nn::Module {
public:
    VGGTImpl(
        int64_t img_size = 518,
        int64_t patch_size = 14,
        int64_t embed_dim = 1024
    );

    /**
     * Forward pass of the VGGT model.
     *
     * @param images Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1]
     * @param query_points Optional query points for tracking, shape [N, 2] or [B, N, 2]
     * @return Dictionary containing predictions:
     *         - pose_enc: Camera pose encoding [B, S, 9]
     *         - depth: Predicted depth maps [B, S, H, W, 1]
     *         - depth_conf: Confidence scores [B, S, H, W]
     *         - world_points: 3D world coordinates [B, S, H, W, 3]
     *         - world_points_conf: Confidence scores [B, S, H, W]
     *         - track: Point tracks [B, S, N, 2] (if query_points provided)
     *         - vis: Visibility scores [B, S, N] (if query_points provided)
     *         - conf: Confidence scores [B, S, N] (if query_points provided)
     *         - images: Original input images
     */
    std::unordered_map<std::string, torch::Tensor> forward(
        torch::Tensor images,
        torch::Tensor query_points = {}
    );

private:
    // Core feature aggregator
    Aggregator aggregator_{nullptr};

    // Prediction heads
    heads::CameraHead camera_head_{nullptr};
    // TODO: heads::DPTHead point_head_{nullptr};  // For 3D world points
    // TODO: heads::DPTHead depth_head_{nullptr};  // For depth
    heads::TrackHead track_head_{nullptr}; // For point tracking
};

TORCH_MODULE(VGGT);

} // namespace models
} // namespace vggt
