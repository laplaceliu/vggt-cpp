/**
 * @file vggt.h
 * @brief VGGT (Visual Geometry and Global Tracking) model implementation
 *
 * This file defines the VGGT class which is used to estimate camera pose, depth maps,
 * and point tracking from image sequences.
 */

#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

#include "aggregator.h"
#include "../heads/camera_head.h"
#include "../heads/dpt_head.h"
#include "../heads/track_head.h"

namespace vggt {
namespace models {

/**
 * @brief VGGT (Visual Geometry and Global Tracking) model
 *
 * This model is used to estimate camera pose, depth maps, and point tracking from image sequences.
 * It combines multiple specialized head networks to handle different visual geometry tasks.
 */
class VGGTImpl : public torch::nn::Module {
public:
    /**
     * @brief Constructor for VGGT model
     *
     * @param img_size Input image size, default is 518
     * @param patch_size Image patch size, default is 14
     * @param embed_dim Embedding dimension, default is 1024
     */
    VGGTImpl(
        int64_t img_size = 518,
        int64_t patch_size = 14,
        int64_t embed_dim = 1024
    );

    /**
     * @brief Forward pass of the VGGT model
     *
     * @param images Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1]
     * @param query_points Optional query points for tracking, in pixel coordinates
     * @return Dictionary containing predictions for camera pose, depth, world points, and tracking
     */
    std::unordered_map<std::string, torch::Tensor> forward(
        const torch::Tensor& images,
        const c10::optional<torch::Tensor>& query_points = c10::nullopt
    );

private:
    Aggregator aggregator_;
    heads::CameraHead camera_head_;
    heads::DPTHead point_head_;
    heads::DPTHead depth_head_;
    heads::TrackHead track_head_;
};

TORCH_MODULE(VGGT);

} // namespace models
} // namespace vggt
