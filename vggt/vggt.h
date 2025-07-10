/**
 * @file vggt.h
 * @brief Main header file for VGGT (Visual Geometry and Graph Tracking) library
 */

#pragma once

#include <torch/torch.h>

// Include all utility headers
#include "utils/geometry.h"
#include "utils/helper.h"
#include "utils/load_fn.h"
#include "utils/pose_enc.h"
#include "utils/rotation.h"
#include "utils/visual_track.h"

// Include layer headers
#include "layers/attention.h"
#include "layers/block.h"
#include "layers/drop_path.h"
#include "layers/layer_scale.h"
#include "layers/mlp.h"
#include "layers/patch_embed.h"
#include "layers/rope.h"
#include "layers/swiglu_ffn.h"
#include "layers/vision_transformer.h"

// Forward declarations
class AggregatorImpl;
class CameraHeadImpl;
class DPTHeadImpl;
class TrackHeadImpl;

/**
 * @namespace vggt
 * @brief Main namespace for VGGT (Visual Geometry and Graph Tracking) library
 */
namespace vggt {

/**
 * @brief Get the version of VGGT library
 * @return std::string Version string
 */
inline std::string version() {
    return "0.1.0";
}

/**
 * @class VGGT
 * @brief VGGT (Visual Geometry and Global Tracking) model implementation
 *
 * This model is used for estimating camera pose, depth maps and point tracking from image sequences.
 * It combines multiple specialized head networks to handle different visual geometry tasks.
 */
class VGGTImpl : public torch::nn::Module {
public:
    /**
     * @brief Constructor for VGGT model
     * @param img_size Image size in pixels, default 518
     * @param patch_size Size of each patch for PatchEmbed, default 14
     * @param embed_dim Dimension of the token embeddings, default 1024
     */
    VGGTImpl(
        int img_size = 518,
        int patch_size = 14,
        int embed_dim = 1024
    );

    /**
     * @brief Forward pass of the VGGT model
     * @param images Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1]
     * @param query_points Query points for tracking, in pixel coordinates (optional)
     * @return std::map containing predictions:
     *   - pose_enc: Camera pose encoding with shape [B, S, 9] (from last iteration)
     *   - depth: Predicted depth maps with shape [B, S, H, W, 1]
     *   - depth_conf: Confidence scores for depth predictions with shape [B, S, H, W]
     *   - world_points: 3D world coordinates for each pixel with shape [B, S, H, W, 3]
     *   - world_points_conf: Confidence scores for world points with shape [B, S, H, W]
     *   - images: Original input images, preserved for visualization
     *   If query_points provided, also includes:
     *   - track: Point tracks with shape [B, S, N, 2] (from last iteration)
     *   - vis: Visibility scores for tracked points with shape [B, S, N]
     *   - conf: Confidence scores for tracked points with shape [B, S, N]
     */
    std::map<std::string, torch::Tensor> forward(
        torch::Tensor images,
        torch::Tensor query_points = torch::Tensor()
    );

private:
    torch::nn::ModuleHolder<AggregatorImpl> aggregator{nullptr};
    torch::nn::ModuleHolder<CameraHeadImpl> camera_head{nullptr};
    torch::nn::ModuleHolder<DPTHeadImpl> point_head{nullptr};
    torch::nn::ModuleHolder<DPTHeadImpl> depth_head{nullptr};
    torch::nn::ModuleHolder<TrackHeadImpl> track_head{nullptr};
};

TORCH_MODULE(VGGT);

} // namespace vggt
