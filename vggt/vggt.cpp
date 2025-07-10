/**
 * @file vggt.cpp
 * @brief Implementation of VGGT (Visual Geometry and Graph Tracking) model
 */

#include "vggt.h"
#include "aggregator.h"
#include "heads/camera_head.h"
#include "heads/dpt_head.h"
#include "heads/track_head.h"

#include <torch/torch.h>
#include <map>
#include <string>

namespace vggt {

VGGTImpl::VGGTImpl(int img_size, int patch_size, int embed_dim) {
    // Initialize aggregator
    aggregator = Aggregator(
        img_size, patch_size, embed_dim,
        24,  // depth
        16,  // num_heads
        4.0f,  // mlp_ratio
        4,  // num_register_tokens
        true,  // qkv_bias
        true,  // proj_bias
        true,  // ffn_bias
        "dinov2_vitl14_reg",  // patch_embed
        {"frame", "global"},  // aa_order
        1,  // aa_block_size
        true,  // qk_norm
        100,  // rope_freq
        0.01f  // init_values
    );

    // Initialize heads
    camera_head = CameraHead(embed_dim);
    point_head = DPTHead(embed_dim, 3);  // 3D points output
    depth_head = DPTHead(embed_dim, 1);  // Depth output
    track_head = TrackHead(embed_dim);

    // Register all submodules
    register_module("aggregator", aggregator);
    register_module("camera_head", camera_head);
    register_module("point_head", point_head);
    register_module("depth_head", depth_head);
    register_module("track_head", track_head);
}

std::map<std::string, torch::Tensor> VGGTImpl::forward(
    torch::Tensor images,
    torch::Tensor query_points
) {
    // Ensure input is 5D [B, S, 3, H, W]
    if (images.dim() == 4) {
        images = images.unsqueeze(0);  // Add batch dimension
    }

    // Get batch and sequence dimensions
    int B = images.size(0);
    int S = images.size(1);

    // Process through aggregator
    auto [aggregator_outputs, patch_start_idx] = aggregator->forward(images);

    // Get last layer outputs for prediction
    auto last_output = aggregator_outputs.back();
    int P = last_output.size(2) - patch_start_idx;  // Number of patch tokens

    // Process through heads
    auto pose_enc = camera_head->forward(last_output);
    auto depth = depth_head->forward(last_output);
    auto world_points = point_head->forward(last_output);

    // Prepare result map
    std::map<std::string, torch::Tensor> result;
    result["pose_enc"] = pose_enc;
    result["depth"] = depth;
    result["world_points"] = world_points;
    result["images"] = images;

    // Add confidence maps
    result["depth_conf"] = torch::ones_like(depth[..., 0]);
    result["world_points_conf"] = torch::ones_like(world_points[..., 0]);

    // Process tracking if query points provided
    if (query_points.defined()) {
        auto [track, vis, conf] = track_head->forward(last_output, query_points);
        result["track"] = track;
        result["vis"] = vis;
        result["conf"] = conf;
    }

    return result;
}

} // namespace vggt
