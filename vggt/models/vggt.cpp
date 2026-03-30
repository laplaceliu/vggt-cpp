#include "vggt.h"
#include "../heads/dpt_head.h"

namespace vggt {
namespace models {

VGGTImpl::VGGTImpl(
    int64_t img_size,
    int64_t patch_size,
    int64_t embed_dim
) {
    // Feature aggregator
    aggregator_ = Aggregator(img_size, patch_size, embed_dim);
    register_module("aggregator", aggregator_);

    // Camera head
    camera_head_ = heads::CameraHead(2 * embed_dim);
    register_module("camera_head", camera_head_);

    // Point head (for 3D world coordinates)
    // DPTHead with output_dim=4, activation="inv_log", conf_activation="expp1"
    std::vector<int64_t> out_channels = {256, 512, 1024, 1024};
    std::vector<int64_t> intermediate_layer_idx = {4, 11, 17, 23};
    point_head_ = heads::DPTHead(
        2 * embed_dim,
        patch_size,
        4,                    // output_dim (xyz + confidence)
        "inv_log",            // activation
        "expp1",              // conf_activation
        256,                  // features
        out_channels,          // out_channels
        intermediate_layer_idx, // intermediate_layer_idx
        true,                  // pos_embed
        false,                 // feature_only
        1                      // down_ratio
    );
    register_module("point_head", point_head_);

    // Depth head
    // DPTHead with output_dim=1, activation="exp", conf_activation="expp1"
    depth_head_ = heads::DPTHead(
        2 * embed_dim,
        patch_size,
        1,                    // output_dim (depth only)
        "exp",                // activation
        "expp1",              // conf_activation
        256,                  // features
        out_channels,          // out_channels
        intermediate_layer_idx, // intermediate_layer_idx
        true,                  // pos_embed
        false,                 // feature_only
        1                      // down_ratio
    );
    register_module("depth_head", depth_head_);

    // Track head
    track_head_ = heads::TrackHead(2 * embed_dim, patch_size);
    register_module("track_head", track_head_);
}

std::unordered_map<std::string, torch::Tensor> VGGTImpl::forward(
    torch::Tensor images,
    torch::Tensor query_points
) {
    // Add batch dimension if not present
    if (images.dim() == 4) {
        images = images.unsqueeze(0);
    }
    if (query_points.defined() && query_points.dim() == 2) {
        query_points = query_points.unsqueeze(0);
    }

    // Forward through aggregator
    auto [aggregated_tokens_list, patch_start_idx] = aggregator_->forward(images);

    std::unordered_map<std::string, torch::Tensor> predictions;

    // Camera head prediction
    if (!camera_head_.is_empty()) {
        auto pose_enc_list = camera_head_->forward(aggregated_tokens_list);
        // Take the last iteration's pose encoding
        predictions["pose_enc"] = pose_enc_list.back();
    }

    // Depth head prediction
    if (!depth_head_.is_empty()) {
        auto [depth, depth_conf] = depth_head_->forward(aggregated_tokens_list, images, patch_start_idx);
        predictions["depth"] = depth;
        predictions["depth_conf"] = depth_conf;
    }

    // Point head prediction
    if (!point_head_.is_empty()) {
        auto [pts3d, pts3d_conf] = point_head_->forward(aggregated_tokens_list, images, patch_start_idx);
        predictions["world_points"] = pts3d;
        predictions["world_points_conf"] = pts3d_conf;
    }

    // Track head prediction
    if (!track_head_.is_empty() && query_points.defined()) {
        auto [track_list, vis, conf] = track_head_->forward(
            aggregated_tokens_list, images, patch_start_idx, query_points);
        // Take the last iteration's track
        predictions["track"] = track_list.back();
        predictions["vis"] = vis;
        predictions["conf"] = conf;
    }

    // Store original images
    predictions["images"] = images;

    return predictions;
}

} // namespace models
} // namespace vggt
