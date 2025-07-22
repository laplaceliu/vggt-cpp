/**
 * @file vggt.cpp
 * @brief Implementation of VGGT (Visual Geometry and Global Tracking) model
 *
 * This file implements the VGGT class which is used to estimate camera pose, depth maps,
 * and point tracking from image sequences.
 */

#include "vggt.h"
#include <iostream>

namespace vggt {
namespace models {

VGGTImpl::VGGTImpl(
    int64_t img_size,
    int64_t patch_size,
    int64_t embed_dim
) {
    // Initialize feature aggregator for extracting and aggregating features from input images
    aggregator_ = Aggregator(
        AggregatorOptions()
            .img_size(img_size)
            .patch_size(patch_size)
            .embed_dim(embed_dim)
    );
    
    // Initialize camera pose estimation head for predicting camera pose encoding
    camera_head_ = heads::CameraHead(
        heads::CameraHeadOptions()
            .dim_in(2 * embed_dim)
    );
    
    // Initialize 3D point prediction head for predicting 3D world coordinates for each pixel
    point_head_ = heads::DPTHead(
        heads::DPTHeadOptions()
            .dim_in(2 * embed_dim)
            .output_dim(4)
            .activation("inv_log")
            .conf_activation("expp1")
    );
    
    // Initialize depth prediction head for predicting depth maps
    depth_head_ = heads::DPTHead(
        heads::DPTHeadOptions()
            .dim_in(2 * embed_dim)
            .output_dim(2)
            .activation("exp")
            .conf_activation("expp1")
    );
    
    // Initialize point tracking head for tracking specified points in image sequences
    track_head_ = heads::TrackHead(
        heads::TrackHeadOptions()
            .dim_in(2 * embed_dim)
            .patch_size(patch_size)
    );
    
    // Register the modules
    register_module("aggregator", aggregator_);
    register_module("camera_head", camera_head_);
    register_module("point_head", point_head_);
    register_module("depth_head", depth_head_);
    register_module("track_head", track_head_);
}

std::unordered_map<std::string, torch::Tensor> VGGTImpl::forward(
    const torch::Tensor& images,
    const c10::optional<torch::Tensor>& query_points
) {
    // If without batch dimension, add it
    torch::Tensor batch_images;
    if (images.dim() == 4) {
        batch_images = images.unsqueeze(0);
    } else {
        batch_images = images;
    }
    
    c10::optional<torch::Tensor> batch_query_points = query_points;
    if (query_points.has_value() && query_points.value().dim() == 2) {
        batch_query_points = query_points.value().unsqueeze(0);
    }
    
    // Use aggregator to process input images, get aggregated feature tokens and patch start index
    auto aggregator_result = aggregator_->forward(batch_images);
    auto aggregated_tokens_list = std::get<0>(aggregator_result);
    auto patch_start_idx = std::get<1>(aggregator_result);
    
    // Initialize prediction results dictionary
    std::unordered_map<std::string, torch::Tensor> predictions;
    
    // Disable automatic mixed precision to ensure computation precision
    // Note: In LibTorch, we don't have direct equivalent to torch.cuda.amp.autocast
    // So we'll just proceed with the computations
    
    if (camera_head_) {
        auto pose_enc_list = camera_head_->forward(aggregated_tokens_list);
        // Store pose encoding of the last iteration
        predictions["pose_enc"] = pose_enc_list.back();
    }
    
    if (depth_head_) {
        auto depth_result = depth_head_->forward(aggregated_tokens_list, batch_images, patch_start_idx);
        predictions["depth"] = std::get<0>(depth_result);      // Depth prediction
        predictions["depth_conf"] = std::get<1>(depth_result); // Depth prediction confidence
    }
    
    if (point_head_) {
        auto pts3d_result = point_head_->forward(aggregated_tokens_list, batch_images, patch_start_idx);
        predictions["world_points"] = std::get<0>(pts3d_result);      // 3D world coordinate points
        predictions["world_points_conf"] = std::get<1>(pts3d_result); // World coordinate points confidence
    }
    
    // If query points are provided, perform point tracking
    if (track_head_ && batch_query_points.has_value()) {
        auto track_result = track_head_->forward(
            aggregated_tokens_list, 
            batch_images, 
            patch_start_idx, 
            batch_query_points.value()
        );
        
        // Track of the last iteration
        predictions["track"] = std::get<0>(track_result).back();
        predictions["vis"] = std::get<1>(track_result);   // Tracked points visibility
        predictions["conf"] = std::get<2>(track_result);  // Tracked points confidence
    }
    
    // Save original input images for visualization
    predictions["images"] = batch_images;
    
    return predictions;
}

} // namespace models
} // namespace vggt