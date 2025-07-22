/**
 * @file track_predict.h
 * @brief Functions for predicting tracks in a sequence of images
 * 
 * This file contains functions for predicting tracks in a sequence of images using feature extraction
 * and tracking algorithms. It is a C++ port of the Python implementation in track_predict.py.
 */

#pragma once

#include <torch/torch.h>
#include <c10/util/Optional.h>
#include <vector>
#include <string>
#include <memory>

namespace vggt {
namespace dependency {

/**
 * @brief Predict tracks for the given images and masks
 * 
 * @param images Tensor of shape [S, 3, H, W] containing the input images
 * @param conf Optional tensor of shape [S, 1, H, W] containing confidence scores
 * @param points_3d Optional tensor containing 3D points
 * @param masks Optional tensor of shape [S, 1, H, W] containing masks
 * @param max_query_pts Maximum number of query points (default: 2048)
 * @param query_frame_num Number of query frames to use (default: 5)
 * @param keypoint_extractor Method for keypoint extraction (default: "aliked+sp")
 * @param max_points_num Maximum number of points to process at once (default: 163840)
 * @param fine_tracking Whether to use fine tracking (default: true)
 * @param complete_non_vis Whether to augment non-visible frames (default: true)
 * @return std::tuple<torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>> 
 *         (pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors)
 */
std::tuple<torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>> 
predict_tracks(
    const torch::Tensor& images,
    const c10::optional<torch::Tensor>& conf = c10::nullopt,
    const c10::optional<torch::Tensor>& points_3d = c10::nullopt,
    const c10::optional<torch::Tensor>& masks = c10::nullopt,
    int max_query_pts = 2048,
    int query_frame_num = 5,
    const std::string& keypoint_extractor = "aliked+sp",
    int max_points_num = 163840,
    bool fine_tracking = true,
    bool complete_non_vis = true
);

/**
 * @brief Process a single query frame for track prediction
 * 
 * @param query_index Index of the query frame
 * @param images Tensor of shape [S, 3, H, W] containing the input images
 * @param conf Optional confidence tensor
 * @param points_3d Optional 3D points tensor
 * @param fmaps_for_tracker Feature maps for the tracker
 * @param keypoint_extractors Initialized feature extractors
 * @param tracker VGG-SFM tracker
 * @param max_points_num Maximum number of points to process at once
 * @param fine_tracking Whether to use fine tracking
 * @param device Device to use for computation
 * @return std::tuple<torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, torch::Tensor> 
 *         (pred_track, pred_vis, pred_conf, pred_point_3d, pred_color)
 */
std::tuple<torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, torch::Tensor> 
_forward_on_query(
    int query_index,
    const torch::Tensor& images,
    const c10::optional<torch::Tensor>& conf,
    const c10::optional<torch::Tensor>& points_3d,
    const torch::Tensor& fmaps_for_tracker,
    const std::unordered_map<std::string, torch::nn::Module>& keypoint_extractors,
    const torch::nn::Module& tracker,
    int max_points_num,
    bool fine_tracking,
    const torch::Device& device
);

/**
 * @brief Augment tracking for frames with insufficient visibility
 * 
 * @param pred_tracks List of numpy arrays containing predicted tracks
 * @param pred_vis_scores List of numpy arrays containing visibility scores
 * @param pred_confs List of numpy arrays containing confidence scores
 * @param pred_points_3d List of numpy arrays containing 3D points
 * @param pred_colors List of numpy arrays containing point colors
 * @param images Tensor of shape [S, 3, H, W] containing the input images
 * @param conf Optional confidence tensor
 * @param points_3d Optional 3D points tensor
 * @param fmaps_for_tracker Feature maps for the tracker
 * @param keypoint_extractors Initialized feature extractors
 * @param tracker VGG-SFM tracker
 * @param max_points_num Maximum number of points to process at once
 * @param fine_tracking Whether to use fine tracking
 * @param min_vis Minimum visibility threshold (default: 500)
 * @param non_vis_thresh Non-visibility threshold (default: 0.1)
 * @param device Device to use for computation
 * @return std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<c10::optional<torch::Tensor>>, 
 *         std::vector<c10::optional<torch::Tensor>>, std::vector<torch::Tensor>> 
 *         Updated pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, and pred_colors
 */
std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<c10::optional<torch::Tensor>>, 
    std::vector<c10::optional<torch::Tensor>>, std::vector<torch::Tensor>> 
_augment_non_visible_frames(
    const std::vector<torch::Tensor>& pred_tracks,
    const std::vector<torch::Tensor>& pred_vis_scores,
    const std::vector<c10::optional<torch::Tensor>>& pred_confs,
    const std::vector<c10::optional<torch::Tensor>>& pred_points_3d,
    const std::vector<torch::Tensor>& pred_colors,
    const torch::Tensor& images,
    const c10::optional<torch::Tensor>& conf,
    const c10::optional<torch::Tensor>& points_3d,
    const torch::Tensor& fmaps_for_tracker,
    const std::unordered_map<std::string, torch::nn::Module>& keypoint_extractors,
    const torch::nn::Module& tracker,
    int max_points_num,
    bool fine_tracking,
    int min_vis = 500,
    float non_vis_thresh = 0.1,
    const torch::Device& device = torch::kCPU
);

/**
 * @brief Initialize feature extractors that can be reused
 * 
 * @param max_query_num Maximum number of keypoints to extract
 * @param det_thres Detection threshold for keypoint extraction (default: 0.005)
 * @param extractor_method String specifying which extractors to use (default: "aliked")
 * @param device Device to run extraction on (default: "cuda")
 * @return std::unordered_map<std::string, torch::nn::Module> Dictionary of initialized extractors
 */
std::unordered_map<std::string, torch::nn::Module> 
initialize_feature_extractors(
    int max_query_num, 
    float det_thres = 0.005, 
    const std::string& extractor_method = "aliked", 
    const torch::Device& device = torch::kCUDA
);

/**
 * @brief Extract keypoints using pre-initialized feature extractors
 * 
 * @param query_image Input image tensor (3xHxW, range [0, 1])
 * @param extractors Dictionary of initialized extractors
 * @param round_keypoints Whether to round keypoint coordinates (default: true)
 * @return torch::Tensor Tensor of keypoint coordinates (1xNx2)
 */
torch::Tensor 
extract_keypoints(
    const torch::Tensor& query_image, 
    const std::unordered_map<std::string, torch::nn::Module>& extractors, 
    bool round_keypoints = true
);

} // namespace dependency
} // namespace vggt