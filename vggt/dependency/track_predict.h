/**
 * @file track_predict.h
 * @brief Functions for predicting tracks in image sequences
 */

#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>

namespace vggt {

/**
 * @brief Predict tracks for the given images
 *
 * @param images Tensor of shape [S, 3, H, W] containing input images
 * @param conf Optional tensor of shape [S, 1, H, W] containing confidence scores
 * @param points_3d Optional tensor containing 3D points
 * @param masks Optional tensor of shape [S, 1, H, W] containing masks
 * @param max_query_pts Maximum number of query points (default: 2048)
 * @param query_frame_num Number of query frames to use (default: 5)
 * @param keypoint_extractor Method for keypoint extraction (default: "aliked+sp")
 * @param max_points_num Maximum number of points to process at once (default: 163840)
 * @param fine_tracking Whether to use fine tracking (default: true)
 * @param complete_non_vis Whether to augment non-visible frames (default: true)
 * @return std::tuple containing:
 *   - pred_tracks: Predicted tracks
 *   - pred_vis_scores: Visibility scores for the tracks
 *   - pred_confs: Confidence scores for the tracks
 *   - pred_points_3d: 3D points for the tracks
 *   - pred_colors: Point colors for the tracks (0-255)
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
predict_tracks(
    torch::Tensor images,
    torch::Tensor conf = {},
    torch::Tensor points_3d = {},
    torch::Tensor masks = {},
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
 * @param images Tensor of shape [S, 3, H, W] containing input images
 * @param conf Optional tensor containing confidence scores
 * @param points_3d Optional tensor containing 3D points
 * @param fmaps_for_tracker Feature maps for the tracker
 * @param keypoint_extractors Initialized feature extractors
 * @param tracker VGG-SFM tracker
 * @param max_points_num Maximum number of points to process at once
 * @param fine_tracking Whether to use fine tracking
 * @return std::tuple containing:
 *   - pred_track: Predicted tracks
 *   - pred_vis: Visibility scores
 *   - pred_conf: Confidence scores
 *   - pred_point_3d: 3D points
 *   - pred_color: Point colors (0-255)
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
forward_on_query(
    int query_index,
    torch::Tensor images,
    torch::Tensor conf,
    torch::Tensor points_3d,
    torch::Tensor fmaps_for_tracker,
    const torch::Tensor& keypoint_extractors,
    const torch::Tensor& tracker,
    int max_points_num,
    bool fine_tracking
);

/**
 * @brief Augment tracking for frames with insufficient visibility
 *
 * @param pred_tracks List of predicted tracks
 * @param pred_vis_scores List of visibility scores
 * @param pred_confs List of confidence scores
 * @param pred_points_3d List of 3D points
 * @param pred_colors List of point colors
 * @param images Tensor of shape [S, 3, H, W] containing input images
 * @param conf Optional tensor containing confidence scores
 * @param points_3d Optional tensor containing 3D points
 * @param fmaps_for_tracker Feature maps for the tracker
 * @param keypoint_extractors Initialized feature extractors
 * @param tracker VGG-SFM tracker
 * @param max_points_num Maximum number of points to process at once
 * @param fine_tracking Whether to use fine tracking
 * @param min_vis Minimum visibility threshold (default: 500)
 * @param non_vis_thresh Non-visibility threshold (default: 0.1)
 * @return std::tuple containing updated lists
 */
std::tuple<
    std::vector<torch::Tensor>,
    std::vector<torch::Tensor>,
    std::vector<torch::Tensor>,
    std::vector<torch::Tensor>,
    std::vector<torch::Tensor>
>
augment_non_visible_frames(
    std::vector<torch::Tensor>& pred_tracks,
    std::vector<torch::Tensor>& pred_vis_scores,
    std::vector<torch::Tensor>& pred_confs,
    std::vector<torch::Tensor>& pred_points_3d,
    std::vector<torch::Tensor>& pred_colors,
    torch::Tensor images,
    torch::Tensor conf,
    torch::Tensor points_3d,
    torch::Tensor fmaps_for_tracker,
    const torch::Tensor& keypoint_extractors,
    const torch::Tensor& tracker,
    int max_points_num,
    bool fine_tracking,
    int min_vis = 500,
    float non_vis_thresh = 0.1f
);

} // namespace vggt
