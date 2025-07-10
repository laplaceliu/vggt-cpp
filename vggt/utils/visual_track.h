/**
 * @file visual_track.h
 * @brief Visualization utilities for tracking results
 */

#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace vggt {
namespace utils {

/**
 * @brief Draw tracking results on images
 *
 * @param images Batch of images tensor of shape (B, C, H, W)
 * @param boxes Bounding boxes tensor of shape (B, N, 4) in format [x1, y1, x2, y2]
 * @param scores Confidence scores tensor of shape (B, N)
 * @param labels Class labels tensor of shape (B, N)
 * @param class_names Vector of class names
 * @param score_threshold Threshold for filtering low confidence detections
 * @return std::vector<cv::Mat> Vector of images with drawn tracking results
 */
std::vector<cv::Mat> draw_tracking_results(
    const torch::Tensor& images,
    const torch::Tensor& boxes,
    const torch::Tensor& scores,
    const torch::Tensor& labels,
    const std::vector<std::string>& class_names,
    float score_threshold = 0.5);

/**
 * @brief Draw bounding boxes on an image
 *
 * @param image OpenCV image
 * @param boxes Bounding boxes tensor of shape (N, 4) in format [x1, y1, x2, y2]
 * @param scores Confidence scores tensor of shape (N)
 * @param labels Class labels tensor of shape (N)
 * @param class_names Vector of class names
 * @param score_threshold Threshold for filtering low confidence detections
 * @return cv::Mat Image with drawn bounding boxes
 */
cv::Mat draw_boxes(
    const cv::Mat& image,
    const torch::Tensor& boxes,
    const torch::Tensor& scores,
    const torch::Tensor& labels,
    const std::vector<std::string>& class_names,
    float score_threshold = 0.5);

/**
 * @brief Convert tensor image to OpenCV image
 *
 * @param tensor_image Tensor image of shape (C, H, W)
 * @param normalize Whether the tensor is normalized
 * @param mean Mean values used for normalization (default: {0.485, 0.456, 0.406})
 * @param std Standard deviation values used for normalization (default: {0.229, 0.224, 0.225})
 * @return cv::Mat OpenCV image in BGR format
 */
cv::Mat tensor_to_cv_image(
    const torch::Tensor& tensor_image,
    bool normalize = true,
    const std::vector<float>& mean = {0.485f, 0.456f, 0.406f},
    const std::vector<float>& std = {0.229f, 0.224f, 0.225f});

/**
 * @brief Save tracking visualization as video
 *
 * @param frames Vector of frames to save
 * @param output_path Output video file path
 * @param fps Frames per second
 * @return bool True if successful, false otherwise
 */
bool save_tracking_video(
    const std::vector<cv::Mat>& frames,
    const std::string& output_path,
    int fps = 30);

} // namespace utils
} // namespace vggt
