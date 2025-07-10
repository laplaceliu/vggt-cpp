/**
 * @file load_fn.h
 * @brief Image loading and preprocessing functions for VGGT
 */

#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace vggt {
namespace utils {

/**
 * @brief Load and preprocess images to make them square by center padding
 *
 * @param image_paths Vector of image file paths
 * @param target_size Target size for the output images
 * @param normalize Whether to normalize the images
 * @param mean Mean values for normalization (default: {0.485, 0.456, 0.406})
 * @param std Standard deviation values for normalization (default: {0.229, 0.224, 0.225})
 * @return torch::Tensor Tensor of preprocessed images of shape (N, C, H, W)
 */
torch::Tensor load_and_preprocess_images_square(
    const std::vector<std::string>& image_paths,
    int target_size,
    bool normalize = true,
    const std::vector<float>& mean = {0.485f, 0.456f, 0.406f},
    const std::vector<float>& std = {0.229f, 0.224f, 0.225f});

/**
 * @brief Quick start function to load and preprocess images for model input
 *
 * @param image_paths Vector of image file paths
 * @param target_size Target size for the output images
 * @param normalize Whether to normalize the images
 * @param mean Mean values for normalization (default: {0.485, 0.456, 0.406})
 * @param std Standard deviation values for normalization (default: {0.229, 0.224, 0.225})
 * @return torch::Tensor Tensor of preprocessed images of shape (N, C, H, W)
 */
torch::Tensor load_and_preprocess_images(
    const std::vector<std::string>& image_paths,
    int target_size,
    bool normalize = true,
    const std::vector<float>& mean = {0.485f, 0.456f, 0.406f},
    const std::vector<float>& std = {0.229f, 0.224f, 0.225f});

} // namespace utils
} // namespace vggt
