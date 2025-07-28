#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace vggt {
namespace utils {

/**
 * Load and preprocess images by center padding to square and resizing to target size.
 * Also returns the position information of original pixels after transformation.
 *
 * @param image_path_list List of paths to image files
 * @param target_size Target size for both width and height. Defaults to 1024.
 * @return Tuple of (batched tensor of preprocessed images, tensor with original coordinates)
 * @throws std::invalid_argument If the input list is empty
 */
std::tuple<torch::Tensor, torch::Tensor> load_and_preprocess_images_square(
    const std::vector<std::string>& image_path_list, 
    int target_size = 1024);

/**
 * A quick start function to load and preprocess images for model input.
 * This assumes the images should have the same shape for easier batching, but the model can also work well with different shapes.
 *
 * @param image_path_list List of paths to image files
 * @param mode Preprocessing mode, either "crop" or "pad".
 *             - "crop" (default): Sets width to 518px and center crops height if needed.
 *             - "pad": Preserves all pixels by making the largest dimension 518px
 *               and padding the smaller dimension to reach a square shape.
 * @return Batched tensor of preprocessed images with shape (N, 3, H, W)
 * @throws std::invalid_argument If the input list is empty or if mode is invalid
 */
torch::Tensor load_and_preprocess_images(
    const std::vector<std::string>& image_path_list, 
    const std::string& mode = "crop");

} // namespace utils
} // namespace vggt