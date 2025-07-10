/**
 * @file helper.h
 * @brief Helper utility functions for VGGT
 */

#pragma once

#include <torch/torch.h>

namespace vggt {
namespace utils {

/**
 * @brief Randomly limit the number of true values in a mask
 *
 * If the number of true values in the mask exceeds max_num_trues,
 * randomly select max_num_trues of them and set the rest to false.
 *
 * @param mask Boolean mask tensor
 * @param max_num_trues Maximum number of true values to keep
 * @return torch::Tensor Modified mask with limited number of true values
 */
torch::Tensor randomly_limit_trues(const torch::Tensor& mask, int64_t max_num_trues);

/**
 * @brief Create pixel coordinate grid and frame indices
 *
 * @param batch_size Batch size
 * @param height Image height
 * @param width Image width
 * @param device Device to create tensors on
 * @return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Tuple of (pixel_x, pixel_y, frame_idx)
 *         where pixel_x and pixel_y are of shape (batch_size, height, width)
 *         and frame_idx is of shape (batch_size, 1, 1)
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> create_pixel_coordinate_grid(
    int64_t batch_size,
    int64_t height,
    int64_t width,
    torch::Device device = torch::kCPU);

} // namespace utils
} // namespace vggt
