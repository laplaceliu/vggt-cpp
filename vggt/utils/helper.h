#pragma once

#include <torch/torch.h>

namespace vggt {
namespace utils {

/**
 * If mask has more than max_trues True values,
 * randomly keep only max_trues of them and set the rest to False.
 */
torch::Tensor randomly_limit_trues(const torch::Tensor& mask, int64_t max_trues);

/**
 * Creates a grid of pixel coordinates and frame indices for all frames.
 * Returns:
 *     A tuple containing:
 *         - points_xyf: Tensor of shape (num_frames, height, width, 3)
 *                       with x, y coordinates and frame indices
 *         - y_coords: Tensor of y coordinates for all frames
 *         - x_coords: Tensor of x coordinates for all frames
 *         - f_coords: Tensor of frame indices for all frames
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
create_pixel_coordinate_grid(int64_t num_frames, int64_t height, int64_t width);

} // namespace utils
} // namespace vggt