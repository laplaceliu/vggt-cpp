/**
 * @file helper.cpp
 * @brief Implementation of helper utility functions for VGGT
 */

#include "helper.h"
#include <torch/torch.h>
#include <random>

namespace vggt {
namespace utils {

torch::Tensor randomly_limit_trues(const torch::Tensor& mask, int64_t max_num_trues) {
    // Count number of true values
    auto num_trues = mask.sum().item<int64_t>();

    // If already within limit, return original mask
    if (num_trues <= max_num_trues) {
        return mask.clone();
    }

    // Create a new mask with the same shape
    auto limited_mask = torch::zeros_like(mask);

    // Get indices of true values
    auto true_indices = torch::nonzero(mask).squeeze();

    // Randomly shuffle the indices
    auto shuffled_indices = torch::randperm(true_indices.size(0),
                                           torch::TensorOptions().device(mask.device()));
    auto selected_indices = true_indices.index_select(0, shuffled_indices.narrow(0, 0, max_num_trues));

    // Set selected indices to true in the new mask
    limited_mask.index_put_({selected_indices}, torch::ones({max_num_trues},
                           torch::TensorOptions().device(mask.device()).dtype(mask.dtype())));

    return limited_mask;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> create_pixel_coordinate_grid(
    int64_t batch_size,
    int64_t height,
    int64_t width,
    torch::Device device) {

    // Create coordinate ranges
    auto y_range = torch::arange(height, torch::TensorOptions().device(device));
    auto x_range = torch::arange(width, torch::TensorOptions().device(device));

    // Create meshgrid
    auto y_grid = y_range.view({-1, 1}).expand({height, width});
    auto x_grid = x_range.view({1, -1}).expand({height, width});

    // Expand for batch dimension
    auto pixel_y = y_grid.unsqueeze(0).expand({batch_size, height, width});
    auto pixel_x = x_grid.unsqueeze(0).expand({batch_size, height, width});

    // Create frame indices
    auto frame_idx = torch::arange(batch_size, torch::TensorOptions().device(device))
                        .view({batch_size, 1, 1});

    return std::make_tuple(pixel_x, pixel_y, frame_idx);
}

} // namespace utils
} // namespace vggt
