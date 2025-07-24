#include "helper.h"
#include <ATen/ATen.h>
#include <ATen/core/TensorBody.h>
#include <c10/util/Optional.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/types.h>

namespace vggt {
namespace utils {

torch::Tensor randomly_limit_trues(const torch::Tensor& mask, int64_t max_trues) {
    // 1D positions of all True entries
    auto true_indices = torch::nonzero(mask).squeeze();

    // if already within budget, return as-is
    if (true_indices.size(0) <= max_trues) {
        return mask.clone();
    }

    // randomly pick which True positions to keep
    auto sampled_indices = torch::randperm(true_indices.size(0), torch::kLong).slice(0, 0, max_trues);
    auto selected_indices = true_indices.index_select(0, sampled_indices);

    // build new flat mask: True only at sampled positions
    auto limited_flat_mask = torch::zeros_like(mask.flatten(), torch::kBool);
    limited_flat_mask.index_put_({selected_indices}, torch::ones_like(selected_indices, torch::kBool));

    // restore original shape
    return limited_flat_mask.reshape(mask.sizes());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
create_pixel_coordinate_grid(int64_t num_frames, int64_t height, int64_t width) {
    // Create coordinate grids for a single frame
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto y_grid = torch::arange(height, options).view({1, height, 1}).expand({1, height, width});
    auto x_grid = torch::arange(width, options).view({1, 1, width}).expand({1, height, width});

    // Broadcast to all frames
    auto x_coords = x_grid.expand({num_frames, height, width});
    auto y_coords = y_grid.expand({num_frames, height, width});

    // Create frame indices and broadcast
    auto f_idx = torch::arange(num_frames, options).view({num_frames, 1, 1}).expand({num_frames, height, width});

    // Stack coordinates and frame indices
    auto points_xyf = torch::stack({x_coords, y_coords, f_idx}, -1);

    return {points_xyf, y_coords, x_coords, f_idx};
}

} // namespace utils
} // namespace vggt