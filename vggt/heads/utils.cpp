/**
 * @file utils.cpp
 * @brief Implementation of utility functions for heads
 */

#include "utils.h"
#include <torch/nn/functional.h>

namespace vggt {

torch::Tensor bilinear_sampler(
    const torch::Tensor& x,
    const torch::Tensor& grid,
    const std::string& padding_mode) {
    // Validate input dimensions
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (B, C, H, W)");
    TORCH_CHECK(grid.dim() == 4 && grid.size(-1) == 2,
                "Grid tensor must be 4D (B, H', W', 2)");

    // Convert padding mode string to enum
    torch::nn::functional::GridSampleFuncOptions::mode_t mode =
        torch::nn::functional::GridSampleFuncOptions::mode_t::Bilinear;

    torch::nn::functional::GridSampleFuncOptions::padding_mode_t pad_mode;
    if (padding_mode == "zeros") {
        pad_mode = torch::nn::functional::GridSampleFuncOptions::padding_mode_t::Zeros;
    } else if (padding_mode == "border") {
        pad_mode = torch::nn::functional::GridSampleFuncOptions::padding_mode_t::Border;
    } else if (padding_mode == "reflection") {
        pad_mode = torch::nn::functional::GridSampleFuncOptions::padding_mode_t::Reflection;
    } else {
        TORCH_CHECK(false, "Unsupported padding mode: ", padding_mode);
    }

    // Create grid sample options
    auto options = torch::nn::functional::GridSampleFuncOptions()
        .mode(mode)
        .padding_mode(pad_mode)
        .align_corners(true);

    // Apply grid sampling
    return torch::nn::functional::grid_sample(x, grid, options);
}

} // namespace vggt
