/**
 * @file utils.h
 * @brief Utility functions for heads
 */

#pragma once

#include <torch/torch.h>
#include <string>

namespace vggt {

/**
 * @brief Bilinear sampler for grid sampling
 *
 * @param x Input tensor (B, C, H, W)
 * @param grid Sampling grid (B, H', W', 2)
 * @param padding_mode Padding mode ("zeros", "border", "reflection")
 * @return torch::Tensor Sampled tensor (B, C, H', W')
 */
torch::Tensor bilinear_sampler(
    const torch::Tensor& x,
    const torch::Tensor& grid,
    const std::string& padding_mode = "zeros");

} // namespace vggt
