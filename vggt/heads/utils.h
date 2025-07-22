/**
 * @file utils.h
 * @brief Utility functions for vision transformer heads
 *
 * This file contains helper functions for positional embeddings and grid generation.
 */

#pragma once

#include <torch/torch.h>

namespace vggt {
namespace heads {

/**
 * @brief Convert 2D position grid to sinusoidal embeddings
 * 
 * @param pos_grid Tensor of shape (H, W, 2) containing 2D coordinates
 * @param embed_dim Output channel dimension for embeddings
 * @param omega_0 Frequency scaling factor (default: 100)
 * @return Tensor of shape (H, W, embed_dim) with positional embeddings
 */
torch::Tensor position_grid_to_embed(
    const torch::Tensor& pos_grid,
    int64_t embed_dim,
    double omega_0 = 100.0
);

/**
 * @brief Generate 1D positional embedding using sine and cosine functions
 * 
 * @param embed_dim The embedding dimension (must be even)
 * @param pos The position to generate the embedding from
 * @param omega_0 Frequency scaling factor (default: 100)
 * @return Tensor of shape (M, embed_dim) where M is the length of pos
 */
torch::Tensor make_sincos_pos_embed(
    int64_t embed_dim,
    const torch::Tensor& pos,
    double omega_0 = 100.0
);

/**
 * @brief Create a normalized UV grid of shape (width, height, 2)
 * 
 * The grid spans horizontally and vertically according to an aspect ratio,
 * ensuring the top-left corner is at (-x_span, -y_span) and the bottom-right
 * corner is at (x_span, y_span), normalized by the diagonal of the plane.
 * 
 * @param width Number of points horizontally
 * @param height Number of points vertically
 * @param aspect_ratio Width-to-height ratio (default: width/height)
 * @param dtype Data type of the resulting tensor (optional)
 * @param device Device on which the tensor is created (optional)
 * @return Tensor of shape (width, height, 2) with UV coordinates
 */
torch::Tensor create_uv_grid(
    int64_t width,
    int64_t height,
    c10::optional<double> aspect_ratio = c10::nullopt,
    c10::optional<torch::Dtype> dtype = c10::nullopt,
    c10::optional<torch::Device> device = c10::nullopt
);

} // namespace heads
} // namespace vggt