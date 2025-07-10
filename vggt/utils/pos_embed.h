/**
 * @file pos_embed.h
 * @brief Positional embedding utility functions for VGGT
 */

#pragma once

#include <torch/torch.h>

namespace vggt {
namespace utils {

/**
 * @brief Convert 2D position grid to sinusoidal embeddings
 *
 * @param pos_grid Tensor of shape (H, W, 2) containing 2D coordinates
 * @param embed_dim Output channel dimension for embeddings
 * @param omega_0 Frequency scaling factor (default=100)
 * @return torch::Tensor of shape (H, W, embed_dim) with positional embeddings
 */
torch::Tensor position_grid_to_embed(
    const torch::Tensor& pos_grid,
    int64_t embed_dim,
    float omega_0 = 100.0f);

/**
 * @brief Generate 1D positional embedding using sine and cosine functions
 *
 * @param embed_dim The embedding dimension (must be even)
 * @param pos The position to generate the embedding from
 * @param omega_0 Frequency scaling factor (default=100)
 * @return torch::Tensor The generated 1D positional embedding
 */
torch::Tensor make_sincos_pos_embed(
    int64_t embed_dim,
    const torch::Tensor& pos,
    float omega_0 = 100.0f);

/**
 * @brief Create a normalized UV grid
 *
 * @param width Number of points horizontally
 * @param height Number of points vertically
 * @param aspect_ratio Width-to-height ratio (default=width/height)
 * @param dtype Data type of the resulting tensor
 * @param device Device on which the tensor is created
 * @return torch::Tensor A (width, height, 2) tensor of UV coordinates
 */
torch::Tensor create_uv_grid(
    int64_t width,
    int64_t height,
    c10::optional<float> aspect_ratio = c10::nullopt,
    c10::optional<torch::Dtype> dtype = c10::nullopt,
    c10::optional<torch::Device> device = c10::nullopt);

} // namespace utils
} // namespace vggt
