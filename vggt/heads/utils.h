#pragma once

/**
 * @file utils.h
 * @brief Utility functions for VGGT prediction heads
 *
 * Provides position embedding and grid creation utilities for
 * camera pose and depth prediction heads.
 */

#include <torch/torch.h>

namespace vggt {
namespace heads {

/**
 * @brief Convert 2D position grid to sinusoidal embedding
 *
 * Takes a 2D grid of (x, y) positions and converts it to a sinusoidal
 * positional embedding, processing x and y coordinates separately and
 * concatenating the results.
 *
 * @param pos_grid Position grid of shape [H, W, 2] containing (x, y) coordinates
 * @param embed_dim Output embedding dimension (must be even)
 * @param omega_0 Base frequency for sinusoidal encoding (default: 100.0)
 * @return Embedding tensor of shape [H, W, embed_dim]
 * @throws c10::Error if pos_grid doesn't have shape [H, W, 2]
 */
torch::Tensor position_grid_to_embed(const torch::Tensor& pos_grid, int64_t embed_dim, float omega_0 = 100.0f);

/**
 * @brief Create 2D sincos positional embedding from coordinates
 *
 * Generates sinusoidal position embeddings following the approach used
 * in DETR and Masked Autoencoder (MAE).
 *
 * @param embed_dim Embedding dimension (must be even)
 * @param pos Position values of shape [M] (1D array of positions)
 * @param omega_0 Base frequency (default: 100.0)
 * @return Embedding tensor of shape [M, embed_dim]
 * @throws c10::Error if embed_dim is not even
 */
torch::Tensor make_sincos_pos_embed(int64_t embed_dim, const torch::Tensor& pos, float omega_0 = 100.0f);

/**
 * @brief Create a normalized UV coordinate grid
 *
 * Generates a grid of normalized UV coordinates suitable for
 * grid sampling operations. The coordinates are normalized to
 * maintain aspect ratio.
 *
 * @param width Grid width
 * @param height Grid height
 * @param aspect_ratio Aspect ratio override. If < 0, computed as width/height (default: -1.0)
 * @param dtype Optional data type override
 * @param device Optional device override
 * @return UV grid of shape [height, width, 2] containing (u, v) coordinates
 */
torch::Tensor create_uv_grid(
    int64_t width,
    int64_t height,
    float aspect_ratio = -1.0f,
    c10::optional<torch::Dtype> dtype = c10::nullopt,
    c10::optional<torch::Device> device = c10::nullopt);

} // namespace heads
} // namespace vggt