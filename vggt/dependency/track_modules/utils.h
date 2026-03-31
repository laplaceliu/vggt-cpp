#pragma once

/**
 * @file utils.h
 * @brief Utility functions for track modules in VGGT
 *
 * Provides positional embedding, bilinear sampling, and feature
 * extraction utilities for point tracking.
 */

#include <torch/torch.h>

namespace vggt {
namespace dependency {
namespace track_modules {

/**
 * @brief Create 2D sinusoidal positional embedding
 *
 * Generates a grid of sinusoidal position embeddings following
 * the approach used in MAE and DETR.
 *
 * @param embed_dim Embedding dimension (must be even)
 * @param grid_size Pair of (height, width) for the grid
 * @param return_grid If true, also return the coordinate grid
 * @return If return_grid=false: [1, embed_dim, H, W]
 *         If return_grid=true: tuple of (pos_embed [1, embed_dim, H, W], grid [2, 1, H, W])
 */
std::tuple<torch::Tensor, torch::Tensor> get_2d_sincos_pos_embed(int embed_dim, const std::pair<int, int>& grid_size, bool return_grid = false);

/**
 * @brief Create 2D sinusoidal positional embedding (simple version)
 *
 * @param embed_dim Embedding dimension (must be even)
 * @param grid_size Pair of (height, width) for the grid
 * @return [1, embed_dim, H, W]
 */
torch::Tensor get_2d_sincos_pos_embed_simple(int embed_dim, const std::pair<int, int>& grid_size);

/**
 * @brief Create 2D sinusoidal embedding from coordinate grid
 *
 * @param embed_dim Embedding dimension (must be even)
 * @param grid Coordinate grid of shape [2, H, W] with (x, y) coordinates
 * @return Embedding tensor of shape [1, H*W, embed_dim]
 */
torch::Tensor get_2d_sincos_pos_embed_from_grid(int embed_dim, const torch::Tensor& grid);

/**
 * @brief Create 1D sinusoidal embedding from positions
 *
 * @param embed_dim Embedding dimension (must be even)
 * @param pos Position values of shape [M]
 * @return Embedding tensor of shape [1, M, embed_dim]
 */
torch::Tensor get_1d_sincos_pos_embed_from_grid(int embed_dim, const torch::Tensor& pos);

/**
 * @brief Create 2D positional embedding from (x, y) coordinates
 *
 * Generates a frequency-based positional encoding for 2D coordinates
 * using sine and cosine functions with different frequencies.
 *
 * @param xy Input coordinates of shape [B, N, 2]
 * @param C Feature dimension for embedding
 * @param cat_coords If true, concatenate original coords with embedding
 * @return Embedding of shape [B, N, C*2] or [B, N, C*2+2] if cat_coords
 */
torch::Tensor get_2d_embedding(const torch::Tensor& xy, int C, bool cat_coords = true);

/**
 * @brief Bilinear sampling from input tensor using coordinates
 *
 * Performs bilinear grid sampling with configurable alignment
 * and padding modes.
 *
 * @param input Input tensor of shape [B, C, H, W]
 * @param coords Sampling coordinates of shape [B, H_out, W_out, 2] in grid format
 * @param align_corners If true, corner pixels are at corners
 * @param padding_mode Padding mode for out-of-bounds coordinates
 * @return Sampled tensor of same shape as input
 */
torch::Tensor bilinear_sampler(const torch::Tensor& input, const torch::Tensor& coords, bool align_corners = true, const torch::nn::functional::GridSampleFuncOptions::padding_mode_t& padding_mode = torch::kBorder);

/**
 * @brief Sample spatial features at given coordinates
 *
 * Extracts features from input tensor at specified 2D coordinates
 * using bilinear sampling.
 *
 * @param input Input features of shape [B, C, H, W]
 * @param coords 2D coordinates of shape [B, R, 2] where R is number of points
 * @return Sampled features of shape [B, R, C]
 */
torch::Tensor sample_features4d(const torch::Tensor& input, const torch::Tensor& coords);

} // namespace track_modules
} // namespace dependency
} // namespace vggt
