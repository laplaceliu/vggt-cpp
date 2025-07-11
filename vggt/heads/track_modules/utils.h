/**
 * @file utils.h
 * @brief Utility functions for tracking heads
 */

#pragma once

#include <torch/torch.h>
#include <vector>

namespace vggt {

/**
 * @brief Generate 2D sine-cosine positional embedding
 *
 * @param embed_dim Embedding dimension
 * @param grid_size Grid size (either single value or pair)
 * @param return_grid Whether to return the grid along with embedding
 * @return std::tuple<torch::Tensor, torch::Tensor> if return_grid is true, else torch::Tensor
 */
torch::Tensor get_2d_sincos_pos_embed(
    int embed_dim,
    const std::vector<int64_t>& grid_size,
    bool return_grid = false
);

/**
 * @brief Generate 2D sine-cosine positional embedding from grid
 *
 * @param embed_dim Embedding dimension
 * @param grid Input grid tensor
 * @return torch::Tensor Generated positional embedding
 */
torch::Tensor get_2d_sincos_pos_embed_from_grid(
    int embed_dim,
    const torch::Tensor& grid
);

/**
 * @brief Generate 1D sine-cosine positional embedding from grid
 *
 * @param embed_dim Embedding dimension
 * @param pos Position tensor
 * @return torch::Tensor Generated positional embedding
 */
torch::Tensor get_1d_sincos_pos_embed_from_grid(
    int embed_dim,
    const torch::Tensor& pos
);

/**
 * @brief Generate 2D positional embedding from coordinates
 *
 * @param xy Input coordinates tensor [B, N, 2]
 * @param C Embedding size
 * @param cat_coords Whether to concatenate original coordinates
 * @return torch::Tensor Generated positional embedding
 */
torch::Tensor get_2d_embedding(
    const torch::Tensor& xy,
    int C,
    bool cat_coords = true
);

/**
 * @brief Bilinear sampler for tensor sampling
 *
 * @param input Input tensor [B, C, H, W] or [B, C, T, H, W]
 * @param coords Coordinates tensor [B, H_o, W_o, 2] or [B, T_o, H_o, W_o, 3]
 * @param align_corners Coordinate convention
 * @param padding_mode Padding mode
 * @return torch::Tensor Sampled tensor
 */
torch::Tensor bilinear_sampler(
    const torch::Tensor& input,
    const torch::Tensor& coords,
    bool align_corners = true,
    const std::string& padding_mode = "border"
);

/**
 * @brief Sample spatial features from 4D tensor
 *
 * @param input Input tensor [B, C, H, W]
 * @param coords Coordinates tensor [B, R, 2]
 * @return torch::Tensor Sampled features [B, R, C]
 */
torch::Tensor sample_features4d(
    const torch::Tensor& input,
    const torch::Tensor& coords
);

} // namespace vggt
