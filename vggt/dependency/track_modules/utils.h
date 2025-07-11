/**
 * @file utils.h
 * @brief Utility functions for tracking modules
 *
 * Modified from https://github.com/facebookresearch/PoseDiffusion
 * and https://github.com/facebookresearch/co-tracker/tree/main
 */

#pragma once

#include <torch/torch.h>
#include <tuple>

namespace vggt {

/**
 * @brief Generate 2D sinusoidal positional embeddings
 *
 * @param embed_dim The embedding dimension (must be even)
 * @param grid_size The grid size (can be int or tuple of ints)
 * @param return_grid Whether to return the grid along with the embeddings
 * @return torch::Tensor or std::tuple<torch::Tensor, torch::Tensor> Positional embeddings (and grid if return_grid is true)
 */
std::variant<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> get_2d_sincos_pos_embed(
    int embed_dim,
    std::variant<int, std::tuple<int, int>> grid_size,
    bool return_grid = false
);

/**
 * @brief Generate 2D sinusoidal positional embeddings from a grid
 *
 * @param embed_dim The embedding dimension (must be even)
 * @param grid The grid to generate embeddings from
 * @return torch::Tensor The positional embeddings
 */
torch::Tensor get_2d_sincos_pos_embed_from_grid(int embed_dim, torch::Tensor grid);

/**
 * @brief Generate 1D sinusoidal positional embeddings from a grid
 *
 * @param embed_dim The embedding dimension (must be even)
 * @param pos The positions to generate embeddings from
 * @return torch::Tensor The positional embeddings
 */
torch::Tensor get_1d_sincos_pos_embed_from_grid(int embed_dim, torch::Tensor pos);

/**
 * @brief Generate 2D positional embeddings from coordinates
 *
 * @param xy The coordinates to generate embeddings from (B, N, 2)
 * @param C The size of the embedding
 * @param cat_coords Whether to concatenate the original coordinates to the embedding
 * @return torch::Tensor The positional embeddings
 */
torch::Tensor get_2d_embedding(torch::Tensor xy, int C, bool cat_coords = true);

/**
 * @brief Sample a tensor using bilinear interpolation
 *
 * @param input The input tensor (B, C, H, W) or (B, C, T, H, W)
 * @param coords The coordinates to sample at (B, H_o, W_o, 2) or (B, H_o, W_o, 3)
 * @param align_corners Whether to align corners
 * @param padding_mode The padding mode ("zeros", "border", or "reflection")
 * @return torch::Tensor The sampled tensor
 */
torch::Tensor bilinear_sampler(
    torch::Tensor input,
    torch::Tensor coords,
    bool align_corners = true,
    const std::string& padding_mode = "border"
);

/**
 * @brief Sample spatial features
 *
 * @param input The input tensor (B, C, H, W)
 * @param coords The coordinates to sample at (B, R, 2)
 * @return torch::Tensor The sampled features (B, R, C)
 */
torch::Tensor sample_features4d(torch::Tensor input, torch::Tensor coords);

} // namespace vggt
