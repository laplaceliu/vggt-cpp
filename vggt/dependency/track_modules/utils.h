#pragma once

#include <torch/torch.h>

namespace vggt {
namespace dependency {
namespace track_modules {

// 2D sine-cosine positional embedding
torch::Tensor get_2d_sincos_pos_embed(int embed_dim, const std::pair<int, int>& grid_size, bool return_grid = false);
torch::Tensor get_2d_sincos_pos_embed_from_grid(int embed_dim, const torch::Tensor& grid);
torch::Tensor get_1d_sincos_pos_embed_from_grid(int embed_dim, const torch::Tensor& pos);

// 2D embedding from coordinates
torch::Tensor get_2d_embedding(const torch::Tensor& xy, int C, bool cat_coords = true);

// Bilinear sampling
torch::Tensor bilinear_sampler(const torch::Tensor& input, const torch::Tensor& coords, bool align_corners = true, const torch::nn::functional::GridSampleFuncOptions::padding_mode_t& padding_mode = torch::kBorder);

// Sample spatial features
torch::Tensor sample_features4d(const torch::Tensor& input, const torch::Tensor& coords);

} // namespace track_modules
} // namespace dependency
} // namespace vggt
