#pragma once

#include <torch/torch.h>

namespace vggt {
namespace heads {

torch::Tensor position_grid_to_embed(const torch::Tensor& pos_grid, int64_t embed_dim, float omega_0 = 100.0f);

torch::Tensor make_sincos_pos_embed(int64_t embed_dim, const torch::Tensor& pos, float omega_0 = 100.0f);

torch::Tensor create_uv_grid(
    int64_t width,
    int64_t height,
    float aspect_ratio = -1.0f,
    c10::optional<torch::Dtype> dtype = c10::nullopt,
    c10::optional<torch::Device> device = c10::nullopt);

} // namespace heads
} // namespace vggt