/**
 * @file pos_embed.cpp
 * @brief Implementation of positional embedding utility functions for VGGT
 */

#include "pos_embed.h"

#include <cmath>
#include <stdexcept>

namespace vggt {
namespace utils {

torch::Tensor position_grid_to_embed(
    const torch::Tensor& pos_grid,
    int64_t embed_dim,
    float omega_0) {
    // Check input dimensions
    TORCH_CHECK(pos_grid.dim() == 3, "pos_grid must be 3D (H, W, 2)");
    TORCH_CHECK(pos_grid.size(2) == 2, "pos_grid last dim must be 2");
    TORCH_CHECK(embed_dim > 0, "embed_dim must be positive");

    auto H = pos_grid.size(0);
    auto W = pos_grid.size(1);

    // Flatten the position grid
    auto pos_flat = pos_grid.reshape({-1, 2});  // (H*W, 2)

    // Process x and y coordinates separately
    auto emb_x = make_sincos_pos_embed(embed_dim / 2, pos_flat.index({torch::indexing::Slice(), 0}), omega_0);
    auto emb_y = make_sincos_pos_embed(embed_dim / 2, pos_flat.index({torch::indexing::Slice(), 1}), omega_0);

    // Combine and reshape
    auto emb = torch::cat({emb_x, emb_y}, -1);  // (H*W, D)
    return emb.view({H, W, embed_dim});  // (H, W, D)
}

torch::Tensor make_sincos_pos_embed(
    int64_t embed_dim,
    const torch::Tensor& pos,
    float omega_0) {
    TORCH_CHECK(embed_dim % 2 == 0, "embed_dim must be even");
    TORCH_CHECK(pos.dim() == 1, "pos must be 1D");

    auto device = pos.device();
    auto dtype = torch::kFloat32;

    // Create omega vector
    auto omega = torch::arange(embed_dim / 2, torch::TensorOptions().dtype(dtype).device(device));
    omega = omega / (embed_dim / 2.0);
    omega = 1.0 / torch::pow(omega_0, omega);  // (D/2,)

    // Outer product of pos and omega
    auto out = torch::einsum("m,d->md", {pos, omega});  // (M, D/2)

    // Compute sin and cos embeddings
    auto emb_sin = torch::sin(out);  // (M, D/2)
    auto emb_cos = torch::cos(out);  // (M, D/2)

    // Concatenate results
    return torch::cat({emb_sin, emb_cos}, 1).to(torch::kFloat32);  // (M, D)
}

torch::Tensor create_uv_grid(
    int64_t width,
    int64_t height,
    c10::optional<float> aspect_ratio,
    c10::optional<torch::Dtype> dtype,
    c10::optional<torch::Device> device) {
    TORCH_CHECK(width > 0, "width must be positive");
    TORCH_CHECK(height > 0, "height must be positive");

    // Calculate aspect ratio if not provided
    float aspect = aspect_ratio.value_or(static_cast<float>(width) / height);

    // Compute normalized spans for X and Y
    float diag_factor = std::sqrt(aspect * aspect + 1.0f);
    float span_x = aspect / diag_factor;
    float span_y = 1.0f / diag_factor;

    // Establish the linspace boundaries
    float left_x = -span_x * (width - 1) / width;
    float right_x = span_x * (width - 1) / width;
    float top_y = -span_y * (height - 1) / height;
    float bottom_y = span_y * (height - 1) / height;

    // Create tensor options
    auto options = torch::TensorOptions()
        .dtype(dtype.value_or(torch::kFloat32))
        .device(device.value_or(torch::kCPU));

    // Generate 1D coordinates
    auto x_coords = torch::linspace(left_x, right_x, width, options);
    auto y_coords = torch::linspace(top_y, bottom_y, height, options);

    // Create 2D meshgrid (width x height) and stack into UV
    auto mesh = torch::meshgrid({x_coords, y_coords}, "xy");
    auto uv_grid = torch::stack({mesh[0], mesh[1]}, -1);

    return uv_grid;
}

} // namespace utils
} // namespace vggt
