/**
 * @file utils.cpp
 * @brief Implementation of utility functions for vision transformer heads
 */

#include "utils.h"

namespace vggt {
namespace heads {

torch::Tensor make_sincos_pos_embed(
    int64_t embed_dim,
    const torch::Tensor& pos,
    double omega_0
) {
    TORCH_CHECK(embed_dim % 2 == 0, "embed_dim must be even");
    
    auto device = pos.device();
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat64)
        .device(device);
    
    // Create omega vector
    auto omega = torch::arange(embed_dim / 2, options);
    omega = omega / (embed_dim / 2.0);
    omega = 1.0 / torch::pow(omega_0, omega);
    
    // Reshape pos and compute outer product
    auto pos_flat = pos.reshape({-1});
    auto out = torch::einsum("m,d->md", {pos_flat, omega});
    
    // Compute sine and cosine embeddings
    auto emb_sin = torch::sin(out);
    auto emb_cos = torch::cos(out);
    
    // Concatenate and convert to float32
    auto emb = torch::cat({emb_sin, emb_cos}, 1);
    return emb.to(torch::kFloat32);
}

torch::Tensor position_grid_to_embed(
    const torch::Tensor& pos_grid,
    int64_t embed_dim,
    double omega_0
) {
    auto sizes = pos_grid.sizes();
    TORCH_CHECK(sizes.size() == 3 && sizes[2] == 2, "pos_grid must be HxWx2");
    
    // Flatten grid and process x/y coordinates
    auto pos_flat = pos_grid.reshape({-1, 2});
    auto emb_x = make_sincos_pos_embed(embed_dim / 2, pos_flat.index({torch::indexing::Slice(), 0}), omega_0);
    auto emb_y = make_sincos_pos_embed(embed_dim / 2, pos_flat.index({torch::indexing::Slice(), 1}), omega_0);
    
    // Combine and reshape
    auto emb = torch::cat({emb_x, emb_y}, -1);
    return emb.reshape({sizes[0], sizes[1], embed_dim});
}

torch::Tensor create_uv_grid(
    int64_t width,
    int64_t height,
    c10::optional<double> aspect_ratio,
    c10::optional<torch::Dtype> dtype,
    c10::optional<torch::Device> device
) {
    // Derive aspect ratio if not provided
    double aspect = aspect_ratio.value_or(static_cast<double>(width) / height);
    
    // Compute normalized spans for X and Y
    double diag_factor = std::sqrt(aspect * aspect + 1.0);
    double span_x = aspect / diag_factor;
    double span_y = 1.0 / diag_factor;
    
    // Establish the linspace boundaries
    double left_x = -span_x * (width - 1) / width;
    double right_x = span_x * (width - 1) / width;
    double top_y = -span_y * (height - 1) / height;
    double bottom_y = span_y * (height - 1) / height;
    
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

} // namespace heads
} // namespace vggt