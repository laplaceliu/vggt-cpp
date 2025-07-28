#include "utils.h"

namespace vggt {
namespace heads {

torch::Tensor position_grid_to_embed(const torch::Tensor& pos_grid, int64_t embed_dim, float omega_0) {
    auto H = pos_grid.size(0);
    auto W = pos_grid.size(1);
    auto grid_dim = pos_grid.size(2);
    TORCH_CHECK(grid_dim == 2, "pos_grid must have shape (H, W, 2)");

    auto pos_flat = pos_grid.reshape({-1, grid_dim}); // Flatten to (H*W, 2)

    // Process x and y coordinates separately
    auto emb_x = make_sincos_pos_embed(embed_dim / 2, pos_flat.index({torch::indexing::Slice(), 0}), omega_0); // [1, H*W, D/2]
    auto emb_y = make_sincos_pos_embed(embed_dim / 2, pos_flat.index({torch::indexing::Slice(), 1}), omega_0); // [1, H*W, D/2]

    // Combine and reshape
    auto emb = torch::cat({emb_x, emb_y}, -1); // [1, H*W, D]

    return emb.view({H, W, embed_dim}); // [H, W, D]
}

torch::Tensor make_sincos_pos_embed(int64_t embed_dim, const torch::Tensor& pos, float omega_0) {
    TORCH_CHECK(embed_dim % 2 == 0, "embed_dim must be even");
    auto device = pos.device();
    auto omega = torch::arange(embed_dim / 2, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    omega /= embed_dim / 2.0;
    omega = 1.0 / torch::pow(omega_0, omega); // (D/2,)

    auto pos_flat = pos.reshape({-1}); // (M,)
    auto out = torch::einsum("m,d->md", {pos_flat, omega}); // (M, D/2), outer product

    auto emb_sin = torch::sin(out); // (M, D/2)
    auto emb_cos = torch::cos(out); // (M, D/2)

    auto emb = torch::cat({emb_sin, emb_cos}, 1); // (M, D)
    return emb.to(torch::kFloat32);
}

torch::Tensor create_uv_grid(
    int64_t width,
    int64_t height,
    float aspect_ratio,
    c10::optional<torch::Dtype> dtype,
    c10::optional<torch::Device> device) {
    if (aspect_ratio < 0.0f) {
        aspect_ratio = static_cast<float>(width) / static_cast<float>(height);
    }

    // Compute normalized spans for X and Y
    float diag_factor = std::sqrt(aspect_ratio * aspect_ratio + 1.0f);
    float span_x = aspect_ratio / diag_factor;
    float span_y = 1.0f / diag_factor;

    // Establish the linspace boundaries
    float left_x = -span_x * (width - 1) / width;
    float right_x = span_x * (width - 1) / width;
    float top_y = -span_y * (height - 1) / height;
    float bottom_y = span_y * (height - 1) / height;

    // Generate 1D coordinates
    auto x_coords = torch::linspace(left_x, right_x, width, torch::TensorOptions().dtype(dtype).device(device));
    auto y_coords = torch::linspace(top_y, bottom_y, height, torch::TensorOptions().dtype(dtype).device(device));

    // Create 2D meshgrid (width x height) and stack into UV
    auto mesh = torch::meshgrid({x_coords, y_coords}, "xy");
    auto uv_grid = torch::stack({mesh[0], mesh[1]}, -1);

    return uv_grid;
}

} // namespace heads
} // namespace vggt