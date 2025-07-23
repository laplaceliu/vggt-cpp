#include "utils.h"

namespace vggt {
namespace dependency {
namespace track_modules {

// 2D sine-cosine positional embedding
torch::Tensor get_2d_sincos_pos_embed(int embed_dim, const std::pair<int, int>& grid_size, bool return_grid) {
    int grid_size_h = grid_size.first;
    int grid_size_w = grid_size.second;

    auto grid_h = torch::arange(grid_size_h, torch::kFloat);
    auto grid_w = torch::arange(grid_size_w, torch::kFloat);
    auto grid = torch::meshgrid({grid_w, grid_h}, "xy");
    auto stacked_grid = torch::stack({grid[0], grid[1]}, 0);
    stacked_grid = stacked_grid.reshape({2, 1, grid_size_h, grid_size_w});
    auto grid_tensor = stacked_grid;

    auto pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, stacked_grid);
    if (return_grid) {
        return torch::cat({pos_embed.reshape({1, grid_size_h, grid_size_w, -1}).permute({0, 3, 1, 2}), stacked_grid}, 0);
    }
    return pos_embed.reshape({1, grid_size_h, grid_size_w, -1}).permute({0, 3, 1, 2});
}

torch::Tensor get_2d_sincos_pos_embed_from_grid(int embed_dim, const torch::Tensor& grid) {
    assert(embed_dim % 2 == 0);

    auto emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim / 2, grid[0]); // (H*W, D/2)
    auto emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim / 2, grid[1]); // (H*W, D/2)

    return torch::cat({emb_h, emb_w}, 2); // (H*W, D)
}

torch::Tensor get_1d_sincos_pos_embed_from_grid(int embed_dim, const torch::Tensor& pos) {
    assert(embed_dim % 2 == 0);

    auto omega = torch::arange(embed_dim / 2, torch::kDouble);
    omega /= embed_dim / 2.0;
    omega = 1.0 / torch::pow(10000.0, omega); // (D/2,)

    auto pos_flat = pos.reshape({-1}); // (M,)
    auto out = torch::einsum("m,d->md", {pos_flat, omega}); // (M, D/2), outer product

    auto emb_sin = torch::sin(out); // (M, D/2)
    auto emb_cos = torch::cos(out); // (M, D/2)

    return torch::cat({emb_sin, emb_cos}, 1).unsqueeze(0).to(torch::kFloat); // (M, D)
}

// 2D embedding from coordinates
torch::Tensor get_2d_embedding(const torch::Tensor& xy, int C, bool cat_coords) {
    auto B = xy.size(0);
    auto N = xy.size(1);
    assert(xy.size(2) == 2);

    auto x = xy.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}).unsqueeze(2);
    auto y = xy.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}).unsqueeze(2);

    auto div_term = torch::arange(0, C, 2, torch::kFloat32).reshape({1, 1, C / 2}).to(xy.device());
    div_term = div_term * (1000.0 / C);

    auto pe_x = torch::zeros({B, N, C}, torch::kFloat32).to(xy.device());
    pe_x.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, C, 2)}, torch::sin(x * div_term));
    pe_x.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, C, 2)}, torch::cos(x * div_term));

    auto pe_y = torch::zeros({B, N, C}, torch::kFloat32).to(xy.device());
    pe_y.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, C, 2)}, torch::sin(y * div_term));
    pe_y.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, C, 2)}, torch::cos(y * div_term));

    auto pe = torch::cat({pe_x, pe_y}, 2); // (B, N, C*2)
    if (cat_coords) {
        pe = torch::cat({xy, pe}, 2); // (B, N, C*2+2)
    }
    return pe;
}

// Bilinear sampling
torch::Tensor bilinear_sampler(const torch::Tensor& input, const torch::Tensor& coords, bool align_corners, const torch::nn::functional::GridSampleFuncOptions::padding_mode_t& padding_mode) {
    auto sizes = input.sizes().slice(2);
    assert(sizes.size() == 2 || sizes.size() == 3);

    auto coords_modified = coords;
    if (sizes.size() == 3) {
        // t x y -> x y t to match dimensions T H W in grid_sample
        coords_modified = coords_modified.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, 3)});
        coords_modified = torch::cat({coords_modified, coords_modified.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(), 0}).unsqueeze(3)}, 3);
    }

    if (align_corners) {
        auto scale_factors = torch::tensor({2.0 / (sizes[1] - 1), 2.0 / (sizes[0] - 1)}, torch::kFloat32).to(coords.device());
        coords_modified = coords_modified * scale_factors;
    } else {
        auto scale_factors = torch::tensor({2.0 / sizes[1], 2.0 / sizes[0]}, torch::kFloat32).to(coords.device());
        coords_modified = coords_modified * scale_factors;
    }

    coords_modified = coords_modified - 1.0;

    return torch::nn::functional::grid_sample(input, coords_modified, torch::nn::functional::GridSampleFuncOptions().align_corners(align_corners).padding_mode(padding_mode));
}

// Sample spatial features
torch::Tensor sample_features4d(const torch::Tensor& input, const torch::Tensor& coords) {
    auto B = input.size(0);
    auto C = input.size(1);

    // B R 2 -> B R 1 2
    auto coords_reshaped = coords.unsqueeze(2);

    // B C R 1
    auto feats = bilinear_sampler(input, coords_reshaped);

    return feats.permute({0, 2, 1, 3}).reshape({B, -1, C * feats.size(3)}); // B R C
}

} // namespace track_modules
} // namespace dependency
} // namespace vggt
