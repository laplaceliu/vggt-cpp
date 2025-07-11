/**
 * @file utils.cpp
 * @brief Implementation of utility functions for tracking heads
 */

#include "utils.h"
#include <stdexcept>
#include <cmath>

namespace vggt {

torch::Tensor get_2d_sincos_pos_embed(
    int embed_dim,
    const std::vector<int64_t>& grid_size,
    bool return_grid
) {
    if (grid_size.size() == 1) {
        return get_2d_sincos_pos_embed(embed_dim, {grid_size[0], grid_size[0]}, return_grid);
    } else if (grid_size.size() != 2) {
        throw std::invalid_argument("grid_size must have 1 or 2 elements");
    }

    int64_t grid_size_h = grid_size[0];
    int64_t grid_size_w = grid_size[1];

    auto grid_h = torch::arange(grid_size_h, torch::kFloat);
    auto grid_w = torch::arange(grid_size_w, torch::kFloat);
    auto grids = torch::meshgrid({grid_w, grid_h}, "xy");
    auto grid = torch::stack(grids, 0);
    grid = grid.reshape({2, 1, grid_size_h, grid_size_w});

    auto pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid);
    pos_embed = pos_embed.reshape({1, grid_size_h, grid_size_w, -1}).permute({0, 3, 1, 2});

    if (return_grid) {
        return torch::stack({pos_embed, grid}, 0);
    }
    return pos_embed;
}

torch::Tensor get_2d_sincos_pos_embed_from_grid(
    int embed_dim,
    const torch::Tensor& grid
) {
    if (embed_dim % 2 != 0) {
        throw std::invalid_argument("embed_dim must be even");
    }

    auto emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim / 2, grid[0]);
    auto emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim / 2, grid[1]);

    return torch::cat({emb_h, emb_w}, 2);
}

torch::Tensor get_1d_sincos_pos_embed_from_grid(
    int embed_dim,
    const torch::Tensor& pos
) {
    if (embed_dim % 2 != 0) {
        throw std::invalid_argument("embed_dim must be even");
    }

    auto omega = torch::arange(embed_dim / 2, torch::kDouble);
    omega = omega / (embed_dim / 2.0);
    omega = 1.0 / torch::pow(10000.0, omega);

    auto pos_flat = pos.reshape({-1});
    auto out = torch::einsum("m,d->md", {pos_flat, omega});

    auto emb_sin = torch::sin(out);
    auto emb_cos = torch::cos(out);

    auto emb = torch::cat({emb_sin, emb_cos}, 1);
    return emb[0].to(torch::kFloat);
}

torch::Tensor get_2d_embedding(
    const torch::Tensor& xy,
    int C,
    bool cat_coords
) {
    if (xy.size(2) != 2) {
        throw std::invalid_argument("xy must have shape [B, N, 2]");
    }

    auto x = xy.index({torch::indexing::Slice(), torch::indexing::Slice(), 0});
    auto y = xy.index({torch::indexing::Slice(), torch::indexing::Slice(), 1});

    auto div_term = torch::arange(0, C, 2, xy.options());
    div_term = div_term.reshape({1, 1, C / 2}) * (1000.0 / C);

    auto pe_x = torch::zeros_like(xy).expand({-1, -1, C});
    auto pe_y = torch::zeros_like(pe_x);

    pe_x.index_put_(
        {torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, C, 2)},
        torch::sin(x.unsqueeze(-1) * div_term)
    );
    pe_x.index_put_(
        {torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, C, 2)},
        torch::cos(x.unsqueeze(-1) * div_term)
    );

    pe_y.index_put_(
        {torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, C, 2)},
        torch::sin(y.unsqueeze(-1) * div_term)
    );
    pe_y.index_put_(
        {torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, C, 2)},
        torch::cos(y.unsqueeze(-1) * div_term)
    );

    auto pe = torch::cat({pe_x, pe_y}, 2);
    if (cat_coords) {
        pe = torch::cat({xy, pe}, 2);
    }
    return pe;
}

torch::Tensor bilinear_sampler(
    const torch::Tensor& input,
    const torch::Tensor& coords,
    bool align_corners,
    const std::string& padding_mode
) {
    auto coords_clone = coords.detach().clone().to(input.device()).to(input.dtype());
    auto sizes = input.sizes().slice(2);

    if (sizes.size() != 2 && sizes.size() != 3) {
        throw std::invalid_argument("input must be 4D or 5D tensor");
    }

    if (sizes.size() == 3) {
        // t x y -> x y t to match dimensions T H W in grid_sample
        coords_clone = coords_clone.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                          torch::indexing::Slice(), torch::indexing::Slice(),
                                          torch::indexing::Slice({1, 2, 0})});
    }

    torch::Tensor scale;
    if (align_corners) {
        std::vector<double> scale_values;
        for (auto it = sizes.rbegin(); it != sizes.rend(); ++it) {
            scale_values.push_back(2.0 / std::max(*it - 1, 1L));
        }
        scale = torch::tensor(scale_values, coords_clone.options());
    } else {
        std::vector<double> scale_values;
        for (auto it = sizes.rbegin(); it != sizes.rend(); ++it) {
            scale_values.push_back(2.0 / *it);
        }
        scale = torch::tensor(scale_values, coords_clone.options());
    }

    coords_clone = coords_clone * scale - 1;

    return torch::nn::functional::grid_sample(
        input,
        coords_clone,
        torch::nn::functional::GridSampleFuncOptions()
            .align_corners(align_corners)
            .padding_mode(padding_mode)
    );
}

torch::Tensor sample_features4d(
    const torch::Tensor& input,
    const torch::Tensor& coords
) {
    if (input.dim() != 4) {
        throw std::invalid_argument("input must be 4D tensor");
    }
    if (coords.dim() != 3 || coords.size(2) != 2) {
        throw std::invalid_argument("coords must have shape [B, R, 2]");
    }

    auto B = input.size(0);
    auto coords_unsqueezed = coords.unsqueeze(2);  // B R 2 -> B R 1 2
    auto feats = bilinear_sampler(input, coords_unsqueezed);

    return feats.permute({0, 2, 1, 3}).reshape({B, -1, feats.size(1) * feats.size(3)});
}

} // namespace vggt
