/**
 * @file utils.cpp
 * @brief Implementation of utility functions for tracking modules
 *
 * Modified from https://github.com/facebookresearch/PoseDiffusion
 * and https://github.com/facebookresearch/co-tracker/tree/main
 */

#include "utils.h"
#include <cmath>
#include <stdexcept>

namespace vggt {

namespace {
    // Helper function to rearrange tensor dimensions (simplified version of einops.rearrange)
    torch::Tensor rearrange(torch::Tensor tensor, const std::string& pattern) {
        if (pattern == "m,d->md") {
            // Outer product
            auto m_size = tensor.sizes()[0];
            auto d_size = tensor.sizes()[1];
            return tensor.unsqueeze(1).expand({m_size, d_size});
        } else {
            throw std::runtime_error("Unsupported rearrange pattern: " + pattern);
        }
    }
}

std::variant<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> get_2d_sincos_pos_embed(
    int embed_dim,
    std::variant<int, std::tuple<int, int>> grid_size,
    bool return_grid
) {
    int grid_size_h, grid_size_w;

    if (std::holds_alternative<int>(grid_size)) {
        grid_size_h = grid_size_w = std::get<int>(grid_size);
    } else {
        auto [h, w] = std::get<std::tuple<int, int>>(grid_size);
        grid_size_h = h;
        grid_size_w = w;
    }

    auto grid_h = torch::arange(grid_size_h, torch::kFloat);
    auto grid_w = torch::arange(grid_size_w, torch::kFloat);

    // Create meshgrid
    auto grid_y = grid_h.unsqueeze(1).expand({grid_size_h, grid_size_w});
    auto grid_x = grid_w.unsqueeze(0).expand({grid_size_h, grid_size_w});

    auto grid = torch::stack({grid_x, grid_y}, 0);
    grid = grid.reshape({2, 1, grid_size_h, grid_size_w});

    auto pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid);
    auto reshaped_embed = pos_embed.reshape({1, grid_size_h, grid_size_w, -1}).permute({0, 3, 1, 2});

    if (return_grid) {
        return std::tuple<torch::Tensor, torch::Tensor>(reshaped_embed, grid);
    }

    return reshaped_embed;
}

torch::Tensor get_2d_sincos_pos_embed_from_grid(int embed_dim, torch::Tensor grid) {
    if (embed_dim % 2 != 0) {
        throw std::invalid_argument("Embedding dimension must be even");
    }

    // Use half of dimensions to encode grid_h
    auto emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim / 2, grid[0]);  // (1, H*W, D/2)
    auto emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim / 2, grid[1]);  // (1, H*W, D/2)

    auto emb = torch::cat({emb_h, emb_w}, 2);  // (1, H*W, D)
    return emb;
}

torch::Tensor get_1d_sincos_pos_embed_from_grid(int embed_dim, torch::Tensor pos) {
    if (embed_dim % 2 != 0) {
        throw std::invalid_argument("Embedding dimension must be even");
    }

    auto omega = torch::arange(embed_dim / 2, torch::kDouble);
    omega = omega / (embed_dim / 2.0);
    omega = 1.0 / torch::pow(10000.0, omega);  // (D/2,)

    pos = pos.reshape({-1});  // (M,)

    // Outer product
    auto out = pos.unsqueeze(1) * omega.unsqueeze(0);  // (M, D/2)

    auto emb_sin = torch::sin(out);  // (M, D/2)
    auto emb_cos = torch::cos(out);  // (M, D/2)

    auto emb = torch::cat({emb_sin, emb_cos}, 1);  // (M, D)
    return emb.unsqueeze(0).to(torch::kFloat);  // (1, M, D)
}

torch::Tensor get_2d_embedding(torch::Tensor xy, int C, bool cat_coords) {
    auto sizes = xy.sizes();
    int B = sizes[0];
    int N = sizes[1];
    int D = sizes[2];

    if (D != 2) {
        throw std::invalid_argument("Input tensor must have shape (B, N, 2)");
    }

    auto x = xy.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}).unsqueeze(-1);
    auto y = xy.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}).unsqueeze(-1);

    auto div_term = torch::arange(0, C, 2, xy.options().dtype(torch::kFloat32)) * (1000.0 / C);
    div_term = div_term.reshape({1, 1, C / 2});

    auto pe_x = torch::zeros({B, N, C}, xy.options().dtype(torch::kFloat32));
    auto pe_y = torch::zeros({B, N, C}, xy.options().dtype(torch::kFloat32));

    pe_x.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)},
                   torch::sin(x * div_term));
    pe_x.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)},
                   torch::cos(x * div_term));

    pe_y.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)},
                   torch::sin(y * div_term));
    pe_y.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)},
                   torch::cos(y * div_term));

    auto pe = torch::cat({pe_x, pe_y}, 2);  // (B, N, C*2)

    if (cat_coords) {
        pe = torch::cat({xy, pe}, 2);  // (B, N, C*2+2)
    }

    return pe;
}

torch::Tensor bilinear_sampler(
    torch::Tensor input,
    torch::Tensor coords,
    bool align_corners,
    const std::string& padding_mode
) {
    auto sizes = input.sizes();
    auto ndim = sizes.size();

    if (ndim != 4 && ndim != 5) {
        throw std::invalid_argument("Input tensor must be 4D or 5D");
    }

    auto coords_sizes = coords.sizes();
    auto coords_ndim = coords_sizes.size();

    if (coords_ndim != 4) {
        throw std::invalid_argument("Coords tensor must be 4D");
    }

    if (ndim == 5) {
        // t x y -> x y t to match dimensions T H W in grid_sample
        coords = coords.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(),
                              torch::indexing::Tensor({1, 2, 0})});
    }

    std::vector<float> scale_factors;
    for (int i = ndim - 1; i >= 2; --i) {
        float scale;
        if (align_corners) {
            scale = 2.0f / std::max(static_cast<int64_t>(1), sizes[i] - 1);
        } else {
            scale = 2.0f / sizes[i];
        }
        scale_factors.push_back(scale);
    }

    auto scale_tensor = torch::tensor(scale_factors, coords.options());
    coords = coords * scale_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0);
    coords = coords - 1;

    return torch::nn::functional::grid_sample(
        input,
        coords,
        torch::nn::functional::GridSampleFuncOptions()
            .align_corners(align_corners)
            .padding_mode(padding_mode)
            .mode("bilinear")
    );
}

torch::Tensor sample_features4d(torch::Tensor input, torch::Tensor coords) {
    auto sizes = input.sizes();
    int B = sizes[0];

    // B R 2 -> B R 1 2
    coords = coords.unsqueeze(2);

    // B C R 1
    auto feats = bilinear_sampler(input, coords);

    return feats.permute({0, 2, 1, 3}).reshape({B, -1, feats.sizes()[1] * feats.sizes()[3]});  // B C R 1 -> B R C
}

} // namespace vggt
