/**
 * @file blocks.cpp
 * @brief Implementation of neural network blocks for tracking heads
 */

#include "blocks.h"
#include <stdexcept>
#include <cmath>

namespace vggt {

EfficientUpdateFormer::EfficientUpdateFormer(
    int space_depth,
    int time_depth,
    int input_dim,
    int hidden_size,
    int num_heads,
    int output_dim,
    double mlp_ratio,
    bool add_space_attn,
    int num_virtual_tracks
) : num_heads(num_heads),
    hidden_size(hidden_size),
    add_space_attn(add_space_attn),
    num_virtual_tracks(num_virtual_tracks) {

    // Register input layers
    input_norm = register_module("input_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({input_dim})));
    input_transform = register_module("input_transform",
        torch::nn::Linear(input_dim, hidden_size));

    // Register output layers
    output_norm = register_module("output_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size})));
    flow_head = register_module("flow_head",
        torch::nn::Linear(hidden_size, output_dim));

    // Register virtual tracks if needed
    if (add_space_attn) {
        virual_tracks = register_parameter("virual_tracks",
            torch::randn({1, num_virtual_tracks, 1, hidden_size}));
    }

    // Register time attention blocks
    time_blocks = register_module("time_blocks", torch::nn::ModuleList());
    for (int i = 0; i < time_depth; ++i) {
        time_blocks->push_back(
            AttnBlock(hidden_size, num_heads, mlp_ratio)
        );
    }

    // Register space attention blocks if needed
    if (add_space_attn) {
        space_virtual_blocks = register_module("space_virtual_blocks", torch::nn::ModuleList());
        space_point2virtual_blocks = register_module("space_point2virtual_blocks", torch::nn::ModuleList());
        space_virtual2point_blocks = register_module("space_virtual2point_blocks", torch::nn::ModuleList());

        for (int i = 0; i < space_depth; ++i) {
            space_virtual_blocks->push_back(
                AttnBlock(hidden_size, num_heads, mlp_ratio)
            );
            space_point2virtual_blocks->push_back(
                CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio)
            );
            space_virtual2point_blocks->push_back(
                CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio)
            );
        }
    }

    initialize_weights();
}

void EfficientUpdateFormer::initialize_weights() {
    // Initialize linear layers with xavier_uniform
    for (auto& module : modules()) {
        if (auto linear = dynamic_cast<torch::nn::Linear*>(module.get())) {
            torch::nn::init::xavier_uniform_(linear->weight);
            if (linear->bias.defined()) {
                torch::nn::init::constant_(linear->bias, 0);
            }
        }
    }
    // Special initialization for flow head
    torch::nn::init::trunc_normal_(flow_head->weight, 0, 0.001);
}

std::tuple<torch::Tensor, torch::Tensor> EfficientUpdateFormer::forward(
    torch::Tensor input_tensor,
    torch::Tensor mask
) {
    // Apply input LayerNorm
    input_tensor = input_norm(input_tensor);
    auto tokens = input_transform(input_tensor);

    auto init_tokens = tokens;

    auto B = tokens.size(0);
    auto T = tokens.size(2);

    if (add_space_attn) {
        auto virtual_tokens = virual_tracks.repeat({B, 1, T, 1});
        tokens = torch::cat({tokens, virtual_tokens}, 1);
    }

    auto N = tokens.size(1);

    int j = 0;
    for (int i = 0; i < time_blocks->size(); ++i) {
        auto time_tokens = tokens.contiguous().view({B * N, T, -1});

        time_tokens = time_blocks[i]->as<AttnBlock>()->forward(time_tokens, mask);

        tokens = time_tokens.view({B, N, T, -1});
        if (add_space_attn && (i % (time_blocks->size() / space_virtual_blocks->size()) == 0)) {
            auto space_tokens = tokens.permute({0, 2, 1, 3}).contiguous().view({B * T, N, -1});
            auto point_tokens = space_tokens.slice(1, 0, N - num_virtual_tracks);
            auto virtual_tokens = space_tokens.slice(1, N - num_virtual_tracks);

            virtual_tokens = space_virtual2point_blocks[j]->as<CrossAttnBlock>()->forward(
                virtual_tokens, point_tokens, mask);
            virtual_tokens = space_virtual_blocks[j]->as<AttnBlock>()->forward(virtual_tokens);
            point_tokens = space_point2virtual_blocks[j]->as<CrossAttnBlock>()->forward(
                point_tokens, virtual_tokens, mask);

            space_tokens = torch::cat({point_tokens, virtual_tokens}, 1);
            tokens = space_tokens.view({B, T, N, -1}).permute({0, 2, 1, 3});
            ++j;
        }
    }

    if (add_space_attn) {
        tokens = tokens.slice(1, 0, N - num_virtual_tracks);
    }

    tokens = tokens + init_tokens;

    // Apply output LayerNorm before final projection
    tokens = output_norm(tokens);
    auto flow = flow_head(tokens);

    return std::make_tuple(flow, torch::Tensor());
}

CorrBlock::CorrBlock(
    torch::Tensor fmaps,
    int num_levels,
    int radius,
    bool multiple_track_feats,
    const std::string& padding_mode
) : S(fmaps.size(1)),
    C(fmaps.size(2)),
    H(fmaps.size(3)),
    W(fmaps.size(4)),
    num_levels(num_levels),
    radius(radius),
    padding_mode(padding_mode),
    multiple_track_feats(multiple_track_feats) {

    // Build pyramid
    fmaps_pyramid.push_back(fmaps);
    auto current_fmaps = fmaps;
    for (int i = 0; i < num_levels - 1; ++i) {
        auto B = current_fmaps.size(0);
        auto S = current_fmaps.size(1);
        auto C = current_fmaps.size(2);
        auto H = current_fmaps.size(3);
        auto W = current_fmaps.size(4);

        current_fmaps = current_fmaps.reshape({B * S, C, H, W});
        current_fmaps = torch::avg_pool2d(current_fmaps, 2, 2);
        auto H_new = current_fmaps.size(2);
        auto W_new = current_fmaps.size(3);
        current_fmaps = current_fmaps.reshape({B, S, C, H_new, W_new});
        fmaps_pyramid.push_back(current_fmaps);
    }

    // Precompute delta grid
    auto dx = torch::linspace(-radius, radius, 2 * radius + 1,
        torch::TensorOptions().device(fmaps.device()).dtype(fmaps.dtype()));
    auto dy = torch::linspace(-radius, radius, 2 * radius + 1,
        torch::TensorOptions().device(fmaps.device()).dtype(fmaps.dtype()));
    delta = torch::stack(torch::meshgrid({dy, dx}, "ij"), -1);
}

torch::Tensor CorrBlock::corr_sample(torch::Tensor targets, torch::Tensor coords) {
    auto B = targets.size(0);
    auto S = targets.size(1);
    auto N = targets.size(2);
    auto C = targets.size(3);

    std::vector<torch::Tensor> targets_split;
    if (multiple_track_feats) {
        targets_split = torch::split(targets, C / num_levels, 3);
    }

    std::vector<torch::Tensor> out_pyramid;
    for (int i = 0; i < fmaps_pyramid.size(); ++i) {
        auto fmaps = fmaps_pyramid[i];
        auto B = fmaps.size(0);
        auto S = fmaps.size(1);
        auto C = fmaps.size(2);
        auto H = fmaps.size(3);
        auto W = fmaps.size(4);

        auto fmap2s = fmaps.view({B, S, C, H * W});
        auto fmap1 = multiple_track_feats ? targets_split[i] : targets;

        auto corrs = compute_corr_level(fmap1, fmap2s, C);
        corrs = corrs.view({B, S, N, H, W});

        auto centroid_lvl = coords.reshape({B * S * N, 1, 1, 2}) / std::pow(2, i);
        auto delta_lvl = delta.to(coords.device()).to(coords.dtype());
        auto coords_lvl = centroid_lvl + delta_lvl.view({1, 2 * radius + 1, 2 * radius + 1, 2});

        auto corrs_sampled = bilinear_sampler(
            corrs.reshape({B * S * N, 1, H, W}),
            coords_lvl,
            true,
            padding_mode
        );
        corrs_sampled = corrs_sampled.view({B, S, N, -1});
        out_pyramid.push_back(corrs_sampled);
    }

    return torch::cat(out_pyramid, 3).contiguous();
}

torch::Tensor compute_corr_level(torch::Tensor fmap1, torch::Tensor fmap2s, int C) {
    auto corrs = torch::matmul(fmap1, fmap2s);
    corrs = corrs.view(fmap1.sizes());
    return corrs / std::sqrt(C);
}

} // namespace vggt
