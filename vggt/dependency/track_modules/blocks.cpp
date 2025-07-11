/**
 * @file blocks.cpp
 * @brief Implementation of neural network blocks for tracking modules
 */

#include "blocks.h"
#include "modules.h" // For ResidualBlock, AttnBlock, CrossAttnBlock
#include "utils.h"   // For bilinear_sampler
#include <stdexcept>
#include <cmath>

namespace vggt {

namespace {
    torch::Tensor _bilinear_interpolate(torch::Tensor x, int stride, int H, int W) {
        return torch::nn::functional::interpolate(
            x,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{H / stride, W / stride})
                .mode(torch::kBilinear)
                .align_corners(true)
        );
    }
}

// BasicEncoder implementation
BasicEncoderImpl::BasicEncoderImpl(int input_dim, int output_dim, int stride)
    : stride(stride), in_planes(output_dim / 2) {

    // Register modules
    conv1 = register_module("conv1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(input_dim, in_planes, 7)
            .stride(2)
            .padding(3)
            .padding_mode(torch::kZeros)
    ));

    norm1 = register_module("norm1", torch::nn::InstanceNorm2d(in_planes));
    relu1 = register_module("relu1", torch::nn::ReLU(true));

    layer1 = register_module("layer1", _make_layer(output_dim / 2, 1));
    layer2 = register_module("layer2", _make_layer(output_dim * 3 / 4, 2));
    layer3 = register_module("layer3", _make_layer(output_dim, 2));
    layer4 = register_module("layer4", _make_layer(output_dim, 2));

    conv2 = register_module("conv2", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(output_dim * 3 + output_dim / 4, output_dim * 2, 3)
            .padding(1)
            .padding_mode(torch::kZeros)
    ));

    norm2 = register_module("norm2", torch::nn::InstanceNorm2d(output_dim * 2));
    relu2 = register_module("relu2", torch::nn::ReLU(true));
    conv3 = register_module("conv3", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(output_dim * 2, output_dim, 1)
    ));

    // Initialize weights
    for (auto& module : modules()) {
        if (auto m = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
            torch::nn::init::kaiming_normal_(m->weight, 0, torch::kFanOut, torch::kReLU);
            if (m->bias.defined()) {
                torch::nn::init::zeros_(m->bias);
            }
        } else if (auto m = dynamic_cast<torch::nn::InstanceNorm2dImpl*>(module.get())) {
            if (m->weight.defined()) {
                torch::nn::init::ones_(m->weight);
            }
            if (m->bias.defined()) {
                torch::nn::init::zeros_(m->bias);
            }
        }
    }
}

torch::nn::Sequential BasicEncoderImpl::_make_layer(int dim, int stride) {
    torch::nn::Sequential seq;
    seq->push_back(ResidualBlock(in_planes, dim, "instance", stride));
    seq->push_back(ResidualBlock(dim, dim, "instance", 1));
    in_planes = dim;
    return seq;
}

torch::Tensor BasicEncoderImpl::forward(torch::Tensor x) {
    auto sizes = x.sizes();
    int H = sizes[2];
    int W = sizes[3];

    x = conv1(x);
    x = norm1(x);
    x = relu1(x);

    auto a = layer1->forward(x);
    auto b = layer2->forward(a);
    auto c = layer3->forward(b);
    auto d = layer4->forward(c);

    a = _bilinear_interpolate(a, stride, H, W);
    b = _bilinear_interpolate(b, stride, H, W);
    c = _bilinear_interpolate(c, stride, H, W);
    d = _bilinear_interpolate(d, stride, H, W);

    x = conv2(torch::cat({a, b, c, d}, 1));
    x = norm2(x);
    x = relu2(x);
    x = conv3(x);

    return x;
}

// ShallowEncoder implementation
ShallowEncoderImpl::ShallowEncoderImpl(int input_dim, int output_dim, int stride,
                                     const std::string& norm_fn)
    : stride(stride), in_planes(output_dim), norm_fn(norm_fn) {

    // Register modules
    conv1 = register_module("conv1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(input_dim, in_planes, 3)
            .stride(2)
            .padding(1)
            .padding_mode(torch::kZeros)
    ));

    if (norm_fn == "group") {
        norm1 = register_module("norm1", torch::nn::GroupNorm(8, in_planes));
        norm2 = register_module("norm2", torch::nn::GroupNorm(8, output_dim * 2));
    } else if (norm_fn == "batch") {
        norm1 = register_module("norm1", torch::nn::BatchNorm2d(in_planes));
        norm2 = register_module("norm2", torch::nn::BatchNorm2d(output_dim * 2));
    } else if (norm_fn == "instance") {
        norm1 = register_module("norm1", torch::nn::InstanceNorm2d(in_planes));
        norm2 = register_module("norm2", torch::nn::InstanceNorm2d(output_dim * 2));
    } else if (norm_fn == "none") {
        norm1 = register_module("norm1", torch::nn::Sequential());
    } else {
        throw std::runtime_error("Invalid norm_fn: " + norm_fn);
    }

    relu1 = register_module("relu1", torch::nn::ReLU(true));
    layer1 = register_module("layer1", _make_layer(output_dim, 2));
    layer2 = register_module("layer2", _make_layer(output_dim, 2));
    conv2 = register_module("conv2", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(output_dim, output_dim, 1)
    ));

    // Initialize weights
    for (auto& module : modules()) {
        if (auto m = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
            torch::nn::init::kaiming_normal_(m->weight, 0, torch::kFanOut, torch::kReLU);
            if (m->bias.defined()) {
                torch::nn::init::zeros_(m->bias);
            }
        } else if (auto m = dynamic_cast<torch::nn::BatchNorm2dImpl*>(module.get())) {
            torch::nn::init::ones_(m->weight);
            torch::nn::init::zeros_(m->bias);
        } else if (auto m = dynamic_cast<torch::nn::InstanceNorm2dImpl*>(module.get())) {
            if (m->weight.defined()) {
                torch::nn::init::ones_(m->weight);
            }
            if (m->bias.defined()) {
                torch::nn::init::zeros_(m->bias);
            }
        } else if (auto m = dynamic_cast<torch::nn::GroupNormImpl*>(module.get())) {
            torch::nn::init::ones_(m->weight);
            torch::nn::init::zeros_(m->bias);
        }
    }
}

torch::nn::Sequential ShallowEncoderImpl::_make_layer(int dim, int stride) {
    torch::nn::Sequential seq;
    seq->push_back(ResidualBlock(in_planes, dim, norm_fn, stride));
    in_planes = dim;
    return seq;
}

torch::Tensor ShallowEncoderImpl::forward(torch::Tensor x) {
    auto sizes = x.sizes();
    int H = sizes[2];
    int W = sizes[3];

    x = conv1(x);
    if (norm_fn != "none") {
        x = norm1->as<torch::nn::Module>()->forward(x).toTensor();
    }
    x = relu1(x);

    auto tmp = layer1->forward(x);
    x = x + torch::nn::functional::interpolate(
        tmp,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{sizes[2], sizes[3]})
            .mode(torch::kBilinear)
            .align_corners(true)
    );

    tmp = layer2->forward(tmp);
    x = x + torch::nn::functional::interpolate(
        tmp,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{sizes[2], sizes[3]})
            .mode(torch::kBilinear)
            .align_corners(true)
    );

    x = conv2(x) + x;

    x = torch::nn::functional::interpolate(
        x,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{H / stride, W / stride})
            .mode(torch::kBilinear)
            .align_corners(true)
    );

    return x;
}

// EfficientUpdateFormer implementation
EfficientUpdateFormerImpl::EfficientUpdateFormerImpl(
    int space_depth,
    int time_depth,
    int input_dim,
    int hidden_size,
    int num_heads,
    int output_dim,
    float mlp_ratio,
    bool add_space_attn,
    int num_virtual_tracks
) : out_channels(2),
    num_heads(num_heads),
    hidden_size(hidden_size),
    add_space_attn(add_space_attn),
    num_virtual_tracks(num_virtual_tracks) {

    // Register modules
    input_transform = register_module("input_transform",
        torch::nn::Linear(input_dim, hidden_size));

    flow_head = register_module("flow_head",
        torch::nn::Linear(hidden_size, output_dim));

    if (add_space_attn) {
        virual_tracks = register_parameter("virual_tracks",
            torch::randn({1, num_virtual_tracks, 1, hidden_size}));
    }

    // Initialize time blocks
    for (int i = 0; i < time_depth; ++i) {
        time_blocks->push_back(AttnBlock(hidden_size, num_heads, mlp_ratio));
    }

    if (add_space_attn) {
        // Initialize space blocks
        for (int i = 0; i < space_depth; ++i) {
            space_virtual_blocks->push_back(AttnBlock(hidden_size, num_heads, mlp_ratio));
            space_point2virtual_blocks->push_back(CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio));
            space_virtual2point_blocks->push_back(CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio));
        }
    }

    initialize_weights();
}

void EfficientUpdateFormerImpl::initialize_weights() {
    for (auto& module : modules()) {
        if (auto m = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
            torch::nn::init::xavier_uniform_(m->weight);
            if (m->bias.defined()) {
                torch::nn::init::zeros_(m->bias);
            }
        }
    }
}

torch::Tensor EfficientUpdateFormerImpl::forward(torch::Tensor input_tensor, torch::Tensor mask) {
    torch::NoGradGuard no_grad;

    auto tokens = input_transform(input_tensor);
    auto init_tokens = tokens;

    auto sizes = tokens.sizes();
    int B = sizes[0];
    int T = sizes[2];

    if (add_space_attn) {
        auto virtual_tokens = virual_tracks.repeat({B, 1, T, 1});
        tokens = torch::cat({tokens, virtual_tokens}, 1);
    }

    sizes = tokens.sizes();
    int N = sizes[1];

    int j = 0;
    for (int i = 0; i < (int)time_blocks->size(); ++i) {
        // Time attention
        auto time_tokens = tokens.contiguous().view({B * N, T, -1});
        time_tokens = time_blocks[i]->as<AttnBlock>()->forward(time_tokens);
        tokens = time_tokens.view({B, N, T, -1});

        // Space attention
        if (add_space_attn && (i % (time_blocks->size() / space_virtual_blocks->size()) == 0)) {
            auto space_tokens = tokens.permute({0, 2, 1, 3}).contiguous().view({B * T, N, -1});
            auto point_tokens = space_tokens.index({torch::indexing::Slice(),
                                                  torch::indexing::Slice(0, N - num_virtual_tracks)});
            auto virtual_tokens = space_tokens.index({torch::indexing::Slice(),
                                                    torch::indexing::Slice(N - num_virtual_tracks, torch::indexing::None)});

            virtual_tokens = space_virtual2point_blocks[j]->as<CrossAttnBlock>()->forward(virtual_tokens, point_tokens, mask);
            virtual_tokens = space_virtual_blocks[j]->as<AttnBlock>()->forward(virtual_tokens);
            point_tokens = space_point2virtual_blocks[j]->as<CrossAttnBlock>()->forward(point_tokens, virtual_tokens, mask);
            space_tokens = torch::cat({point_tokens, virtual_tokens}, 1);
            tokens = space_tokens.view({B, T, N, -1}).permute({0, 2, 1, 3});
            j++;
        }
    }

    if (add_space_attn) {
        tokens = tokens.index({torch::indexing::Slice(),
                             torch::indexing::Slice(0, N - num_virtual_tracks)});
    }

    tokens = tokens + init_tokens;
    auto flow = flow_head(tokens);

    return flow;
}

// CorrBlock implementation
CorrBlock::CorrBlock(
    torch::Tensor fmaps,
    int num_levels,
    int radius,
    bool multiple_track_feats,
    const std::string& padding_mode
) : num_levels(num_levels),
    radius(radius),
    multiple_track_feats(multiple_track_feats),
    padding_mode(padding_mode) {

    auto sizes = fmaps.sizes();
    S = sizes[1];
    C = sizes[2];
    H = sizes[3];
    W = sizes[4];

    fmaps_pyramid.push_back(fmaps);
    for (int i = 0; i < num_levels - 1; ++i) {
        auto fmaps_ = fmaps.reshape({-1, C, H, W});
        fmaps_ = torch::avg_pool2d(fmaps_, 2, 2);
        sizes = fmaps_.sizes();
        H = sizes[2];
        W = sizes[3];
        fmaps = fmaps_.reshape({sizes[0] / S, S, C, H, W});
        fmaps_pyramid.push_back(fmaps);
    }
}

torch::Tensor CorrBlock::sample(torch::Tensor coords) {
    int r = radius;
    auto sizes = coords.sizes();
    int B = sizes[0];
    int N = sizes[2];

    std::vector<torch::Tensor> out_pyramid;
    for (int i = 0; i < num_levels; ++i) {
        auto corrs = corrs_pyramid[i];
        sizes = corrs.sizes();
        int H = sizes[3];
        int W = sizes[4];

        auto dx = torch::linspace(-r, r, 2 * r + 1, coords.device());
        auto dy = torch::linspace(-r, r, 2 * r + 1, coords.device());
        auto delta = torch::meshgrid({dy, dx}, "ij");
        delta = torch::stack({delta[0], delta[1]}, -1);

        auto centroid_lvl = coords.reshape({B * S * N, 1, 1, 2}) / std::pow(2, i);
        auto delta_lvl = delta.view({1, 2 * r + 1, 2 * r + 1, 2});
        auto coords_lvl = centroid_lvl + delta_lvl;

        auto corrs_sampled = bilinear_sampler(
            corrs.reshape({B * S * N, 1, H, W}),
            coords_lvl,
            padding_mode
        );

        corrs_sampled = corrs_sampled.view({B, S, N, -1});
        out_pyramid.push_back(corrs_sampled);
    }

    auto out = torch::cat(out_pyramid, -1);
    return out;
}

void CorrBlock::corr(torch::Tensor targets) {
    corrs_pyramid.clear();

    for (int i = 0; i < num_levels; ++i) {
        auto fmaps = fmaps_pyramid[i];
        auto sizes = fmaps.sizes();
        int B = sizes[0];
        int S = sizes[1];
        int C = sizes[2];
        int H = sizes[3];
        int W = sizes[4];

        auto fmaps_ = fmaps.reshape({B * S, C, H, W});
        auto targets_ = targets.reshape({B * S, C, H, W});

        auto corrs = torch::einsum("bchw,bcHW->bhwHW", {fmaps_, targets_});
        corrs = corrs / std::sqrt(C);

        if (multiple_track_feats) {
            corrs = corrs.view({B, S, H, W, H, W});
        } else {
            corrs = corrs.view({B, 1, H, W, H, W}).repeat({1, S, 1, 1, 1, 1});
        }

        corrs_pyramid.push_back(corrs);
    }
}

} // namespace vggt
