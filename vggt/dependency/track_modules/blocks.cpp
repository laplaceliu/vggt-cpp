#include "blocks.h"

namespace vggt {
namespace dependency {
namespace track_modules {

BasicEncoderImpl::BasicEncoderImpl(int64_t input_dim, int64_t output_dim, int64_t stride) {
    this->stride = stride;
    this->norm_fn = "instance";
    this->in_planes = output_dim / 2;

    norm1 = register_module("norm1", torch::nn::InstanceNorm2d(this->in_planes));
    norm2 = register_module("norm2", torch::nn::InstanceNorm2d(output_dim * 2));

    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_dim, this->in_planes, 7)
        .stride(2).padding(3).padding_mode(torch::kZeros)));
    relu1 = register_module("relu1", torch::nn::ReLU(true));
    layer1 = register_module("layer1", _make_layer(output_dim / 2, 1));
    layer2 = register_module("layer2", _make_layer(output_dim / 4 * 3, 2));
    layer3 = register_module("layer3", _make_layer(output_dim, 2));
    layer4 = register_module("layer4", _make_layer(output_dim, 2));

    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(output_dim * 3 + output_dim / 4, output_dim * 2, 3)
        .padding(1).padding_mode(torch::kZeros)));
    relu2 = register_module("relu2", torch::nn::ReLU(true));
    conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(output_dim * 2, output_dim, 1)));

    // Initialize weights
    for (auto& module : modules(false)) {
        if (auto* conv = module->as<torch::nn::Conv2d>()) {
            torch::nn::init::kaiming_normal_(conv->weight, 0.0, torch::kFanOut, torch::kReLU);
        } else if (auto* norm = module->as<torch::nn::InstanceNorm2d>()) {
            if (norm->weight.defined()) {
                torch::nn::init::constant_(norm->weight, 1.0);
            }
            if (norm->bias.defined()) {
                torch::nn::init::constant_(norm->bias, 0.0);
            }
        }
    }
}

torch::nn::Sequential BasicEncoderImpl::_make_layer(int64_t dim, int64_t stride) {
    auto layer1 = ResidualBlock(this->in_planes, dim, this->norm_fn, stride);
    auto layer2 = ResidualBlock(dim, dim, this->norm_fn, 1);

    this->in_planes = dim;
    return torch::nn::Sequential(layer1, layer2);
}

torch::Tensor BasicEncoderImpl::forward(torch::Tensor x) {
    auto sizes = x.sizes();
    int64_t H = sizes[2];
    int64_t W = sizes[3];

    x = conv1->forward(x);
    x = norm1(x);
    x = relu1->forward(x);

    auto a = layer1->forward(x);
    auto b = layer2->forward(a);
    auto c = layer3->forward(b);
    auto d = layer4->forward(c);

    a = _bilinear_intepolate(a, this->stride, H, W);
    b = _bilinear_intepolate(b, this->stride, H, W);
    c = _bilinear_intepolate(c, this->stride, H, W);
    d = _bilinear_intepolate(d, this->stride, H, W);

    x = conv2->forward(torch::cat({a, b, c, d}, 1));
    x = norm2->forward(x);
    x = relu2->forward(x);
    x = conv3->forward(x);
    return x;
}

ShallowEncoderImpl::ShallowEncoderImpl(int64_t input_dim, int64_t output_dim, int64_t stride, const std::string& norm_fn) {
    this->stride = stride;
    this->norm_fn = norm_fn;
    this->in_planes = output_dim;

    if (this->norm_fn == "group") {
        norm1 = register_module("norm1", torch::nn::GroupNorm(8, this->in_planes));
        norm2 = register_module("norm2", torch::nn::GroupNorm(8, output_dim * 2));
    } else if (this->norm_fn == "batch") {
        norm1 = register_module("norm1", torch::nn::BatchNorm2d(this->in_planes));
        norm2 = register_module("norm2", torch::nn::BatchNorm2d(output_dim * 2));
    } else if (this->norm_fn == "instance") {
        norm1 = register_module("norm1", torch::nn::InstanceNorm2d(this->in_planes));
        norm2 = register_module("norm2", torch::nn::InstanceNorm2d(output_dim * 2));
    } else if (this->norm_fn == "none") {
        norm1 = register_module("norm1", StackSequential());
    }

    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_dim, this->in_planes, 3)
        .stride(2).padding(1).padding_mode(torch::kZeros)));
    relu1 = register_module("relu1", torch::nn::ReLU(true));

    layer1 = _make_layer(output_dim, 2);
    register_module("layer1", layer1);
    layer2 = _make_layer(output_dim, 2);
    register_module("layer2", layer2);
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(output_dim, output_dim, 1)));

    // Initialize weights
    for (auto& module : modules(false)) {
        if (auto* conv = module->as<torch::nn::Conv2d>()) {
            torch::nn::init::kaiming_normal_(conv->weight, 0.0, torch::kFanOut, torch::kReLU);
        } else if (auto* norm = module->as<torch::nn::BatchNorm2d>()) {
            if (norm->weight.defined()) {
                torch::nn::init::constant_(norm->weight, 1.0);
            }
            if (norm->bias.defined()) {
                torch::nn::init::constant_(norm->bias, 0.0);
            }
        } else if (auto* norm = module->as<torch::nn::InstanceNorm2d>()) {
            if (norm->weight.defined()) {
                torch::nn::init::constant_(norm->weight, 1.0);
            }
            if (norm->bias.defined()) {
                torch::nn::init::constant_(norm->bias, 0.0);
            }
        } else if (auto* norm = module->as<torch::nn::GroupNorm>()) {
            if (norm->weight.defined()) {
                torch::nn::init::constant_(norm->weight, 1.0);
            }
            if (norm->bias.defined()) {
                torch::nn::init::constant_(norm->bias, 0.0);
            }
        }
    }
}

ResidualBlock ShallowEncoderImpl::_make_layer(int64_t dim, int64_t stride) {
    this->in_planes = dim;
    return ResidualBlock(this->in_planes, dim, this->norm_fn, stride);
}

torch::Tensor ShallowEncoderImpl::forward(torch::Tensor x) {
    auto sizes = x.sizes();
    int64_t H = sizes[2];
    int64_t W = sizes[3];

    x = conv1->forward(x);
    x = norm1.forward(x);
    x = relu1->forward(x);

    auto tmp = layer1->forward(x);
    x = x + torch::nn::functional::interpolate(tmp, torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<int64_t>{x.size(2), x.size(3)})
        .mode(torch::kBilinear)
        .align_corners(true));

    tmp = layer2->forward(tmp);
    x = x + torch::nn::functional::interpolate(tmp, torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<int64_t>{x.size(2), x.size(3)})
        .mode(torch::kBilinear)
        .align_corners(true));

    tmp = torch::Tensor(); // Set to None
    x = conv2->forward(x) + x;

    x = torch::nn::functional::interpolate(x, torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<int64_t>{H / this->stride, W / this->stride})
        .mode(torch::kBilinear)
        .align_corners(true));

    return x;
}

torch::Tensor _bilinear_intepolate(const torch::Tensor& x, int64_t stride, int64_t H, int64_t W) {
    return torch::nn::functional::interpolate(x, torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<int64_t>{H / stride, W / stride})
        .mode(torch::kBilinear)
        .align_corners(true));
}

EfficientUpdateFormerImpl::EfficientUpdateFormerImpl(
    int64_t space_depth,
    int64_t time_depth,
    int64_t input_dim,
    int64_t hidden_size,
    int64_t num_heads,
    int64_t output_dim,
    double mlp_ratio,
    bool add_space_attn,
    int64_t num_virtual_tracks
) {
    this->out_channels = 2;
    this->num_heads = num_heads;
    this->hidden_size = hidden_size;
    this->add_space_attn = add_space_attn;
    this->num_virtual_tracks = num_virtual_tracks;

    input_transform = register_module("input_transform", torch::nn::Linear(torch::nn::LinearOptions(input_dim, hidden_size).bias(true)));
    flow_head = register_module("flow_head", torch::nn::Linear(torch::nn::LinearOptions(hidden_size, output_dim).bias(true)));

    if (this->add_space_attn) {
        virual_tracks = register_parameter("virual_tracks", torch::randn({1, num_virtual_tracks, 1, hidden_size}));
    }

    // Create time blocks
    time_blocks = register_module("time_blocks", torch::nn::ModuleList());
    for (int64_t i = 0; i < time_depth; ++i) {
        time_blocks->push_back(AttnBlock(hidden_size, num_heads, torch::nn::AnyModule(torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(hidden_size, num_heads))), mlp_ratio));
    }

    if (add_space_attn) {
        // Create space virtual blocks
        space_virtual_blocks = register_module("space_virtual_blocks", torch::nn::ModuleList());
        for (int64_t i = 0; i < space_depth; ++i) {
            space_virtual_blocks->push_back(AttnBlock(hidden_size, num_heads, torch::nn::AnyModule(torch::nn::MultiheadAttention(
                torch::nn::MultiheadAttentionOptions(hidden_size, num_heads))), mlp_ratio));
        }

        // Create space point2virtual blocks
        space_point2virtual_blocks = register_module("space_point2virtual_blocks", torch::nn::ModuleList());
        for (int64_t i = 0; i < space_depth; ++i) {
            space_point2virtual_blocks->push_back(CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio));
        }

        // Create space virtual2point blocks
        space_virtual2point_blocks = register_module("space_virtual2point_blocks", torch::nn::ModuleList());
        for (int64_t i = 0; i < space_depth; ++i) {
            space_virtual2point_blocks->push_back(CrossAttnBlock(hidden_size, hidden_size, num_heads, mlp_ratio));
        }
    }

    initialize_weights();
}

void EfficientUpdateFormerImpl::initialize_weights() {
    // Basic initialization for all modules
    for (auto& module : modules(false)) {
        if (auto* linear = module->as<torch::nn::Linear>()) {
            torch::nn::init::xavier_uniform_(linear->weight);
            if (linear->bias.defined()) {
                torch::nn::init::constant_(linear->bias, 0);
            }
        }
    }
}

torch::Tensor EfficientUpdateFormerImpl::forward(torch::Tensor input_tensor, torch::Tensor mask) {
    auto tokens = input_transform->forward(input_tensor);
    auto init_tokens = tokens;

    auto B = tokens.size(0);
    auto T = tokens.size(2);

    if (add_space_attn) {
        auto virtual_tokens = virual_tracks.repeat({B, 1, T, 1});
        tokens = torch::cat({tokens, virtual_tokens}, 1);
    }

    auto N = tokens.size(1);

    int64_t j = 0;
    for (int64_t i = 0; i < time_blocks->size(); ++i) {
        auto time_tokens = tokens.contiguous().view({B * N, T, -1});  // B N T C -> (B N) T C
        time_tokens = time_blocks[i]->as<AttnBlockImpl>()->forward(time_tokens);

        tokens = time_tokens.view({B, N, T, -1});  // (B N) T C -> B N T C
        if (add_space_attn && (i % (time_blocks->size() / space_virtual_blocks->size()) == 0)) {
            auto space_tokens = tokens.permute({0, 2, 1, 3}).contiguous().view({B * T, N, -1});  // B N T C -> (B T) N C
            auto point_tokens = space_tokens.index({torch::indexing::Slice(), torch::indexing::Slice(0, N - num_virtual_tracks)});
            auto virtual_tokens = space_tokens.index({torch::indexing::Slice(), torch::indexing::Slice(N - num_virtual_tracks, N)});

            virtual_tokens = space_virtual2point_blocks[j]->as<CrossAttnBlockImpl>()->forward(virtual_tokens, point_tokens, mask);
            virtual_tokens = space_virtual_blocks[j]->as<AttnBlockImpl>()->forward(virtual_tokens);
            point_tokens = space_point2virtual_blocks[j]->as<CrossAttnBlockImpl>()->forward(point_tokens, virtual_tokens, mask);
            space_tokens = torch::cat({point_tokens, virtual_tokens}, 1);
            tokens = space_tokens.view({B, T, N, -1}).permute({0, 2, 1, 3});  // (B T) N C -> B N T C
            j += 1;
        }
    }

    if (add_space_attn) {
        tokens = tokens.index({torch::indexing::Slice(), torch::indexing::Slice(0, N - num_virtual_tracks)});
    }

    tokens = tokens + init_tokens;

    auto flow = flow_head->forward(tokens);
    return flow;
}

CorrBlock::CorrBlock(const torch::Tensor& fmaps, int64_t num_levels, int64_t radius, bool multiple_track_feats, const torch::nn::functional::GridSampleFuncOptions::padding_mode_t& padding_mode) {
    auto sizes = fmaps.sizes();
    B = sizes[0];
    S = sizes[1];
    C = sizes[2];
    H = sizes[3];
    W = sizes[4];

    this->padding_mode = padding_mode;
    this->num_levels = num_levels;
    this->radius = radius;
    this->multiple_track_feats = multiple_track_feats;

    fmaps_pyramid.push_back(fmaps);
    for (int64_t i = 0; i < num_levels - 1; ++i) {
        auto fmaps_ = fmaps.reshape(std::vector<int64_t>{B * S, C, H, W});
        fmaps_ = torch::nn::functional::avg_pool2d(fmaps_, torch::nn::functional::AvgPool2dFuncOptions(2).stride(2));
        auto new_sizes = fmaps_.sizes();
        H = new_sizes[2];
        W = new_sizes[3];
        auto new_fmaps = fmaps_.reshape(std::vector<int64_t>{B, S, C, H, W});
        fmaps_pyramid.push_back(new_fmaps);
    }
}

torch::Tensor CorrBlock::sample(const torch::Tensor& coords) {
    int64_t r = radius;
    auto sizes = coords.sizes();
    int64_t B = sizes[0];
    int64_t S = sizes[1];
    int64_t N = sizes[2];
    int64_t D = sizes[3];
    assert(D == 2);

    int64_t H = this->H;
    int64_t W = this->W;
    std::vector<torch::Tensor> out_pyramid;

    for (int64_t i = 0; i < num_levels; ++i) {
        auto corrs = corrs_pyramid[i];  // B, S, N, H, W
        auto corr_sizes = corrs.sizes();
        H = corr_sizes[3];
        W = corr_sizes[4];

        auto dx = torch::linspace(-r, r, 2 * r + 1, torch::kFloat);
        auto dy = torch::linspace(-r, r, 2 * r + 1, torch::kFloat);
        auto grid = torch::meshgrid({dy, dx}, "ij");
        auto delta = torch::stack({grid[0], grid[1]}, -1).to(coords.device());

        auto centroid_lvl = coords.reshape({B * S * N, 1, 1, 2}) / std::pow(2, i);
        auto delta_lvl = delta.view({1, 2 * r + 1, 2 * r + 1, 2});
        auto coords_lvl = centroid_lvl + delta_lvl;

        auto sampled_corrs = bilinear_sampler(corrs.reshape({B * S * N, 1, H, W}), coords_lvl, true,
            padding_mode);
        sampled_corrs = sampled_corrs.view({B, S, N, -1});

        out_pyramid.push_back(sampled_corrs);
    }

    auto out = torch::cat(out_pyramid, -1).contiguous();  // B, S, N, LRR*2
    return out;
}

void CorrBlock::corr(const torch::Tensor& targets) {
    auto sizes = targets.sizes();
    int64_t B = sizes[0];
    int64_t S = sizes[1];
    int64_t N = sizes[2];
    int64_t C = sizes[3];

    std::vector<torch::Tensor> targets_split;
    if (multiple_track_feats) {
        int64_t split_size = C / num_levels;
        for (int64_t i = 0; i < num_levels; ++i) {
            targets_split.push_back(targets.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(),
                torch::indexing::Slice(i * split_size, (i + 1) * split_size)}));
        }
        auto split_sizes = targets_split[0].sizes();
        C = split_sizes[3];
    }

    assert(C == this->C);
    assert(S == this->S);

    auto fmap1 = targets;
    corrs_pyramid.clear();

    for (int64_t i = 0; i < fmaps_pyramid.size(); ++i) {
        auto fmaps = fmaps_pyramid[i];
        auto fmap_sizes = fmaps.sizes();
        int64_t H = fmap_sizes[3];
        int64_t W = fmap_sizes[4];

        auto fmap2s = fmaps.view({B, S, C, H * W});  // B S C H W ->  B S C (H W)

        if (multiple_track_feats) {
            fmap1 = targets_split[i];
        }

        auto corrs = torch::matmul(fmap1, fmap2s);
        corrs = corrs.view({B, S, N, H, W});  // B S N (H W) -> B S N H W
        corrs = corrs / torch::sqrt(torch::tensor(static_cast<double>(C)));

        corrs_pyramid.push_back(corrs);
    }
}

} // namespace track_modules
} // namespace dependency
} // namespace vggt
