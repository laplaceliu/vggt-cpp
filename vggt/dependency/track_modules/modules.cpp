/**
 * @file modules.cpp
 * @brief Implementation of neural network modules for tracking
 */

#include "modules.h"
#include <stdexcept>

namespace vggt {

ResidualBlock::ResidualBlock(int in_planes, int planes,
                           const std::string& norm_fn,
                           int stride, int kernel_size)
    : norm_type(norm_fn) {
    // Register conv layers
    conv1 = register_module("conv1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, planes, kernel_size)
            .stride(stride).padding(1).padding_mode(torch::kZeros)));
    conv2 = register_module("conv2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(planes, planes, kernel_size)
            .padding(1).padding_mode(torch::kZeros)));
    relu = register_module("relu", torch::nn::ReLU());

    // Register normalization layers
    int num_groups = planes / 8;
    if (norm_fn == "group") {
        norm1 = register_module("norm1", torch::nn::GroupNorm(num_groups, planes));
        norm2 = register_module("norm2", torch::nn::GroupNorm(num_groups, planes));
        if (stride != 1) {
            norm3 = register_module("norm3", torch::nn::GroupNorm(num_groups, planes));
        }
    } else if (norm_fn == "batch") {
        norm1 = register_module("norm1", torch::nn::BatchNorm2d(planes));
        norm2 = register_module("norm2", torch::nn::BatchNorm2d(planes));
        if (stride != 1) {
            norm3 = register_module("norm3", torch::nn::BatchNorm2d(planes));
        }
    } else if (norm_fn == "instance") {
        norm1 = register_module("norm1", torch::nn::InstanceNorm2d(planes));
        norm2 = register_module("norm2", torch::nn::InstanceNorm2d(planes));
        if (stride != 1) {
            norm3 = register_module("norm3", torch::nn::InstanceNorm2d(planes));
        }
    } else if (norm_fn != "none") {
        throw std::invalid_argument("Unknown normalization type: " + norm_fn);
    }

    // Register downsample if needed
    if (stride != 1) {
        torch::nn::Sequential downsample_seq;
        downsample_seq->push_back(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_planes, planes, 1).stride(stride)));
        if (norm_fn != "none") {
            if (norm_fn == "group") {
                downsample_seq->push_back(torch::nn::GroupNorm(num_groups, planes));
            } else if (norm_fn == "batch") {
                downsample_seq->push_back(torch::nn::BatchNorm2d(planes));
            } else if (norm_fn == "instance") {
                downsample_seq->push_back(torch::nn::InstanceNorm2d(planes));
            }
        }
        downsample = register_module("downsample", downsample_seq);
    }
}

torch::Tensor ResidualBlock::forward(torch::Tensor x) {
    torch::Tensor y = x;
    y = relu(norm1(conv1(y)));
    y = relu(norm2(conv2(y)));

    if (downsample) {
        x = downsample->forward(x);
    }

    return relu(x + y);
}

Mlp::Mlp(int in_features,
        int hidden_features,
        int out_features,
        const torch::nn::AnyModule& act_layer,
        const torch::nn::AnyModule& norm_layer,
        bool bias,
        double drop,
        bool use_conv) {
    out_features = out_features == -1 ? in_features : out_features;
    hidden_features = hidden_features == -1 ? in_features : hidden_features;

    if (use_conv) {
        fc1 = register_module("fc1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_features, hidden_features, 1).bias(bias)));
        fc2 = register_module("fc2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(hidden_features, out_features, 1).bias(bias)));
    } else {
        fc1 = register_module("fc1",
            torch::nn::Linear(torch::nn::LinearOptions(in_features, hidden_features).bias(bias)));
        fc2 = register_module("fc2",
            torch::nn::Linear(torch::nn::LinearOptions(hidden_features, out_features).bias(bias)));
    }

    act = register_module("act", act_layer);
    drop1 = register_module("drop1", torch::nn::Dropout(drop));
    drop2 = register_module("drop2", torch::nn::Dropout(drop));
}

torch::Tensor Mlp::forward(torch::Tensor x) {
    x = fc1->forward(x);
    x = act->forward(x);
    x = drop1->forward(x);
    x = fc2->forward(x);
    return drop2->forward(x);
}

AttnBlock::AttnBlock(int hidden_size,
                   int num_heads,
                   double mlp_ratio) {
    norm1 = register_module("norm1",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size}).elementwise_affine(false).eps(1e-6)));
    norm2 = register_module("norm2",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size}).elementwise_affine(false).eps(1e-6)));

    attn = register_module("attn",
        torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(hidden_size, num_heads).batch_first(true)));

    int mlp_hidden_dim = static_cast<int>(hidden_size * mlp_ratio);
    mlp = register_module("mlp",
        Mlp(hidden_size, mlp_hidden_dim, hidden_size, torch::nn::AnyModule(torch::nn::GELU())));
}

torch::Tensor AttnBlock::forward(torch::Tensor x, torch::Tensor mask) {
    x = norm1(x);
    torch::Tensor attn_output;
    if (mask.defined()) {
        attn_output = std::get<0>(attn->forward(x, x, x, mask));
    } else {
        attn_output = std::get<0>(attn->forward(x, x, x));
    }
    x = x + attn_output;
    x = x + mlp->forward(norm2(x));
    return x;
}

CrossAttnBlock::CrossAttnBlock(int hidden_size,
                             int context_dim,
                             int num_heads,
                             double mlp_ratio) {
    norm1 = register_module("norm1",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size}).elementwise_affine(false).eps(1e-6)));
    norm_context = register_module("norm_context",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({context_dim})));
    norm2 = register_module("norm2",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size}).elementwise_affine(false).eps(1e-6)));

    cross_attn = register_module("cross_attn",
        torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(hidden_size, num_heads).batch_first(true)));

    int mlp_hidden_dim = static_cast<int>(hidden_size * mlp_ratio);
    mlp = register_module("mlp",
        Mlp(hidden_size, mlp_hidden_dim, hidden_size, torch::nn::AnyModule(torch::nn::GELU())));
}

torch::Tensor CrossAttnBlock::forward(torch::Tensor x, torch::Tensor context, torch::Tensor mask) {
    x = norm1(x);
    context = norm_context(context);
    torch::Tensor attn_output;
    if (mask.defined()) {
        attn_output = std::get<0>(cross_attn->forward(x, context, context, mask));
    } else {
        attn_output = std::get<0>(cross_attn->forward(x, context, context));
    }
    x = x + attn_output;
    x = x + mlp->forward(norm2(x));
    return x;
}

} // namespace vggt
