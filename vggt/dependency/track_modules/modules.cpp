

#include "modules.h"

namespace vggt {
namespace track_modules {

ResidualBlockImpl::ResidualBlockImpl(int64_t in_planes, int64_t planes, const std::string& norm_fn, int64_t stride, int64_t kernel_size) {
    // Conv layers
    conv1 = register_module("conv1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_planes, planes, kernel_size)
            .stride(stride)
            .padding(1)
            .padding_mode(torch::kZeros)
    ));

    conv2 = register_module("conv2", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(planes, planes, kernel_size)
            .padding(1)
            .padding_mode(torch::kZeros)
    ));

    relu = register_module("relu", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));

    // Norm layers
    int64_t num_groups = planes / 8;

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
    } else if (norm_fn == "none") {
        norm1 = register_module("norm1", torch::nn::Sequential());
        norm2 = register_module("norm2", torch::nn::Sequential());
        if (stride != 1) {
            norm3 = register_module("norm3", torch::nn::Sequential());
        }
    } else {
        throw std::runtime_error("Unsupported norm type: " + norm_fn);
    }

    // Downsample
    if (stride != 1) {
        downsample = register_module("downsample", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, planes, 1).stride(stride)),
            norm3
        ));
    }
}

torch::Tensor ResidualBlockImpl::forward(torch::Tensor x) {
    auto y = x;
    y = relu(norm1(conv1(y)));
    y = relu(norm2(conv2(y)));

    if (downsample) {
        x = downsample(x);
    }

    return relu(x + y);
}

MlpImpl::MlpImpl(int64_t in_features, int64_t hidden_features, int64_t out_features, const torch::nn::AnyModule& act_layer, const torch::nn::AnyModule& norm_layer, bool bias, double drop, bool use_conv) {
    out_features = out_features == 0 ? in_features : out_features;
    hidden_features = hidden_features == 0 ? in_features : hidden_features;

    if (use_conv) {
        fc1 = register_module("fc1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_features, hidden_features, 1).bias(bias)));
        fc2 = register_module("fc2", torch::nn::Conv2d(torch::nn::Conv2dOptions(hidden_features, out_features, 1).bias(bias)));
    } else {
        fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(in_features, hidden_features).bias(bias)));
        fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(hidden_features, out_features).bias(bias)));
    }

    act = register_module("act", act_layer);
    drop1 = register_module("drop1", torch::nn::Dropout(drop));
    drop2 = register_module("drop2", torch::nn::Dropout(drop));
}

torch::Tensor MlpImpl::forward(torch::Tensor x) {
    x = fc1(x);
    x = act(x);
    x = drop1(x);
    x = fc2(x);
    x = drop2(x);
    return x;
}

AttnBlockImpl::AttnBlockImpl(int64_t hidden_size, int64_t num_heads, const torch::nn::AnyModule& attn_class, double mlp_ratio) {
    norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size}).elementwise_affine(false).eps(1e-6)));
    norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size}).elementwise_affine(false).eps(1e-6)));

    attn = register_module("attn", attn_class);

    int64_t mlp_hidden_dim = static_cast<int64_t>(hidden_size * mlp_ratio);
    mlp = register_module("mlp", Mlp(hidden_size, mlp_hidden_dim, hidden_size, torch::nn::AnyModule(torch::nn::GELU()), torch::nn::AnyModule(), true, 0.0));
}

torch::Tensor AttnBlockImpl::forward(torch::Tensor x, torch::Tensor mask) {
    x = norm1(x);
    auto attn_output = attn(x, x, x).output;
    x = x + attn_output;
    x = x + mlp(norm2(x));
    return x;
}

CrossAttnBlockImpl::CrossAttnBlockImpl(int64_t hidden_size, int64_t context_dim, int64_t num_heads, double mlp_ratio) {
    norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size}).elementwise_affine(false).eps(1e-6)));
    norm_context = register_module("norm_context", torch::nn::LayerNorm(hidden_size));
    norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size}).elementwise_affine(false).eps(1e-6)));

    cross_attn = register_module("cross_attn", torch::nn::MultiheadAttention(
        torch::nn::MultiheadAttentionOptions(hidden_size, num_heads).batch_first(true)
    ));

    int64_t mlp_hidden_dim = static_cast<int64_t>(hidden_size * mlp_ratio);
    mlp = register_module("mlp", Mlp(hidden_size, mlp_hidden_dim, hidden_size, torch::nn::AnyModule(torch::nn::GELU()), torch::nn::AnyModule(), true, 0.0));
}

torch::Tensor CrossAttnBlockImpl::forward(torch::Tensor x, torch::Tensor context, torch::Tensor mask) {
    x = norm1(x);
    context = norm_context(context);

    auto attn_output = cross_attn(x, context, context, mask).output;
    x = x + attn_output;
    x = x + mlp(norm2(x));
    return x;
}

} // namespace track_modules
} // namespace vggt
