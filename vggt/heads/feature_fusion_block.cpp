#include "feature_fusion_block.h"

namespace vggt {
namespace heads {

ResidualConvUnitImpl::ResidualConvUnitImpl(
    int features,
    const torch::nn::Functional& activation,
    bool bn,
    int groups
) : bn(bn),
    groups(groups),
    conv1(torch::nn::Conv2dOptions(features, features, 3).stride(1).padding(1).bias(true).groups(groups)),
    conv2(torch::nn::Conv2dOptions(features, features, 3).stride(1).padding(1).bias(true).groups(groups)),
    activation(activation) {
    if (bn) {
        norm1 = torch::nn::LayerNorm(features);
        norm2 = torch::nn::LayerNorm(features);
    }
    register_module("conv1", conv1);
    register_module("conv2", conv2);
    register_module("activation", activation);
    register_module("skip_add", skip_add);
}

torch::Tensor ResidualConvUnitImpl::forward(const torch::Tensor& x) {
    auto out = activation(x);
    out = conv1(out);
    if (bn) {
        out = norm1(out);
    }

    out = activation(out);
    out = conv2(out);
    if (bn) {
        out = norm2(out);
    }

    return skip_add.add(out, x);
}

FeatureFusionBlockImpl::FeatureFusionBlockImpl(
    int features,
    const torch::nn::Functional& activation,
    bool deconv,
    bool bn,
    bool expand,
    bool align_corners,
    bool has_residual,
    int groups
) : deconv(deconv),
    bn(bn),
    expand(expand),
    align_corners(align_corners),
    has_residual(has_residual),
    groups(groups),
    activation(activation),
    rcu(ResidualConvUnit(features, activation, bn, groups)),
    project(torch::nn::Conv2dOptions(features, features, 1).stride(1).padding(0).bias(false)) {
    register_module("rcu", rcu);
    register_module("project", project);
}

torch::Tensor FeatureFusionBlockImpl::forward(const torch::Tensor& x, const torch::Tensor& y) {
    auto out = x;
    if (y.defined()) {
        if (deconv) {
            out = torch::nn::functional::interpolate(
                out,
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{y.size(2), y.size(3)})
                    .mode(torch::kNearest)
                    .align_corners(align_corners)
            );
        }
        out = project(out + y);
    }
    if (has_residual) {
        out = rcu(out);
    }
    return out;
}

} // namespace heads
} // namespace vggt