#include "dpt_head.h"

namespace vggt {
namespace heads {

torch::nn::Module _make_scratch(const std::vector<int64_t>& in_shape, int64_t out_shape, int64_t groups, bool expand) {
    torch::nn::Module scratch;
    int64_t out_shape1 = out_shape;
    int64_t out_shape2 = out_shape;
    int64_t out_shape3 = out_shape;
    int64_t out_shape4 = out_shape;

    if (expand) {
        out_shape1 = out_shape;
        out_shape2 = out_shape * 2;
        out_shape3 = out_shape * 4;
        if (in_shape.size() >= 4) {
            out_shape4 = out_shape * 8;
        }
    }

    scratch.register_module("layer1_rn", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_shape[0], out_shape1, 3).stride(1).padding(1).bias(false).groups(groups)
    ));

    scratch.register_module("layer2_rn", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_shape[1], out_shape2, 3).stride(1).padding(1).bias(false).groups(groups)
    ));

    scratch.register_module("layer3_rn", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_shape[2], out_shape3, 3).stride(1).padding(1).bias(false).groups(groups)
    ));

    if (in_shape.size() >= 4) {
        scratch.register_module("layer4_rn", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_shape[3], out_shape4, 3).stride(1).padding(1).bias(false).groups(groups)
        ));
    }

    return scratch;
}

torch::Tensor custom_interpolate(
    torch::Tensor x,
    std::pair<int64_t, int64_t> size,
    float scale_factor,
    torch::nn::functional::InterpolateFuncOptions::mode_t mode,
    bool align_corners) {
    if (size.first == 0 && size.second == 0) {
        size = std::make_pair(
            static_cast<int64_t>(x.size(-2) * scale_factor),
            static_cast<int64_t>(x.size(-1) * scale_factor)
        );
    }

    constexpr int64_t int_max = 1610612736;
    int64_t input_elements = size.first * size.second * x.size(0) * x.size(1);

    if (input_elements > int_max) {
        auto chunks = torch::chunk(x, (input_elements / int_max) + 1, 0);
        std::vector<torch::Tensor> interpolated_chunks;
        for (const auto& chunk : chunks) {
            interpolated_chunks.push_back(
                torch::nn::functional::interpolate(
                    chunk,
                    torch::nn::functional::InterpolateFuncOptions()
                        .size(std::vector<int64_t>({size.first, size.second}))
                        .mode(mode)
                        .align_corners(align_corners)
                )
            );
        }
        return torch::cat(interpolated_chunks, 0).contiguous();
    } else {
        return torch::nn::functional::interpolate(
            x,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>({size.first, size.second}))
                .mode(mode)
                .align_corners(align_corners)
        );
    }
}

ResidualConvUnitImpl::ResidualConvUnitImpl(int64_t features, torch::nn::AnyModule activation, bool bn, int64_t groups)
    : groups(groups),
      activation(std::move(activation)),
      bn(bn) {
    conv1 = register_module("conv1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(features, features, 3).stride(1).padding(1).groups(groups).bias(true)
    ));
    conv2 = register_module("conv2", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(features, features, 3).stride(1).padding(1).groups(groups).bias(true)
    ));
    register_module("activation", activation.ptr());
}

torch::Tensor ResidualConvUnitImpl::forward(torch::Tensor x) {
    auto out = activation.forward<torch::Tensor>(x);
    out = conv1->forward(out);
    if (norm1.is_empty()) {
      out = norm1.forward<torch::Tensor>(out);
    }
    out = activation.forward<torch::Tensor>(out);
    out = conv2->forward(out);
    if (norm2.is_empty()) {
      out = norm2.forward<torch::Tensor>(out);
    }
    return torch::add(out, x);
}

} // namespace heads
} // namespace vggt
