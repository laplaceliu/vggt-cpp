#include "dpt_head.h"
#include <torch/nn/functional.h>

namespace vggt {
namespace heads {

utils::StackSequential _make_fusion_block(int64_t features, int64_t size, bool has_residual, int64_t groups) {
    utils::StackSequential block;
    if (has_residual) {
        block->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
        block->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(features, features, 3).stride(1).padding(1).groups(groups).bias(false)));
        block->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
        block->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(features, features, 3).stride(1).padding(1).groups(groups).bias(false)));
    }
    return block;
}

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

    scratch.register_module("layer1_rn", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_shape[0], out_shape1, 3).stride(1).padding(1).groups(groups).bias(false)));
    scratch.register_module("layer2_rn", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_shape[1], out_shape2, 3).stride(1).padding(1).groups(groups).bias(false)));
    scratch.register_module("layer3_rn", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_shape[2], out_shape3, 3).stride(1).padding(1).groups(groups).bias(false)));
    if (in_shape.size() >= 4) {
        scratch.register_module("layer4_rn", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_shape[3], out_shape4, 3).stride(1).padding(1).groups(groups).bias(false)));
    }
    return scratch;
}

torch::Tensor custom_interpolate(const torch::Tensor& input, const std::pair<int64_t, int64_t>& size, const torch::nn::functional::InterpolateFuncOptions::mode_t& mode, bool align_corners) {
    return torch::nn::functional::interpolate(
        input,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{size.first, size.second})
            .mode(mode)
            .align_corners(align_corners)
    );
}

} // namespace heads
} // namespace vggt
