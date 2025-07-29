#pragma once

#include <torch/torch.h>
#include <vector>
#include <utility>

namespace vggt {
namespace heads {

torch::nn::Module _make_scratch(const std::vector<int64_t>& in_shape, int64_t out_shape, int64_t groups = 1, bool expand = false);
torch::Tensor custom_interpolate(
    torch::Tensor x,
    std::pair<int64_t, int64_t> size = std::pair<int64_t, int64_t>(),
    float scale_factor = -1.0f,
    torch::nn::functional::InterpolateFuncOptions::mode_t mode = torch::kBilinear,
    bool align_corners = true);

class ResidualConvUnitImpl : public torch::nn::Module {
public:
    ResidualConvUnitImpl(int64_t features, torch::nn::AnyModule activation, bool bn, int64_t groups = 1);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::BatchNorm2d bn{nullptr};
    int64_t groups;
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::AnyModule norm1, norm2;
    torch::nn::AnyModule activation;
    torch::nn::Functional skip_add{nullptr};
};
TORCH_MODULE(ResidualConvUnit);

} // namespace heads
} // namespace vggt
