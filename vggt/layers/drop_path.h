#pragma once

#include <torch/torch.h>

namespace vggt {
namespace layers {

torch::Tensor drop_path(torch::Tensor x, float drop_prob = 0.0f, bool training = false);

class DropPathImpl : public torch::nn::Module {
public:
    explicit DropPathImpl(float drop_prob = 0.0f);
    torch::Tensor forward(torch::Tensor x);

private:
    float drop_prob;
};

TORCH_MODULE(DropPath);

} // namespace layers
} // namespace vggt