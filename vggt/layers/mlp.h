#pragma once

#include <torch/torch.h>

namespace vggt {
namespace layers {

class MlpImpl : public torch::nn::Module {
public:
    MlpImpl(int64_t in_features, int64_t hidden_features = -1, int64_t out_features = -1,
            torch::nn::AnyModule act_layer = torch::nn::AnyModule(torch::nn::GELU()),
            double drop = 0.0, bool bias = true);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::AnyModule act;
    torch::nn::Dropout drop1{nullptr}, drop2{nullptr};
};

TORCH_MODULE(Mlp);

} // namespace layers
} // namespace vggt
