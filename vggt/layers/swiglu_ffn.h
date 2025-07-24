#pragma once

#include <torch/torch.h>

namespace vggt {
namespace layers {
class SwiGLUFFNImpl : public torch::nn::Module {
public:
    SwiGLUFFNImpl(int64_t in_features,
                  int64_t hidden_features = -1,
                  int64_t out_features = -1,
                  torch::nn::AnyModule act_layer = torch::nn::AnyModule(),
                  double drop = 0.0,
                  bool bias = true);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear w12{nullptr}, w3{nullptr};
};
TORCH_MODULE(SwiGLUFFN);

class SwiGLUFFNFusedImpl : public SwiGLUFFNImpl {
public:
    SwiGLUFFNFusedImpl(int64_t in_features,
                       int64_t hidden_features = -1,
                       int64_t out_features = -1,
                       torch::nn::AnyModule act_layer = torch::nn::AnyModule(),
                       double drop = 0.0,
                       bool bias = true);
};
TORCH_MODULE(SwiGLUFFNFused);
}
} // namespace vggt
