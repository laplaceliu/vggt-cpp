#include "swiglu_ffn.h"

namespace vggt {
namespace layers {
SwiGLUFFNImpl::SwiGLUFFNImpl(int64_t in_features,
                             int64_t hidden_features,
                             int64_t out_features,
                             torch::nn::AnyModule act_layer,
                             double drop,
                             bool bias)
    : w12(register_module("w12", torch::nn::Linear(torch::nn::LinearOptions(in_features, 2 * hidden_features).bias(bias)))),
      w3(register_module("w3", torch::nn::Linear(torch::nn::LinearOptions(hidden_features, out_features).bias(bias)))) {
    out_features = out_features == -1 ? in_features : out_features;
    hidden_features = hidden_features == -1 ? in_features : hidden_features;
}

torch::Tensor SwiGLUFFNImpl::forward(torch::Tensor x) {
    auto x12 = w12->forward(x);
    auto chunks = x12.chunk(2, -1);
    auto hidden = torch::silu(chunks[0]) * chunks[1];
    return w3->forward(hidden);
}

SwiGLUFFNFusedImpl::SwiGLUFFNFusedImpl(int64_t in_features,
                                      int64_t hidden_features,
                                      int64_t out_features,
                                      torch::nn::AnyModule act_layer,
                                      double drop,
                                      bool bias)
    : SwiGLUFFNImpl(in_features, (hidden_features * 2 / 3 + 7) / 8 * 8, out_features, act_layer, drop, bias) {}
}
} // namespace vggt
