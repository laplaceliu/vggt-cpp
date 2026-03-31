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
    // NOTE: The member initializer list uses the original hidden_features/out_features values
    // before they are adjusted in the constructor body. This means hidden_features=-1
    // will cause a negative dimension error during Linear construction.
    // Only pass positive values for hidden_features when using this constructor.
    out_features = out_features == -1 ? in_features : out_features;
    hidden_features = hidden_features == -1 ? in_features : hidden_features;
}

torch::Tensor SwiGLUFFNImpl::forward(torch::Tensor x) {
    // SwiGLU computation: f(x) = silu(x @ W1) * (x @ W2) @ W3
    // where the first linear projects to 2*hidden_features, then split into two halves
    auto x12 = w12->forward(x);
    auto chunks = x12.chunk(2, -1);  // Split along last dimension
    auto hidden = torch::silu(chunks[0]) * chunks[1];  // SiLU gate
    return w3->forward(hidden);
}

SwiGLUFFNFusedImpl::SwiGLUFFNFusedImpl(int64_t in_features,
                                      int64_t hidden_features,
                                      int64_t out_features,
                                      torch::nn::AnyModule act_layer,
                                      double drop,
                                      bool bias)
    // Computes: (hidden_features * 2 / 3 + 7) / 8 * 8
    // This rounds up to nearest multiple of 8 for better hardware utilization
    : SwiGLUFFNImpl(in_features, (hidden_features * 2 / 3 + 7) / 8 * 8, out_features, act_layer, drop, bias) {}
}
} // namespace vggt
