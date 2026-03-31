#pragma once

#include <torch/torch.h>

namespace vggt {
namespace layers {

/**
 * @class SwiGLUFFNImpl
 * @brief SwiGLU Feed-Forward Network implementation
 * 
 * Implements the SwiGLU (Swish-Gated Linear Unit) feed-forward network
 * as described in "GLU Variants Improve Transformer" (https://arxiv.org/abs/2002.05202).
 * 
 * The SwiGLU formula: f(X) = silu(X @ W1) * (X @ W2) @ W3
 * where silu(x) = x * sigmoid(x)
 * 
 * @param in_features Input feature dimension
 * @param hidden_features Hidden layer dimension (should be positive value, not -1)
 * @param out_features Output feature dimension (defaults to in_features if -1)
 * @param act_layer Activation layer (currently unused, silu is hardcoded)
 * @param drop Dropout rate
 * @param bias Whether to use bias in linear layers
 */
class SwiGLUFFNImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new SwiGLUFFN layer
     * @param in_features Input feature dimension
     * @param hidden_features Hidden layer dimension (must be positive, not -1)
     * @param out_features Output feature dimension (-1 means same as in_features)
     * @param act_layer Activation layer (unused, silu is hardcoded)
     * @param drop Dropout rate
     * @param bias Whether to use bias in linear layers
     */
    SwiGLUFFNImpl(int64_t in_features,
                  int64_t hidden_features = -1,
                  int64_t out_features = -1,
                  torch::nn::AnyModule act_layer = torch::nn::AnyModule(),
                  double drop = 0.0,
                  bool bias = true);
    
    /**
     * @brief Forward pass of SwiGLUFFN
     * @param x Input tensor of shape [*, in_features]
     * @return Output tensor of shape [*, out_features]
     */
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear w12{nullptr}, w3{nullptr};  ///< First projection and gate; w12 = [W1, W2]
};
TORCH_MODULE(SwiGLUFFN);

/**
 * @class SwiGLUFFNFusedImpl
 * @brief Fused SwiGLUFFN with optimized hidden dimension calculation
 * 
 * Computes hidden_features as: (hidden_features * 2 / 3 + 7) / 8 * 8
 * This ensures the hidden dimension is a multiple of 8 and uses the
 * standard SwiGLU hidden dimension formula.
 * 
 * @param in_features Input feature dimension
 * @param hidden_features Base hidden dimension for calculation
 * @param out_features Output feature dimension
 * @param act_layer Activation layer (unused)
 * @param drop Dropout rate
 * @param bias Whether to use bias
 */
class SwiGLUFFNFusedImpl : public SwiGLUFFNImpl {
public:
    /**
     * @brief Construct a fused SwiGLUFFN layer
     * @param in_features Input feature dimension
     * @param hidden_features Base hidden dimension
     * @param out_features Output feature dimension
     * @param act_layer Activation layer
     * @param drop Dropout rate
     * @param bias Whether to use bias
     */
    SwiGLUFFNFusedImpl(int64_t in_features,
                       int64_t hidden_features = -1,
                       int64_t out_features = -1,
                       torch::nn::AnyModule act_layer = torch::nn::AnyModule(),
                       double drop = 0.0,
                       bool bias = true);
};
TORCH_MODULE(SwiGLUFFNFused);

} // namespace layers
} // namespace vggt
