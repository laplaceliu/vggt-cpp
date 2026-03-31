#pragma once

/**
 * @file mlp.h
 * @brief Multi-Layer Perceptron (MLP) implementation for VGGT
 */

#include <torch/torch.h>

namespace vggt {
namespace layers {

/**
 * @class MlpImpl
 * @brief Multi-Layer Perceptron module with dropout and configurable activation
 * 
 * Implements a standard two-layer MLP with optional dropout regularization.
 * Supports custom activation functions and automatic dimension inference.
 * 
 * @param in_features Input feature dimension
 * @param hidden_features Hidden layer dimension (defaults to in_features)
 * @param out_features Output dimension (defaults to in_features)
 * @param act_layer Activation function module (default: GELU)
 * @param drop Dropout probability (default: 0.0)
 * @param bias Whether to use bias in linear layers (default: true)
 */
class MlpImpl : public torch::nn::Module {
public:
    MlpImpl(int64_t in_features, int64_t hidden_features = -1, int64_t out_features = -1,
            torch::nn::AnyModule act_layer = torch::nn::AnyModule(torch::nn::GELU()),
            double drop = 0.0, bool bias = true);
    
    /**
     * @brief Forward pass through the MLP
     * @param x Input tensor of shape [..., in_features]
     * @return Output tensor of shape [..., out_features]
     */
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::AnyModule act;
    torch::nn::Dropout drop1{nullptr}, drop2{nullptr};
};

TORCH_MODULE(Mlp);

} // namespace layers
} // namespace vggt
