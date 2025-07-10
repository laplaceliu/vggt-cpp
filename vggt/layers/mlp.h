/**
 * @file mlp.h
 * @brief Multi-layer perceptron implementation for vision transformers
 *
 * This file defines the Mlp module which implements a standard multi-layer perceptron
 * used in vision transformers. The MLP consists of:
 *
 * 1. A first linear layer that projects from input dimension to hidden dimension
 * 2. An activation function (default: GELU)
 * 3. Optional dropout for regularization
 * 4. A second linear layer that projects from hidden dimension to output dimension
 *
 * This implementation follows the standard MLP architecture used in transformer models,
 * typically applied after the attention mechanism in each transformer block.
 */

#pragma once

#include <torch/nn/module.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/options/activation.h>

namespace vggt {
namespace layers {

class MlpImpl : public torch::nn::Module {
public:
    MlpImpl(
        int64_t in_features,
        int64_t hidden_features = -1,
        int64_t out_features = -1,
        const torch::nn::Functional& act_layer = torch::nn::Functional(torch::gelu),
        double drop = 0.0,
        bool bias = true);

    torch::Tensor forward(const torch::Tensor& x);

private:
    torch::nn::Linear fc1{nullptr};
    torch::nn::Functional act{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Dropout drop{nullptr};
    int64_t in_features_;
    int64_t hidden_features_;
    int64_t out_features_;
};

TORCH_MODULE(Mlp);

} // namespace layers
} // namespace vggt
