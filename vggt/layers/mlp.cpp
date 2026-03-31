/**
 * @file mlp.cpp
 * @brief Implementation of Multi-Layer Perceptron module
 * @see mlp.h
 */

#include "mlp.h"

namespace vggt {
namespace layers {

/**
 * @brief Constructor for MlpImpl
 * 
 * Initializes the MLP with two linear layers, activation function, and dropout.
 * If hidden_features or out_features are -1, they default to in_features.
 * 
 * @param in_features Input feature dimension
 * @param hidden_features Hidden layer dimension
 * @param out_features Output dimension
 * @param act_layer Activation function module
 * @param drop Dropout probability
 * @param bias Whether to use bias in linear layers
 */
MlpImpl::MlpImpl(int64_t in_features, int64_t hidden_features, int64_t out_features,
                torch::nn::AnyModule act_layer, double drop, bool bias)
    : act(std::move(act_layer)) {
    out_features = out_features == -1 ? in_features : out_features;
    hidden_features = hidden_features == -1 ? in_features : hidden_features;

    fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(in_features, hidden_features).bias(bias)));
    register_module("act", act.ptr());
    drop1 = register_module("drop1", torch::nn::Dropout(drop));
    fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(hidden_features, out_features).bias(bias)));
    drop2 = register_module("drop2", torch::nn::Dropout(drop));
}

/**
 * @brief Forward pass
 * 
 * Passes input through: fc1 -> activation -> dropout1 -> fc2 -> dropout2
 * 
 * @param x Input tensor of shape [*, in_features]
 * @return Output tensor of shape [*, out_features]
 */
torch::Tensor MlpImpl::forward(torch::Tensor x) {
    x = fc1->forward(x);
    x = act.forward(x);
    x = drop1->forward(x);
    x = fc2->forward(x);
    x = drop2->forward(x);
    return x;
}

} // namespace layers
} // namespace vggt
