/**
 * @file mlp.cpp
 * @brief Implementation of the Multi-layer perceptron for vision transformers
 *
 * This file implements the MlpImpl class methods defined in mlp.h.
 * It provides the core functionality for the MLP component used in
 * vision transformers, including:
 *
 * 1. Constructor for initializing the MLP with specified parameters
 * 2. Forward method that implements the MLP computation logic
 * 3. Two linear layers with configurable dimensions
 * 4. Activation function (default: GELU)
 * 5. Optional dropout for regularization
 * 6. Automatic feature dimension handling
 * 7. Module registration for PyTorch serialization
 * 8. Support for both training and inference modes
 *
 * The implementation follows the standard MLP architecture used in
 * transformer models, typically applied after the attention mechanism
 * in each transformer block. Key features include:
 * - Flexible input/output feature dimensions
 * - Optional hidden dimension expansion
 * - Support for custom activation functions
 * - Dropout applied after each linear layer
 * - Efficient memory layout for tensor operations
 */

#include "mlp.h"

namespace vggt {
namespace layers {

MlpImpl::MlpImpl(
    int64_t in_features,
    int64_t hidden_features,
    int64_t out_features,
    const torch::nn::Functional& act_layer,
    double drop,
    bool bias)
    : in_features_(in_features),
      hidden_features_(hidden_features == -1 ? in_features : hidden_features),
      out_features_(out_features == -1 ? in_features : out_features),
      act(act_layer),
      drop(torch::nn::DropoutOptions(drop)) {
    // Register layers
    fc1 = register_module("fc1",
        torch::nn::Linear(torch::nn::LinearOptions(in_features_, hidden_features_).bias(bias)));
    fc2 = register_module("fc2",
        torch::nn::Linear(torch::nn::LinearOptions(hidden_features_, out_features_).bias(bias)));
}

torch::Tensor MlpImpl::forward(const torch::Tensor& x) {
    x = fc1->forward(x);
    x = act->forward(x);
    x = drop->forward(x);
    x = fc2->forward(x);
    x = drop->forward(x);
    return x;
}

} // namespace layers
} // namespace vggt
