/**
 * @file drop_path.cpp
 * @brief Implementation of the DropPath module for vision transformers
 *
 * This file implements the DropPathImpl class methods defined in drop_path.h.
 * It provides the core functionality for stochastic depth regularization
 * used in vision transformers, including:
 *
 * 1. A standalone drop_path function that can be used independently
 * 2. A DropPathImpl class that wraps the function as a module
 * 3. Support for both training and inference modes
 *
 * The implementation uses PyTorch's tensor operations to ensure
 * compatibility with the PyTorch ecosystem and efficient execution on both
 * CPU and GPU devices.
 */

#include "drop_path.h"

namespace vggt {
namespace layers {

torch::Tensor drop_path(const torch::Tensor& x, double drop_prob, bool training) {
    if (drop_prob == 0.0 || !training) {
        return x;
    }
    
    double keep_prob = 1.0 - drop_prob;
    
    // Create shape for the random tensor: (batch_size, 1, 1, ..., 1)
    // This works with tensors of any dimension, not just 2D ConvNets
    std::vector<int64_t> shape = {x.size(0)};
    for (int64_t i = 1; i < x.dim(); i++) {
        shape.push_back(1);
    }
    
    // Create a random tensor with the same device and dtype as x
    auto random_tensor = torch::empty(shape, x.options()).bernoulli_(keep_prob);
    
    if (keep_prob > 0.0) {
        random_tensor.div_(keep_prob);
    }
    
    return x * random_tensor;
}

DropPathImpl::DropPathImpl(double drop_prob)
    : drop_prob_(drop_prob) {
}

torch::Tensor DropPathImpl::forward(const torch::Tensor& x) {
    return drop_path(x, drop_prob_, is_training());
}

} // namespace layers
} // namespace vggt