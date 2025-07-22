/**
 * @file layer_scale.cpp
 * @brief Implementation of the LayerScale module for vision transformers
 *
 * This file implements the LayerScaleImpl class methods defined in layer_scale.h.
 * It provides the functionality for applying learnable per-channel scaling factors
 * to input tensors, which helps stabilize training in vision transformers.
 */

#include "layer_scale.h"

namespace vggt {
namespace layers {

LayerScaleImpl::LayerScaleImpl(
    int64_t dim,
    double init_values,
    bool inplace)
    : inplace_(inplace) {
    
    // Initialize gamma parameter with ones multiplied by init_values
    gamma = register_parameter("gamma", torch::ones(dim) * init_values);
}

torch::Tensor LayerScaleImpl::forward(const torch::Tensor& x) {
    if (inplace_) {
        return x.mul_(gamma);
    } else {
        return x * gamma;
    }
}

} // namespace layers
} // namespace vggt