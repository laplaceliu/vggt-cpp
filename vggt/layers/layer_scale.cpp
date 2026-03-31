/**
 * @file layer_scale.cpp
 * @brief Implementation of LayerScale module
 * @see layer_scale.h
 */

#include "layer_scale.h"

namespace vggt {
namespace layers {

/**
 * @brief Constructor with custom initial values tensor
 * @param dim Feature dimension
 * @param init_values Tensor of initial values, will be broadcast to dim
 * @param inplace Whether to use in-place multiplication
 */
LayerScaleImpl::LayerScaleImpl(int64_t dim, torch::Tensor init_values, bool inplace)
    : inplace(inplace) {
    gamma = register_parameter("gamma", init_values * torch::ones(dim));
}

/**
 * @brief Constructor with scalar initial value
 * @param dim Feature dimension
 * @param init_value Scalar initial value for gamma
 * @param inplace Whether to use in-place multiplication
 */
LayerScaleImpl::LayerScaleImpl(int64_t dim, double init_value, bool inplace)
    : inplace(inplace) {
    gamma = register_parameter("gamma", init_value * torch::ones(dim));
}

/**
 * @brief Forward pass
 * 
 * Multiplies input by gamma: y = x * gamma
 * 
 * @param x Input tensor of shape [..., dim]
 * @return Scaled tensor of shape [..., dim]
 */
torch::Tensor LayerScaleImpl::forward(torch::Tensor x) {
    return inplace ? x.mul_(gamma) : x.mul(gamma);
}
}
} // namespace vggt
