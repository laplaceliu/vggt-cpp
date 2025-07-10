/**
 * @brief Implementation of layer scaling module for vision transformers
 *
 * This file implements the LayerScaleImpl class methods defined in layer_scale.h.
 * It provides the core functionality for applying learnable per-channel scaling
 * factors to transformer block outputs.
 *
 * The implementation includes:
 * 1. Constructor that initializes the scaling parameters (gamma) with the specified
 *    initial values and dimension
 * 2. Forward method that applies the scaling factors to the input tensor
 * 3. Support for both in-place operations (to optimize memory usage) and
 *    standard out-of-place operations
 *
 * Layer scaling is a simple yet effective technique for improving training
 * stability and performance in deep transformer networks.
 */

#include "layer_scale.h"

namespace vggt {
LayerScaleImpl::LayerScaleImpl(
    int64_t dim,
    torch::Tensor init_values,
    bool inplace)
    : inplace_(inplace),
      gamma_(register_parameter("gamma", init_values * torch::ones({dim}))) {}

torch::Tensor LayerScaleImpl::forward(const torch::Tensor& x) {
    if (inplace_) {
        return x.mul_(gamma_);
    }
    return x * gamma_;
}

} // namespace layers
} // namespace vggt
