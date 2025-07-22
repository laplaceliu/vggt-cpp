/**
 * @file layer_scale.h
 * @brief Layer scale implementation for vision transformers
 *
 * This file defines the LayerScale module which applies a learnable per-channel scaling
 * factor to the input tensor. It is commonly used in vision transformers to stabilize training.
 */

#pragma once

#include <torch/nn/module.h>

namespace vggt {
namespace layers {

class LayerScaleImpl : public torch::nn::Module {
public:
    /**
     * Layer scale module that applies a learnable per-channel scaling factor
     *
     * @param dim Number of input channels
     * @param init_values Initial value for the scaling factors
     * @param inplace Whether to perform the operation in-place
     */
    LayerScaleImpl(
        int64_t dim,
        double init_values = 1e-5,
        bool inplace = false);

    /**
     * Forward pass of the layer scale module
     *
     * @param x Input tensor
     * @return Scaled tensor
     */
    torch::Tensor forward(const torch::Tensor& x);

private:
    bool inplace_;
    torch::Tensor gamma;
};

TORCH_MODULE(LayerScale);

} // namespace layers
} // namespace vggt