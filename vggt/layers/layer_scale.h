#pragma once

/**
 * @file layer_scale.h
 * @brief LayerScale implementation for VGGT
 * 
 * LayerScale is a technique that multiplies the output of a sublayer
 * by a learnable parameter gamma, improving training stability in deep networks.
 */

#include <torch/torch.h>

namespace vggt {
namespace layers {

/**
 * @class LayerScaleImpl
 * @brief Learnable scaling module for layer outputs
 * 
 * Multiplies the input by a learnable gamma parameter, initialized to a small value.
 * Can operate in-place for memory efficiency.
 */
class LayerScaleImpl : public torch::nn::Module {
public:
    /**
     * @brief Constructor with custom initial values
     * @param dim Feature dimension
     * @param init_values Initial values for gamma scaling
     * @param inplace Whether to modify input in-place (default: false)
     */
    LayerScaleImpl(int64_t dim, torch::Tensor init_values, bool inplace = false);
    
    /**
     * @brief Constructor with scalar initial value
     * @param dim Feature dimension
     * @param init_value Initial value for gamma (default: 1e-5)
     * @param inplace Whether to modify input in-place (default: false)
     */
    LayerScaleImpl(int64_t dim, double init_value = 1e-5, bool inplace = false);
    
    /**
     * @brief Forward pass
     * @param x Input tensor of shape [..., dim]
     * @return Scaled tensor of shape [..., dim]
     */
    torch::Tensor forward(torch::Tensor x);

private:
    torch::Tensor gamma;  ///< Learnable scaling parameter
    bool inplace;        ///< Whether to use in-place operation
};

TORCH_MODULE(LayerScale);
}
} // namespace vggt
