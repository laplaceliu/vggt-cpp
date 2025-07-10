/**
 * @brief Layer scaling module for vision transformers
 *
 * This file defines the LayerScale module which implements the layer scaling
 * technique introduced in "Going deeper with Image Transformers" (Touvron et al., 2021).
 * Layer scaling applies learnable per-channel scaling factors to the outputs of
 * transformer blocks, which helps stabilize training of deep transformer networks.
 *
 * The implementation includes:
 * 1. A LayerScaleImpl class that applies per-channel scaling to input tensors
 * 2. Support for both in-place and out-of-place operations
 * 3. Configurable initialization values for the scaling parameters
 *
 * Layer scaling is typically applied after attention and MLP blocks in transformer
 * architectures to control the contribution of each block to the overall network.
 */

#pragma once

#include <torch/nn/module.h>
#include <torch/types.h>

namespace vggt {
namespace layers {

class LayerScaleImpl : public torch::nn::Module {
public:
    LayerScaleImpl(
        int64_t dim,
        torch::Tensor init_values = torch::ones({1}) * 1e-5,
    torch::Tensor forward(const torch::Tensor& x);

private:
    bool inplace_;
    torch::Tensor gamma_;
};

TORCH_MODULE(LayerScale);

} // namespace layers
} // namespace vggt
