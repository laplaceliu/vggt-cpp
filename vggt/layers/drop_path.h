/**
 * @brief Drop path (Stochastic Depth) regularization for vision transformers
 *
 * This file defines the DropPath module which implements stochastic depth
 * regularization as described in "Deep Networks with Stochastic Depth"
 * (Huang et al., 2016). This technique randomly drops entire layers during
 * training to improve generalization and reduce overfitting.
 *
 * The implementation includes:
 * 1. A standalone drop_path function that can be applied to any tensor
 * 2. A DropPathImpl module class that can be integrated into neural network architectures
 *
 * During inference (when training=false), this module acts as an identity function.
 * During training, it randomly drops entire paths (channels) with probability drop_prob.
 */

#pragma once

#include <torch/nn/module.h>

namespace vggt {
namespace layers {

torch::Tensor drop_path(
    torch::Tensor x,
    double drop_prob = 0.0,
class DropPathImpl : public torch::nn::Module {
public:
    explicit DropPathImpl(double drop_prob = 0.0);

    torch::Tensor forward(const torch::Tensor& x);

private:
    double drop_prob_;
};

TORCH_MODULE(DropPath);

} // namespace layers
} // namespace vggt
