/**
 * @file drop_path.h
 * @brief Drop path (stochastic depth) implementation for vision transformers
 *
 * This file defines the DropPath module which implements stochastic depth regularization
 * technique used in vision transformers. It randomly drops entire paths (channels) during
 * training to improve generalization.
 *
 * The module takes an input tensor and applies:
 * 1. During training: randomly zeroes entire channels with probability drop_prob
 * 2. During inference: passes the input unchanged
 *
 * This is particularly useful when applied in the main path of residual blocks.
 */

#pragma once

#include <torch/nn/module.h>

namespace vggt {
namespace layers {

/**
 * Drop paths (Stochastic Depth) per sample when applied in main path of residual blocks
 */
torch::Tensor drop_path(
    const torch::Tensor& x,
    double drop_prob = 0.0,
    bool training = false);

class DropPathImpl : public torch::nn::Module {
public:
    DropPathImpl(double drop_prob = 0.0);

    torch::Tensor forward(const torch::Tensor& x);

private:
    double drop_prob_;
};

TORCH_MODULE(DropPath);

} // namespace layers
} // namespace vggt