#pragma once

/**
 * @file drop_path.h
 * @brief DropPath (Stochastic Depth) implementation for VGGT
 * 
 * DropPath randomly drops entire residual branches during training
 * to improve regularization and training stability.
 */

#include <torch/torch.h>

namespace vggt {
namespace layers {

/**
 * @brief Stochastic depth drop_path function
 * 
 * Randomly drops entire paths during training with probability drop_prob.
 * When a path is dropped, it returns zeros scaled by 1/(1-drop_prob) to
 * maintain expected value during training.
 * 
 * @param x Input tensor
 * @param drop_prob Probability of dropping the path (default: 0.0)
 * @param training Whether currently in training mode (default: false)
 * @return Tensor with dropped paths scaled appropriately
 */
torch::Tensor drop_path(torch::Tensor x, float drop_prob = 0.0f, bool training = false);

/**
 * @class DropPathImpl
 * @brief Module wrapper for drop_path with configurable drop probability
 */
class DropPathImpl : public torch::nn::Module {
public:
    /** @brief Constructor with drop probability */
    explicit DropPathImpl(float drop_prob = 0.0f);
    
    /** @brief Forward pass using module's training state */
    torch::Tensor forward(torch::Tensor x);

private:
    float drop_prob;
};

TORCH_MODULE(DropPath);

} // namespace layers
} // namespace vggt