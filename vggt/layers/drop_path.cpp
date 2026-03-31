/**
 * @file drop_path.cpp
 * @brief Implementation of DropPath (Stochastic Depth)
 * @see drop_path.h
 */

#include "drop_path.h"

namespace vggt {
namespace layers {

/**
 * @brief Stochastic depth drop_path implementation
 * 
 * During training, randomly drops entire residual paths with probability drop_prob.
 * The surviving paths are scaled by 1/(1-drop_prob) to maintain the expected sum.
 * During inference, returns the input unchanged.
 * 
 * @param x Input tensor of shape [B, ...] where B is batch size
 * @param drop_prob Probability of dropping a path (0.0 to 1.0)
 * @param training Whether in training mode
 * @return Tensor with stochastic depth applied
 */
torch::Tensor drop_path(torch::Tensor x, float drop_prob, bool training) {
    if (drop_prob == 0.0f || !training) {
        return x;
    }
    float keep_prob = 1 - drop_prob;
    auto shape = std::vector<int64_t>(x.dim(), 1);
    shape[0] = x.size(0);
    auto random_tensor = torch::empty(shape, x.options()).bernoulli_(keep_prob);
    if (keep_prob > 0.0f) {
        random_tensor.div_(keep_prob);
    }
    return x * random_tensor;
}

/**
 * @brief Constructor
 * @param drop_prob Probability of dropping a path during training
 */
DropPathImpl::DropPathImpl(float drop_prob) : drop_prob(drop_prob) {}

/**
 * @brief Forward pass
 * @param x Input tensor
 * @return Tensor with drop_path applied based on training mode
 */
torch::Tensor DropPathImpl::forward(torch::Tensor x) {
    return drop_path(x, drop_prob, is_training());
}

} // namespace layers
} // namespace vggt