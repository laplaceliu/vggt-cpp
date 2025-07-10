/**
 * @brief Implementation of Drop path (Stochastic Depth) regularization
 *
 * This file implements the drop_path function and DropPathImpl class methods
 * defined in drop_path.h. It provides the core functionality for stochastic
 * depth regularization, which randomly drops entire paths during training.
 *
 * The implementation includes:
 * 1. A drop_path function that applies stochastic depth to input tensors
 *    by generating a random mask and scaling the remaining paths
 * 2. The DropPathImpl forward method that wraps the drop_path function
 *    for use as a module in neural network architectures
 *
 * During inference or when drop_prob is 0, this implementation acts as
 * an identity function, passing the input through unchanged.
 */

#include "drop_path.h"

namespace vggt {
namespace layers {

torch::Tensor drop_path(torch::Tensor x, double drop_prob, bool training) {
    if (drop_prob == 0.0 || !training) {
        return x;
    }
    double keep_prob = 1 - drop_prob;
    std::vector<int64_t> shape(x.dim(), 1);
    shape[0] = x.size(0);
    auto random_tensor = torch::empty(shape, x.options()).bernoulli_(keep_prob);
    if (keep_prob > 0.0) {
        random_tensor.div_(keep_prob);
    }
    return x * random_tensor;
DropPathImpl::DropPathImpl(double drop_prob) : drop_prob_(drop_prob) {}

torch::Tensor DropPathImpl::forward(const torch::Tensor& x) {
    return drop_path(x, drop_prob_, this->is_training());
}

} // namespace layers
} // namespace vggt
