#include "drop_path.h"

namespace vggt {
namespace layers {

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

DropPathImpl::DropPathImpl(float drop_prob) : drop_prob(drop_prob) {}

torch::Tensor DropPathImpl::forward(torch::Tensor x) {
    return drop_path(x, drop_prob, is_training());
}

} // namespace layers
} // namespace vggt