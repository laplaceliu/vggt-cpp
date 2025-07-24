#include "layer_scale.h"

namespace vggt {

LayerScaleImpl::LayerScaleImpl(int64_t dim, torch::Tensor init_values, bool inplace)
    : inplace(inplace) {
    gamma = register_parameter("gamma", init_values * torch::ones(dim));
}

LayerScaleImpl::LayerScaleImpl(int64_t dim, double init_value, bool inplace)
    : inplace(inplace) {
    gamma = register_parameter("gamma", init_value * torch::ones(dim));
}

torch::Tensor LayerScaleImpl::forward(torch::Tensor x) {
    return inplace ? x.mul_(gamma) : x.mul(gamma);
}

} // namespace vggt
