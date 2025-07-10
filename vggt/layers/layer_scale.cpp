// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the Apache License, Version 2.0
// found in the LICENSE file in the root directory of this source tree.

#include "layer_scale.h"

namespace vggt {
namespace layers {

LayerScaleImpl::LayerScaleImpl(
    int64_t dim,
    torch::Tensor init_values,
    bool inplace)
    : inplace_(inplace),
      gamma_(register_parameter("gamma", init_values * torch::ones({dim}))) {}

torch::Tensor LayerScaleImpl::forward(const torch::Tensor& x) {
    if (inplace_) {
        return x.mul_(gamma_);
    }
    return x * gamma_;
}

} // namespace layers
} // namespace vggt
