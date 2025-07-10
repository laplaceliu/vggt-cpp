// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the Apache License, Version 2.0
// found in the LICENSE file in the root directory of this source tree.

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
}

DropPathImpl::DropPathImpl(double drop_prob) : drop_prob_(drop_prob) {}

torch::Tensor DropPathImpl::forward(const torch::Tensor& x) {
    return drop_path(x, drop_prob_, this->is_training());
}

} // namespace layers
} // namespace vggt
