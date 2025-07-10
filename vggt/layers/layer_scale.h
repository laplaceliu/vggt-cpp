// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the Apache License, Version 2.0
// found in the LICENSE file in the root directory of this source tree.

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
        bool inplace = false);

    torch::Tensor forward(const torch::Tensor& x);

private:
    bool inplace_;
    torch::Tensor gamma_;
};

TORCH_MODULE(LayerScale);

} // namespace layers
} // namespace vggt
