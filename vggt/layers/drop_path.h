// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the Apache License, Version 2.0
// found in the LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/nn/module.h>

namespace vggt {
namespace layers {

torch::Tensor drop_path(
    torch::Tensor x,
    double drop_prob = 0.0,
    bool training = false);

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
