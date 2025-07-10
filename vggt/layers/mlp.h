// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the Apache License, Version 2.0
// found in the LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/nn/module.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/options/activation.h>

namespace vggt {
namespace layers {

class MlpImpl : public torch::nn::Module {
public:
    MlpImpl(
        int64_t in_features,
        int64_t hidden_features = -1,
        int64_t out_features = -1,
        const torch::nn::Functional& act_layer = torch::nn::Functional(torch::gelu),
        double drop = 0.0,
        bool bias = true);

    torch::Tensor forward(const torch::Tensor& x);

private:
    torch::nn::Linear fc1{nullptr};
    torch::nn::Functional act{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Dropout drop{nullptr};
    int64_t in_features_;
    int64_t hidden_features_;
    int64_t out_features_;
};

TORCH_MODULE(Mlp);

} // namespace layers
} // namespace vggt
