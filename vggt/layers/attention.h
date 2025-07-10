// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the Apache License, Version 2.0
// found in the LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/nn/module.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/normalization.h>

namespace vggt {
namespace layers {

class AttentionImpl : public torch::nn::Module {
public:
    AttentionImpl(
        int64_t dim,
        int64_t num_heads = 8,
        bool qkv_bias = true,
        bool proj_bias = true,
        double attn_drop = 0.0,
        double proj_drop = 0.0,
        bool qk_norm = false,
        bool fused_attn = true);

    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& pos = {});

private:
    int64_t num_heads_;
    int64_t head_dim_;
    double scale_;
    bool fused_attn_;
    torch::nn::Linear qkv{nullptr};
    torch::nn::LayerNorm q_norm{nullptr};
    torch::nn::LayerNorm k_norm{nullptr};
    torch::nn::Dropout attn_drop{nullptr};
    torch::nn::Linear proj{nullptr};
    torch::nn::Dropout proj_drop{nullptr};
};

TORCH_MODULE(Attention);

} // namespace layers
} // namespace vggt
