// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the Apache License, Version 2.0
// found in the LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/nn/module.h>
#include <torch/nn/modules/normalization.h>
#include "attention.h"
#include "drop_path.h"
#include "layer_scale.h"
#include "mlp.h"

namespace vggt {
namespace layers {

class BlockImpl : public torch::nn::Module {
public:
    BlockImpl(
        int64_t dim,
        int64_t num_heads,
        double mlp_ratio = 4.0,
        bool qkv_bias = true,
        bool proj_bias = true,
        bool ffn_bias = true,
        double drop = 0.0,
        double attn_drop = 0.0,
        torch::Tensor init_values = {},
        double drop_path = 0.0,
        bool qk_norm = false,
        bool fused_attn = true);

    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& pos = {});

private:
    torch::nn::LayerNorm norm1{nullptr};
    Attention attn{nullptr};
    LayerScale ls1{nullptr};
    DropPath drop_path1{nullptr};
    torch::nn::LayerNorm norm2{nullptr};
    Mlp mlp{nullptr};
    LayerScale ls2{nullptr};
    DropPath drop_path2{nullptr};
    double sample_drop_ratio;
};

TORCH_MODULE(Block);

} // namespace layers
} // namespace vggt
