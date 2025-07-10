// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the Apache License, Version 2.0
// found in the LICENSE file in the root directory of this source tree.

#include "block.h"

namespace vggt {
namespace layers {

BlockImpl::BlockImpl(
    int64_t dim,
    int64_t num_heads,
    double mlp_ratio,
    bool qkv_bias,
    bool proj_bias,
    bool ffn_bias,
    double drop,
    double attn_drop,
    torch::Tensor init_values,
    double drop_path,
    bool qk_norm,
    bool fused_attn)
    : norm1(torch::nn::LayerNormOptions({dim})),
      attn(Attention(dim, num_heads, qkv_bias, proj_bias, attn_drop, drop, qk_norm, fused_attn)),
      ls1(init_values.defined() ? LayerScale(dim, init_values) : nullptr),
      drop_path1(drop_path > 0.0 ? DropPath(drop_path) : nullptr),
      norm2(torch::nn::LayerNormOptions({dim})),
      mlp(Mlp(dim, static_cast<int64_t>(dim * mlp_ratio), ffn_bias, drop)),
      ls2(init_values.defined() ? LayerScale(dim, init_values) : nullptr),
      drop_path2(drop_path > 0.0 ? DropPath(drop_path) : nullptr),
      sample_drop_ratio(drop_path) {

    register_module("norm1", norm1);
    register_module("attn", attn);
    if (ls1 != nullptr) {
        register_module("ls1", ls1);
    }
    if (drop_path1 != nullptr) {
        register_module("drop_path1", drop_path1);
    }
    register_module("norm2", norm2);
    register_module("mlp", mlp);
    if (ls2 != nullptr) {
        register_module("ls2", ls2);
    }
    if (drop_path2 != nullptr) {
        register_module("drop_path2", drop_path2);
    }
}

torch::Tensor BlockImpl::forward(const torch::Tensor& x, const torch::Tensor& pos) {
    auto attn_residual = [&](const torch::Tensor& x) {
        auto normed = norm1->forward(x);
        auto attn_out = attn->forward(normed, pos);
        if (ls1 != nullptr) {
            attn_out = ls1->forward(attn_out);
        }
        return attn_out;
    };

    auto ffn_residual = [&](const torch::Tensor& x) {
        auto normed = norm2->forward(x);
        auto mlp_out = mlp->forward(normed);
        if (ls2 != nullptr) {
            mlp_out = ls2->forward(mlp_out);
        }
        return mlp_out;
    };

    torch::Tensor out = x;
    if (this->is_training() && sample_drop_ratio > 0.0) {
        if (drop_path1 != nullptr) {
            out = out + drop_path1->forward(attn_residual(out));
        } else {
            out = out + attn_residual(out);
        }
        if (drop_path2 != nullptr) {
            out = out + drop_path2->forward(ffn_residual(out));
        } else {
            out = out + ffn_residual(out);
        }
    } else {
        out = out + attn_residual(out);
        out = out + ffn_residual(out);
    }

    return out;
}

} // namespace layers
} // namespace vggt
