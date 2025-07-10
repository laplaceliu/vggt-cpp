// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the Apache License, Version 2.0
// found in the LICENSE file in the root directory of this source tree.

#include "attention.h"

namespace vggt {
namespace layers {

AttentionImpl::AttentionImpl(
    int64_t dim,
    int64_t num_heads,
    bool qkv_bias,
    bool proj_bias,
    double attn_drop,
    double proj_drop,
    bool qk_norm,
    bool fused_attn)
    : num_heads_(num_heads),
      head_dim_(dim / num_heads),
      scale_(std::pow(head_dim_, -0.5)),
      fused_attn_(fused_attn),
      qkv(torch::nn::LinearOptions(dim, dim * 3).bias(qkv_bias)),
      attn_drop(attn_drop),
      proj(torch::nn::LinearOptions(dim, dim).bias(proj_bias)),
      proj_drop(proj_drop) {

    if (qk_norm) {
        q_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({head_dim_}));
        k_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({head_dim_}));
    } else {
        q_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({0}));
        k_norm = torch::nn::LayerNorm(torch::nn::LayerNormOptions({0}));
    }

    register_module("qkv", qkv);
    register_module("q_norm", q_norm);
    register_module("k_norm", k_norm);
    register_module("proj", proj);
}

torch::Tensor AttentionImpl::forward(const torch::Tensor& x, const torch::Tensor& pos) {
    auto B = x.size(0);
    auto N = x.size(1);
    auto C = x.size(2);

    auto qkv = this->qkv->forward(x)
        .reshape({B, N, 3, this->num_heads_, this->head_dim_})
        .permute({2, 0, 3, 1, 4});

    auto q = qkv[0];
    auto k = qkv[1];
    auto v = qkv[2];

    q = this->q_norm->forward(q);
    k = this->k_norm->forward(k);

    torch::Tensor attn_output;
    if (this->fused_attn_) {
        attn_output = torch::scaled_dot_product_attention(
            q, k, v,
            {},
            this->is_training() ? this->attn_drop->options().p() : 0.0);
    } else {
        q = q * this->scale_;
        auto attn = (q @ k.transpose(-2, -1));
        attn = torch::softmax(attn, -1);
        attn = this->attn_drop->forward(attn);
        attn_output = attn @ v;
    }

    attn_output = attn_output.transpose(1, 2).reshape({B, N, C});
    attn_output = this->proj->forward(attn_output);
    attn_output = this->proj_drop->forward(attn_output);

    return attn_output;
}

} // namespace layers
} // namespace vggt
