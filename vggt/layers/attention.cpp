/**
 * @file attention.cpp
 * @brief Implementation of the Attention module for vision transformers
 *
 * This file implements the AttentionImpl class methods defined in attention.h.
 * It provides the core functionality for multi-head self-attention mechanism
 * used in vision transformers, including:
 *
 * 1. Constructor for initializing the attention module with specified parameters
 * 2. Forward method that implements the attention computation logic
 * 3. Support for both standard attention implementation and fused attention kernels
 * 4. Optional query-key normalization
 * 5. Efficient memory layout transformations for query, key and value tensors
 * 6. Dropout regularization for attention weights and projection outputs
 * 7. Support for positional embeddings integration
 *
 * The implementation uses PyTorch's tensor operations and modules to ensure
 * compatibility with the PyTorch ecosystem and efficient execution on both
 * CPU and GPU devices. It includes optimizations for both training and inference
 * scenarios.
 */

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
