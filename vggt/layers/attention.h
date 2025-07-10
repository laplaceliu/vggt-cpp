/**
 * @file attention.h
 * @brief Attention layer implementation for vision transformers
 *
 * This file defines the Attention module which implements the multi-head self-attention
 * mechanism used in vision transformers. It supports both standard and fused attention
 * implementations, with optional query-key normalization.
 *
 * The module takes an input tensor and applies:
 * 1. Linear projection to queries, keys and values
 * 2. Optional layer normalization for queries and keys
 * 3. Scaled dot-product attention
 * 4. Linear projection of the output
 *
 * It supports both standard PyTorch implementation and fused attention kernels when available.
 */

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
