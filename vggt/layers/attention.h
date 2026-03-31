#pragma once

/**
 * @file attention.h
 * @brief Multi-head self-attention module for VGGT
 */

#include <torch/torch.h>

namespace vggt {
namespace layers {

// Forward declaration only
class RotaryPositionEmbedding2DImpl;

/**
 * @class AttentionImpl
 * @brief Multi-head self-attention module with optional RoPE and QK normalization
 * 
 * Implements standard multi-head attention with support for:
 * - Optional rotary position embedding (RoPE)
 * - Optional QK normalization
 * - Configurable dropout rates
 * - Fused or unfused attention computation
 */
class AttentionImpl : public torch::nn::Module {
public:
    /**
     * @brief Constructor for Attention module
     * @param dim Input and output dimension
     * @param num_heads Number of attention heads
     * @param qkv_bias Whether to use bias in QKV projection
     * @param proj_bias Whether to use bias in output projection
     * @param attn_drop Dropout probability for attention weights
     * @param proj_drop Dropout probability for output
     * @param norm_layer Normalization layer for QK normalization
     * @param qk_norm Whether to apply QK normalization
     * @param fused_attn Whether to use fused attention
     * @param rope Rotary position embedding module
     */
    AttentionImpl(
        int64_t dim,
        int64_t num_heads = 8,
        bool qkv_bias = true,
        bool proj_bias = true,
        double attn_drop = 0.0,
        double proj_drop = 0.0,
        torch::nn::AnyModule norm_layer = torch::nn::AnyModule(torch::nn::LayerNorm(torch::nn::LayerNormOptions({}))),
        bool qk_norm = false,
        bool fused_attn = true,
        torch::nn::AnyModule rope = torch::nn::AnyModule()
    );

    /**
     * @brief Forward pass
     * @param x Input tensor of shape [B, N, C]
     * @param pos Position embeddings (optional, for RoPE)
     * @return Output tensor of shape [B, N, C]
     */
    torch::Tensor forward(torch::Tensor x, torch::Tensor pos = {});

private:
    int64_t num_heads;
    int64_t head_dim;
    double scale;
    bool fused_attn;

    torch::nn::Linear qkv{nullptr};
    torch::nn::LayerNorm q_norm{nullptr};
    torch::nn::LayerNorm k_norm{nullptr};
    torch::nn::Identity q_norm_identity{nullptr};
    torch::nn::Identity k_norm_identity{nullptr};
    torch::nn::Dropout attn_drop{nullptr};
    torch::nn::Linear proj{nullptr};
    torch::nn::Dropout proj_drop{nullptr};
    torch::nn::AnyModule rope;
    bool use_qk_norm;
};
TORCH_MODULE(Attention);

/**
 * @class MemEffAttentionImpl
 * @brief Memory-efficient attention implementation
 * 
 * Uses xFormers for memory-efficient attention when available.
 * Falls back to standard attention otherwise.
 */
class MemEffAttentionImpl : public AttentionImpl {
public:
    using AttentionImpl::AttentionImpl;

    /**
     * @brief Forward pass for memory-efficient attention
     * @param x Input tensor of shape [B, N, C]
     * @param attn_bias Attention bias (for xFormers)
     * @param pos Position embeddings (optional)
     * @return Output tensor of shape [B, N, C]
     */
    torch::Tensor forward(torch::Tensor x, torch::Tensor attn_bias = {}, torch::Tensor pos = {});
};
TORCH_MODULE(MemEffAttention);

} // namespace layers
} // namespace vggt
