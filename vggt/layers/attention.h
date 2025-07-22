/**
 * @file attention.h
 * @brief Attention mechanism implementation for vision transformers
 *
 * This file contains the implementation of the Attention module used in vision transformers.
 * References:
 *   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
 *   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py
 */

#pragma once

#include <torch/torch.h>

namespace vggt {
namespace layers {

/**
 * @brief Attention module for vision transformers
 *
 * This class implements the standard multi-head self-attention mechanism used in transformers.
 */
class AttentionImpl : public torch::nn::Module {
public:
    /**
     * @brief Constructor for Attention module
     *
     * @param dim Input dimension
     * @param num_heads Number of attention heads
     * @param qkv_bias Whether to use bias in QKV projection
     * @param proj_bias Whether to use bias in output projection
     * @param attn_drop Dropout rate for attention weights
     * @param proj_drop Dropout rate for output projection
     * @param norm_layer Normalization layer type
     * @param qk_norm Whether to apply normalization to query and key
     * @param fused_attn Whether to use fused attention (scaled_dot_product_attention)
     * @param rope Optional rotary position embedding
     */
    AttentionImpl(
        int64_t dim,
        int64_t num_heads = 8,
        bool qkv_bias = true,
        bool proj_bias = true,
        double attn_drop = 0.0,
        double proj_drop = 0.0,
        bool qk_norm = false,
        bool fused_attn = true,
        c10::optional<torch::nn::Module> rope = c10::nullopt
    );

    /**
     * @brief Forward pass of the attention module
     *
     * @param x Input tensor of shape [B, N, C]
     * @param pos Optional position tensor
     * @return torch::Tensor Output tensor of shape [B, N, C]
     */
    torch::Tensor forward(torch::Tensor x, c10::optional<torch::Tensor> pos = c10::nullopt);

    /**
     * @brief Returns a string describing the module
     *
     * @return std::string Module description
     */
    std::string pretty_print(int64_t indent) const;

private:
    int64_t num_heads_;
    int64_t head_dim_;
    float scale_;
    bool fused_attn_;

    torch::nn::Linear qkv_;
    torch::nn::Module q_norm_;
    torch::nn::Module k_norm_;
    torch::nn::Dropout attn_drop_;
    torch::nn::Linear proj_;
    torch::nn::Dropout proj_drop_;
    c10::optional<torch::nn::Module> rope_{c10::nullopt};
};

TORCH_MODULE(Attention);

/**
 * @brief Memory efficient attention implementation
 *
 * This class extends the standard Attention module with memory-efficient implementation
 * using xFormers if available.
 */
class MemEffAttentionImpl : public Attention {
public:
    using Attention::Attention;

    /**
     * @brief Forward pass of the memory-efficient attention module
     *
     * @param x Input tensor of shape [B, N, C]
     * @param attn_bias Optional attention bias tensor
     * @param pos Optional position tensor
     * @return torch::Tensor Output tensor of shape [B, N, C]
     */
    torch::Tensor forward(
        torch::Tensor x,
        c10::optional<torch::Tensor> attn_bias = c10::nullopt,
        c10::optional<torch::Tensor> pos = c10::nullopt
    );
};

TORCH_MODULE(MemEffAttention);

} // namespace layers
} // namespace vggt
