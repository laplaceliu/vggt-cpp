/**
 * @file block.cpp
 * @brief Implementation of the Transformer block for vision transformers
 *
 * This file implements the BlockImpl class methods defined in block.h.
 * It provides the core functionality for a standard transformer block,
 * including:
 *
 * 1. Constructor for initializing the block with specified parameters
 * 2. Forward method that implements the block computation logic
 * 3. Layer normalization and residual connections
 * 4. Multi-head self-attention and MLP operations
 * 5. Optional layer scaling and drop path regularization
 * 6. Training/inference mode handling
 * 7. Efficient residual connection implementation
 * 8. Module registration for PyTorch serialization
 *
 * The implementation follows the pre-norm architecture with optional
 * stochastic depth (drop path) during training. Key features include:
 * - Pre-layer normalization for improved training stability
 * - Optional layer scaling (gamma parameters) for better optimization
 * - Drop path regularization for improved generalization
 * - Lambda functions for clean residual connection implementation
 * - Support for both training and inference modes
 */

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
