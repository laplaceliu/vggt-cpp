/**
 * @file block.h
 * @brief Transformer block implementation for vision transformers
 *
 * This file defines the Block module which implements a standard transformer block
 * used in vision transformers. Each block consists of:
 *
 * 1. Layer normalization followed by multi-head self-attention
 * 2. Optional layer scaling and drop path for regularization
 * 3. Layer normalization followed by MLP (feed-forward network)
 * 4. Optional layer scaling and drop path for the MLP output
 *
 * The block follows the pre-norm architecture where normalization is applied
 * before the attention and MLP operations, with residual connections around each.
 */

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
    torch::nn::LayerNorm norm1;
    Attention attn;
    LayerScale ls1;
    DropPath drop_path1;
    torch::nn::LayerNorm norm2;
    Mlp mlp;
    LayerScale ls2;
    DropPath drop_path2;
    double sample_drop_ratio;
};

TORCH_MODULE(Block);

} // namespace layers
} // namespace vggt
