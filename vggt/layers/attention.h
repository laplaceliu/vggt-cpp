#pragma once

#include <torch/torch.h>

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
        torch::nn::AnyModule norm_layer = torch::nn::AnyModule(torch::nn::LayerNorm(torch::nn::LayerNormOptions({}))),
        bool qk_norm = false,
        bool fused_attn = true,
        torch::nn::AnyModule rope = torch::nn::AnyModule()
    );

    torch::Tensor forward(torch::Tensor x, torch::Tensor pos = {});

private:
    int64_t num_heads;
    int64_t head_dim;
    double scale;
    bool fused_attn;

    torch::nn::Linear qkv{nullptr};
    torch::nn::AnyModule q_norm;
    torch::nn::AnyModule k_norm;
    torch::nn::Dropout attn_drop{nullptr};
    torch::nn::Linear proj{nullptr};
    torch::nn::Dropout proj_drop{nullptr};
    torch::nn::AnyModule rope;
};
TORCH_MODULE(Attention);

class MemEffAttentionImpl : public AttentionImpl {
public:
    using AttentionImpl::AttentionImpl;

    torch::Tensor forward(torch::Tensor x, torch::Tensor attn_bias = {}, torch::Tensor pos = {});
};
TORCH_MODULE(MemEffAttention);

} // namespace layers
} // namespace vggt
