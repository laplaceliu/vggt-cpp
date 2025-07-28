#include "attention.h"
#include <torch/torch.h>

namespace vggt {
namespace layers {

AttentionImpl::AttentionImpl(
    int64_t dim,
    int64_t num_heads,
    bool qkv_bias,
    bool proj_bias,
    double attn_drop_prob,
    double proj_drop_prob,
    torch::nn::AnyModule norm_layer,
    bool qk_norm,
    bool fused_attn,
    torch::nn::AnyModule rope
) : num_heads(num_heads),
    head_dim(dim / num_heads),
    scale(std::pow(dim / num_heads, -0.5)),
    fused_attn(fused_attn),
    q_norm(qk_norm ? norm_layer : torch::nn::AnyModule(torch::nn::Identity())),
    k_norm(qk_norm ? norm_layer : torch::nn::AnyModule(torch::nn::Identity())),
    rope(std::move(rope)) {

    // Check if dim is divisible by num_heads
    TORCH_CHECK(dim % num_heads == 0, "dim should be divisible by num_heads");

    // Initialize layers
    qkv = register_module("qkv", torch::nn::Linear(torch::nn::LinearOptions(dim, dim * 3).bias(qkv_bias)));
    register_module("q_norm", q_norm.ptr());
    register_module("k_norm", k_norm.ptr());
    attn_drop = register_module("attn_drop", torch::nn::Dropout(attn_drop_prob));
    proj = register_module("proj", torch::nn::Linear(torch::nn::LinearOptions(dim, dim).bias(proj_bias)));
    proj_drop = register_module("proj_drop", torch::nn::Dropout(proj_drop_prob));
    if (rope.ptr()) {
        register_module("rope", rope.ptr());
    }
}

torch::Tensor AttentionImpl::forward(torch::Tensor x, torch::Tensor pos) {
    auto B = x.size(0);
    auto N = x.size(1);
    auto C = x.size(2);

    // qkv projection and reshape
    auto qkv = this->qkv(x).reshape({B, N, 3, num_heads, head_dim}).permute({2, 0, 3, 1, 4});
    auto qkv_unbind = qkv.unbind(0);
    auto q = qkv_unbind[0];
    auto k = qkv_unbind[1];
    auto v = qkv_unbind[2];

    // Apply normalization if enabled
    q = q_norm.forward(q);
    k = k_norm.forward(k);

    // Apply rope if available
    if (rope.ptr() && pos.defined()) {
        q = rope.forward<torch::Tensor>(std::move(q), std::move(pos));
        k = rope.forward<torch::Tensor>(std::move(k), std::move(pos.clone()));
    }

    torch::Tensor x_out;
    if (fused_attn) {
        // Use scaled dot product attention
        q = q * scale;
        auto attn = torch::matmul(q, k.transpose(-2, -1));
        attn = torch::softmax(attn, -1);
        if (is_training()) {
            attn = attn_drop(attn);
        }
        x_out = torch::matmul(attn, v);
    } else {
        // Manual attention implementation
        q = q * scale;
        auto attn = torch::matmul(q, k.transpose(-2, -1));
        attn = torch::softmax(attn, -1);
        attn = attn_drop(attn);
        x_out = torch::matmul(attn, v);
    }

    // Reshape and project
    x_out = x_out.transpose(1, 2).reshape({B, N, C});
    x_out = proj(x_out);
    x_out = proj_drop(x_out);

    return x_out;
}

torch::Tensor MemEffAttentionImpl::forward(torch::Tensor x, torch::Tensor attn_bias, torch::Tensor pos) {
    // Check if xFormers is available (which it isn't in this implementation)
    if (attn_bias.defined()) {
        TORCH_CHECK(false, "xFormers is required for using nested tensors");
    }

    // If pos is defined, we can't use the parent implementation
    TORCH_CHECK(!pos.defined(), "pos must be None for MemEffAttention without xFormers");

    // Fall back to the parent implementation
    return AttentionImpl::forward(x, {});
}

} // namespace layers
} // namespace vggt
