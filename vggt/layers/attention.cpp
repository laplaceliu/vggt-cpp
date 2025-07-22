/**
 * @file attention.cpp
 * @brief Implementation of attention mechanisms for vision transformers
 */

#include "attention.h"
#include <torch/torch.h>
#include <c10/util/Exception.h>

namespace vggt {
namespace layers {

namespace {
// Flag to check if xFormers is available
bool XFORMERS_AVAILABLE = false;

// Helper function to unbind tensor along a dimension
std::vector<torch::Tensor> unbind(const torch::Tensor& tensor, int64_t dim) {
    std::vector<torch::Tensor> outputs;
    for (int64_t i = 0; i < tensor.size(dim); ++i) {
        outputs.push_back(tensor.select(dim, i));
    }
    return outputs;
}

// Memory efficient attention implementation (placeholder for xFormers integration)
torch::Tensor memory_efficient_attention(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const c10::optional<torch::Tensor>& attn_bias = c10::nullopt
) {
    TORCH_CHECK(false, "xFormers is not available. Please install xFormers for memory efficient attention.");
    return torch::Tensor();
}
} // anonymous namespace

AttentionImpl::AttentionImpl(
    int64_t dim,
    int64_t num_heads,
    bool qkv_bias,
    bool proj_bias,
    double attn_drop,
    double proj_drop,
    bool qk_norm,
    bool fused_attn,
    c10::optional<torch::nn::Module> rope
) : num_heads_(num_heads), fused_attn_(fused_attn) {
    TORCH_CHECK(dim % num_heads == 0, "dim should be divisible by num_heads");

    head_dim_ = dim / num_heads;
    scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    // Initialize QKV projection
    qkv_ = register_module("qkv", torch::nn::Linear(torch::nn::LinearOptions(dim, dim * 3).bias(qkv_bias)));

    // Initialize normalization layers for Q and K if needed
    if (qk_norm) {
        // Create normalization layers with the correct feature size
        q_norm_ = register_module("q_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({head_dim_})));
        k_norm_ = register_module("k_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({head_dim_})));
    } else {
        q_norm_ = register_module("q_norm", torch::nn::Identity());
        k_norm_ = register_module("k_norm", torch::nn::Identity());
    }

    // Initialize dropout layers
    attn_drop_ = register_module("attn_drop", torch::nn::Dropout(torch::nn::DropoutOptions(attn_drop)));
    proj_drop_ = register_module("proj_drop", torch::nn::Dropout(torch::nn::DropoutOptions(proj_drop)));

    // Initialize output projection
    proj_ = register_module("proj", torch::nn::Linear(torch::nn::LinearOptions(dim, dim).bias(proj_bias)));

    // Store rotary position embedding if provided
    if (rope.has_value()) {
        rope_ = rope;
        register_module("rope", *rope_);
    }
}

torch::Tensor AttentionImpl::forward(torch::Tensor x, c10::optional<torch::Tensor> pos) {
    auto B = x.size(0);
    auto N = x.size(1);
    auto C = x.size(2);

    // QKV projection and reshape
    auto qkv = qkv_(x).reshape({B, N, 3, num_heads_, head_dim_}).permute({2, 0, 3, 1, 4});
    auto qkv_unbind = unbind(qkv, 0);
    auto q = qkv_unbind[0];
    auto k = qkv_unbind[1];
    auto v = qkv_unbind[2];

    // Apply normalization to Q and K
    q = q_norm_(q);
    k = k_norm_(k);

    // Apply rotary position embedding if available
    if (rope_.has_value() && pos.has_value()) {
        q = rope_->as<torch::nn::Module>().forward({q, pos.value()}).toTensor();
        k = rope_->as<torch::nn::Module>().forward({k, pos.value()}).toTensor();
    }

    torch::Tensor out;

    if (fused_attn_) {
        // Use PyTorch's scaled_dot_product_attention if available
        #if TORCH_VERSION_MAJOR > 1 || (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 12)
            float dropout_p = is_training() ? attn_drop_->options.p() : 0.0;
            out = torch::scaled_dot_product_attention(
                q, k, v,
                /*attn_mask=*/c10::nullopt,
                /*dropout_p=*/dropout_p,
                /*is_causal=*/false
            );
        #else
            // Fallback for older PyTorch versions
            fused_attn_ = false;
            q = q * scale_;
            auto attn = torch::matmul(q, k.transpose(-2, -1));
            attn = torch::softmax(attn, -1);
            attn = attn_drop_(attn);
            out = torch::matmul(attn, v);
        #endif
    } else {
        // Manual implementation of attention
        q = q * scale_;
        auto attn = torch::matmul(q, k.transpose(-2, -1));
        attn = torch::softmax(attn, -1);
        attn = attn_drop_(attn);
        out = torch::matmul(attn, v);
    }

    // Reshape and project output
    out = out.transpose(1, 2).reshape({B, N, C});
    out = proj_(out);
    out = proj_drop_(out);

    return out;
}

std::string AttentionImpl::pretty_print(int64_t indent) const {
    std::ostringstream ss;
    ss << "AttentionImpl(dim=" << qkv_->weight.size(1)
       << ", num_heads=" << num_heads_
       << ", head_dim=" << head_dim_
       << ", fused_attn=" << (fused_attn_ ? "True" : "False")
       << ")";
    return ss.str();
}

torch::Tensor MemEffAttention::forward(
    torch::Tensor x,
    c10::optional<torch::Tensor> attn_bias,
    c10::optional<torch::Tensor> pos
) {
    // If position is provided or xFormers is not available, fall back to standard attention
    if (pos.has_value() || !XFORMERS_AVAILABLE) {
        if (attn_bias.has_value()) {
            TORCH_CHECK(XFORMERS_AVAILABLE, "xFormers is required for using attention bias");
        }
        return AttentionImpl::forward(x, pos);
    }

    auto B = x.size(0);
    auto N = x.size(1);
    auto C = x.size(2);

    // QKV projection and reshape
    auto qkv = qkv_(x).reshape({B, N, 3, num_heads_, C / num_heads_});

    // Unbind Q, K, V
    auto qkv_unbind = unbind(qkv, 2);
    auto q = qkv_unbind[0];
    auto k = qkv_unbind[1];
    auto v = qkv_unbind[2];

    // Use memory efficient attention
    auto out = memory_efficient_attention(q, k, v, attn_bias);
    out = out.reshape({B, N, C});

    // Project and dropout
    out = proj_(out);
    out = proj_drop_(out);

    return out;
}

} // namespace layers
} // namespace vggt
