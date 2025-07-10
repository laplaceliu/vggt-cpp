/**
 * @file efficient_update_former.cpp
 * @brief Implementation of efficient update transformer for iterative refinement
 */

#include "efficient_update_former.h"
#include <torch/nn/functional.h>

namespace vggt {
namespace track_modules {

EfficientUpdateFormerImpl::EfficientUpdateFormerImpl(
    int64_t hidden_dim,
    int64_t num_heads,
    int64_t num_layers,
    int64_t dim_feedforward,
    double dropout,
    const std::string& activation,
    bool norm_first,
    bool return_intermediate) {
    // Initialize parameters
    hidden_dim_ = hidden_dim;
    num_heads_ = num_heads;
    num_layers_ = num_layers;
    return_intermediate_ = return_intermediate;

    // Create transformer layers
    transformer_layers_ = register_module("transformer_layers", torch::nn::ModuleList());
    for (int64_t i = 0; i < num_layers; ++i) {
        transformer_layers_->push_back(
            EfficientTransformerEncoderLayer(
                hidden_dim, num_heads, dim_feedforward, dropout, activation, norm_first));
    }

    // Create final normalization layer
    norm_ = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_dim})));
}

torch::Tensor EfficientUpdateFormerImpl::forward(
    const torch::Tensor& src,
    const torch::Tensor& pos,
    const torch::Tensor& src_key_padding_mask) {
    auto output = src;
    std::vector<torch::Tensor> intermediate_outputs;

    // Process through transformer layers
    for (int64_t i = 0; i < num_layers_; ++i) {
        auto& layer = transformer_layers_[i];
        output = layer->as<EfficientTransformerEncoderLayer>()->forward(
            output, {}, src_key_padding_mask, pos);

        if (return_intermediate_) {
            intermediate_outputs.push_back(norm_(output));
        }
    }

    // Apply final normalization
    output = norm_(output);

    // Return intermediate outputs if requested
    if (return_intermediate_) {
        return torch::stack(intermediate_outputs, 0);
    }

    return output;
}

EfficientTransformerEncoderLayerImpl::EfficientTransformerEncoderLayerImpl(
    int64_t d_model,
    int64_t nhead,
    int64_t dim_feedforward,
    double dropout,
    const std::string& activation,
    bool norm_first) {
    // Initialize parameters
    d_model_ = d_model;
    nhead_ = nhead;
    norm_first_ = norm_first;

    // Initialize attention and feedforward layers
    self_attn_ = register_module("self_attn",
        torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(d_model, nhead).dropout(dropout)));

    // Initialize feedforward network
    linear1_ = register_module("linear1", torch::nn::Linear(d_model, dim_feedforward));
    linear2_ = register_module("linear2", torch::nn::Linear(dim_feedforward, d_model));

    // Initialize normalization layers
    norm1_ = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
    norm2_ = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));

    // Initialize dropout layers
    dropout1_ = register_module("dropout1", torch::nn::Dropout(dropout));
    dropout2_ = register_module("dropout2", torch::nn::Dropout(dropout));
    dropout3_ = register_module("dropout3", torch::nn::Dropout(dropout));

    // Initialize activation function
    if (activation == "relu") {
        activation_ = register_module("activation", torch::nn::ReLU());
    } else if (activation == "gelu") {
        activation_ = register_module("activation", torch::nn::GELU());
    } else {
        throw std::runtime_error("Unsupported activation function: " + activation);
    }
}

torch::Tensor EfficientTransformerEncoderLayerImpl::forward(
    const torch::Tensor& src,
    const torch::Tensor& src_mask,
    const torch::Tensor& src_key_padding_mask,
    const torch::Tensor& pos) {
    if (norm_first_) {
        // Pre-norm architecture
        auto x = src;
        x = x + _sa_block(norm1_(x), src_mask, src_key_padding_mask, pos);
        x = x + _ff_block(norm2_(x));
        return x;
    } else {
        // Post-norm architecture
        auto x = _sa_block(src, src_mask, src_key_padding_mask, pos);
        x = norm1_(src + x);
        x = norm2_(x + _ff_block(x));
        return x;
    }
}

torch::Tensor EfficientTransformerEncoderLayerImpl::_sa_block(
    const torch::Tensor& x,
    const torch::Tensor& attn_mask,
    const torch::Tensor& key_padding_mask,
    const torch::Tensor& pos) {
    auto B = x.size(0);
    auto S = x.size(1);
    auto N = x.size(2);

    // Reshape input for multihead attention
    auto x_flat = x.reshape({B * S, N, d_model_});

    // Add position encoding if provided
    torch::Tensor q = x_flat;
    torch::Tensor k = x_flat;
    torch::Tensor v = x_flat;

    if (pos.defined() && !pos.numel() == 0) {
        auto pos_flat = pos.reshape({B * S, N, d_model_});
        q = q + pos_flat;
        k = k + pos_flat;
    }

    // Reshape key padding mask if provided
    torch::Tensor key_padding_mask_flat;
    if (key_padding_mask.defined() && !key_padding_mask.numel() == 0) {
        key_padding_mask_flat = key_padding_mask.reshape({B * S, N});
    }

    // Apply multihead attention
    auto attn_output = std::get<0>(self_attn_->forward(
        q.transpose(0, 1),  // (N, B*S, d_model)
        k.transpose(0, 1),  // (N, B*S, d_model)
        v.transpose(0, 1),  // (N, B*S, d_model)
        key_padding_mask_flat,
        false,  // need_weights
        attn_mask));

    // Reshape output back to original dimensions
    attn_output = attn_output.transpose(0, 1).reshape({B, S, N, d_model_});

    return dropout1_(attn_output);
}

torch::Tensor EfficientTransformerEncoderLayerImpl::_ff_block(const torch::Tensor& x) {
    auto B = x.size(0);
    auto S = x.size(1);
    auto N = x.size(2);

    // Reshape for linear layers
    auto x_flat = x.reshape({B * S * N, -1});

    // Apply feedforward network
    auto out = linear2_(dropout2_(activation_->forward(linear1_->forward(x_flat))));

    // Reshape back to original dimensions
    out = out.reshape({B, S, N, d_model_});

    return dropout3_(out);
}

} // namespace track_modules
} // namespace vggt
