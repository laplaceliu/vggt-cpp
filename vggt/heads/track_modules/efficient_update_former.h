/**
 * @file efficient_update_former.h
 * @brief Efficient update transformer for iterative refinement
 */

#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>

namespace vggt {
namespace track_modules {

class EfficientUpdateFormerImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new EfficientUpdateFormerImpl object
     *
     * @param hidden_dim Hidden dimension size
     * @param num_heads Number of attention heads
     * @param num_layers Number of transformer layers
     * @param dim_feedforward Dimension of feedforward network
     * @param dropout Dropout rate
     * @param activation Activation function
     * @param norm_first Whether to apply normalization before attention/feedforward
     * @param return_intermediate Whether to return intermediate outputs
     */
    EfficientUpdateFormerImpl(
        int64_t hidden_dim = 128,
        int64_t num_heads = 8,
        int64_t num_layers = 6,
        int64_t dim_feedforward = 1024,
        double dropout = 0.1,
        const std::string& activation = "relu",
        bool norm_first = false,
        bool return_intermediate = false);

    /**
     * @brief Forward pass through the transformer
     *
     * @param src Input tensor (B, S, N, C)
     * @param pos Position encoding (B, S, N, C)
     * @param src_key_padding_mask Padding mask for source (B, S, N)
     * @return torch::Tensor Output tensor (B, S, N, C)
     */
    torch::Tensor forward(
        const torch::Tensor& src,
        const torch::Tensor& pos = {},
        const torch::Tensor& src_key_padding_mask = {});

private:
    int64_t hidden_dim_;
    int64_t num_heads_;
    int64_t num_layers_;
    bool return_intermediate_;

    torch::nn::ModuleList transformer_layers_;
    torch::nn::LayerNorm norm_;
};

TORCH_MODULE(EfficientUpdateFormer);

/**
 * @brief Transformer encoder layer with efficient implementation
 */
class EfficientTransformerEncoderLayerImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new EfficientTransformerEncoderLayerImpl object
     *
     * @param d_model Model dimension
     * @param nhead Number of attention heads
     * @param dim_feedforward Dimension of feedforward network
     * @param dropout Dropout rate
     * @param activation Activation function
     * @param norm_first Whether to apply normalization before attention/feedforward
     */
    EfficientTransformerEncoderLayerImpl(
        int64_t d_model,
        int64_t nhead,
        int64_t dim_feedforward = 2048,
        double dropout = 0.1,
        const std::string& activation = "relu",
        bool norm_first = false);

    /**
     * @brief Forward pass through the encoder layer
     *
     * @param src Input tensor (B, S, N, C)
     * @param src_mask Attention mask (optional)
     * @param src_key_padding_mask Key padding mask (optional)
     * @param pos Position encoding (optional)
     * @return torch::Tensor Output tensor (B, S, N, C)
     */
    torch::Tensor forward(
        const torch::Tensor& src,
        const torch::Tensor& src_mask = {},
        const torch::Tensor& src_key_padding_mask = {},
        const torch::Tensor& pos = {});

private:
    int64_t d_model_;
    int64_t nhead_;
    bool norm_first_;

    torch::nn::MultiheadAttention self_attn_{nullptr};
    torch::nn::Linear linear1_{nullptr}, linear2_{nullptr};
    torch::nn::Dropout dropout1_{nullptr}, dropout2_{nullptr}, dropout3_{nullptr};
    torch::nn::LayerNorm norm1_{nullptr}, norm2_{nullptr};
    torch::nn::Module activation_{nullptr};

    /**
     * @brief Apply self-attention with position encoding
     */
    torch::Tensor _sa_block(
        const torch::Tensor& x,
        const torch::Tensor& attn_mask,
        const torch::Tensor& key_padding_mask,
        const torch::Tensor& pos);

    /**
     * @brief Apply feedforward network
     */
    torch::Tensor _ff_block(const torch::Tensor& x);
};

TORCH_MODULE(EfficientTransformerEncoderLayer);

} // namespace track_modules
} // namespace vggt
