/**
 * @file vision_transformer.h
 * @brief Vision Transformer implementation
 *
 * This file defines the DinoVisionTransformer class and related helper classes
 * for implementing Vision Transformers based on the DINOv2 architecture.
 */

#pragma once

#include <torch/torch.h>
#include <vector>
#include <functional>
#include <memory>
#include <unordered_map>
#include <string>

#include "mlp.h"
#include "patch_embed.h"
#include "swiglu_ffn.h"
#include "attention.h"
#include "block.h"

namespace vggt {
namespace layers {

/**
 * @brief Helper class for applying a named function to modules
 *
 * @param fn Function to apply
 * @param module Module to apply the function to
 * @param name Module name
 * @param depth_first Whether to apply the function depth-first
 * @param include_root Whether to include the root module
 * @return Module after applying the function
 */
torch::nn::Module named_apply(
    std::function<void(torch::nn::Module&, const std::string&)> fn,
    torch::nn::Module& module,
    const std::string& name = "",
    bool depth_first = true,
    bool include_root = false);

/**
 * @brief Block chunk implementation for grouping transformer blocks
 */
class BlockChunkImpl : public torch::nn::ModuleList {
public:
    BlockChunkImpl() = default;

    /**
     * @brief Forward pass for the block chunk
     *
     * @param x Input tensor
     * @return Output tensor after passing through all blocks
     */
    torch::Tensor forward(const torch::Tensor& x);
};

TORCH_MODULE(BlockChunk);

/**
 * @brief DinoVisionTransformer implementation
 *
 * This class implements the Vision Transformer architecture used in DINOv2.
 */
class DinoVisionTransformerImpl : public torch::nn::Module {
public:
    /**
     * @brief Constructor for DinoVisionTransformer
     *
     * @param img_size Input image size
     * @param patch_size Patch size
     * @param in_chans Number of input channels
     * @param embed_dim Embedding dimension
     * @param depth Transformer depth
     * @param num_heads Number of attention heads
     * @param mlp_ratio MLP ratio
     * @param qkv_bias Whether to use bias for QKV
     * @param ffn_bias Whether to use bias for FFN
     * @param proj_bias Whether to use bias for projection
     * @param drop_path_rate Drop path rate
     * @param drop_path_uniform Whether to use uniform drop path
     * @param init_values Layer scale initialization values
     * @param ffn_layer FFN layer type
     * @param block_chunks Number of block chunks
     * @param num_register_tokens Number of register tokens
     * @param interpolate_antialias Whether to use antialiasing for interpolation
     * @param interpolate_offset Interpolation offset
     * @param qk_norm Whether to use QK normalization
     */
    DinoVisionTransformerImpl(
        int64_t img_size = 224,
        int64_t patch_size = 16,
        int64_t in_chans = 3,
        int64_t embed_dim = 768,
        int64_t depth = 12,
        int64_t num_heads = 12,
        double mlp_ratio = 4.0,
        bool qkv_bias = true,
        bool ffn_bias = true,
        bool proj_bias = true,
        double drop_path_rate = 0.0,
        bool drop_path_uniform = false,
        c10::optional<double> init_values = c10::nullopt,
        const std::string& ffn_layer = "mlp",
        int64_t block_chunks = 1,
        int64_t num_register_tokens = 0,
        bool interpolate_antialias = false,
        double interpolate_offset = 0.1,
        bool qk_norm = false);

    /**
     * @brief Initialize weights
     */
    void init_weights();

    /**
     * @brief Interpolate position encoding
     *
     * @param x Input tensor
     * @param w Width
     * @param h Height
     * @return Interpolated position encoding
     */
    torch::Tensor interpolate_pos_encoding(const torch::Tensor& x, int64_t w, int64_t h);

    /**
     * @brief Prepare tokens with masks
     *
     * @param x Input tensor
     * @param masks Optional mask tensor
     * @return Prepared tokens
     */
    torch::Tensor prepare_tokens_with_masks(const torch::Tensor& x, const c10::optional<torch::Tensor>& masks = c10::nullopt);

    /**
     * @brief Forward pass for a list of features
     *
     * @param x_list List of input tensors
     * @param masks_list List of mask tensors
     * @return List of output dictionaries
     */
    std::vector<std::unordered_map<std::string, torch::Tensor>> forward_features_list(
        const std::vector<torch::Tensor>& x_list,
        const std::vector<torch::Tensor>& masks_list);

    /**
     * @brief Forward pass for features
     *
     * @param x Input tensor or list of tensors
     * @param masks Optional mask tensor
     * @return Output dictionary
     */
    std::unordered_map<std::string, torch::Tensor> forward_features(
        const torch::Tensor& x,
        const c10::optional<torch::Tensor>& masks = c10::nullopt);

    /**
     * @brief Get intermediate layers (non-chunked version)
     *
     * @param x Input tensor
     * @param n Number of layers or indices
     * @return Vector of intermediate layer outputs
     */
    std::vector<torch::Tensor> _get_intermediate_layers_not_chunked(
        const torch::Tensor& x,
        const torch::Tensor& n = torch::tensor(1));

    /**
     * @brief Get intermediate layers (chunked version)
     *
     * @param x Input tensor
     * @param n Number of layers or indices
     * @return Vector of intermediate layer outputs
     */
    std::vector<torch::Tensor> _get_intermediate_layers_chunked(
        const torch::Tensor& x,
        const torch::Tensor& n = torch::tensor(1));

    /**
     * @brief Get intermediate layers
     *
     * @param x Input tensor
     * @param n Number of layers or indices
     * @param reshape Whether to reshape outputs
     * @param return_class_token Whether to return class tokens
     * @param norm Whether to apply normalization
     * @return Tuple of intermediate layer outputs
     */
    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> get_intermediate_layers(
        const torch::Tensor& x,
        const torch::Tensor& n = torch::tensor(1),
        bool reshape = false,
        bool return_class_token = false,
        bool norm = true);

    /**
     * @brief Forward pass
     *
     * @param args Input arguments
     * @param is_training Whether in training mode
     * @param kwargs Keyword arguments
     * @return Output tensor or dictionary
     */
    torch::Tensor forward(
        torch::Tensor x,
        c10::optional<torch::Tensor> masks = c10::nullopt,
        bool is_training = true);

    // Getters
    int64_t num_features() const { return embed_dim_; }
    int64_t embed_dim() const { return embed_dim_; }
    int64_t num_tokens() const { return num_tokens_; }
    int64_t n_blocks() const { return n_blocks_; }
    int64_t num_heads() const { return num_heads_; }
    int64_t patch_size() const { return patch_size_; }
    int64_t num_register_tokens() const { return num_register_tokens_; }

private:
    int64_t embed_dim_;
    int64_t num_tokens_;
    int64_t n_blocks_;
    int64_t num_heads_;
    int64_t patch_size_;
    int64_t num_register_tokens_;
    bool interpolate_antialias_;
    double interpolate_offset_;
    bool use_reentrant_;
    bool chunked_blocks_;

    PatchEmbed patch_embed_;
    torch::nn::ParameterList cls_token_;
    torch::nn::ParameterList pos_embed_;
    torch::nn::ParameterList register_tokens_;
    torch::nn::ModuleList blocks_;
    torch::nn::LayerNorm norm_;
    torch::nn::Module head_;
    torch::nn::ParameterList mask_token_;
};

TORCH_MODULE(DinoVisionTransformer);

/**
 * @brief Initialize weights for Vision Transformer
 *
 * @param module Module to initialize
 * @param name Module name
 */
void init_weights_vit_timm(torch::nn::Module& module, const std::string& name = "");

/**
 * @brief Create a ViT-Small model
 *
 * @param patch_size Patch size
 * @param num_register_tokens Number of register tokens
 * @param kwargs Additional arguments
 * @return ViT-Small model
 */
DinoVisionTransformer vit_small(
    int64_t patch_size = 16,
    int64_t num_register_tokens = 0);

/**
 * @brief Create a ViT-Base model
 *
 * @param patch_size Patch size
 * @param num_register_tokens Number of register tokens
 * @param kwargs Additional arguments
 * @return ViT-Base model
 */
DinoVisionTransformer vit_base(
    int64_t patch_size = 16,
    int64_t num_register_tokens = 0);

/**
 * @brief Create a ViT-Large model
 *
 * @param patch_size Patch size
 * @param num_register_tokens Number of register tokens
 * @param kwargs Additional arguments
 * @return ViT-Large model
 */
DinoVisionTransformer vit_large(
    int64_t patch_size = 16,
    int64_t num_register_tokens = 0);

/**
 * @brief Create a ViT-Giant2 model
 *
 * @param patch_size Patch size
 * @param num_register_tokens Number of register tokens
 * @param kwargs Additional arguments
 * @return ViT-Giant2 model
 */
DinoVisionTransformer vit_giant2(
    int64_t patch_size = 16,
    int64_t num_register_tokens = 0);

} // namespace layers
} // namespace vggt
