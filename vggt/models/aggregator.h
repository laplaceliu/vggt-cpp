/**
 * @file aggregator.h
 * @brief Aggregator model for VGGT
 *
 * This file defines the Aggregator class which applies alternating-attention over input frames,
 * as described in VGGT: Visual Geometry Grounded Transformer.
 */

#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>
#include <tuple>

#include "../layers/patch_embed.h"
#include "../layers/block.h"
#include "../layers/rope.h"
#include "../layers/vision_transformer.h"

namespace vggt {
namespace models {

/**
 * @brief Helper function to process specialized tokens
 *
 * Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
 * 1) Uses the first position (index=0) for the first frame only
 * 2) Uses the second position (index=1) for all remaining frames (S-1 frames)
 * 3) Expands both to match batch size B
 * 4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
 *    followed by (S-1) second-position tokens
 * 5) Flattens to (B*S, X, C) for processing
 *
 * @param token_tensor Token tensor with shape (1, 2, X, C)
 * @param B Batch size
 * @param S Sequence length
 * @return Processed tokens with shape (B*S, X, C)
 */
torch::Tensor slice_expand_and_flatten(const torch::Tensor& token_tensor, int64_t B, int64_t S);

/**
 * @brief Aggregator model for VGGT
 *
 * The Aggregator applies alternating-attention over input frames,
 * as described in VGGT: Visual Geometry Grounded Transformer.
 */
class AggregatorImpl : public torch::nn::Module {
public:
    /**
     * @brief Constructor for Aggregator
     *
     * @param img_size Image size in pixels
     * @param patch_size Size of each patch for PatchEmbed
     * @param embed_dim Dimension of the token embeddings
     * @param depth Number of blocks
     * @param num_heads Number of attention heads
     * @param mlp_ratio Ratio of MLP hidden dim to embedding dim
     * @param num_register_tokens Number of register tokens
     * @param qkv_bias Whether to include bias in QKV projections
     * @param proj_bias Whether to include bias in the output projection
     * @param ffn_bias Whether to include bias in MLP layers
     * @param patch_embed Type of patch embed, e.g., "conv" or "dinov2_vitl14_reg"
     * @param aa_order The order of alternating attention, e.g. ["frame", "global"]
     * @param aa_block_size How many blocks to group under each attention type before switching
     * @param qk_norm Whether to apply QK normalization
     * @param rope_freq Base frequency for rotary embedding. -1 to disable
     * @param init_values Init scale for layer scale
     */
    AggregatorImpl(
        int64_t img_size = 518,
        int64_t patch_size = 14,
        int64_t embed_dim = 1024,
        int64_t depth = 24,
        int64_t num_heads = 16,
        double mlp_ratio = 4.0,
        int64_t num_register_tokens = 4,
        bool qkv_bias = true,
        bool proj_bias = true,
        bool ffn_bias = true,
        const std::string& patch_embed = "dinov2_vitl14_reg",
        const std::vector<std::string>& aa_order = {"frame", "global"},
        int64_t aa_block_size = 1,
        bool qk_norm = true,
        int64_t rope_freq = 100,
        double init_values = 0.01
    );

    /**
     * @brief Forward pass
     *
     * @param images Input images with shape [B, S, 3, H, W], in range [0, 1]
     * @return Tuple of list of outputs from the attention blocks and patch_start_idx
     */
    std::tuple<std::vector<torch::Tensor>, int64_t> forward(const torch::Tensor& images);

private:
    /**
     * @brief Build the patch embed layer
     *
     * @param patch_embed_type Type of patch embed
     * @param img_size Image size
     * @param patch_size Patch size
     * @param num_register_tokens Number of register tokens
     * @param interpolate_antialias Whether to use antialiasing for interpolation
     * @param interpolate_offset Interpolation offset
     * @param block_chunks Number of block chunks
     * @param init_values Init scale for layer scale
     * @param embed_dim Embedding dimension
     */
    void build_patch_embed(
        const std::string& patch_embed_type,
        int64_t img_size,
        int64_t patch_size,
        int64_t num_register_tokens,
        bool interpolate_antialias = true,
        double interpolate_offset = 0.0,
        int64_t block_chunks = 0,
        double init_values = 1.0,
        int64_t embed_dim = 1024
    );

    /**
     * @brief Process frame attention blocks
     *
     * @param tokens Input tokens
     * @param B Batch size
     * @param S Sequence length
     * @param P Number of patches
     * @param C Embedding dimension
     * @param frame_idx Current frame index
     * @param pos Optional position tensor
     * @return Tuple of processed tokens, updated frame index, and intermediates
     */
    std::tuple<torch::Tensor, int64_t, std::vector<torch::Tensor>> process_frame_attention(
        torch::Tensor tokens,
        int64_t B,
        int64_t S,
        int64_t P,
        int64_t C,
        int64_t frame_idx,
        const c10::optional<torch::Tensor>& pos = c10::nullopt
    );

    /**
     * @brief Process global attention blocks
     *
     * @param tokens Input tokens
     * @param B Batch size
     * @param S Sequence length
     * @param P Number of patches
     * @param C Embedding dimension
     * @param global_idx Current global index
     * @param pos Optional position tensor
     * @return Tuple of processed tokens, updated global index, and intermediates
     */
    std::tuple<torch::Tensor, int64_t, std::vector<torch::Tensor>> process_global_attention(
        torch::Tensor tokens,
        int64_t B,
        int64_t S,
        int64_t P,
        int64_t C,
        int64_t global_idx,
        const c10::optional<torch::Tensor>& pos = c10::nullopt
    );

private:
    torch::nn::Module patch_embed_;
    layers::RotaryPositionEmbedding2D rope_;
    layers::PositionGetter position_getter_;
    torch::nn::ModuleList frame_blocks_;
    torch::nn::ModuleList global_blocks_;
    torch::nn::Parameter camera_token_;
    torch::nn::Parameter register_token_;

    int64_t depth_;
    std::vector<std::string> aa_order_;
    int64_t patch_size_;
    int64_t aa_block_size_;
    int64_t aa_block_num_;
    int64_t patch_start_idx_;
    bool use_reentrant_;

    torch::Tensor _resnet_mean;
    torch::Tensor _resnet_std;
};

TORCH_MODULE(Aggregator);

} // namespace models
} // namespace vggt
