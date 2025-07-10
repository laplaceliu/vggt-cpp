/**
 * @file aggregator.h
 * @brief Aggregator module for VGGT (Visual Geometry and Graph Tracking) library
 */

#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>

#include "layers/block.h"
#include "layers/patch_embed.h"
#include "layers/rope.h"
#include "layers/vision_transformer.h"

namespace vggt {

/**
 * @class Aggregator
 * @brief The Aggregator applies alternating-attention over input frames
 *
 * Remember to set model.train() to enable gradient checkpointing to reduce memory usage.
 */
class AggregatorImpl : public torch::nn::Module {
public:
    /**
     * @brief Constructor for Aggregator module
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
     * @param patch_embed Type of patch embed (e.g., "conv" or "dinov2_vitl14_reg")
     * @param aa_order The order of alternating attention (e.g. ["frame", "global"])
     * @param aa_block_size How many blocks to group under each attention type before switching
     * @param qk_norm Whether to apply QK normalization
     * @param rope_freq Base frequency for rotary embedding (-1 to disable)
     * @param init_values Init scale for layer scale
     */
    AggregatorImpl(
        int img_size = 518,
        int patch_size = 14,
        int embed_dim = 1024,
        int depth = 24,
        int num_heads = 16,
        float mlp_ratio = 4.0f,
        int num_register_tokens = 4,
        bool qkv_bias = true,
        bool proj_bias = true,
        bool ffn_bias = true,
        const std::string& patch_embed = "dinov2_vitl14_reg",
        const std::vector<std::string>& aa_order = {"frame", "global"},
        int aa_block_size = 1,
        bool qk_norm = true,
        int rope_freq = 100,
        float init_values = 0.01f
    );

    /**
     * @brief Forward pass of the Aggregator
     * @param images Input images with shape [B, S, 3, H, W], in range [0, 1]
     * @return std::tuple containing:
     *         1. List of outputs from the attention blocks
     *         2. The patch_start_idx indicating where patch tokens begin
     */
    std::tuple<std::vector<torch::Tensor>, int> forward(torch::Tensor images);

private:
    void build_patch_embed(
        const std::string& patch_embed,
        int img_size,
        int patch_size,
        int num_register_tokens,
        bool interpolate_antialias = true,
        float interpolate_offset = 0.0f,
        int block_chunks = 0,
        float init_values = 1.0f,
        int embed_dim = 1024
    );

    std::tuple<torch::Tensor, int, std::vector<torch::Tensor>> process_frame_attention(
        torch::Tensor tokens,
        int B,
        int S,
        int P,
        int C,
        int frame_idx,
        torch::Tensor pos = torch::Tensor()
    );

    std::tuple<torch::Tensor, int, std::vector<torch::Tensor>> process_global_attention(
        torch::Tensor tokens,
        int B,
        int S,
        int P,
        int C,
        int global_idx,
        torch::Tensor pos = torch::Tensor()
    );

    torch::nn::ModuleHolder<vggt::PatchEmbedImpl> patch_embed{nullptr};
    torch::nn::ModuleHolder<vggt::RotaryPositionEmbedding2DImpl> rope{nullptr};
    torch::nn::ModuleHolder<vggt::PositionGetterImpl> position_getter{nullptr};

    torch::nn::ModuleList frame_blocks;
    torch::nn::ModuleList global_blocks;

    torch::Tensor camera_token;
    torch::Tensor register_token;

    int depth;
    std::vector<std::string> aa_order;
    int patch_size;
    int aa_block_size;
    int aa_block_num;
    int patch_start_idx;

    torch::Tensor _resnet_mean;
    torch::Tensor _resnet_std;

    bool use_reentrant = false;
};

TORCH_MODULE(Aggregator);

} // namespace vggt
