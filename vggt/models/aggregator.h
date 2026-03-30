#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>

#include "../layers/block.h"
#include "../layers/patch_embed.h"
#include "../layers/rope.h"

namespace vggt {
namespace models {

/**
 * The Aggregator applies alternating-attention over input frames,
 * as described in VGGT: Visual Geometry Grounded Transformer.
 */
class AggregatorImpl : public torch::nn::Module {
public:
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
        double rope_freq = 100.0,
        double init_values = 0.01
    );

    /**
     * Forward pass of the Aggregator.
     *
     * @param images Input images with shape [B, S, 3, H, W], in range [0, 1]
     * @return Pair of (list of aggregated tokens, patch_start_idx)
     */
    std::pair<std::vector<torch::Tensor>, int64_t> forward(torch::Tensor images);

private:
    void buildPatchEmbed(
        const std::string& patch_embed,
        int64_t img_size,
        int64_t patch_size,
        int64_t num_register_tokens,
        bool interpolate_antialias = true,
        double interpolate_offset = 0.0,
        int64_t block_chunks = 0,
        double init_values = 1.0,
        int64_t embed_dim = 1024
    );

    std::tuple<torch::Tensor, int64_t, std::vector<torch::Tensor>> processFrameAttention(
        torch::Tensor tokens,
        int64_t B,
        int64_t S,
        int64_t P,
        int64_t C,
        int64_t frame_idx,
        torch::Tensor pos
    );

    std::tuple<torch::Tensor, int64_t, std::vector<torch::Tensor>> processGlobalAttention(
        torch::Tensor tokens,
        int64_t B,
        int64_t S,
        int64_t P,
        int64_t C,
        int64_t global_idx,
        torch::Tensor pos
    );

    // Configuration
    int64_t depth_;
    int64_t aa_block_size_;
    int64_t aa_block_num_;
    int64_t patch_size_;
    int64_t patch_start_idx_;
    std::vector<std::string> aa_order_;
    bool use_reentrant_ = false;

    // Patch embedding
    torch::nn::AnyModule patch_embed_;

    // Rotary position embedding
    layers::RotaryPositionEmbedding2D rope_{nullptr};
    std::shared_ptr<layers::PositionGetter> position_getter_;

    // Attention blocks - stored as vector of Block modules
    std::vector<layers::Block> frame_blocks_;
    std::vector<layers::Block> global_blocks_;

    // Special tokens
    torch::Tensor camera_token_;
    torch::Tensor register_token_;

    // Normalization constants (registered as buffers)
    torch::Tensor _resnet_mean;
    torch::Tensor _resnet_std;
};

TORCH_MODULE(Aggregator);

/**
 * Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
 * 1) Uses the first position (index=0) for the first frame only
 * 2) Uses the second position (index=1) for all remaining frames (S-1 frames)
 * 3) Expands both to match batch size B
 * 4) Concatenates to form (B, S, X, C)
 * 5) Flattens to (B*S, X, C) for processing
 */
torch::Tensor sliceExpandAndFlatten(const torch::Tensor& token_tensor, int64_t B, int64_t S);

} // namespace models
} // namespace vggt
