/**
 * @file blocks.h
 * @brief Neural network blocks for tracking heads
 */

#pragma once

#include <torch/torch.h>
#include <vector>
#include "modules.h"
#include "utils.h"

namespace vggt {

/**
 * @brief Transformer model that updates track estimates
 */
class EfficientUpdateFormer : public torch::nn::Module {
public:
    /**
     * @brief Construct a new EfficientUpdateFormer object
     *
     * @param space_depth Number of space attention blocks
     * @param time_depth Number of time attention blocks
     * @param input_dim Input dimension
     * @param hidden_size Hidden size
     * @param num_heads Number of attention heads
     * @param output_dim Output dimension
     * @param mlp_ratio MLP expansion ratio
     * @param add_space_attn Whether to add space attention
     * @param num_virtual_tracks Number of virtual tracks
     */
    EfficientUpdateFormer(
        int space_depth = 6,
        int time_depth = 6,
        int input_dim = 320,
        int hidden_size = 384,
        int num_heads = 8,
        int output_dim = 130,
        double mlp_ratio = 4.0,
        bool add_space_attn = true,
        int num_virtual_tracks = 64
    );

    /**
     * @brief Forward pass
     *
     * @param input_tensor Input tensor
     * @param mask Optional attention mask
     * @return std::tuple<torch::Tensor, torch::Tensor> Output flow and optional extra output
     */
    std::tuple<torch::Tensor, torch::Tensor> forward(
        torch::Tensor input_tensor,
        torch::Tensor mask = {}
    );

private:
    void initialize_weights();

    int num_heads;
    int hidden_size;
    bool add_space_attn;
    int num_virtual_tracks;

    torch::nn::LayerNorm input_norm{nullptr};
    torch::nn::Linear input_transform{nullptr};
    torch::nn::LayerNorm output_norm{nullptr};
    torch::nn::Linear flow_head{nullptr};

    torch::Tensor virual_tracks;
    torch::nn::ModuleList time_blocks{nullptr};
    torch::nn::ModuleList space_virtual_blocks{nullptr};
    torch::nn::ModuleList space_point2virtual_blocks{nullptr};
    torch::nn::ModuleList space_virtual2point_blocks{nullptr};
};

/**
 * @brief Correlation block for feature pyramid and correlation computation
 */
class CorrBlock {
public:
    /**
     * @brief Construct a new CorrBlock object
     *
     * @param fmaps Input feature maps (B, S, C, H, W)
     * @param num_levels Number of pyramid levels
     * @param radius Search radius for sampling correlation
     * @param multiple_track_feats Whether to split target features per level
     * @param padding_mode Padding mode for sampling
     */
    CorrBlock(
        torch::Tensor fmaps,
        int num_levels = 4,
        int radius = 4,
        bool multiple_track_feats = false,
        const std::string& padding_mode = "zeros"
    );

    /**
     * @brief Sample correlation volumes
     *
     * @param targets Target features (B, S, N, C)
     * @param coords Coordinates at full resolution (B, S, N, 2)
     * @return torch::Tensor Sampled correlations (B, S, N, L)
     */
    torch::Tensor corr_sample(torch::Tensor targets, torch::Tensor coords);

private:
    int S, C, H, W;
    int num_levels;
    int radius;
    std::string padding_mode;
    bool multiple_track_feats;
    std::vector<torch::Tensor> fmaps_pyramid;
    torch::Tensor delta;
};

/**
 * @brief Compute correlation between two feature maps
 *
 * @param fmap1 First feature map (B, S, N, C)
 * @param fmap2s Second feature maps (B, S, C, H*W)
 * @param C Channel dimension
 * @return torch::Tensor Computed correlations (B, S, N, H*W)
 */
torch::Tensor compute_corr_level(torch::Tensor fmap1, torch::Tensor fmap2s, int C);

} // namespace vggt
