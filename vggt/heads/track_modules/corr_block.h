/**
 * @file corr_block.h
 * @brief Correlation block for feature matching
 */

#pragma once

#include <torch/torch.h>
#include <vector>

namespace vggt {
namespace track_modules {

class CorrBlock {
public:
    /**
     * @brief Construct a new CorrBlock object
     *
     * @param fmaps Input feature maps (B, S, C, H, W)
     * @param num_levels Number of pyramid levels (default=4)
     * @param radius Search radius for sampling correlation (default=4)
     * @param multiple_track_feats Whether to split target features per level (default=false)
     * @param padding_mode Padding mode for sampling (default="zeros")
     */
    CorrBlock(
        const torch::Tensor& fmaps,
        int64_t num_levels = 4,
        int64_t radius = 4,
        bool multiple_track_feats = false,
        const std::string& padding_mode = "zeros");

    /**
     * @brief Sample correlation volumes for given targets and coordinates
     *
     * @param targets Target features (B, S, N, C)
     * @param coords Coordinates at full resolution (B, S, N, 2)
     * @return torch::Tensor Sampled correlations (B, S, N, L) where L = num_levels * (2*radius+1)^2
     */
    torch::Tensor corr_sample(
        const torch::Tensor& targets,
        const torch::Tensor& coords);

private:
    int64_t S_, C_, H_, W_;
    int64_t num_levels_;
    int64_t radius_;
    std::string padding_mode_;
    bool multiple_track_feats_;
    torch::Tensor delta_;
    std::vector<torch::Tensor> fmaps_pyramid_;

    /**
     * @brief Compute correlation for a single level
     */
    static torch::Tensor compute_corr_level(
        const torch::Tensor& fmap1,
        const torch::Tensor& fmap2s,
        int64_t C);
};

} // namespace track_modules
} // namespace vggt
