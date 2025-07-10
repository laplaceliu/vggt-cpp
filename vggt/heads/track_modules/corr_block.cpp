/**
 * @file corr_block.cpp
 * @brief Implementation of correlation block for feature matching
 */

#include "corr_block.h"
#include "../utils.h"
#include <torch/nn/functional.h>
#include <cmath>

namespace vggt {
namespace track_modules {

CorrBlock::CorrBlock(
    const torch::Tensor& fmaps,
    int64_t num_levels,
    int64_t radius,
    bool multiple_track_feats,
    const std::string& padding_mode) {
    // Initialize parameters
    auto sizes = fmaps.sizes();
    S_ = sizes[1];
    C_ = sizes[2];
    H_ = sizes[3];
    W_ = sizes[4];
    num_levels_ = num_levels;
    radius_ = radius;
    padding_mode_ = padding_mode;
    multiple_track_feats_ = multiple_track_feats;

    // Build feature pyramid
    fmaps_pyramid_.push_back(fmaps); // Level 0 is full resolution
    auto current_fmaps = fmaps;

    for (int64_t i = 0; i < num_levels - 1; ++i) {
        auto current_sizes = current_fmaps.sizes();
        // Merge batch & sequence dimensions
        current_fmaps = current_fmaps.reshape({current_sizes[0] * current_sizes[1],
                                             current_sizes[2],
                                             current_sizes[3],
                                             current_sizes[4]});
        // Avg pool down by factor 2
        current_fmaps = torch::nn::functional::avg_pool2d(
            current_fmaps,
            torch::nn::functional::AvgPool2dFuncOptions(2).stride(2));
        // Reshape back
        auto new_sizes = current_fmaps.sizes();
        current_fmaps = current_fmaps.reshape({sizes[0], sizes[1],
                                              new_sizes[1],
                                              new_sizes[2],
                                              new_sizes[3]});
        fmaps_pyramid_.push_back(current_fmaps);
    }

    // Precompute delta grid for sampling
    auto options = torch::TensorOptions()
        .device(fmaps.device())
        .dtype(fmaps.dtype());

    auto dx = torch::linspace(-radius, radius, 2 * radius + 1, options);
    auto dy = torch::linspace(-radius, radius, 2 * radius + 1, options);

    // delta: for every (dy,dx) displacement
    auto mesh = torch::meshgrid({dy, dx}, "ij");
    delta_ = torch::stack({mesh[0], mesh[1]}, -1); // shape: (2r+1, 2r+1, 2)
}

torch::Tensor CorrBlock::corr_sample(
    const torch::Tensor& targets,
    const torch::Tensor& coords) {
    auto B = targets.size(0);
    auto S = targets.size(1);
    auto N = targets.size(2);
    auto C = targets.size(3);

    // Split targets if using multiple track features
    std::vector<torch::Tensor> targets_split;
    if (multiple_track_feats_) {
        auto split_size = C / num_levels_;
        targets_split = torch::split(targets, split_size, -1);
    }

    std::vector<torch::Tensor> out_pyramid;
    for (int64_t i = 0; i < num_levels_; ++i) {
        auto& fmaps = fmaps_pyramid_[i];
        auto fmaps_sizes = fmaps.sizes();
        auto H = fmaps_sizes[3];
        auto W = fmaps_sizes[4];

        // Reshape feature maps for correlation computation
        auto fmap2s = fmaps.view({B, S, C_, H * W});

        // Choose appropriate target features
        auto fmap1 = multiple_track_feats_ ? targets_split[i] : targets;

        // Compute correlation
        auto corrs = compute_corr_level(fmap1, fmap2s, C_);
        corrs = corrs.view({B, S, N, H, W});

        // Prepare sampling grid
        auto centroid_lvl = coords.reshape({B * S * N, 1, 1, 2}) / std::pow(2, i);
        auto delta_lvl = delta_.to(coords.device()).to(coords.dtype());
        auto coords_lvl = centroid_lvl + delta_lvl.view({1, 2 * radius_ + 1, 2 * radius_ + 1, 2});

        // Sample from correlation volume using bilinear interpolation
        auto corrs_sampled = bilinear_sampler(
            corrs.reshape({B * S * N, 1, H, W}),
            coords_lvl,
            padding_mode_);

        // Flatten the last two dimensions
        corrs_sampled = corrs_sampled.view({B, S, N, -1});
        out_pyramid.push_back(corrs_sampled);
    }

    // Concatenate all levels
    return torch::cat(out_pyramid, -1).contiguous();
}

torch::Tensor CorrBlock::compute_corr_level(
    const torch::Tensor& fmap1,
    const torch::Tensor& fmap2s,
    int64_t C) {
    // Compute correlation
    auto corrs = torch::matmul(fmap1, fmap2s);
    corrs = corrs.view(fmap1.sizes());
    return corrs / std::sqrt(C);
}

} // namespace track_modules
} // namespace vggt
