// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/torch.h>
#include <memory>

namespace vggt {
namespace track_modules {

/**
 * @brief Base class for track predictors
 * 
 * This class provides the core functionality for predicting tracks in a sequence of images.
 */
class BaseTrackerPredictor : public torch::nn::Module {
public:
    /**
     * @brief Construct a new BaseTrackerPredictor object
     * 
     * @param stride Stride of the feature maps
     * @param corr_levels Number of correlation levels
     * @param corr_radius Correlation radius
     * @param latent_dim Latent dimension size
     * @param hidden_size Hidden layer size
     * @param use_spaceatt Whether to use spatial attention
     * @param depth Depth of the transformer
     * @param fine Whether this is a fine predictor
     */
    BaseTrackerPredictor(
        int stride = 4,
        int corr_levels = 5,
        int corr_radius = 4,
        int latent_dim = 128,
        int hidden_size = 384,
        bool use_spaceatt = true,
        int depth = 6,
        bool fine = false
    );

    /**
     * @brief Forward pass of the predictor
     * 
     * @param query_points Query points tensor with shape [B, N, 2]
     * @param fmaps Feature maps tensor with shape [B, S, C, HH, WW]
     * @param iters Number of iterations
     * @param down_ratio Downsampling ratio
     * @return std::tuple<std::vector<torch::Tensor>, torch::Tensor> 
     *         (coord_preds, vis_e)
     */
    std::tuple<std::vector<torch::Tensor>, torch::Tensor> forward(
        const torch::Tensor& query_points,
        const torch::Tensor& fmaps,
        int iters = 4,
        int down_ratio = 1
    );

private:
    int stride;
    int corr_levels;
    int corr_radius;
    int latent_dim;
    int hidden_size;
    bool use_spaceatt;
    int depth;
    bool fine;
    int flows_emb_dim;
    int transformer_dim;
};

} // namespace track_modules
} // namespace vggt