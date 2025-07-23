// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/torch.h>
#include <memory>
#include <string>
#include <tuple>

namespace vggt {

/**
 * @brief C++ implementation of the TrackerPredictor class from Python
 * 
 * This class is responsible for predicting tracks in a sequence of images.
 */
class TrackerPredictor : public torch::nn::Module {
public:
    /**
     * @brief Construct a new Tracker Predictor object
     */
    TrackerPredictor();

    /**
     * @brief Load model weights from a file
     * 
     * @param model_path Path to the model weights file
     */
    void load(const std::string& model_path);

    /**
     * @brief Process images to feature maps
     * 
     * @param images Images tensor with shape [S, 3, H, W]
     * @return torch::Tensor Feature maps
     */
    torch::Tensor process_images_to_fmaps(const torch::Tensor& images);

    /**
     * @brief Forward pass of the tracker predictor
     * 
     * @param images Images tensor with shape [B, S, 3, H, W]
     * @param query_points Query points tensor with shape [B, N, 2]
     * @param fmaps Precomputed feature maps (optional)
     * @param fine_tracking Whether to perform fine tracking
     * @param fine_chunk Chunk size for fine tracking
     * @param coarse_iters Number of iterations for coarse prediction
     * @param inference Whether to perform inference
     * @return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
     *         (fine_pred_track, coarse_pred_track, pred_vis, pred_score)
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
        const torch::Tensor& images,
        const torch::Tensor& query_points,
        const torch::Tensor& fmaps = {},
        bool fine_tracking = true,
        int fine_chunk = 40960,
        int coarse_iters = 6,
        bool inference = true
    );

private:
    // Coarse predictor components
    torch::nn::Module coarse_fnet{nullptr};
    torch::nn::Module coarse_predictor{nullptr};
    int coarse_down_ratio = 2;

    // Fine predictor components
    torch::nn::Module fine_fnet{nullptr};
    torch::nn::Module fine_predictor{nullptr};
};

} // namespace vggt