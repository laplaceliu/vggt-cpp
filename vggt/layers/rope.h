// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the Apache License, Version 2.0
// found in the LICENSE file in the root directory of this source tree.

// Implementation of 2D Rotary Position Embeddings (RoPE) in C++

#pragma once

#include <map>
#include <utility>
#include <vector>
#include <memory>
#include "Eigen/Dense"

class PositionGetter {
public:
    PositionGetter();

    // Generates spatial positions for a batch of patches
    // Returns tensor of shape (batch_size, height*width, 2)
    Eigen::Tensor<float, 3> operator()(int batch_size, int height, int width);

private:
    std::map<std::pair<int, int>, Eigen::Tensor<float, 2>> position_cache_;
};

class RotaryPositionEmbedding2D {
public:
    RotaryPositionEmbedding2D(float frequency = 100.0f, float scaling_factor = 1.0f);

    // Applies 2D rotary position embeddings to input tokens
    // tokens shape: (batch_size, n_heads, n_tokens, dim)
    // positions shape: (batch_size, n_tokens, 2)
    Eigen::Tensor<float, 4> forward(const Eigen::Tensor<float, 4>& tokens,
                                  const Eigen::Tensor<float, 3>& positions);

private:
    // Computes frequency components for rotary embeddings
    std::pair<Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>>
    compute_frequency_components(int dim, int seq_len);

    // Performs feature rotation
    Eigen::Tensor<float, 4> rotate_features(const Eigen::Tensor<float, 4>& x);

    // Applies 1D rotary position embeddings
    Eigen::Tensor<float, 4> apply_1d_rope(const Eigen::Tensor<float, 4>& tokens,
                                         const Eigen::Tensor<float, 3>& positions,
                                         const Eigen::Tensor<float, 2>& cos_comp,
                                         const Eigen::Tensor<float, 2>& sin_comp);

    float base_frequency_;
    float scaling_factor_;
    std::map<std::tuple<int, int>,
             std::pair<Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>>> frequency_cache_;
};
