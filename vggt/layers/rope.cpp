// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the Apache License, Version 2.0
// found in the LICENSE file in the root directory of this source tree.

#include "rope.h"
#include <cmath>
#include <iostream>

using namespace Eigen;

PositionGetter::PositionGetter() {}

Eigen::Tensor<float, 3> PositionGetter::operator()(int batch_size, int height, int width) {
    auto key = std::make_pair(height, width);
    if (position_cache_.find(key) == position_cache_.end()) {
        // Generate y and x coordinates
        Tensor<float, 1> y_coords = Tensor<float, 1>(height);
        Tensor<float, 1> x_coords = Tensor<float, 1>(width);
        for (int i = 0; i < height; ++i) y_coords(i) = static_cast<float>(i);
        for (int i = 0; i < width; ++i) x_coords(i) = static_cast<float>(i);

        // Create cartesian product of positions
        Tensor<float, 2> positions(height * width, 2);
        int idx = 0;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                positions(idx, 0) = y_coords(y);
                positions(idx, 1) = x_coords(x);
                idx++;
            }
        }
        position_cache_[key] = positions;
    }

    // Expand for batch size
    auto cached_positions = position_cache_[key];
    Tensor<float, 3> result(batch_size, height * width, 2);
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < height * width; ++i) {
            result(b, i, 0) = cached_positions(i, 0);
            result(b, i, 1) = cached_positions(i, 1);
        }
    }
    return result;
}

RotaryPositionEmbedding2D::RotaryPositionEmbedding2D(float frequency, float scaling_factor)
    : base_frequency_(frequency), scaling_factor_(scaling_factor) {}

std::pair<Eigen::Tensor<float, 2>, Eigen::Tensor<float, 2>>
RotaryPositionEmbedding2D::compute_frequency_components(int dim, int seq_len) {
    auto key = std::make_tuple(dim, seq_len);
    if (frequency_cache_.find(key) == frequency_cache_.end()) {
        // Compute frequency bands
        Tensor<float, 1> exponents(dim / 2);
        Tensor<float, 1> inv_freq(dim / 2);
        for (int i = 0; i < dim / 2; ++i) {
            exponents(i) = static_cast<float>(i) / dim;
            inv_freq(i) = 1.0f / std::pow(base_frequency_, exponents(i));
        }

        // Generate position-dependent frequencies
        Tensor<float, 2> angles(seq_len, dim / 2);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < dim / 2; ++j) {
                angles(i, j) = static_cast<float>(i) * inv_freq(j);
            }
        }

        // Compute cosine and sine components
        Tensor<float, 2> cos_comp = angles.cos();
        Tensor<float, 2> sin_comp = angles.sin();

        // Duplicate for full dimension
        Tensor<float, 2> full_cos(seq_len, dim);
        Tensor<float, 2> full_sin(seq_len, dim);
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < dim / 2; ++j) {
                full_cos(i, j) = cos_comp(i, j);
                full_cos(i, j + dim / 2) = cos_comp(i, j);
                full_sin(i, j) = sin_comp(i, j);
                full_sin(i, j + dim / 2) = sin_comp(i, j);
            }
        }

        frequency_cache_[key] = std::make_pair(full_cos, full_sin);
    }
    return frequency_cache_[key];
}

Eigen::Tensor<float, 4> RotaryPositionEmbedding2D::rotate_features(const Eigen::Tensor<float, 4>& x) {
    int feature_dim = x.dimension(3);
    Tensor<float, 4> result = x;

    // Split and rotate features
    for (int b = 0; b < x.dimension(0); ++b) {
        for (int h = 0; h < x.dimension(1); ++h) {
            for (int t = 0; t < x.dimension(2); ++t) {
                for (int d = 0; d < feature_dim / 2; ++d) {
                    float x1 = x(b, h, t, d);
                    float x2 = x(b, h, t, d + feature_dim / 2);
                    result(b, h, t, d) = -x2;
                    result(b, h, t, d + feature_dim / 2) = x1;
                }
            }
        }
    }
    return result;
}

Eigen::Tensor<float, 4> RotaryPositionEmbedding2D::apply_1d_rope(
    const Eigen::Tensor<float, 4>& tokens,
    const Eigen::Tensor<float, 3>& positions,
    const Eigen::Tensor<float, 2>& cos_comp,
    const Eigen::Tensor<float, 2>& sin_comp) {

    Tensor<float, 4> cos_embedding(tokens.dimension(0), 1, tokens.dimension(2), tokens.dimension(3));
    Tensor<float, 4> sin_embedding(tokens.dimension(0), 1, tokens.dimension(2), tokens.dimension(3));

    // Create embeddings from position indices
    for (int b = 0; b < tokens.dimension(0); ++b) {
        for (int t = 0; t < tokens.dimension(2); ++t) {
            int pos = static_cast<int>(positions(b, t, 0));
            for (int d = 0; d < tokens.dimension(3); ++d) {
                cos_embedding(b, 0, t, d) = cos_comp(pos, d);
                sin_embedding(b, 0, t, d) = sin_comp(pos, d);
            }
        }
    }

    // Apply rotation
    return tokens * cos_embedding + rotate_features(tokens) * sin_embedding;
}

Eigen::Tensor<float, 4> RotaryPositionEmbedding2D::forward(
    const Eigen::Tensor<float, 4>& tokens,
    const Eigen::Tensor<float, 3>& positions) {

    // Validate inputs
    if (tokens.dimension(3) % 2 != 0) {
        throw std::invalid_argument("Feature dimension must be even");
    }
    if (positions.dimension(2) != 2) {
        throw std::invalid_argument("Positions must have shape (batch_size, n_tokens, 2)");
    }

    // Compute feature dimension for each spatial direction
    int feature_dim = tokens.dimension(3) / 2;

    // Get max position for frequency computation
    float max_pos = 0;
    for (int i = 0; i < positions.size(); ++i) {
        if (positions.data()[i] > max_pos) {
            max_pos = positions.data()[i];
        }
    }
    int max_position = static_cast<int>(max_pos) + 1;

    // Get frequency components
    auto [cos_comp, sin_comp] = compute_frequency_components(feature_dim, max_position);

    // Split features for vertical and horizontal processing
    Tensor<float, 4> vertical_features(tokens.dimension(0), tokens.dimension(1),
                                      tokens.dimension(2), feature_dim);
    Tensor<float, 4> horizontal_features(tokens.dimension(0), tokens.dimension(1),
                                        tokens.dimension(2), feature_dim);

    for (int i = 0; i < vertical_features.size(); ++i) {
        vertical_features.data()[i] = tokens.data()[i];
        horizontal_features.data()[i] = tokens.data()[i + feature_dim * tokens.dimension(0) * tokens.dimension(1) * tokens.dimension(2)];
    }

    // Apply RoPE separately for each dimension
    vertical_features = apply_1d_rope(vertical_features, positions, cos_comp, sin_comp);
    horizontal_features = apply_1d_rope(horizontal_features, positions, cos_comp, sin_comp);

    // Combine processed features
    Tensor<float, 4> result(tokens.dimension(0), tokens.dimension(1),
                           tokens.dimension(2), tokens.dimension(3));
    for (int i = 0; i < vertical_features.size(); ++i) {
        result.data()[i] = vertical_features.data()[i];
        result.data()[i + feature_dim * tokens.dimension(0) * tokens.dimension(1) * tokens.dimension(2)] =
            horizontal_features.data()[i];
    }

    return result;
}
