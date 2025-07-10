// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the Apache License, Version 2.0
// found in the LICENSE file in the root directory of this source tree.

#pragma once

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include "patch_embed.h"
#include "attention.h"
#include "mlp.h"
#include "drop_path.h"
#include "layer_scale.h"

class DinoVisionTransformer {
public:
    DinoVisionTransformer(
        int img_size = 224,
        int patch_size = 16,
        int in_chans = 3,
        int embed_dim = 768,
        int depth = 12,
        int num_heads = 12,
        float mlp_ratio = 4.0f,
        bool qkv_bias = true,
        bool proj_bias = true,
        bool ffn_bias = true,
        float drop_path_rate = 0.0f,
        bool drop_path_uniform = false,
        float init_values = 0.0f,
        const std::string& ffn_layer = "mlp",
        int block_chunks = 1,
        int num_register_tokens = 0,
        bool interpolate_antialias = false,
        float interpolate_offset = 0.1f,
        bool qk_norm = false
    );

    // Forward pass
    Eigen::Tensor<float, 4> forward(const Eigen::Tensor<float, 4>& x);

private:
    // Helper functions
    Eigen::Tensor<float, 3> interpolate_pos_encoding(const Eigen::Tensor<float, 4>& x, int w, int h);
    Eigen::Tensor<float, 3> prepare_tokens(const Eigen::Tensor<float, 4>& x);

    // Model parameters
    int img_size_;
    int patch_size_;
    int in_chans_;
    int embed_dim_;
    int depth_;
    int num_heads_;
    float mlp_ratio_;
    int num_register_tokens_;
    bool interpolate_antialias_;
    float interpolate_offset_;

    // Model components
    PatchEmbed patch_embed_;
    Eigen::MatrixXf cls_token_;
    Eigen::MatrixXf pos_embed_;
    Eigen::MatrixXf register_tokens_;
    Eigen::MatrixXf mask_token_;
    std::vector<std::shared_ptr<Block>> blocks_;
    LayerNorm norm_;
};
