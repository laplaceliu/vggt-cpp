// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/torch.h>
#include "../layers/conv.h"
#include "head_act.h"
#include "../utils/position_embed.h"

namespace vggt {
namespace heads {

class DPTHeadImpl : public torch::nn::Module {
public:
    DPTHeadImpl(
        int64_t dim_in,
        int64_t patch_size = 14,
        int64_t output_dim = 4,
        const std::string& activation = "inv_log",
        const std::string& conf_activation = "expp1",
        int64_t features = 256,
        const std::vector<int64_t>& out_channels = {256, 512, 1024, 1024},
        const std::vector<int64_t>& intermediate_layer_idx = {4, 11, 17, 23},
        bool pos_embed = true,
        bool feature_only = false,
        int64_t down_ratio = 1
    );

    torch::Tensor forward(
        const std::vector<torch::Tensor>& aggregated_tokens_list,
        const torch::Tensor& images,
        int64_t patch_start_idx,
        int64_t frames_chunk_size = 8
    );

private:
    torch::Tensor _forward_impl(
        const std::vector<torch::Tensor>& aggregated_tokens_list,
        const torch::Tensor& images,
        int64_t patch_start_idx,
        int64_t frames_start_idx = -1,
        int64_t frames_end_idx = -1
    );

    torch::Tensor _apply_pos_embed(const torch::Tensor& x, int64_t W, int64_t H, float ratio = 0.1);
    torch::Tensor scratch_forward(const std::vector<torch::Tensor>& features);

    int64_t patch_size;
    std::string activation, conf_activation;
    bool pos_embed, feature_only;
    int64_t down_ratio;
    std::vector<int64_t> intermediate_layer_idx;

    torch::nn::LayerNorm norm;
    torch::nn::ModuleList projects;
    torch::nn::ModuleList resize_layers;
    torch::nn::Module scratch;
};
TORCH_MODULE(DPTHead);

} // namespace heads
} // namespace vggt