// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the Apache License, Version 2.0
// found in the LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/nn/module.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/modules/normalization.h>

namespace vggt {
namespace layers {

class PatchEmbedImpl : public torch::nn::Module {
public:
    PatchEmbedImpl(
        int64_t img_size = 224,
        int64_t patch_size = 16,
        int64_t in_chans = 3,
        int64_t embed_dim = 768,
        torch::nn::AnyModule norm = nullptr,
        bool flatten_embedding = true);

    torch::Tensor forward(const torch::Tensor& x);

private:
    std::pair<int64_t, int64_t> img_size_;
    std::pair<int64_t, int64_t> patch_size_;
    std::pair<int64_t, int64_t> patches_resolution_;
    int64_t num_patches_;
    int64_t in_chans_;
    int64_t embed_dim_;
    bool flatten_embedding_;
    torch::nn::Conv2d proj{nullptr};
    torch::nn::AnyModule norm{nullptr};
};

TORCH_MODULE(PatchEmbed);

} // namespace layers
} // namespace vggt
