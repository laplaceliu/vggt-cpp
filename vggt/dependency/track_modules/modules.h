// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/torch.h>

namespace vggt {
namespace track_modules {

class ResidualBlockImpl : public torch::nn::Module {
public:
    ResidualBlockImpl(int64_t in_planes, int64_t planes, const std::string& norm_fn = "group", int64_t stride = 1, int64_t kernel_size = 3);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::ReLU relu{nullptr};
    torch::nn::GroupNorm norm1{nullptr}, norm2{nullptr}, norm3{nullptr};
    torch::nn::Sequential downsample{nullptr};
};
TORCH_MODULE(ResidualBlock);

class MlpImpl : public torch::nn::Module {
public:
    MlpImpl(int64_t in_features, 
            int64_t hidden_features = 0, 
            int64_t out_features = 0, 
            const torch::nn::AnyModule& act_layer = torch::nn::AnyModule(torch::nn::GELU()), 
            const torch::nn::AnyModule& norm_layer = torch::nn::AnyModule(), 
            bool bias = true, 
            double drop = 0.0, 
            bool use_conv = false);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::AnyModule fc1{nullptr}, fc2{nullptr};
    torch::nn::AnyModule act{nullptr};
    torch::nn::Dropout drop1{nullptr}, drop2{nullptr};
};
TORCH_MODULE(Mlp);

class AttnBlockImpl : public torch::nn::Module {
public:
    AttnBlockImpl(int64_t hidden_size, 
                 int64_t num_heads, 
                 const torch::nn::AnyModule& attn_class = torch::nn::AnyModule(torch::nn::MultiheadAttentionOptions(hidden_size, num_heads).batch_first(true)), 
                 double mlp_ratio = 4.0);
    torch::Tensor forward(torch::Tensor x, torch::Tensor mask = {});

private:
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    torch::nn::AnyModule attn{nullptr};
    Mlp mlp{nullptr};
};
TORCH_MODULE(AttnBlock);

class CrossAttnBlockImpl : public torch::nn::Module {
public:
    CrossAttnBlockImpl(int64_t hidden_size, 
                      int64_t context_dim, 
                      int64_t num_heads = 1, 
                      double mlp_ratio = 4.0);
    torch::Tensor forward(torch::Tensor x, torch::Tensor context, torch::Tensor mask = {});

private:
    torch::nn::LayerNorm norm1{nullptr}, norm_context{nullptr}, norm2{nullptr};
    torch::nn::MultiheadAttention cross_attn{nullptr};
    Mlp mlp{nullptr};
};
TORCH_MODULE(CrossAttnBlock);

} // namespace track_modules
} // namespace vggt