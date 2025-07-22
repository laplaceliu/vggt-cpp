

#pragma once

#include <torch/torch.h>

namespace vggt {
namespace track_modules {

class ResidualBlockImpl : public torch::nn::Module {
public:
    ResidualBlockImpl(int64_t in_planes, int64_t planes, const std::string& norm_fn = "group", int64_t stride = 1, int64_t kernel_size = 3);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1, conv2;
    torch::nn::ReLU relu;
    torch::nn::GroupNorm norm1, norm2, norm3;
    torch::nn::Sequential downsample;
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
    torch::nn::AnyModule fc1, fc2;
    torch::nn::AnyModule act;
    torch::nn::Dropout drop1, drop2;
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
    torch::nn::LayerNorm norm1, norm2;
    torch::nn::AnyModule attn;
    Mlp mlp;
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
    torch::nn::LayerNorm norm1, norm_context, norm2;
    torch::nn::MultiheadAttention cross_attn;
    Mlp mlp;
};
TORCH_MODULE(CrossAttnBlock);

} // namespace track_modules
} // namespace vggt
