#pragma once

#include <torch/torch.h>
#include <vector>

namespace vggt {
namespace layers {

class BlockImpl : public torch::nn::Module {
public:
    BlockImpl(
        int64_t dim,
        int64_t num_heads,
        double mlp_ratio = 4.0,
        bool qkv_bias = true,
        bool proj_bias = true,
        bool ffn_bias = true,
        double drop = 0.0,
        double attn_drop = 0.0,
        torch::Tensor init_values = {},
        double drop_path = 0.0,
        torch::nn::AnyModule act_layer = torch::nn::AnyModule(torch::nn::GELU()),
        torch::nn::AnyModule norm_layer = torch::nn::AnyModule(torch::nn::LayerNorm(torch::nn::LayerNormOptions({}))),
        torch::nn::AnyModule attn_class = torch::nn::AnyModule(),
        torch::nn::AnyModule ffn_layer = torch::nn::AnyModule(),
        bool qk_norm = false,
        bool fused_attn = true,
        torch::nn::AnyModule rope = torch::nn::AnyModule()
    );

    torch::Tensor forward(torch::Tensor x, torch::Tensor pos = {});

protected:
    torch::nn::AnyModule norm1;
    torch::nn::AnyModule attn;
    torch::nn::AnyModule ls1;
    torch::nn::AnyModule drop_path1;
    torch::nn::AnyModule norm2;
    torch::nn::AnyModule mlp;
    torch::nn::AnyModule ls2;
    torch::nn::AnyModule drop_path2;
    double sample_drop_ratio;
};

TORCH_MODULE(Block);

torch::Tensor drop_add_residual_stochastic_depth(
    torch::Tensor x, 
    std::function<torch::Tensor(torch::Tensor, torch::Tensor)> residual_func, 
    double sample_drop_ratio = 0.0, 
    torch::Tensor pos = {}
);

std::tuple<torch::Tensor, double> get_branges_scales(torch::Tensor x, double sample_drop_ratio = 0.0);

torch::Tensor add_residual(
    torch::Tensor x, 
    torch::Tensor brange, 
    torch::Tensor residual, 
    double residual_scale_factor, 
    torch::Tensor scaling_vector = {}
);

class NestedTensorBlockImpl : public BlockImpl {
public:
    using BlockImpl::BlockImpl;

    std::vector<torch::Tensor> forward_nested(std::vector<torch::Tensor> x_list);
    torch::Tensor forward(torch::Tensor x_or_x_list);
};

TORCH_MODULE(NestedTensorBlock);

} // namespace layers
} // namespace vggt