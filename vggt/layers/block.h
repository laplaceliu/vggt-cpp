#pragma once

#include <torch/torch.h>
#include <vector>
#include <memory>

#include "attention.h"
#include "mlp.h"
#include "layer_scale.h"
#include "drop_path.h"

namespace vggt {
namespace layers {

/**
 * A standard transformer block with support for various attention mechanisms,
 * layer scaling, and stochastic depth.
 */
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
    FORWARD_HAS_DEFAULT_ARGS({1, torch::nn::AnyValue(torch::Tensor())})

protected:
    // Submodules - use TORCH_MODULE types for proper registration
    torch::nn::LayerNorm norm1{nullptr};
    Attention attn{nullptr};
    LayerScale ls1{nullptr};
    torch::nn::Identity drop_path1_identity{nullptr};
    DropPath drop_path1_droppath{nullptr};
    torch::nn::LayerNorm norm2{nullptr};
    Mlp mlp{nullptr};
    LayerScale ls2{nullptr};
    torch::nn::Identity drop_path2_identity{nullptr};
    DropPath drop_path2_droppath{nullptr};
    
    double sample_drop_ratio;
    bool use_drop_path1;
    bool use_drop_path2;
    bool use_ls1;
    bool use_ls2;
};

TORCH_MODULE(Block);

// Helper functions for residual connections with stochastic depth
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

// Nested tensor version for efficient batch processing
class NestedTensorBlockImpl : public BlockImpl {
public:
    using BlockImpl::BlockImpl;

    std::vector<torch::Tensor> forward_nested(std::vector<torch::Tensor> x_list);
    torch::Tensor forward(torch::Tensor x_or_x_list);
};

TORCH_MODULE(NestedTensorBlock);

} // namespace layers
} // namespace vggt
