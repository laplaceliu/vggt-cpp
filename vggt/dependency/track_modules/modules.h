#pragma once

#include <torch/torch.h>
#include <vector>
#include <functional>
#include "utils/stack_sequential.h"

namespace vggt {
namespace dependency {
namespace track_modules {

// From PyTorch internals
inline std::vector<int64_t> _ntuple(int n, int64_t x) {
    if (x == 1) {
        return std::vector<int64_t>(n, x);
    }
    return {x, x};
}

inline bool exists(torch::Tensor val) {
    return val.defined();
}

inline torch::Tensor default_val(torch::Tensor val, torch::Tensor d) {
    return exists(val) ? val : d;
}

inline std::vector<int64_t> to_2tuple(int64_t x) {
    return _ntuple(2, x);
}

class ResidualBlockImpl : public torch::nn::Module {
public:
    ResidualBlockImpl(int64_t in_planes, int64_t planes, const std::string& norm_fn = "group", int64_t stride = 1, int64_t kernel_size = 3);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::ReLU relu{nullptr};
    torch::nn::AnyModule norm1, norm2, norm3;
    utils::StackSequential downsample{nullptr};
};
TORCH_MODULE(ResidualBlock);

class MlpImpl : public torch::nn::Module {
public:
    MlpImpl(int64_t in_features, int64_t hidden_features = -1, int64_t out_features = -1,
            torch::nn::AnyModule act_layer = torch::nn::AnyModule(torch::nn::GELU()),
            torch::nn::AnyModule norm_layer = torch::nn::AnyModule(), bool bias = true,
            double drop = 0.0, bool use_conv = false);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::AnyModule act;
    torch::nn::Dropout drop1{nullptr}, drop2{nullptr};
};
TORCH_MODULE(Mlp);

class AttnBlockImpl : public torch::nn::Module {
public:
    AttnBlockImpl(int64_t hidden_size, int64_t num_heads,
                 torch::nn::AnyModule attn_class,
                 double mlp_ratio = 4.0);
    torch::Tensor forward(torch::Tensor x, torch::Tensor mask = torch::Tensor());

private:
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    torch::nn::AnyModule attn;
    Mlp mlp{nullptr};
};
TORCH_MODULE(AttnBlock);

class CrossAttnBlockImpl : public torch::nn::Module {
public:
    CrossAttnBlockImpl(int64_t hidden_size, int64_t context_dim, int64_t num_heads = 1, double mlp_ratio = 4.0);
    torch::Tensor forward(torch::Tensor x, torch::Tensor context, torch::Tensor mask = torch::Tensor());

private:
    torch::nn::LayerNorm norm1{nullptr}, norm_context{nullptr}, norm2{nullptr};
    torch::nn::MultiheadAttention cross_attn{nullptr};
    Mlp mlp{nullptr};
};
TORCH_MODULE(CrossAttnBlock);

} // namespace track_modules
} // namespace dependency
} // namespace vggt
