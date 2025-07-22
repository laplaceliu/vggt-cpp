/**
 * @file modules.h
 * @brief Neural network modules for tracking heads
 */

#pragma once

#include <torch/torch.h>
#include <string>
#include <functional>

namespace vggt {

/**
 * @brief Residual block with two convolutional layers and residual connection
 */
class ResidualBlock : public torch::nn::Module {
public:
    /**
     * @brief Construct a new Residual Block object
     *
     * @param in_planes Input channels
     * @param planes Output channels
     * @param norm_fn Normalization type ("group", "batch", "instance" or "none")
     * @param stride Stride for convolution
     * @param kernel_size Kernel size
     */
    ResidualBlock(int in_planes, int planes,
                const std::string& norm_fn = "group",
                int stride = 1, int kernel_size = 3);

    /**
     * @brief Forward pass
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1, conv2;
    torch::nn::ReLU relu;
    torch::nn::GroupNorm norm1, norm2, norm3;
    torch::nn::Sequential downsample;
    std::string norm_type;
};

/**
 * @brief MLP module as used in Vision Transformer
 */
class Mlp : public torch::nn::Module {
public:
    /**
     * @brief Construct a new Mlp object
     *
     * @param in_features Input features
     * @param hidden_features Hidden layer features (default: in_features)
     * @param out_features Output features (default: in_features)
     * @param act_layer Activation layer (default: torch::nn::GELU)
     * @param norm_layer Normalization layer (default: nullptr)
     * @param bias Use bias (default: true)
     * @param drop Dropout rate (default: 0.0)
     * @param use_conv Use convolutional MLP (default: false)
     */
    Mlp(int in_features,
       int hidden_features = -1,
       int out_features = -1,
       const torch::nn::AnyModule& act_layer = torch::nn::AnyModule(torch::nn::GELU()),
       const torch::nn::AnyModule& norm_layer = torch::nn::AnyModule(),
       bool bias = true,
       double drop = 0.0,
       bool use_conv = false);

    /**
     * @brief Forward pass
     *
     * @param x Input tensor
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::AnyModule fc1, fc2;
    torch::nn::AnyModule act;
    torch::nn::Dropout drop1, drop2;
};

/**
 * @brief Self-attention block
 */
class AttnBlock : public torch::nn::Module {
public:
    /**
     * @brief Construct a new Attn Block object
     *
     * @param hidden_size Hidden size
     * @param num_heads Number of attention heads
     * @param mlp_ratio MLP expansion ratio (default: 4.0)
     */
    AttnBlock(int hidden_size,
             int num_heads,
             double mlp_ratio = 4.0);

    /**
     * @brief Forward pass
     *
     * @param x Input tensor
     * @param mask Optional attention mask
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x, torch::Tensor mask = {});

private:
    torch::nn::LayerNorm norm1, norm2;
    torch::nn::MultiheadAttention attn;
    Mlp mlp;
};

/**
 * @brief Cross-attention block
 */
class CrossAttnBlock : public torch::nn::Module {
public:
    /**
     * @brief Construct a new Cross Attn Block object
     *
     * @param hidden_size Hidden size
     * @param context_dim Context dimension
     * @param num_heads Number of attention heads (default: 1)
     * @param mlp_ratio MLP expansion ratio (default: 4.0)
     */
    CrossAttnBlock(int hidden_size,
                 int context_dim,
                 int num_heads = 1,
                 double mlp_ratio = 4.0);

    /**
     * @brief Forward pass
     *
     * @param x Input tensor
     * @param context Context tensor
     * @param mask Optional attention mask
     * @return torch::Tensor Output tensor
     */
    torch::Tensor forward(torch::Tensor x, torch::Tensor context, torch::Tensor mask = {});

private:
    torch::nn::LayerNorm norm1, norm_context, norm2;
    torch::nn::MultiheadAttention cross_attn;
    Mlp mlp;
};

} // namespace vggt
