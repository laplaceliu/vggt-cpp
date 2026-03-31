#pragma once

#include <torch/torch.h>
#include <vector>
#include <functional>
#include "utils/stack_sequential.h"

namespace vggt {
namespace dependency {
namespace track_modules {

/**
 * @brief Create n-tuple from a single value or pair
 * @param n Size of the tuple to create
 * @param x Value or pair to expand
 * @return std::vector<int64_t> n-tuple
 * 
 * If x equals 1, returns a vector of n copies of 1.
 * Otherwise returns {x, x} for n=2.
 */
inline std::vector<int64_t> _ntuple(int n, int64_t x) {
    if (x == 1) {
        return std::vector<int64_t>(n, x);
    }
    return {x, x};
}

/**
 * @brief Check if a tensor is defined
 * @param val Tensor to check
 * @return true if tensor is defined, false otherwise
 */
inline bool exists(torch::Tensor val) {
    return val.defined();
}

/**
 * @brief Return tensor if defined, otherwise return default
 * @param val Tensor to return if defined
 * @param d Default tensor to return if val is undefined
 * @return val if defined, otherwise d
 */
inline torch::Tensor default_val(torch::Tensor val, torch::Tensor d) {
    return exists(val) ? val : d;
}

/**
 * @brief Convert single value to 2-tuple
 * @param x Single value
 * @return std::vector<int64_t> 2-tuple
 */
inline std::vector<int64_t> to_2tuple(int64_t x) {
    return _ntuple(2, x);
}

/**
 * @brief Residual block with convolutions and normalization
 * 
 * A residual block with two convolutional layers, optional downsampling,
 * and support for different normalization functions (group, batch, instance, none).
 * 
 * Architecture:
 * - conv1 -> norm1 -> relu
 * - conv2 -> norm2 -> relu
 * - Optional downsample for stride > 1
 * - Residual connection: output = relu(x + y)
 */
class ResidualBlockImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a ResidualBlock
     * @param in_planes Number of input channels
     * @param planes Number of output channels
     * @param norm_fn Normalization function: "group", "batch", "instance", or "none"
     * @param stride Stride for first convolution (default: 1)
     * @param kernel_size Kernel size for convolutions (default: 3)
     */
    ResidualBlockImpl(int64_t in_planes, int64_t planes, const std::string& norm_fn = "group", int64_t stride = 1, int64_t kernel_size = 3);
    
    /**
     * @brief Forward pass through residual block
     * @param x Input tensor [B, C, H, W]
     * @return Output tensor [B, planes, H/stride, W/stride]
     */
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::ReLU relu{nullptr};
    torch::nn::AnyModule norm1, norm2, norm3;
    utils::StackSequential downsample{nullptr};
};
TORCH_MODULE(ResidualBlock);

/**
 * @brief Multi-layer perceptron with optional dropout and conv support
 * 
 * Standard MLP with two linear layers, activation function, and dropout.
 * Supports optional weight normalization and convolution-based computation.
 */
class MlpImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct an MLP
     * @param in_features Input feature dimension
     * @param hidden_features Hidden layer dimension (default: same as in_features)
     * @param out_features Output dimension (default: same as in_features)
     * @param act_layer Activation function (default: GELU)
     * @param norm_layer Normalization layer (optional)
     * @param bias Whether to use bias (default: true)
     * @param drop Dropout probability (default: 0.0)
     * @param use_conv Use conv instead of linear (default: false)
     */
    MlpImpl(int64_t in_features, int64_t hidden_features = -1, int64_t out_features = -1,
            torch::nn::AnyModule act_layer = torch::nn::AnyModule(torch::nn::GELU()),
            torch::nn::AnyModule norm_layer = torch::nn::AnyModule(), bool bias = true,
            double drop = 0.0, bool use_conv = false);
    
    /**
     * @brief Forward pass through MLP
     * @param x Input tensor [..., in_features]
     * @return Output tensor [..., out_features]
     */
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::AnyModule act;
    torch::nn::Dropout drop1{nullptr}, drop2{nullptr};
};
TORCH_MODULE(Mlp);

/**
 * @brief Self-attention block with MLP and layer normalization
 * 
 * Standard transformer-style attention block with pre-norm,
 * multihead attention, and MLP path.
 * 
 * Architecture:
 * - x = norm1(x); attn_out = attn(x, x, x); x = x + attn_out
 * - x = x + mlp(norm2(x))
 */
class AttnBlockImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct an AttnBlock
     * @param hidden_size Feature dimension
     * @param num_heads Number of attention heads
     * @param attn_class Attention module to use
     * @param mlp_ratio MLP hidden dim multiplier (default: 4.0)
     */
    AttnBlockImpl(int64_t hidden_size, int64_t num_heads,
                 torch::nn::AnyModule attn_class,
                 double mlp_ratio = 4.0);
    
    /**
     * @brief Forward pass through attention block
     * @param x Input tensor [batch, seq, hidden_size]
     * @param mask Optional attention mask
     * @return Output tensor [batch, seq, hidden_size]
     */
    torch::Tensor forward(torch::Tensor x, torch::Tensor mask = torch::Tensor());

private:
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    torch::nn::AnyModule attn;
    Mlp mlp{nullptr};
};
TORCH_MODULE(AttnBlock);

/**
 * @brief Cross-attention block with MLP
 * 
 * Cross-attention block where query comes from x and key/value come from context.
 * Uses separate layer norms for query and context.
 */
class CrossAttnBlockImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a CrossAttnBlock
     * @param hidden_size Query feature dimension
     * @param context_dim Context key/value dimension
     * @param num_heads Number of attention heads (default: 1)
     * @param mlp_ratio MLP hidden dim multiplier (default: 4.0)
     */
    CrossAttnBlockImpl(int64_t hidden_size, int64_t context_dim, int64_t num_heads = 1, double mlp_ratio = 4.0);
    
    /**
     * @brief Forward pass through cross-attention block
     * @param x Query tensor [batch, seq_x, hidden_size]
     * @param context Key/value tensor [batch, seq_ctx, context_dim]
     * @param mask Optional attention mask
     * @return Output tensor [batch, seq_x, hidden_size]
     */
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
