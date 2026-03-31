#pragma once

#include <torch/torch.h>
#include "modules.h"
#include "utils.h"

namespace vggt {
namespace dependency {
namespace track_modules {

/**
 * @brief Basic feature encoder with multi-scale output
 * 
 * Encodes input images into features at multiple scales using a series of
 * residual blocks, then fuses them at the original resolution using
 * bilinear interpolation.
 * 
 * Architecture:
 * - conv1 (7x7, stride 2) -> instance norm -> ReLU
 * - layer1, layer2, layer3, layer4 (ResidualBlocks with strides 1, 2, 2, 2)
 * - Each layer output bilinearly interpolated to original resolution
 * - All interpolated outputs concatenated
 * - conv2 (3x3) -> instance norm -> ReLU -> conv3 (1x1)
 */
class BasicEncoderImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a BasicEncoder
     * @param input_dim Number of input channels (default: 3 for RGB)
     * @param output_dim Output feature dimension (default: 128)
     * @param stride Initial stride for first conv (default: 4)
     */
    BasicEncoderImpl(int64_t input_dim = 3, int64_t output_dim = 128, int64_t stride = 4);
    
    /**
     * @brief Forward pass through basic encoder
     * @param x Input tensor [B, input_dim, H, W]
     * @return Encoded features [B, output_dim, H/stride, W/stride]
     */
    torch::Tensor forward(torch::Tensor x);

private:
    /** @brief Create a residual layer with two blocks */
    torch::nn::Sequential _make_layer(int64_t dim, int64_t stride = 1);

    int64_t stride;
    std::string norm_fn;
    int64_t in_planes;

    torch::nn::InstanceNorm2d norm1{nullptr}, norm2{nullptr};
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::ReLU relu1{nullptr}, relu2{nullptr};
    torch::nn::Sequential layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
};
TORCH_MODULE(BasicEncoder);

/**
 * @brief Shallow feature encoder with residual connections
 * 
 * A lightweight encoder with residual blocks and skip connections
 * that maintains spatial resolution through the network.
 * 
 * Architecture:
 * - conv1 (3x3, stride 2) -> norm -> ReLU
 * - layer1: ResidualBlock with stride 2
 * - Add skip connection from layer1 output
 * - layer2: ResidualBlock with stride 2  
 * - Add skip connection from layer2 output
 * - conv2 (1x1) with residual connection
 * - Final bilinear interpolation to output resolution
 */
class ShallowEncoderImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a ShallowEncoder
     * @param input_dim Number of input channels (default: 3)
     * @param output_dim Output feature dimension (default: 32)
     * @param stride Output stride (default: 1)
     * @param norm_fn Normalization: "group", "batch", "instance", "none"
     */
    ShallowEncoderImpl(int64_t input_dim = 3, int64_t output_dim = 32, int64_t stride = 1, const std::string& norm_fn = "instance");
    
    /**
     * @brief Forward pass through shallow encoder
     * @param x Input tensor [B, input_dim, H, W]
     * @return Encoded features [B, output_dim, H/stride, W/stride]
     */
    torch::Tensor forward(torch::Tensor x);

private:
    /** @brief Create a residual block with given dimensions and stride */
    ResidualBlock _make_layer(int64_t dim, int64_t stride = 1);

    int64_t stride;
    std::string norm_fn;
    int64_t in_planes;

    torch::nn::AnyModule norm1, norm2;
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::ReLU relu1{nullptr};
    ResidualBlock layer1{nullptr}, layer2{nullptr};
};
TORCH_MODULE(ShallowEncoder);

/**
 * @brief Bilinear interpolation helper for downsampling
 * @param x Input tensor [..., H, W]
 * @param stride Downsampling factor
 * @param H Original height
 * @param W Original width
 * @return Downsampled tensor [..., H/stride, W/stride]
 */
torch::Tensor _bilinear_intepolate(const torch::Tensor& x, int64_t stride, int64_t H, int64_t W);

/**
 * @brief Efficient Update Transformer for point tracking refinement
 * 
 * Transformer-based module that refines point tracks using alternating
 * temporal (time blocks) and spatial (space blocks) attention.
 * 
 * Architecture:
 * - Input projection: Linear(input_dim -> hidden_size)
 * - Time attention: Self-attention across temporal dimension
 * - Space attention (optional): Point-to-virtual and virtual-to-point attention
 * - Flow prediction: Linear(hidden_size -> output_dim)
 * 
 * Virtual tracks are optional auxiliary tokens that help with attention.
 */
class EfficientUpdateFormerImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct an EfficientUpdateFormer
     * @param space_depth Number of spatial attention layers
     * @param time_depth Number of temporal attention layers
     * @param input_dim Input feature dimension
     * @param hidden_size Hidden dimension for transformer
     * @param num_heads Number of attention heads
     * @param output_dim Output dimension (e.g., flow vectors)
     * @param mlp_ratio MLP hidden dim multiplier
     * @param add_space_attn Whether to use spatial attention
     * @param num_virtual_tracks Number of virtual track tokens
     */
    EfficientUpdateFormerImpl(
        int64_t space_depth = 6,
        int64_t time_depth = 6,
        int64_t input_dim = 320,
        int64_t hidden_size = 384,
        int64_t num_heads = 8,
        int64_t output_dim = 130,
        double mlp_ratio = 4.0,
        bool add_space_attn = true,
        int64_t num_virtual_tracks = 64
    );
    
    /**
     * @brief Forward pass through update former
     * @param input_tensor Input tensor [B, N, T, input_dim]
     * @param mask Optional attention mask
     * @return Flow predictions [B, N, T, output_dim]
     */
    torch::Tensor forward(torch::Tensor input_tensor, torch::Tensor mask = torch::Tensor());
    
    /** @brief Initialize network weights with xavier uniform */
    void initialize_weights();

private:
    int64_t out_channels;
    int64_t num_heads;
    int64_t hidden_size;
    bool add_space_attn;
    int64_t num_virtual_tracks;

    torch::nn::Linear input_transform{nullptr}, flow_head{nullptr};
    torch::Tensor virual_tracks;
    torch::nn::ModuleList time_blocks{nullptr};
    torch::nn::ModuleList space_virtual_blocks{nullptr};
    torch::nn::ModuleList space_point2virtual_blocks{nullptr};
    torch::nn::ModuleList space_virtual2point_blocks{nullptr};
};
TORCH_MODULE(EfficientUpdateFormer);

/**
 * @brief Correlation block for computing correlation pyramids
 * 
 * Computes correlation features between feature maps and query coordinates.
 * Uses a correlation pyramid approach for multi-scale matching.
 * 
 * The correlation is computed as:
 * corr = matmul(features1, features2) / sqrt(C)
 * 
 * Supports multi-scale pyramids via progressive downsampling.
 */
class CorrBlock {
public:
    /**
     * @brief Construct a CorrBlock
     * @param fmaps Feature maps [B, S, C, H, W]
     * @param num_levels Number of pyramid levels (default: 4)
     * @param radius Local correlation radius (default: 4)
     * @param multiple_track_feats Use multiple feature splits
     * @param padding_mode Grid sample padding mode (default: zeros)
     */
    CorrBlock(const torch::Tensor& fmaps, int64_t num_levels = 4, int64_t radius = 4, bool multiple_track_feats = false, const torch::nn::functional::GridSampleFuncOptions::padding_mode_t& padding_mode = torch::kZeros);
    
    /**
     * @brief Sample correlation features at given coordinates
     * @param coords Normalized coordinates [B, S, N, 2]
     * @return Correlation features [B, S, N, LRR*2] where LRR = (2*radius+1)^2
     */
    torch::Tensor sample(const torch::Tensor& coords);
    
    /**
     * @brief Compute correlation between stored fmaps and target features
     * @param targets Target features [B, S, N, C]
     */
    void corr(const torch::Tensor& targets);

private:
    int64_t B, S, C, H, W;
    torch::nn::functional::GridSampleFuncOptions::padding_mode_t padding_mode;
    int64_t num_levels;
    int64_t radius;
    std::vector<torch::Tensor> fmaps_pyramid;
    std::vector<torch::Tensor> corrs_pyramid;
    bool multiple_track_feats;
};

} // namespace track_modules
} // namespace dependency
} // namespace vggt