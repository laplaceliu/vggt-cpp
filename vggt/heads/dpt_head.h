#pragma once

#include <torch/torch.h>
#include <vector>
#include <utility>
#include <tuple>

namespace vggt {
namespace heads {

/**
 * @brief Custom interpolation with INT_MAX protection
 * 
 * Provides interpolation functionality similar to torch.nn.functional.interpolate
 * but with protection against integer overflow for large tensor sizes.
 * 
 * @param x Input tensor [..., C, H, W]
 * @param size Target size as (height, width) pair (optional)
 * @param scale_factor Scale factor for resizing (optional)
 * @param mode Interpolation mode (default: bilinear)
 * @param align_corners Whether to align corners (default: true)
 * @return Resized tensor [..., C, target_h, target_w]
 * @throws std::runtime_error if neither size nor scale_factor is provided
 */
torch::Tensor custom_interpolate(
    torch::Tensor x,
    c10::optional<std::pair<int64_t, int64_t>> size = c10::nullopt,
    c10::optional<double> scale_factor = c10::nullopt,
    torch::nn::functional::InterpolateFuncOptions::mode_t mode = torch::kBilinear,
    bool align_corners = true);

/**
 * @brief Create scratch module for multi-scale feature fusion
 * 
 * Creates a module with multiple convolutional layers for processing
 * features from different network stages at various scales.
 * 
 * @param in_shape Vector of input channel sizes for each stage
 * @param out_shape Output channel size for each layer
 * @param groups Number of groups for grouped convolution
 * @param expand Whether to expand channels (out_shape * {1, 2, 4, 8})
 * @return torch::nn::Module with registered convolution layers
 */
torch::nn::Module _make_scratch(
    const std::vector<int64_t>& in_shape,
    int64_t out_shape,
    int64_t groups = 1,
    bool expand = false);

/**
 * @brief Create a fusion block with residual convolution units
 * 
 * @param features Number of input/output features
 * @param size Optional target size for interpolation
 * @param has_residual Whether to include residual connection
 * @param groups Number of groups for grouped convolution
 * @return torch::nn::Module with fusion block
 */
torch::nn::Module _make_fusion_block(
    int64_t features,
    c10::optional<std::pair<int64_t, int64_t>> size = c10::nullopt,
    bool has_residual = true,
    int64_t groups = 1);

/**
 * @brief Residual convolution unit with activation and normalization
 * 
 * Architecture: activation -> conv1 -> norm1 -> activation -> conv2 -> norm2 -> add -> input
 * 
 * This implements a residual unit where the input is added to the output
 * after two convolution-normalization-activation blocks.
 */
class ResidualConvUnitImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a ResidualConvUnit
     * @param features Number of input/output channels
     * @param activation Activation function module
     * @param bn Whether to use batch normalization
     * @param groups Number of groups for grouped convolution
     */
    ResidualConvUnitImpl(int64_t features, torch::nn::AnyModule activation, bool bn, int64_t groups = 1);
    
    /**
     * @brief Forward pass through residual convolution unit
     * @param x Input tensor [B, C, H, W]
     * @return Output tensor [B, C, H, W] with residual added
     */
    torch::Tensor forward(torch::Tensor x);

private:
    bool use_bn;
    int64_t groups_;
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::AnyModule norm1, norm2;
    torch::nn::AnyModule activation_;
};
TORCH_MODULE(ResidualConvUnit);

/**
 * @brief Feature fusion block for combining multi-scale features
 * 
 * Fuses features from multiple sources using residual convolution units
 * and optional interpolation to match spatial resolutions.
 * 
 * When has_residual=true with 2+ inputs:
 * - output = input0 + resConfUnit1(input1), then processed through resConfUnit2
 * 
 * When has_residual=false:
 * - output = resConfUnit2(input0)
 */
class FeatureFusionBlockImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a FeatureFusionBlock
     * @param features Number of input/output features
     * @param activation Activation function to use
     * @param deconv Whether to use deconvolution (default: false)
     * @param bn Whether to use batch normalization (default: false)
     * @param expand Whether to expand channels by 2x (default: false)
     * @param align_corners Interpolation corner alignment (default: true)
     * @param size Optional fixed output size
     * @param has_residual Whether to add residual from input1 to input0 (default: true)
     * @param groups Number of groups for grouped convolution
     */
    FeatureFusionBlockImpl(
        int64_t features,
        torch::nn::AnyModule activation,
        bool deconv = false,
        bool bn = false,
        bool expand = false,
        bool align_corners = true,
        c10::optional<std::pair<int64_t, int64_t>> size = c10::nullopt,
        bool has_residual = true,
        int64_t groups = 1);

    /**
     * @brief Forward pass through feature fusion block
     * @param xs Vector of input tensors [B, C, H, W], must have at least 1 element
     * @param size Optional size override for final interpolation
     * @return Fused output tensor
     * @throws c10::Error if has_residual=true but fewer than 2 inputs provided
     */
    torch::Tensor forward(const std::vector<torch::Tensor>& xs, c10::optional<std::pair<int64_t, int64_t>> size = c10::nullopt);

private:
    bool deconv_;
    bool align_corners_;
    int64_t groups_;
    bool expand_;
    bool has_residual_;
    c10::optional<std::pair<int64_t, int64_t>> size_;

    torch::nn::Conv2d out_conv{nullptr};
    ResidualConvUnit resConfUnit1{nullptr};
    ResidualConvUnit resConfUnit2{nullptr};
};
TORCH_MODULE(FeatureFusionBlock);

/**
 * @brief Dense Prediction Head (DPT) for depth and point prediction
 * 
 * DPTHead implements the dense prediction head from the DPT paper (Ranftl et al.)
 * adapted for VGGT. It fuses features from multiple intermediate layers of
 * the vision transformer and produces dense predictions.
 * 
 * Architecture:
 * 1. Project and resize intermediate features to common resolution
 * 2. Fuse features through refinement blocks (refinenet1-4)
 * 3. Apply output convolutions and activation functions
 * 
 * Output: tuple of (predictions, confidence) or (features, empty) if feature_only=true
 */
class DPTHeadImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a DPTHead
     * @param dim_in Input feature dimension from vision transformer
     * @param patch_size Patch size used in ViT (default: 14)
     * @param output_dim Output dimension (default: 4 for depth + 3D points)
     * @param activation Activation function for predictions (default: "inv_log")
     * @param conf_activation Activation for confidence (default: "expp1")
     * @param features Feature dimension for refinement blocks (default: 256)
     * @param out_channels Channels for each intermediate layer projection
     * @param intermediate_layer_idx Indices of intermediate layers to use
     * @param pos_embed Whether to add positional embedding (default: true)
     * @param feature_only If true, only return features without prediction head
     * @param down_ratio Downsampling ratio for output (default: 1)
     */
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
        int64_t down_ratio = 1);

    /**
     * @brief Forward pass through DPT head
     * @param aggregated_tokens_list List of tensors from intermediate ViT layers
     * @param images Input images [B, S, 3, H, W]
     * @param patch_start_idx Starting index for patch tokens
     * @param frames_chunk_size Chunk size for processing frames (0 = all at once)
     * @return Tuple of (predictions [B, S, output_dim, H/out_ratio, W/out_ratio], 
     *                  confidence [B, S, 1, H/out_ratio, W/out_ratio])
     */
    std::tuple<torch::Tensor, torch::Tensor> forward(
        const std::vector<torch::Tensor>& aggregated_tokens_list,
        torch::Tensor images,
        int64_t patch_start_idx,
        int64_t frames_chunk_size = 8);

private:
    std::tuple<torch::Tensor, torch::Tensor> forward_impl(
        const std::vector<torch::Tensor>& aggregated_tokens_list,
        torch::Tensor images,
        int64_t patch_start_idx,
        c10::optional<int64_t> frames_start_idx = c10::nullopt,
        c10::optional<int64_t> frames_end_idx = c10::nullopt);

    /**
     * @brief Process multi-scale features through scratch network
     * @param features Vector of 4 feature tensors at different scales
     * @return Fused feature tensor
     */
    torch::Tensor scratch_forward(const std::vector<torch::Tensor>& features);
    
    /**
     * @brief Apply positional embedding to feature map
     * @param x Feature tensor [B, C, H, W]
     * @param W Original image width
     * @param H Original image height
     * @param ratio Scaling ratio for embedding
     * @return Feature tensor with positional embedding added
     */
    torch::Tensor apply_pos_embed(const torch::Tensor& x, int64_t W, int64_t H, float ratio = 0.1f);

    int64_t patch_size_;
    std::string activation_;
    std::string conf_activation_;
    bool pos_embed_;
    bool feature_only_;
    int64_t down_ratio_;
    std::vector<int64_t> intermediate_layer_idx_;

    torch::nn::LayerNorm norm_{nullptr};
    std::vector<torch::nn::Conv2d> projects_;
    std::vector<torch::nn::AnyModule> resize_layers_;

    // Scratch modules
    torch::nn::Conv2d layer1_rn{nullptr}, layer2_rn{nullptr}, layer3_rn{nullptr}, layer4_rn{nullptr};
    FeatureFusionBlock refinenet1{nullptr}, refinenet2{nullptr}, refinenet3{nullptr}, refinenet4{nullptr};
    torch::nn::Conv2d output_conv1{nullptr};
    torch::nn::Sequential output_conv2_{nullptr};
};
TORCH_MODULE(DPTHead);

} // namespace heads
} // namespace vggt
