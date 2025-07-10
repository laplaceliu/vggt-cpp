/**
 * @file dpt_head.h
 * @brief DPT head module for dense prediction tasks
 */

#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>

namespace vggt {
namespace heads {

class DPTHeadImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new DPTHeadImpl object
     *
     * @param dim_in Input dimension (channels)
     * @param patch_size Patch size (default=14)
     * @param output_dim Number of output channels (default=4)
     * @param activation Activation type (default="inv_log")
     * @param conf_activation Confidence activation type (default="expp1")
     * @param features Feature channels for intermediate representations (default=256)
     * @param out_channels Output channels for each intermediate layer
     * @param intermediate_layer_idx Indices of layers from aggregated tokens used for DPT
     * @param pos_embed Whether to use positional embedding (default=true)
     * @param feature_only If true, return features only (default=false)
     * @param down_ratio Downscaling factor for output resolution (default=1)
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
     * @brief Forward pass through the DPT head
     *
     * @param aggregated_tokens_list List of token tensors from different transformer layers
     * @param images Input images with shape [B, S, 3, H, W], in range [0, 1]
     * @param patch_start_idx Starting index for patch tokens in the token sequence
     * @param frames_chunk_size Number of frames to process in each chunk (default=8)
     * @return torch::Tensor or std::tuple<torch::Tensor, torch::Tensor>
     *         Feature maps or tuple of (predictions, confidence)
     */
    torch::Tensor forward(
        const std::vector<torch::Tensor>& aggregated_tokens_list,
        const torch::Tensor& images,
        int64_t patch_start_idx,
        int64_t frames_chunk_size = 8);

private:
    int64_t patch_size_;
    std::string activation_;
    std::string conf_activation_;
    bool pos_embed_;
    bool feature_only_;
    int64_t down_ratio_;
    std::vector<int64_t> intermediate_layer_idx_;

    // Submodules
    torch::nn::LayerNorm norm_;
    torch::nn::ModuleList projects_;
    torch::nn::ModuleList resize_layers_;
    Scratch scratch_;

    /**
     * @brief Implementation of the forward pass
     */
    torch::Tensor _forward_impl(
        const std::vector<torch::Tensor>& aggregated_tokens_list,
        const torch::Tensor& images,
        int64_t patch_start_idx,
        int64_t frames_start_idx = -1,
        int64_t frames_end_idx = -1);

    /**
     * @brief Apply positional embedding to tensor
     */
    torch::Tensor _apply_pos_embed(
        const torch::Tensor& x,
        int64_t W,
        int64_t H,
        float ratio = 0.1f);

    /**
     * @brief Forward pass through the fusion blocks
     */
    torch::Tensor scratch_forward(
        const std::vector<torch::Tensor>& features);
};

TORCH_MODULE(DPTHead);

// Helper classes
struct Scratch : torch::nn::Module {
    torch::nn::Conv2d layer1_rn{nullptr}, layer2_rn{nullptr}, layer3_rn{nullptr}, layer4_rn{nullptr};
    torch::nn::Module refinenet1{nullptr}, refinenet2{nullptr}, refinenet3{nullptr}, refinenet4{nullptr};
    torch::nn::Module output_conv1{nullptr}, output_conv2{nullptr};
    torch::nn::Module stem_transpose{nullptr};

    void reset() override;
};

class ResidualConvUnitImpl : public torch::nn::Module {
public:
    ResidualConvUnitImpl(
        int64_t features,
        const torch::nn::AnyModule& activation,
        bool bn,
        int64_t groups = 1);

    torch::Tensor forward(const torch::Tensor& x);

private:
    torch::nn::Conv2d conv1_{nullptr}, conv2_{nullptr};
    torch::nn::Module norm1_{nullptr}, norm2_{nullptr};
    torch::nn::AnyModule activation_;
    bool bn_;
    int64_t groups_;
};

TORCH_MODULE(ResidualConvUnit);

class FeatureFusionBlockImpl : public torch::nn::Module {
public:
    FeatureFusionBlockImpl(
        int64_t features,
        const torch::nn::AnyModule& activation,
        bool deconv = false,
        bool bn = false,
        bool expand = false,
        bool align_corners = true,
        const std::vector<int64_t>& size = {},
        bool has_residual = true,
        int64_t groups = 1);

    torch::Tensor forward(
        const torch::Tensor& x0,
        const torch::Tensor& x1 = torch::Tensor(),
        const std::vector<int64_t>& size = {});

private:
    bool deconv_;
    bool align_corners_;
    int64_t groups_;
    bool expand_;
    std::vector<int64_t> size_;
    bool has_residual_;

    torch::nn::Conv2d out_conv_{nullptr};
    ResidualConvUnit resConfUnit1_{nullptr}, resConfUnit2_{nullptr};
};

TORCH_MODULE(FeatureFusionBlock);

} // namespace heads
} // namespace vggt
