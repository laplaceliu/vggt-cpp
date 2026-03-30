#pragma once

#include <torch/torch.h>
#include <vector>
#include <utility>
#include <tuple>

namespace vggt {
namespace heads {

// Custom interpolate function to avoid INT_MAX issues
torch::Tensor custom_interpolate(
    torch::Tensor x,
    c10::optional<std::pair<int64_t, int64_t>> size = c10::nullopt,
    c10::optional<double> scale_factor = c10::nullopt,
    torch::nn::functional::InterpolateFuncOptions::mode_t mode = torch::kBilinear,
    bool align_corners = true);

// Create scratch module for feature fusion
torch::nn::Module _make_scratch(
    const std::vector<int64_t>& in_shape,
    int64_t out_shape,
    int64_t groups = 1,
    bool expand = false);

// Create fusion block
torch::nn::Module _make_fusion_block(
    int64_t features,
    c10::optional<std::pair<int64_t, int64_t>> size = c10::nullopt,
    bool has_residual = true,
    int64_t groups = 1);

// Residual convolution unit
class ResidualConvUnitImpl : public torch::nn::Module {
public:
    ResidualConvUnitImpl(int64_t features, torch::nn::AnyModule activation, bool bn, int64_t groups = 1);
    torch::Tensor forward(torch::Tensor x);

private:
    bool use_bn;
    int64_t groups_;
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::AnyModule norm1, norm2;
    torch::nn::AnyModule activation_;
};
TORCH_MODULE(ResidualConvUnit);

// Feature fusion block
class FeatureFusionBlockImpl : public torch::nn::Module {
public:
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

// DPT Head for dense prediction
class DPTHeadImpl : public torch::nn::Module {
public:
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

    torch::Tensor scratch_forward(const std::vector<torch::Tensor>& features);
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
