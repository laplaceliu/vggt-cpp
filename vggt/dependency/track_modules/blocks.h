#pragma once

#include <torch/torch.h>
#include "modules.h"
#include "utils.h"

namespace vggt {
namespace dependency {
namespace track_modules {

class BasicEncoderImpl : public torch::nn::Module {
public:
    BasicEncoderImpl(int64_t input_dim = 3, int64_t output_dim = 128, int64_t stride = 4);
    torch::Tensor forward(torch::Tensor x);

private:
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

class ShallowEncoderImpl : public torch::nn::Module {
public:
    ShallowEncoderImpl(int64_t input_dim = 3, int64_t output_dim = 32, int64_t stride = 1, const std::string& norm_fn = "instance");
    torch::Tensor forward(torch::Tensor x);

private:
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

torch::Tensor _bilinear_intepolate(const torch::Tensor& x, int64_t stride, int64_t H, int64_t W);

class EfficientUpdateFormerImpl : public torch::nn::Module {
public:
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
    torch::Tensor forward(torch::Tensor input_tensor, torch::Tensor mask = torch::Tensor());
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

class CorrBlock {
public:
    CorrBlock(const torch::Tensor& fmaps, int64_t num_levels = 4, int64_t radius = 4, bool multiple_track_feats = false, const torch::nn::functional::GridSampleFuncOptions::padding_mode_t& padding_mode = torch::kZeros);
    torch::Tensor sample(const torch::Tensor& coords);
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