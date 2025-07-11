/**
 * @file blocks.h
 * @brief Neural network blocks for tracking modules
 */

#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>

namespace vggt {

/**
 * @brief Basic encoder network for feature extraction
 */
class BasicEncoderImpl : public torch::nn::Module {
public:
    BasicEncoderImpl(int input_dim = 3, int output_dim = 128, int stride = 4);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::InstanceNorm2d norm1{nullptr}, norm2{nullptr};
    torch::nn::ReLU relu1{nullptr}, relu2{nullptr};
    torch::nn::Sequential layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
    int stride;
    int in_planes;

    torch::nn::Sequential _make_layer(int dim, int stride = 1);
};
TORCH_MODULE(BasicEncoder);

/**
 * @brief Shallow encoder network for feature extraction
 */
class ShallowEncoderImpl : public torch::nn::Module {
public:
    ShallowEncoderImpl(int input_dim = 3, int output_dim = 32, int stride = 1,
                      const std::string& norm_fn = "instance");
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::ReLU relu1{nullptr};
    torch::nn::Module norm1{nullptr}, norm2{nullptr};
    torch::nn::Sequential layer1{nullptr}, layer2{nullptr};
    int stride;
    int in_planes;
    std::string norm_fn;

    torch::nn::Sequential _make_layer(int dim, int stride = 1);
};
TORCH_MODULE(ShallowEncoder);

/**
 * @brief Transformer model that updates track estimates
 */
class EfficientUpdateFormerImpl : public torch::nn::Module {
public:
    EfficientUpdateFormerImpl(
        int space_depth = 6,
        int time_depth = 6,
        int input_dim = 320,
        int hidden_size = 384,
        int num_heads = 8,
        int output_dim = 130,
        float mlp_ratio = 4.0,
        bool add_space_attn = true,
        int num_virtual_tracks = 64
    );

    torch::Tensor forward(torch::Tensor input_tensor, torch::Tensor mask = torch::Tensor());

private:
    void initialize_weights();

    int out_channels;
    int num_heads;
    int hidden_size;
    bool add_space_attn;
    int num_virtual_tracks;

    torch::nn::Linear input_transform{nullptr};
    torch::nn::Linear flow_head{nullptr};
    torch::Tensor virual_tracks;

    torch::nn::ModuleList time_blocks{nullptr};
    torch::nn::ModuleList space_virtual_blocks{nullptr};
    torch::nn::ModuleList space_point2virtual_blocks{nullptr};
    torch::nn::ModuleList space_virtual2point_blocks{nullptr};
};
TORCH_MODULE(EfficientUpdateFormer);

/**
 * @brief Correlation block for computing and sampling correlations
 */
class CorrBlock {
public:
    CorrBlock(
        torch::Tensor fmaps,
        int num_levels = 4,
        int radius = 4,
        bool multiple_track_feats = false,
        const std::string& padding_mode = "zeros"
    );

    torch::Tensor sample(torch::Tensor coords);
    void corr(torch::Tensor targets);

private:
    int S, C, H, W;
    int num_levels;
    int radius;
    bool multiple_track_feats;
    std::string padding_mode;
    std::vector<torch::Tensor> fmaps_pyramid;
    std::vector<torch::Tensor> corrs_pyramid;
};

} // namespace vggt
