#ifndef VGGT_HEADS_FEATURE_FUSION_BLOCK_H
#define VGGT_HEADS_FEATURE_FUSION_BLOCK_H

#include <torch/torch.h>

namespace vggt {
namespace heads {

class ResidualConvUnitImpl : public torch::nn::Module {
public:
    ResidualConvUnitImpl(int features, const torch::nn::Functional& activation, bool bn, int groups = 1);
    torch::Tensor forward(const torch::Tensor& x);

private:
    bool bn;
    int groups;
    torch::nn::Conv2d conv1;
    torch::nn::Conv2d conv2;
    torch::nn::LayerNorm norm1;
    torch::nn::LayerNorm norm2;
    torch::nn::Functional activation;
    torch::nn::quantized::FloatFunctional skip_add;
};

TORCH_MODULE(ResidualConvUnit);

class FeatureFusionBlockImpl : public torch::nn::Module {
public:
    FeatureFusionBlockImpl(
        int features,
        const torch::nn::Functional& activation,
        bool deconv = false,
        bool bn = false,
        bool expand = false,
        bool align_corners = true,
        bool has_residual = true,
        int groups = 1
    );
    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& y = torch::Tensor());

private:
    bool deconv;
    bool bn;
    bool expand;
    bool align_corners;
    bool has_residual;
    int groups;
    torch::nn::Functional activation;
    ResidualConvUnit rcu;
    torch::nn::Conv2d project;
};

TORCH_MODULE(FeatureFusionBlock);

} // namespace heads
} // namespace vggt

#endif // VGGT_HEADS_FEATURE_FUSION_BLOCK_H