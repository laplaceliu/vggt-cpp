#pragma once

#include <torch/torch.h>

namespace vggt {
namespace layers {
class LayerScaleImpl : public torch::nn::Module {
public:
    LayerScaleImpl(int64_t dim, torch::Tensor init_values, bool inplace = false);
    LayerScaleImpl(int64_t dim, double init_value = 1e-5, bool inplace = false);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::Tensor gamma;
    bool inplace;
};

TORCH_MODULE(LayerScale);
}
} // namespace vggt
