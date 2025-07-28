#pragma once

#include <torch/torch.h>

namespace vggt {
namespace dependency {

class TrackerPredictorImpl : public torch::nn::Module {
public:
    TrackerPredictorImpl();

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
        torch::Tensor images,
        torch::Tensor query_points,
        torch::Tensor fmaps = torch::Tensor(),
        int64_t coarse_iters = 6,
        bool inference = true,
        bool fine_tracking = true,
        int64_t fine_chunk = 40960
    );

    torch::Tensor process_images_to_fmaps(torch::Tensor images);

private:
    int64_t coarse_down_ratio;
    torch::nn::AnyModule coarse_fnet;
    torch::nn::AnyModule coarse_predictor;
    torch::nn::AnyModule fine_fnet;
    torch::nn::AnyModule fine_predictor;
};

TORCH_MODULE(TrackerPredictor);

} // namespace dependency
} // namespace vggt