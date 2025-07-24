#pragma once

#include <torch/torch.h>
#include <einops.hpp>
#include <variant>
#include "blocks.h"
#include "utils.h"

namespace vggt {
namespace dependency {
namespace track_modules {

class BaseTrackerPredictorImpl : public torch::nn::Module {
public:
    BaseTrackerPredictorImpl(
        int64_t stride = 4,
        int64_t corr_levels = 5,
        int64_t corr_radius = 4,
        int64_t latent_dim = 128,
        int64_t hidden_size = 384,
        bool use_spaceatt = true,
        int64_t depth = 6,
        bool fine = false
    );

    std::variant<
        std::tuple<std::vector<torch::Tensor>, torch::Tensor>,
        std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor>
    > forward(
        torch::Tensor query_points,
        torch::Tensor fmaps,
        int64_t iters = 4,
        bool return_feat = false,
        int64_t down_ratio = 1
    );

    std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor, torch::Tensor> forward_with_feat(
        torch::Tensor query_points,
        torch::Tensor fmaps,
        int64_t iters = 4,
        int64_t down_ratio = 1
    );

private:
    int64_t stride;
    int64_t latent_dim;
    int64_t corr_levels;
    int64_t corr_radius;
    int64_t hidden_size;
    bool fine;
    int64_t flows_emb_dim;
    int64_t transformer_dim;

    EfficientUpdateFormer updateformer{nullptr};
    torch::nn::GroupNorm norm{nullptr};
    torch::nn::Sequential ffeat_updater{nullptr};
    torch::nn::Sequential vis_predictor{nullptr};
};
TORCH_MODULE(BaseTrackerPredictor);

} // namespace track_modules
} // namespace dependency
} // namespace vggt