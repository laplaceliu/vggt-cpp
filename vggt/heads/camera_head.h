#pragma once

#include <torch/torch.h>
#include "utils/stack_sequential.h"

namespace vggt {
namespace heads {

class CameraHeadImpl : public torch::nn::Module {
public:
    CameraHeadImpl(
        int64_t dim_in = 2048,
        int64_t trunk_depth = 4,
        const std::string& pose_encoding_type = "absT_quaR_FoV",
        int64_t num_heads = 16,
        int64_t mlp_ratio = 4,
        double init_values = 0.01,
        const std::string& trans_act = "linear",
        const std::string& quat_act = "linear",
        const std::string& fl_act = "relu"
    );

    std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& aggregated_tokens_list, int64_t num_iterations = 4);

private:
    std::vector<torch::Tensor> trunk_fn(const torch::Tensor& pose_tokens, int64_t num_iterations);

    int64_t target_dim;
    std::string trans_act;
    std::string quat_act;
    std::string fl_act;
    int64_t trunk_depth;

    utils::StackSequential trunk;
    torch::nn::LayerNorm token_norm;
    torch::nn::LayerNorm trunk_norm;
    torch::Tensor empty_pose_tokens;
    torch::nn::Linear embed_pose;
    utils::StackSequential poseLN_modulation;
    torch::nn::LayerNorm adaln_norm;
    torch::nn::AnyModule pose_branch;
};

TORCH_MODULE(CameraHead);

} // namespace heads
} // namespace vggt
