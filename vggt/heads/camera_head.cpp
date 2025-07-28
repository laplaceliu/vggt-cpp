#include "camera_head.h"
#include "layers/mlp.h"
#include "layers/block.h"
#include "heads/head_act.h"

namespace vggt {
namespace heads {

torch::Tensor modulate(const torch::Tensor& x, const torch::Tensor& shift, const torch::Tensor& scale);

CameraHeadImpl::CameraHeadImpl(
    int64_t dim_in,
    int64_t trunk_depth,
    const std::string& pose_encoding_type,
    int64_t num_heads,
    int64_t mlp_ratio,
    double init_values,
    const std::string& trans_act,
    const std::string& quat_act,
    const std::string& fl_act
) : target_dim(9),
    trans_act(trans_act),
    quat_act(quat_act),
    fl_act(fl_act),
    trunk_depth(trunk_depth),
    token_norm(torch::nn::LayerNormOptions({dim_in})),
    trunk_norm(torch::nn::LayerNormOptions({dim_in})),
    embed_pose(torch::nn::LinearOptions(target_dim, dim_in)),
    adaln_norm(torch::nn::LayerNormOptions({dim_in}).elementwise_affine(false).eps(1e-6)) {

    if (pose_encoding_type != "absT_quaR_FoV") {
        throw std::runtime_error("Unsupported camera encoding type: " + pose_encoding_type);
    }

    // Build the trunk
    utils::StackSequential trunk;
    for (int64_t i = 0; i < trunk_depth; ++i) {
        trunk->push_back(layers::Block(dim_in, num_heads, mlp_ratio, init_values));
    }
    register_module("trunk", trunk);

    // Learnable empty camera pose token
    empty_pose_tokens = register_parameter("empty_pose_tokens", torch::zeros({1, 1, target_dim}));

    // PoseLN modulation
    poseLN_modulation = utils::StackSequential(
        torch::nn::SiLU(),
        torch::nn::Linear(torch::nn::LinearOptions(dim_in, 3 * dim_in).bias(true))
    );
    register_module("poseLN_modulation", poseLN_modulation);

    // Pose branch
    pose_branch = torch::nn::AnyModule(layers::Mlp(dim_in, dim_in / 2, target_dim));
    register_module("pose_branch", pose_branch.ptr());
}

std::vector<torch::Tensor> CameraHeadImpl::forward(const std::vector<torch::Tensor>& aggregated_tokens_list, int64_t num_iterations) {
    auto tokens = aggregated_tokens_list.back();
    auto pose_tokens = tokens.index({torch::indexing::Slice(), torch::indexing::Slice(), 0});
    pose_tokens = token_norm(pose_tokens);
    return trunk_fn(pose_tokens, num_iterations);
}

std::vector<torch::Tensor> CameraHeadImpl::trunk_fn(const torch::Tensor& pose_tokens, int64_t num_iterations) {
    auto B = pose_tokens.size(0);
    auto S = pose_tokens.size(1);
    auto C = pose_tokens.size(2);

    torch::Tensor pred_pose_enc;
    std::vector<torch::Tensor> pred_pose_enc_list;

    for (int64_t i = 0; i < num_iterations; ++i) {
        torch::Tensor module_input;
        if (!pred_pose_enc.defined()) {
            module_input = embed_pose(empty_pose_tokens.expand({B, S, -1}));
        } else {
            module_input = embed_pose(pred_pose_enc.detach());
        }

        torch::Tensor modulation = poseLN_modulation(module_input);
        auto shift_msa = modulation.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, C)});
        auto scale_msa = modulation.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(C, 2 * C)});
        auto gate_msa = modulation.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(2 * C, 3 * C)});

        auto pose_tokens_modulated = gate_msa * modulate(adaln_norm(pose_tokens), shift_msa, scale_msa);
        pose_tokens_modulated = pose_tokens_modulated + pose_tokens;

        pose_tokens_modulated = trunk(pose_tokens_modulated);
        auto pred_pose_enc_delta = pose_branch.forward(trunk_norm(pose_tokens_modulated));

        if (!pred_pose_enc.defined()) {
            pred_pose_enc = pred_pose_enc_delta;
        } else {
            pred_pose_enc = pred_pose_enc + pred_pose_enc_delta;
        }

        auto activated_pose = heads::activate_pose(pred_pose_enc, trans_act, quat_act, fl_act);
        pred_pose_enc_list.push_back(activated_pose);
    }

    return pred_pose_enc_list;
}

torch::Tensor modulate(const torch::Tensor& x, const torch::Tensor& shift, const torch::Tensor& scale) {
    return x * (1 + scale) + shift;
}

} // namespace heads
} // namespace vggt
