

#include "camera_head.h"

namespace vggt {
namespace heads {

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
) {
    if (pose_encoding_type == "absT_quaR_FoV") {
        target_dim = 9;
    } else {
        throw std::runtime_error("Unsupported camera encoding type: " + pose_encoding_type);
    }

    trans_act = trans_act;
    quat_act = quat_act;
    fl_act = fl_act;
    trunk_depth = trunk_depth;

    // Build the trunk using a sequence of transformer blocks.
    for (int64_t i = 0; i < trunk_depth; ++i) {
        trunk->push_back(Block(dim_in, num_heads, mlp_ratio, init_values));
    }

    // Normalizations for camera token and trunk output.
    token_norm = register_module("token_norm", torch::nn::LayerNorm(dim_in));
    trunk_norm = register_module("trunk_norm", torch::nn::LayerNorm(dim_in));

    // Learnable empty camera pose token.
    empty_pose_tokens = register_parameter("empty_pose_tokens", torch::zeros({1, 1, target_dim}));
    embed_pose = register_module("embed_pose", torch::nn::Linear(target_dim, dim_in));

    // Module for producing modulation parameters: shift, scale, and a gate.
    poseLN_modulation = register_module("poseLN_modulation", torch::nn::Sequential(
        torch::nn::SiLU(),
        torch::nn::Linear(dim_in, 3 * dim_in, true)
    ));

    // Adaptive layer normalization without affine parameters.
    adaln_norm = register_module("adaln_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim_in}).elementwise_affine(false).eps(1e-6)));
    pose_branch = register_module("pose_branch", Mlp(dim_in, dim_in / 2, target_dim, 0));
}

std::vector<torch::Tensor> CameraHeadImpl::forward(const std::vector<torch::Tensor>& aggregated_tokens_list, int64_t num_iterations) {
    // Use tokens from the last block for camera prediction.
    auto tokens = aggregated_tokens_list.back();

    // Extract the camera tokens
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
        // Use a learned empty pose for the first iteration.
        torch::Tensor module_input;
        if (i == 0) {
            module_input = embed_pose(empty_pose_tokens.expand({B, S, -1}));
        } else {
            // Detach the previous prediction to avoid backprop through time.
            pred_pose_enc = pred_pose_enc.detach();
            module_input = embed_pose(pred_pose_enc);
        }

        // Generate modulation parameters and split them into shift, scale, and gate components.
        auto modulation = poseLN_modulation(module_input);
        auto shift_msa = modulation.slice(-1, 0, C);
        auto scale_msa = modulation.slice(-1, C, 2 * C);
        auto gate_msa = modulation.slice(-1, 2 * C, 3 * C);

        // Adaptive layer normalization and modulation.
        auto pose_tokens_modulated = gate_msa * modulate(adaln_norm(pose_tokens), shift_msa, scale_msa);
        pose_tokens_modulated = pose_tokens_modulated + pose_tokens;

        pose_tokens_modulated = trunk(pose_tokens_modulated);
        // Compute the delta update for the pose encoding.
        auto pred_pose_enc_delta = pose_branch(trunk_norm(pose_tokens_modulated));

        if (i == 0) {
            pred_pose_enc = pred_pose_enc_delta;
        } else {
            pred_pose_enc = pred_pose_enc + pred_pose_enc_delta;
        }

        // Apply final activation functions for translation, quaternion, and field-of-view.
        auto activated_pose = activate_pose(pred_pose_enc, trans_act, quat_act, fl_act);
        pred_pose_enc_list.push_back(activated_pose);
    }

    return pred_pose_enc_list;
}

} // namespace heads
} // namespace vggt
