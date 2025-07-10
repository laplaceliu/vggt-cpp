/**
 * @file camera_head.cpp
 * @brief Implementation of camera head module for VGGT
 */

#include "camera_head.h"
#include <torch/nn/init.h>
#include <stdexcept>

namespace vggt {
namespace heads {

CameraHeadImpl::CameraHeadImpl(
    int64_t dim_in,
    int64_t trunk_depth,
    const std::string& pose_encoding_type,
    int64_t num_heads,
    int64_t mlp_ratio,
    float init_values,
    const std::string& trans_act,
    const std::string& quat_act,
    const std::string& fl_act) {
    // Initialize parameters
    target_dim_ = 8; // translation(3) + quaternion(4) + focal_length(1)
    trans_act_ = trans_act;
    quat_act_ = quat_act;
    fl_act_ = fl_act;
    trunk_depth_ = trunk_depth;

    // Initialize empty pose tokens
    empty_pose_tokens_ = torch::zeros({1, 1, target_dim_});

    // Initialize submodules
    token_norm_ = register_module("token_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim_in})));
    trunk_norm_ = register_module("trunk_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({target_dim_})));

    // Pose embedding layer
    embed_pose_ = register_module("embed_pose", torch::nn::Linear(dim_in, target_dim_));

    // Pose LN modulation layers
    poseLN_modulation_ = register_module("poseLN_modulation",
        torch::nn::Sequential(
            torch::nn::Linear(dim_in, dim_in),
            torch::nn::SiLU(),
            torch::nn::Linear(dim_in, target_dim_ * 2)
        ));

    // AdaLN norm
    adaln_norm_ = register_module("adaln_norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({target_dim_})));

    // Pose branch
    pose_branch_ = register_module("pose_branch",
        torch::nn::Sequential(
            torch::nn::Linear(target_dim_, target_dim_ * 4),
            torch::nn::SiLU(),
            torch::nn::Linear(target_dim_ * 4, target_dim_)
        ));

    // Initialize weights
    for (auto& module : modules()) {
        if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
            torch::nn::init::xavier_uniform_(M->weight);
            torch::nn::init::zeros_(M->bias);
        }
    }
}

std::vector<torch::Tensor> CameraHeadImpl::forward(
    const std::vector<torch::Tensor>& aggregated_tokens_list,
    int64_t num_iterations) {
    // Check input
    if (aggregated_tokens_list.empty()) {
        throw std::runtime_error("aggregated_tokens_list cannot be empty");
    }

    // Process each token in the list
    std::vector<torch::Tensor> results;
    for (const auto& tokens : aggregated_tokens_list) {
        // Normalize tokens
        auto tokens_norm = token_norm_(tokens);

        // Get pose tokens
        auto pose_tokens = embed_pose_(tokens_norm);

        // Get modulation parameters
        auto modulation_params = poseLN_modulation_(tokens_norm);
        auto scale = modulation_params.index({torch::indexing::Slice(), torch::indexing::Slice(0, target_dim_)});
        auto shift = modulation_params.index({torch::indexing::Slice(), torch::indexing::Slice(target_dim_, target_dim_ * 2)});

        // Apply modulation
        pose_tokens = pose_tokens * (1.0 + scale) + shift;

        // Process through trunk
        auto trunk_results = trunk_fn(pose_tokens, num_iterations);
        results.insert(results.end(), trunk_results.begin(), trunk_results.end());
    }

    return results;
}

std::vector<torch::Tensor> CameraHeadImpl::trunk_fn(
    const torch::Tensor& pose_tokens,
    int64_t num_iterations) {
    // Check input
    TORCH_CHECK(pose_tokens.defined(), "pose_tokens must be defined");

    std::vector<torch::Tensor> results;
    auto current_pose = pose_tokens.clone();

    for (int64_t i = 0; i < num_iterations; ++i) {
        // Normalize
        current_pose = trunk_norm_(current_pose);

        // Process through pose branch
        auto delta_pose = pose_branch_(current_pose);

        // Update pose
        current_pose = current_pose + delta_pose;

        // Store result
        results.push_back(current_pose.clone());
    }

    return results;
}

} // namespace heads
} // namespace vggt
