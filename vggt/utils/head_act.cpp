/**
 * @file head_act.cpp
 * @brief Implementation of activation functions for VGGT heads
 */

#include "head_act.h"
#include <stdexcept>
#include <cmath>

namespace vggt {
namespace utils {

torch::Tensor inverse_log_transform(const torch::Tensor& y) {
    TORCH_CHECK(y.defined(), "Input tensor must be defined");
    return y.sign() * (torch::exp(torch::abs(y)) - 1.0);
}

torch::Tensor base_pose_act(
    const torch::Tensor& pose_enc,
    const std::string& act_type) {
    TORCH_CHECK(pose_enc.defined(), "Input tensor must be defined");

    if (act_type == "linear") {
        return pose_enc.clone();
    } else if (act_type == "inv_log") {
        return inverse_log_transform(pose_enc);
    } else if (act_type == "exp") {
        return torch::exp(pose_enc);
    } else if (act_type == "relu") {
        return torch::relu(pose_enc);
    } else {
        throw std::runtime_error("Unknown activation type: " + act_type);
    }
}

torch::Tensor activate_pose(
    const torch::Tensor& pred_pose_enc,
    const std::string& trans_act,
    const std::string& quat_act,
    const std::string& fl_act) {
    TORCH_CHECK(pred_pose_enc.defined(), "Input tensor must be defined");
    TORCH_CHECK(pred_pose_enc.dim() == 2, "Input tensor must be 2D");

    // Split into translation, quaternion and focal length parts
    auto trans = pred_pose_enc.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
    auto quat = pred_pose_enc.index({torch::indexing::Slice(), torch::indexing::Slice(3, 7)});
    auto fl = pred_pose_enc.index({torch::indexing::Slice(), torch::indexing::Slice(7, 8)});

    // Apply activations
    trans = base_pose_act(trans, trans_act);
    quat = base_pose_act(quat, quat_act);
    fl = base_pose_act(fl, fl_act);

    // Normalize quaternion
    quat = quat / torch::norm(quat, 2, -1, true);

    // Concatenate results
    return torch::cat({trans, quat, fl}, -1);
}

std::tuple<torch::Tensor, torch::Tensor> activate_head(
    const torch::Tensor& out,
    const std::string& activation,
    const std::string& conf_activation) {
    TORCH_CHECK(out.defined(), "Input tensor must be defined");
    TORCH_CHECK(out.dim() == 4, "Input tensor must be 4D (B, C, H, W)");

    // Split into 3D points and confidence
    auto points = out.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
    auto conf = out.index({torch::indexing::Slice(), torch::indexing::Slice(3, 4)});

    // Apply activation to points
    if (activation == "norm_exp") {
        points = torch::exp(points);
        auto norm = torch::norm(points, 2, 1, true);
        points = points / (norm + 1e-6);
    } else if (activation == "linear") {
        // No activation
    } else {
        throw std::runtime_error("Unknown activation type for points: " + activation);
    }

    // Apply activation to confidence
    if (conf_activation == "expp1") {
        conf = torch::exp(conf) + 1.0;
    } else if (conf_activation == "sigmoid") {
        conf = torch::sigmoid(conf);
    } else if (conf_activation == "linear") {
        // No activation
    } else {
        throw std::runtime_error("Unknown activation type for confidence: " + conf_activation);
    }

    return {points, conf};
}

} // namespace utils
} // namespace vggt
