/**
 * @file head_act.cpp
 * @brief Implementation of activation functions for head modules
 */

#include "head_act.h"
#include <stdexcept>

namespace vggt {

torch::Tensor activate_pose(
    torch::Tensor pred_pose_enc,
    const std::string& trans_act,
    const std::string& quat_act,
    const std::string& fl_act
) {
    // Split the pose encoding into translation, quaternion, and focal length components
    auto T = pred_pose_enc.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, 3)});
    auto quat = pred_pose_enc.index({torch::indexing::Ellipsis, torch::indexing::Slice(3, 7)});
    auto fl = pred_pose_enc.index({torch::indexing::Ellipsis, torch::indexing::Slice(7, pred_pose_enc.size(-1))});

    // Apply activations to each component
    T = base_pose_act(T, trans_act);
    quat = base_pose_act(quat, quat_act);
    fl = base_pose_act(fl, fl_act);

    // Concatenate the activated components
    return torch::cat({T, quat, fl}, -1);
}

torch::Tensor base_pose_act(
    torch::Tensor pose_enc,
    const std::string& act_type
) {
    if (act_type == "linear") {
        return pose_enc;
    } else if (act_type == "inv_log") {
        return inverse_log_transform(pose_enc);
    } else if (act_type == "exp") {
        return torch::exp(pose_enc);
    } else if (act_type == "relu") {
        return torch::relu(pose_enc);
    } else {
        throw std::invalid_argument("Unknown act_type: " + act_type);
    }
}

std::tuple<torch::Tensor, torch::Tensor> activate_head(
    torch::Tensor out,
    const std::string& activation,
    const std::string& conf_activation
) {
    // Move channels from last dim to the 4th dimension => (B, H, W, C)
    auto fmap = out.permute({0, 2, 3, 1});  // B,H,W,C expected

    // Split into xyz (first C-1 channels) and confidence (last channel)
    auto xyz = fmap.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, -1)});
    auto conf = fmap.index({torch::indexing::Ellipsis, -1});

    torch::Tensor pts3d;

    if (activation == "norm_exp") {
        auto d = xyz.norm(-1, true).clamp(1e-8);
        auto xyz_normed = xyz / d;
        pts3d = xyz_normed * torch::expm1(d);
    } else if (activation == "norm") {
        pts3d = xyz / xyz.norm(-1, true);
    } else if (activation == "exp") {
        pts3d = torch::exp(xyz);
    } else if (activation == "relu") {
        pts3d = torch::relu(xyz);
    } else if (activation == "inv_log") {
        pts3d = inverse_log_transform(xyz);
    } else if (activation == "xy_inv_log") {
        auto xy = xyz.index({torch::indexing::Ellipsis, torch::indexing::Slice(0, 2)});
        auto z = xyz.index({torch::indexing::Ellipsis, torch::indexing::Slice(2, 3)});
        z = inverse_log_transform(z);
        pts3d = torch::cat({xy * z, z}, -1);
    } else if (activation == "sigmoid") {
        pts3d = torch::sigmoid(xyz);
    } else if (activation == "linear") {
        pts3d = xyz;
    } else {
        throw std::invalid_argument("Unknown activation: " + activation);
    }

    torch::Tensor conf_out;

    if (conf_activation == "expp1") {
        conf_out = 1 + conf.exp();
    } else if (conf_activation == "expp0") {
        conf_out = conf.exp();
    } else if (conf_activation == "sigmoid") {
        conf_out = torch::sigmoid(conf);
    } else {
        throw std::invalid_argument("Unknown conf_activation: " + conf_activation);
    }

    return std::make_tuple(pts3d, conf_out);
}

torch::Tensor inverse_log_transform(torch::Tensor y) {
    return torch::sign(y) * torch::expm1(torch::abs(y));
}

} // namespace vggt
