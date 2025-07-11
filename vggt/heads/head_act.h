/**
 * @file head_act.h
 * @brief Activation functions for head modules
 */

#pragma once

#include <torch/torch.h>
#include <string>

namespace vggt {

/**
 * @brief Activate pose parameters with specified activation functions
 *
 * @param pred_pose_enc Tensor containing encoded pose parameters [translation, quaternion, focal length]
 * @param trans_act Activation type for translation component
 * @param quat_act Activation type for quaternion component
 * @param fl_act Activation type for focal length component
 * @return torch::Tensor Activated pose parameters tensor
 */
torch::Tensor activate_pose(
    torch::Tensor pred_pose_enc,
    const std::string& trans_act = "linear",
    const std::string& quat_act = "linear",
    const std::string& fl_act = "linear"
);

/**
 * @brief Apply basic activation function to pose parameters
 *
 * @param pose_enc Tensor containing encoded pose parameters
 * @param act_type Activation type ("linear", "inv_log", "exp", "relu")
 * @return torch::Tensor Activated pose parameters
 */
torch::Tensor base_pose_act(
    torch::Tensor pose_enc,
    const std::string& act_type = "linear"
);

/**
 * @brief Process network output to extract 3D points and confidence values
 *
 * @param out Network output tensor (B, C, H, W)
 * @param activation Activation type for 3D points
 * @param conf_activation Activation type for confidence values
 * @return std::tuple<torch::Tensor, torch::Tensor> Tuple of (3D points tensor, confidence tensor)
 */
std::tuple<torch::Tensor, torch::Tensor> activate_head(
    torch::Tensor out,
    const std::string& activation = "norm_exp",
    const std::string& conf_activation = "expp1"
);

/**
 * @brief Apply inverse log transform: sign(y) * (exp(|y|) - 1)
 *
 * @param y Input tensor
 * @return torch::Tensor Transformed tensor
 */
torch::Tensor inverse_log_transform(torch::Tensor y);

} // namespace vggt
