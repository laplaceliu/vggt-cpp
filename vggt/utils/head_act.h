/**
 * @file head_act.h
 * @brief Activation functions for VGGT heads
 */

#pragma once

#include <torch/torch.h>

namespace vggt {
namespace utils {

/**
 * @brief Apply inverse log transform: sign(y) * (exp(|y|) - 1)
 *
 * @param y Input tensor
 * @return torch::Tensor Transformed tensor
 */
torch::Tensor inverse_log_transform(const torch::Tensor& y);

/**
 * @brief Apply basic activation function to pose parameters
 *
 * @param pose_enc Tensor containing encoded pose parameters
 * @param act_type Activation type ("linear", "inv_log", "exp", "relu")
 * @return torch::Tensor Activated pose parameters
 */
torch::Tensor base_pose_act(
    const torch::Tensor& pose_enc,
    const std::string& act_type = "linear");

/**
 * @brief Activate pose parameters with specified activation functions
 *
 * @param pred_pose_enc Tensor containing encoded pose parameters [translation, quaternion, focal length]
 * @param trans_act Activation type for translation component (default="linear")
 * @param quat_act Activation type for quaternion component (default="linear")
 * @param fl_act Activation type for focal length component (default="linear")
 * @return torch::Tensor Activated pose parameters tensor
 */
torch::Tensor activate_pose(
    const torch::Tensor& pred_pose_enc,
    const std::string& trans_act = "linear",
    const std::string& quat_act = "linear",
    const std::string& fl_act = "linear");

/**
 * @brief Process network output to extract 3D points and confidence values
 *
 * @param out Network output tensor (B, C, H, W)
 * @param activation Activation type for 3D points (default="norm_exp")
 * @param conf_activation Activation type for confidence values (default="expp1")
 * @return std::tuple<torch::Tensor, torch::Tensor> Tuple of (3D points tensor, confidence tensor)
 */
std::tuple<torch::Tensor, torch::Tensor> activate_head(
    const torch::Tensor& out,
    const std::string& activation = "norm_exp",
    const std::string& conf_activation = "expp1");

} // namespace utils
} // namespace vggt
