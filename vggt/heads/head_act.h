#pragma once

/**
 * @file head_act.h
 * @brief Activation functions for VGGT prediction heads
 *
 * Provides activation functions for camera pose encoding, depth prediction,
 * and confidence estimation in the VGGT model.
 */

#include <torch/torch.h>

namespace vggt {
namespace heads {

/**
 * @brief Apply activation to pose encoding with separate transforms for T, quat, and fl
 *
 * Splits the pose encoding into translation (T), quaternion (quat), and focal length (fl),
 * then applies separate activation functions to each component.
 *
 * @param pred_pose_enc Pose encoding tensor of shape [..., 8] where 8 = T(3) + quat(4) + fl(1)
 * @param trans_act Activation type for translation (default: "linear")
 * @param quat_act Activation type for quaternion (default: "linear")
 * @param fl_act Activation type for focal length (default: "linear")
 * @return Activated pose encoding with same shape as input
 */
torch::Tensor activate_pose(
    torch::Tensor pred_pose_enc,
    const std::string& trans_act = "linear",
    const std::string& quat_act = "linear",
    const std::string& fl_act = "linear");

/**
 * @brief Apply activation function to pose encoding
 *
 * Supported activation types:
 * - "linear": No activation (identity)
 * - "inv_log": Inverse log transform using sign(y) * (exp(|y|) - 1)
 * - "exp": Exponential activation exp(y)
 * - "relu": Rectified Linear Unit max(0, y)
 *
 * @param pose_enc Input pose encoding tensor
 * @param act_type Activation type string (default: "linear")
 * @return Activated tensor with same shape as input
 * @throws std::runtime_error if act_type is unknown
 */
torch::Tensor base_pose_act(
    torch::Tensor pose_enc,
    const std::string& act_type = "linear");

/**
 * @brief Apply activation to dense prediction head outputs
 *
 * Handles both single-channel (e.g., depth) and multi-channel (e.g., 3D points) outputs.
 * For multi-channel outputs, the last channel is treated as confidence.
 *
 * Supported activations for prediction:
 * - "norm_exp": Normalized exponential: xyz/norm * expm1(norm)
 * - "norm": Normalized: xyz/norm
 * - "exp": Exponential
 * - "relu": ReLU
 * - "inv_log": Inverse log transform
 * - "xy_inv_log": Separate xy and z with inv_log on z
 * - "sigmoid": Sigmoid activation
 * - "linear": No activation
 *
 * Supported confidence activations:
 * - "expp1": exp(x) + 1 (ensures positive)
 * - "expp0": exp(x)
 * - "sigmoid": Sigmoid activation
 *
 * @param out Output tensor from prediction head [B, C, H, W]
 * @param activation Activation type for predictions (default: "norm_exp")
 * @param conf_activation Activation type for confidence (default: "expp1")
 * @return Tuple of (activated predictions, confidence scores)
 */
std::tuple<torch::Tensor, torch::Tensor> activate_head(
    torch::Tensor out,
    const std::string& activation = "norm_exp",
    const std::string& conf_activation = "expp1");

/**
 * @brief Inverse log transform for depth/disparity
 *
 * Computes: sign(y) * (exp(|y|) - 1)
 * This is the inverse of log1p = log(1 + x)
 *
 * @param y Input tensor (typically log-depth or log-disparity)
 * @return Tensor with inverse log transform applied
 */
torch::Tensor inverse_log_transform(torch::Tensor y);

} // namespace heads
} // namespace vggt