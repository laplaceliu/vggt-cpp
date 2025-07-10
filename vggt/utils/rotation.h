/**
 * @file rotation.h
 * @brief Rotation utility functions for VGGT
 */

#pragma once

#include <torch/torch.h>

namespace vggt {
namespace utils {

/**
 * @brief Convert quaternion to rotation matrix
 *
 * @param quaternion Quaternion tensor of shape (..., 4)
 * @return torch::Tensor Rotation matrix tensor of shape (..., 3, 3)
 */
torch::Tensor quat_to_mat(const torch::Tensor& quaternion);

/**
 * @brief Convert rotation matrix to quaternion
 *
 * @param rotation_matrix Rotation matrix tensor of shape (..., 3, 3)
 * @return torch::Tensor Quaternion tensor of shape (..., 4)
 */
torch::Tensor mat_to_quat(const torch::Tensor& rotation_matrix);

/**
 * @brief Standardize quaternion to have non-negative real part
 *
 * @param quaternion Quaternion tensor of shape (..., 4)
 * @return torch::Tensor Standardized quaternion tensor of shape (..., 4)
 */
torch::Tensor standardize_quaternion(const torch::Tensor& quaternion);

/**
 * @brief Helper function to compute the square root of the positive part
 *
 * @param x Input tensor
 * @return torch::Tensor Square root of max(0, x)
 */
torch::Tensor _sqrt_positive_part(const torch::Tensor& x);

} // namespace utils
} // namespace vggt
