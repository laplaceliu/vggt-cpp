/**
 * @file pose_enc.h
 * @brief Camera pose encoding utility functions for VGGT
 */

#pragma once

#include <torch/torch.h>
#include <tuple>

namespace vggt {
namespace utils {

/**
 * @brief Convert camera extrinsics and intrinsics to compact pose encoding
 *
 * @param extrinsics Camera extrinsics tensor of shape (B, 4, 4)
 * @param intrinsics Camera intrinsics tensor of shape (B, 3, 3)
 * @return torch::Tensor Pose encoding tensor of shape (B, 16)
 */
torch::Tensor extri_intri_to_pose_encoding(
    const torch::Tensor& extrinsics,
    const torch::Tensor& intrinsics);

/**
 * @brief Convert pose encoding back to camera extrinsics and intrinsics
 *
 * @param pose_encoding Pose encoding tensor of shape (B, 16)
 * @return std::tuple<torch::Tensor, torch::Tensor> Tuple of (extrinsics, intrinsics)
 *         where extrinsics is of shape (B, 4, 4) and intrinsics is of shape (B, 3, 3)
 */
std::tuple<torch::Tensor, torch::Tensor> pose_encoding_to_extri_intri(
    const torch::Tensor& pose_encoding);

} // namespace utils
} // namespace vggt
