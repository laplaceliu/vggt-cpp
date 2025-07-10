/**
 * @file geometry.h
 * @brief Geometry utility functions for VGGT
 */

#pragma once

#include <torch/torch.h>

namespace vggt {
namespace utils {

/**
 * @brief Unproject depth map to point map
 *
 * @param depth_map Depth map tensor of shape (B, 1, H, W)
 * @param intrinsics Camera intrinsics tensor of shape (B, 3, 3)
 * @param extrinsics Camera extrinsics tensor of shape (B, 4, 4)
 * @return torch::Tensor Point map tensor of shape (B, 3, H, W)
 */
torch::Tensor unproject_depth_map_to_point_map(
    const torch::Tensor& depth_map,
    const torch::Tensor& intrinsics,
    const torch::Tensor& extrinsics);

/**
 * @brief Convert depth map to world coordinates
 *
 * @param depth_map Depth map tensor of shape (B, 1, H, W)
 * @param intrinsics Camera intrinsics tensor of shape (B, 3, 3)
 * @param extrinsics Camera extrinsics tensor of shape (B, 4, 4)
 * @return torch::Tensor World coordinates tensor of shape (B, H*W, 3)
 */
torch::Tensor depth_to_world_coords_points(
    const torch::Tensor& depth_map,
    const torch::Tensor& intrinsics,
    const torch::Tensor& extrinsics);

/**
 * @brief Convert depth map to camera coordinates
 *
 * @param depth_map Depth map tensor of shape (B, 1, H, W)
 * @param intrinsics Camera intrinsics tensor of shape (B, 3, 3)
 * @return torch::Tensor Camera coordinates tensor of shape (B, H*W, 3)
 */
torch::Tensor depth_to_cam_coords_points(
    const torch::Tensor& depth_map,
    const torch::Tensor& intrinsics);

/**
 * @brief Compute the inverse of an SE3 matrix
 *
 * @param se3_matrix SE3 matrix tensor of shape (B, 4, 4)
 * @return torch::Tensor Inverse SE3 matrix tensor of shape (B, 4, 4)
 */
torch::Tensor closed_form_inverse_se3(const torch::Tensor& se3_matrix);

} // namespace utils
} // namespace vggt
