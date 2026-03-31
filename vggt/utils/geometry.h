#pragma once

/**
 * @file geometry.h
 * @brief Geometry utilities for VGGT camera pose estimation
 *
 * Provides functions for converting between depth maps, camera coordinates,
 * and world coordinates, as well as SE(3) matrix operations.
 */

#include <torch/torch.h>

namespace vggt {
namespace utils {

/**
 * @brief Unproject a depth map to a 3D point map in world coordinates
 *
 * Processes each frame in the depth map independently and transforms
 * the camera-space points to world coordinates using the provided
 * extrinsic and intrinsic parameters.
 *
 * @param depth_map Depth map tensor of shape [S, H, W] or [S, H, W, 1]
 * @param extrinsics_cam Camera extrinsics of shape [S, 3, 4]
 * @param intrinsics_cam Camera intrinsics of shape [S, 3, 3]
 * @return World coordinates tensor of shape [S, H, W, 3]
 */
torch::Tensor unproject_depth_map_to_point_map(
    const torch::Tensor& depth_map,
    const torch::Tensor& extrinsics_cam,
    const torch::Tensor& intrinsics_cam);

/**
 * @brief Convert depth map to 3D world coordinates
 *
 * Transforms depth values and pixel coordinates into 3D points
 * in the world coordinate frame using camera extrinsics and intrinsics.
 *
 * @param depth_map 2D depth map of shape [H, W]
 * @param extrinsic Camera extrinsic matrix [3, 4]
 * @param intrinsic Camera intrinsic matrix [3, 3]
 * @param eps Small value to avoid division by zero (default: 1e-8)
 * @return Tuple of (world_coords, cam_coords, mask) where:
 *         - world_coords: [H, W, 3] world coordinates
 *         - cam_coords: [H, W, 3] camera coordinates
 *         - mask: [H, W] boolean mask of valid depth points
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> depth_to_world_coords_points(
    const torch::Tensor& depth_map,
    const torch::Tensor& extrinsic,
    const torch::Tensor& intrinsic,
    double eps = 1e-8);

/**
 * @brief Convert depth map to camera-space 3D coordinates
 *
 * Uses the pinhole camera model to unproject 2D depth values
 * into 3D camera coordinates.
 *
 * @param depth_map 2D depth map of shape [H, W]
 * @param intrinsic Camera intrinsic matrix [3, 3]
 * @return Camera coordinates tensor of shape [H, W, 3]
 * @note The intrinsic matrix must have zero skew
 */
torch::Tensor depth_to_cam_coords_points(
    const torch::Tensor& depth_map,
    const torch::Tensor& intrinsic);

/**
 * @brief Compute the closed-form inverse of an SE(3) transformation matrix
 *
 * For a transformation matrix [R|t] where R is rotation and t is translation,
 * computes the inverse [R^T|-R^T * t].
 *
 * @param se3 SE(3) transformation matrix of shape [N, 4, 4] or [N, 3, 4]
 * @param R Optional pre-extracted rotation matrix (if not provided, extracted from se3)
 * @param T Optional pre-extracted translation vector (if not provided, extracted from se3)
 * @return Inverse SE(3) matrix of shape [N, 4, 4]
 */
torch::Tensor closed_form_inverse_se3(
    const torch::Tensor& se3,
    const torch::Tensor& R = torch::Tensor(),
    const torch::Tensor& T = torch::Tensor());

} // namespace utils
} // namespace vggt