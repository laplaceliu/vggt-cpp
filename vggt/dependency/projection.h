// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <Eigen/Dense>
#include <vector>

namespace vggt {

/**
 * @brief Apply camera intrinsics (and optional radial distortion) to camera-space points.
 *
 * @param intrinsics Bx3x3 camera matrix K
 * @param points_cam Bx3xN homogeneous camera coordinates (x, y, z)ᵀ
 * @param extra_params BxN or Bxk distortion parameters (k = 1,2,4) or empty vector if no distortion
 * @param default value used for NaN replacement
 * @return std::vector<Eigen::MatrixXd> BxNx2 pixel coordinates
 */
std::vector<Eigen::MatrixXd> imgFromCam(
    const std::vector<Eigen::Matrix3d>& intrinsics,
    const std::vector<Eigen::MatrixXd>& points_cam,
    const std::vector<Eigen::VectorXd>& extra_params = {},
    double default_val = 0.0);

/**
 * @brief Project 3D world points to 2D image coordinates.
 *
 * @param points3D Nx3 world-space points
 * @param extrinsics Bx3x4 [R|t] matrix for each of B cameras
 * @param intrinsics Bx3x3 K matrix (optional if only_points_cam=true)
 * @param extra_params Bxk or BxN distortion parameters (k ∈ {1,2,4}) or empty vector
 * @param default_val value used to replace NaNs
 * @param only_points_cam if true, skip the projection and return points2D as empty
 * @return std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>>
 *         (points2D, points_cam) where points2D is BxNx2 or empty, points_cam is Bx3xN
 */
std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> project3DPoints(
    const Eigen::MatrixXd& points3D,
    const std::vector<Eigen::Matrix<double, 3, 4>>& extrinsics,
    const std::vector<Eigen::Matrix3d>& intrinsics = {},
    const std::vector<Eigen::VectorXd>& extra_params = {},
    double default_val = 0.0,
    bool only_points_cam = false);

} // namespace vggt
