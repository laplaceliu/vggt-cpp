// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "projection.h"
#include <Eigen/Dense>
#include <vector>
#include <stdexcept>

namespace vggt {

namespace { // helper functions

Eigen::MatrixXd applyDistortion(
    const Eigen::VectorXd& params,
    const Eigen::VectorXd& u,
    const Eigen::VectorXd& v) {
    // Convert vectors to matrices for batch processing
    Eigen::MatrixXd params_mat(1, params.size());
    params_mat.row(0) = params;

    Eigen::MatrixXd u_mat(1, u.size());
    u_mat.row(0) = u;

    Eigen::MatrixXd v_mat(1, v.size());
    v_mat.row(0) = v;

    // Call the distortion function
    auto [u_dist, v_dist] = vggt::applyDistortion(params_mat, u_mat, v_mat);

    // Combine results into single matrix
    Eigen::MatrixXd result(2, u.size());
    result << u_dist.row(0), v_dist.row(0);
    return result;
}

} // namespace

std::vector<Eigen::MatrixXd> imgFromCam(
    const std::vector<Eigen::Matrix3d>& intrinsics,
    const std::vector<Eigen::MatrixXd>& points_cam,
    const std::vector<Eigen::VectorXd>& extra_params,
    double default_val) {

    if (intrinsics.size() != points_cam.size()) {
        throw std::invalid_argument("intrinsics and points_cam must have same batch size");
    }

    if (!extra_params.empty() && extra_params.size() != intrinsics.size()) {
        throw std::invalid_argument("extra_params must be empty or match batch size");
    }

    const size_t B = intrinsics.size();
    std::vector<Eigen::MatrixXd> points2D(B);

    for (size_t b = 0; b < B; ++b) {
        const auto& K = intrinsics[b];
        const auto& pc = points_cam[b];
        const size_t N = pc.cols();

        // 1. Perspective divide
        Eigen::MatrixXd points_cam_norm = pc.array().rowwise() / pc.row(2).array();
        Eigen::MatrixXd uv = points_cam_norm.topRows(2);

        // 2. Apply distortion if needed
        if (!extra_params.empty()) {
            const auto& params = extra_params[b];
            Eigen::MatrixXd distorted_uv = applyDistortion(params, uv.row(0), uv.row(1));
            uv = distorted_uv;
        }

        // 3. Homogeneous coords then K multiplication
        Eigen::MatrixXd points_cam_h(3, N);
        points_cam_h << uv, Eigen::RowVectorXd::Ones(N);

        Eigen::MatrixXd points2D_h = K * points_cam_h;
        Eigen::MatrixXd result = points2D_h.topRows(2);

        // Replace NaNs with default value
        for (int i = 0; i < result.rows(); ++i) {
            for (int j = 0; j < result.cols(); ++j) {
                if (std::isnan(result(i,j))) {
                    result(i,j) = default_val;
                }
            }
        }

        points2D[b] = result.transpose(); // Convert to BxNx2 format
    }

    return points2D;
}

std::pair<std::vector<Eigen::MatrixXd>, std::vector<Eigen::MatrixXd>> project3DPoints(
    const Eigen::MatrixXd& points3D,
    const std::vector<Eigen::Matrix<double, 3, 4>>& extrinsics,
    const std::vector<Eigen::Matrix3d>& intrinsics,
    const std::vector<Eigen::VectorXd>& extra_params,
    double default_val,
    bool only_points_cam) {

    const size_t B = extrinsics.size();
    const size_t N = points3D.rows();

    if (!intrinsics.empty() && intrinsics.size() != B) {
        throw std::invalid_argument("intrinsics must be empty or match extrinsics batch size");
    }

    if (!extra_params.empty() && extra_params.size() != B) {
        throw std::invalid_argument("extra_params must be empty or match batch size");
    }

    // 1. Convert to homogeneous coordinates
    Eigen::MatrixXd points3D_h(N, 4);
    points3D_h << points3D, Eigen::VectorXd::Ones(N);

    // 2. Apply extrinsics (world to camera)
    std::vector<Eigen::MatrixXd> points_cam(B);
    for (size_t b = 0; b < B; ++b) {
        points_cam[b] = extrinsics[b] * points3D_h.transpose(); // 3xN
    }

    if (only_points_cam) {
        return {{}, points_cam};
    }

    if (intrinsics.empty()) {
        throw std::invalid_argument("intrinsics must be provided unless only_points_cam=true");
    }

    // 3. Apply intrinsics and distortion
    auto points2D = imgFromCam(intrinsics, points_cam, extra_params, default_val);

    return {points2D, points_cam};
}

} // namespace vggt
