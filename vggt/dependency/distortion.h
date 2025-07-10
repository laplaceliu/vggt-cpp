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
 * @brief Apply radial or OpenCV distortion to normalized 2D points
 *
 * @param params Distortion parameters (Bx1, Bx2 or Bx4)
 * @param u Normalized x coordinates (BxN)
 * @param v Normalized y coordinates (BxN)
 * @return std::pair<Eigen::MatrixXd, Eigen::MatrixXd> Distorted (u, v) coordinates
 */
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> applyDistortion(
    const Eigen::MatrixXd& params,
    const Eigen::MatrixXd& u,
    const Eigen::MatrixXd& v);

} // namespace vggt
