

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <torch/torch.h>

namespace vggt {

/**
 * @brief Apply radial or OpenCV distortion to normalized 2D points (Eigen version)
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

/**
 * @brief Apply radial or OpenCV distortion to normalized 2D points (PyTorch version)
 *
 * @param params Distortion parameters (Bx1, Bx2 or Bx4)
 * @param u Normalized x coordinates (BxN)
 * @param v Normalized y coordinates (BxN)
 * @return std::tuple<torch::Tensor, torch::Tensor> Distorted (u, v) coordinates
 */
std::tuple<torch::Tensor, torch::Tensor> apply_distortion(
    const torch::Tensor& params,
    const torch::Tensor& u,
    const torch::Tensor& v);

} // namespace vggt
