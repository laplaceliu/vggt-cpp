

#include "distortion.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <torch/torch.h>

namespace vggt {

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> applyDistortion(
    const Eigen::MatrixXd& params,
    const Eigen::MatrixXd& u,
    const Eigen::MatrixXd& v) {

    // Check input dimensions
    if (params.rows() != u.rows() || params.rows() != v.rows()) {
        throw std::invalid_argument("params, u and v must have same batch size");
    }
    if (u.cols() != v.cols()) {
        throw std::invalid_argument("u and v must have same number of points");
    }

    const size_t B = params.rows();
    const size_t N = u.cols();
    const size_t num_params = params.cols();

    Eigen::MatrixXd u_dist = u;
    Eigen::MatrixXd v_dist = v;

    for (size_t b = 0; b < B; ++b) {
        // Get batch parameters
        Eigen::VectorXd batch_params = params.row(b);

        // Compute common terms
        Eigen::ArrayXd u_arr = u.row(b).array();
        Eigen::ArrayXd v_arr = v.row(b).array();
        Eigen::ArrayXd u2 = u_arr * u_arr;
        Eigen::ArrayXd v2 = v_arr * v_arr;
        Eigen::ArrayXd r2 = u2 + v2;

        // Apply distortion based on number of parameters
        if (num_params == 1) {
            // Simple radial distortion (1 parameter)
            double k = batch_params(0);
            Eigen::ArrayXd radial = k * r2;
            u_dist.row(b).array() += u_arr * radial;
            v_dist.row(b).array() += v_arr * radial;

        } else if (num_params == 2) {
            // RadialCameraModel distortion (2 parameters)
            double k1 = batch_params(0);
            double k2 = batch_params(1);
            Eigen::ArrayXd radial = k1 * r2 + k2 * r2 * r2;
            u_dist.row(b).array() += u_arr * radial;
            v_dist.row(b).array() += v_arr * radial;

        } else if (num_params == 4) {
            // OpenCVCameraModel distortion (4 parameters)
            double k1 = batch_params(0);
            double k2 = batch_params(1);
            double p1 = batch_params(2);
            double p2 = batch_params(3);

            Eigen::ArrayXd radial = k1 * r2 + k2 * r2 * r2;
            Eigen::ArrayXd uv = u_arr * v_arr;

            u_dist.row(b).array() += u_arr * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2);
            v_dist.row(b).array() += v_arr * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2);

        } else {
            throw std::invalid_argument("Unsupported number of distortion parameters (must be 1, 2 or 4)");
        }
    }

    return {u_dist, v_dist};
}

std::tuple<torch::Tensor, torch::Tensor> apply_distortion(
    const torch::Tensor& params,
    const torch::Tensor& u,
    const torch::Tensor& v) {

    // Check input dimensions
    if (params.size(0) != u.size(0) || params.size(0) != v.size(0)) {
        throw std::invalid_argument("params, u and v must have same batch size");
    }
    if (u.size(1) != v.size(1)) {
        throw std::invalid_argument("u and v must have same number of points");
    }

    const int64_t B = params.size(0);
    const int64_t N = u.size(1);
    const int64_t num_params = params.size(1);

    // Initialize output tensors with input values
    auto u_dist = u.clone();
    auto v_dist = v.clone();

    // Process each batch
    for (int64_t b = 0; b < B; ++b) {
        // Get batch parameters
        auto batch_params = params[b];

        // Compute common terms
        auto u_b = u[b];
        auto v_b = v[b];
        auto u2 = u_b * u_b;
        auto v2 = v_b * v_b;
        auto r2 = u2 + v2;

        // Apply distortion based on number of parameters
        if (num_params == 1) {
            // Simple radial distortion (1 parameter)
            auto k = batch_params[0].item<float>();
            auto radial = k * r2;
            u_dist[b] += u_b * radial;
            v_dist[b] += v_b * radial;

        } else if (num_params == 2) {
            // RadialCameraModel distortion (2 parameters)
            auto k1 = batch_params[0].item<float>();
            auto k2 = batch_params[1].item<float>();
            auto radial = k1 * r2 + k2 * r2 * r2;
            u_dist[b] += u_b * radial;
            v_dist[b] += v_b * radial;

        } else if (num_params == 4) {
            // OpenCVCameraModel distortion (4 parameters)
            auto k1 = batch_params[0].item<float>();
            auto k2 = batch_params[1].item<float>();
            auto p1 = batch_params[2].item<float>();
            auto p2 = batch_params[3].item<float>();

            auto radial = k1 * r2 + k2 * r2 * r2;
            auto uv = u_b * v_b;

            u_dist[b] += u_b * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2);
            v_dist[b] += v_b * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2);

        } else {
            throw std::invalid_argument("Unsupported number of distortion parameters (must be 1, 2 or 4)");
        }
    }

    return std::make_tuple(u_dist, v_dist);
}

} // namespace vggt
