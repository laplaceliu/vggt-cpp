#include "distortion.h"
#include <stdexcept>

namespace vggt {
namespace dependency {

bool is_torch(const torch::Tensor& x) {
    return x.defined();
}

torch::Tensor ensure_torch(const torch::Tensor& x) {
    // In C++ implementation, we directly accept torch::Tensor
    // so we just need to check if it's defined
    if (!x.defined()) {
        throw std::runtime_error("Input tensor is not defined");
    }
    return x;
}

torch::Tensor single_undistortion(const torch::Tensor& params, const torch::Tensor& tracks_normalized) {
    torch::Tensor params_torch = ensure_torch(params);
    torch::Tensor tracks_normalized_torch = ensure_torch(tracks_normalized);

    torch::Tensor u = tracks_normalized_torch.index({torch::indexing::Ellipsis, 0}).clone();
    torch::Tensor v = tracks_normalized_torch.index({torch::indexing::Ellipsis, 1}).clone();

    auto [u_undist, v_undist] = apply_distortion(params_torch, u, v);

    return torch::stack({u_undist, v_undist}, -1);
}

torch::Tensor iterative_undistortion(const torch::Tensor& params, const torch::Tensor& tracks_normalized,
                                    int64_t max_iterations, double max_step_norm, double rel_step_size) {
    torch::Tensor params_torch = ensure_torch(params);
    torch::Tensor tracks_normalized_torch = ensure_torch(tracks_normalized);

    auto B = tracks_normalized_torch.size(0);
    auto N = tracks_normalized_torch.size(1);

    torch::Tensor u = tracks_normalized_torch.index({torch::indexing::Ellipsis, 0}).clone();
    torch::Tensor v = tracks_normalized_torch.index({torch::indexing::Ellipsis, 1}).clone();
    torch::Tensor original_u = u.clone();
    torch::Tensor original_v = v.clone();

    double eps = std::numeric_limits<float>::epsilon();

    for (int64_t idx = 0; idx < max_iterations; ++idx) {
        auto [u_undist, v_undist] = apply_distortion(params_torch, u, v);
        torch::Tensor dx = original_u - u_undist;
        torch::Tensor dy = original_v - v_undist;

        torch::Tensor step_u = torch::clamp(torch::abs(u) * rel_step_size, eps);
        torch::Tensor step_v = torch::clamp(torch::abs(v) * rel_step_size, eps);

        // Calculate Jacobian
        auto [u_plus_step_u_0, v_plus_step_u_0] = apply_distortion(params_torch, u + step_u, v);
        auto [u_minus_step_u_0, v_minus_step_u_0] = apply_distortion(params_torch, u - step_u, v);
        auto [u_plus_step_v_0, v_plus_step_v_0] = apply_distortion(params_torch, u, v + step_v);
        auto [u_minus_step_v_0, v_minus_step_v_0] = apply_distortion(params_torch, u, v - step_v);

        torch::Tensor J_00 = (u_plus_step_u_0 - u_minus_step_u_0) / (2 * step_u);
        torch::Tensor J_01 = (u_plus_step_v_0 - u_minus_step_v_0) / (2 * step_v);
        torch::Tensor J_10 = (v_plus_step_u_0 - v_minus_step_u_0) / (2 * step_u);
        torch::Tensor J_11 = (v_plus_step_v_0 - v_minus_step_v_0) / (2 * step_v);

        // Add identity to Jacobian
        J_00 = J_00 + 1;
        J_11 = J_11 + 1;

        // Stack Jacobian
        torch::Tensor J_row0 = torch::stack({J_00, J_01}, -1);
        torch::Tensor J_row1 = torch::stack({J_10, J_11}, -1);
        torch::Tensor J = torch::stack({J_row0, J_row1}, -2);

        // Solve linear system
        torch::Tensor rhs = torch::stack({dx, dy}, -1);
        torch::Tensor delta = torch::linalg_solve(J, rhs);

        // Update u and v
        u = u + delta.index({torch::indexing::Ellipsis, 0});
        v = v + delta.index({torch::indexing::Ellipsis, 1});

        // Check convergence
        if (torch::max(torch::sum(delta * delta, -1)).item<double>() < max_step_norm) {
            break;
        }
    }

    return torch::stack({u, v}, -1);
}

std::pair<torch::Tensor, torch::Tensor> apply_distortion(const torch::Tensor& extra_params,
                                                        const torch::Tensor& u,
                                                        const torch::Tensor& v) {
    torch::Tensor params = ensure_torch(extra_params);
    torch::Tensor u_tensor = ensure_torch(u);
    torch::Tensor v_tensor = ensure_torch(v);

    int64_t num_params = params.size(1);
    torch::Tensor du, dv;

    if (num_params == 1) {
        // Simple radial distortion
        torch::Tensor k = params.index({torch::indexing::Slice(), 0});
        torch::Tensor u2 = u_tensor * u_tensor;
        torch::Tensor v2 = v_tensor * v_tensor;
        torch::Tensor r2 = u2 + v2;
        torch::Tensor radial = k.unsqueeze(1) * r2;
        du = u_tensor * radial;
        dv = v_tensor * radial;
    }
    else if (num_params == 2) {
        // RadialCameraModel distortion
        torch::Tensor k1 = params.index({torch::indexing::Slice(), 0});
        torch::Tensor k2 = params.index({torch::indexing::Slice(), 1});
        torch::Tensor u2 = u_tensor * u_tensor;
        torch::Tensor v2 = v_tensor * v_tensor;
        torch::Tensor r2 = u2 + v2;
        torch::Tensor radial = k1.unsqueeze(1) * r2 + k2.unsqueeze(1) * r2 * r2;
        du = u_tensor * radial;
        dv = v_tensor * radial;
    }
    else if (num_params == 4) {
        // OpenCVCameraModel distortion
        torch::Tensor k1 = params.index({torch::indexing::Slice(), 0});
        torch::Tensor k2 = params.index({torch::indexing::Slice(), 1});
        torch::Tensor p1 = params.index({torch::indexing::Slice(), 2});
        torch::Tensor p2 = params.index({torch::indexing::Slice(), 3});

        torch::Tensor u2 = u_tensor * u_tensor;
        torch::Tensor v2 = v_tensor * v_tensor;
        torch::Tensor uv = u_tensor * v_tensor;
        torch::Tensor r2 = u2 + v2;

        torch::Tensor radial = k1.unsqueeze(1) * r2 + k2.unsqueeze(1) * r2 * r2;
        du = u_tensor * radial + 2 * p1.unsqueeze(1) * uv + p2.unsqueeze(1) * (r2 + 2 * u2);
        dv = v_tensor * radial + 2 * p2.unsqueeze(1) * uv + p1.unsqueeze(1) * (r2 + 2 * v2);
    }
    else {
        throw std::runtime_error("Unsupported number of distortion parameters");
    }

    torch::Tensor u_out = u_tensor.clone() + du;
    torch::Tensor v_out = v_tensor.clone() + dv;

    return std::make_pair(u_out, v_out);
}

} // namespace dependency
} // namespace vggt
