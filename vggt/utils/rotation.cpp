#include "rotation.h"

namespace vggt {
namespace utils {

torch::Tensor quat_to_mat(const torch::Tensor& quaternions) {
    auto i_j_k_r = quaternions.unbind(-1);
    auto i = i_j_k_r[0];
    auto j = i_j_k_r[1];
    auto k = i_j_k_r[2];
    auto r = i_j_k_r[3];

    auto two_s = 2.0 / (quaternions * quaternions).sum(-1);

    auto o = torch::stack({
        1 - two_s * (j * j + k * k),
        two_s * (i * j - k * r),
        two_s * (i * k + j * r),
        two_s * (i * j + k * r),
        1 - two_s * (i * i + k * k),
        two_s * (j * k - i * r),
        two_s * (i * k - j * r),
        two_s * (j * k + i * r),
        1 - two_s * (i * i + j * j)
    }, -1);

    return o.reshape(quaternions.sizes().slice(0, -1)).reshape({-1, 3, 3});
}

torch::Tensor mat_to_quat(const torch::Tensor& matrix) {
    if (matrix.size(-1) != 3 || matrix.size(-2) != 3) {
        throw std::runtime_error("Invalid rotation matrix shape.");
    }

    auto batch_dim = matrix.sizes().slice(0, -2);
    auto batch_vec = batch_dim.vec();
    batch_vec.push_back(9);
    auto m = matrix.reshape(batch_vec);
    auto m00 = m.select(-1, 0);
    auto m01 = m.select(-1, 1);
    auto m02 = m.select(-1, 2);
    auto m10 = m.select(-1, 3);
    auto m11 = m.select(-1, 4);
    auto m12 = m.select(-1, 5);
    auto m20 = m.select(-1, 6);
    auto m21 = m.select(-1, 7);
    auto m22 = m.select(-1, 8);

    auto q_abs = _sqrt_positive_part(
        torch::stack({
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22
        }, -1)
    );

    auto quat_by_rijk = torch::stack({
        torch::stack({q_abs.select(-1, 0).pow(2), m21 - m12, m02 - m20, m10 - m01}, -1),
        torch::stack({m21 - m12, q_abs.select(-1, 1).pow(2), m10 + m01, m02 + m20}, -1),
        torch::stack({m02 - m20, m10 + m01, q_abs.select(-1, 2).pow(2), m12 + m21}, -1),
        torch::stack({m10 - m01, m20 + m02, m21 + m12, q_abs.select(-1, 3).pow(2)}, -1)
    }, -2);

    auto flr = torch::tensor(0.1, torch::TensorOptions().dtype(q_abs.dtype()).device(q_abs.device()));
    auto quat_candidates = quat_by_rijk / (2.0 * q_abs.unsqueeze(-1).max(flr));

    auto out = quat_candidates.index_select(-2, q_abs.argmax(-1)).squeeze(-2);
    out = out.index_select(-1, torch::tensor({1, 2, 3, 0}, torch::TensorOptions().device(out.device())));

    return standardize_quaternion(out);
}

torch::Tensor _sqrt_positive_part(const torch::Tensor& x) {
    auto ret = torch::zeros_like(x);
    auto positive_mask = x > 0;
    if (torch::GradMode::is_enabled()) {
        ret.masked_scatter_(positive_mask, torch::sqrt(x.masked_select(positive_mask)));
    } else {
        ret = torch::where(positive_mask, torch::sqrt(x), ret);
    }
    return ret;
}

torch::Tensor standardize_quaternion(const torch::Tensor& quaternions) {
    return torch::where(quaternions.index_select(-1, torch::tensor({3}, torch::TensorOptions().device(quaternions.device()))) < 0,
                       -quaternions, quaternions);
}

} // namespace utils
} // namespace vggt