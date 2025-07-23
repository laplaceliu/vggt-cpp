// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "rotation.h"
#include <torch/torch.h>

namespace vggt {
namespace utils {

torch::Tensor quat_to_mat(const torch::Tensor& quaternions) {
    auto i = quaternions.index({torch::indexing::Slice(), 0});
    auto j = quaternions.index({torch::indexing::Slice(), 1});
    auto k = quaternions.index({torch::indexing::Slice(), 2});
    auto r = quaternions.index({torch::indexing::Slice(), 3});

    auto two_s = 2.0 / (quaternions * quaternions).sum(-1);

    auto o = torch::stack(
        {
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j)
        },
        -1
    );
    return o.reshape(quaternions.sizes().vec() + std::vector<int64_t>{3, 3});
}

torch::Tensor mat_to_quat(const torch::Tensor& matrix) {
    if (matrix.size(-1) != 3 || matrix.size(-2) != 3) {
        throw std::runtime_error("Invalid rotation matrix shape.");
    }

    auto batch_dim = matrix.sizes().vec();
    batch_dim.pop_back();
    batch_dim.pop_back();

    auto m = matrix.reshape(batch_dim + std::vector<int64_t>{9});
    auto m00 = m.index({torch::indexing::Slice(), 0});
    auto m01 = m.index({torch::indexing::Slice(), 1});
    auto m02 = m.index({torch::indexing::Slice(), 2});
    auto m10 = m.index({torch::indexing::Slice(), 3});
    auto m11 = m.index({torch::indexing::Slice(), 4});
    auto m12 = m.index({torch::indexing::Slice(), 5});
    auto m20 = m.index({torch::indexing::Slice(), 6});
    auto m21 = m.index({torch::indexing::Slice(), 7});
    auto m22 = m.index({torch::indexing::Slice(), 8});

    auto q_abs = _sqrt_positive_part(
        torch::stack(
            {
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22
            },
            -1
        )
    );

    auto quat_by_rijk = torch::stack(
        {
            torch::stack({q_abs.index({torch::indexing::Slice(), 0}).pow(2), m21 - m12, m02 - m20, m10 - m01}, -1),
            torch::stack({m21 - m12, q_abs.index({torch::indexing::Slice(), 1}).pow(2), m10 + m01, m02 + m20}, -1),
            torch::stack({m02 - m20, m10 + m01, q_abs.index({torch::indexing::Slice(), 2}).pow(2), m12 + m21}, -1),
            torch::stack({m10 - m01, m20 + m02, m21 + m12, q_abs.index({torch::indexing::Slice(), 3}).pow(2)}, -1)
        },
        -2
    );

    auto flr = torch::tensor(0.1, torch::TensorOptions().dtype(q_abs.dtype()).device(q_abs.device()));
    auto quat_candidates = quat_by_rijk / (2.0 * q_abs.index({torch::indexing::Slice(), torch::indexing::None}).max(flr));

    auto out = quat_candidates.index(
        {torch::indexing::Slice(), torch::one_hot(q_abs.argmax(-1), 4).to(torch::kBool)}
    ).reshape(batch_dim + std::vector<int64_t>{4});

    // Convert from rijk to ijkr
    out = out.index({torch::indexing::Slice(), std::vector<int64_t>{1, 2, 3, 0}});

    return standardize_quaternion(out);
}

torch::Tensor _sqrt_positive_part(const torch::Tensor& x) {
    auto ret = torch::zeros_like(x);
    auto positive_mask = x > 0;
    if (torch::is_grad_enabled()) {
        ret.index_put_({positive_mask}, x.index({positive_mask}).sqrt());
    } else {
        ret = torch::where(positive_mask, x.sqrt(), ret);
    }
    return ret;
}

torch::Tensor standardize_quaternion(const torch::Tensor& quaternions) {
    return torch::where(quaternions.index({torch::indexing::Slice(), 3}) < 0, -quaternions, quaternions);
}

} // namespace utils
} // namespace vggt