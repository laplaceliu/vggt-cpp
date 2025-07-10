/**
 * @file rotation.cpp
 * @brief Implementation of rotation utility functions for VGGT
 */

#include "rotation.h"
#include <torch/torch.h>

namespace vggt {
namespace utils {

torch::Tensor _sqrt_positive_part(const torch::Tensor& x) {
    // Return sqrt(max(0, x))
    return torch::sqrt(torch::max(x, torch::zeros_like(x)));
}

torch::Tensor quat_to_mat(const torch::Tensor& quaternion) {
    // Normalize quaternion
    auto q = quaternion / torch::norm(quaternion, 2, -1, true);

    // Extract quaternion components
    auto w = q.index({Ellipsis, 0});
    auto x = q.index({Ellipsis, 1});
    auto y = q.index({Ellipsis, 2});
    auto z = q.index({Ellipsis, 3});

    // Compute rotation matrix elements
    auto batch_size = quaternion.size(0);
    auto device = quaternion.device();
    auto dtype = quaternion.dtype();

    auto zeros = torch::zeros({batch_size}, torch::TensorOptions().device(device).dtype(dtype));
    auto ones = torch::ones({batch_size}, torch::TensorOptions().device(device).dtype(dtype));

    // First row
    auto r00 = 1 - 2 * (y * y + z * z);
    auto r01 = 2 * (x * y - w * z);
    auto r02 = 2 * (x * z + w * y);

    // Second row
    auto r10 = 2 * (x * y + w * z);
    auto r11 = 1 - 2 * (x * x + z * z);
    auto r12 = 2 * (y * z - w * x);

    // Third row
    auto r20 = 2 * (x * z - w * y);
    auto r21 = 2 * (y * z + w * x);
    auto r22 = 1 - 2 * (x * x + y * y);

    // Stack to form rotation matrix
    auto row0 = torch::stack({r00, r01, r02}, -1);
    auto row1 = torch::stack({r10, r11, r12}, -1);
    auto row2 = torch::stack({r20, r21, r22}, -1);

    auto rotation_matrix = torch::stack({row0, row1, row2}, -2);

    return rotation_matrix;
}

torch::Tensor mat_to_quat(const torch::Tensor& rotation_matrix) {
    // Extract rotation matrix elements
    auto m00 = rotation_matrix.index({Ellipsis, 0, 0});
    auto m01 = rotation_matrix.index({Ellipsis, 0, 1});
    auto m02 = rotation_matrix.index({Ellipsis, 0, 2});
    auto m10 = rotation_matrix.index({Ellipsis, 1, 0});
    auto m11 = rotation_matrix.index({Ellipsis, 1, 1});
    auto m12 = rotation_matrix.index({Ellipsis, 1, 2});
    auto m20 = rotation_matrix.index({Ellipsis, 2, 0});
    auto m21 = rotation_matrix.index({Ellipsis, 2, 1});
    auto m22 = rotation_matrix.index({Ellipsis, 2, 2});

    // Compute quaternion components
    auto q0 = _sqrt_positive_part(1 + m00 + m11 + m22);
    auto q1 = _sqrt_positive_part(1 + m00 - m11 - m22);
    auto q2 = _sqrt_positive_part(1 - m00 + m11 - m22);
    auto q3 = _sqrt_positive_part(1 - m00 - m11 + m22);

    auto q0q1 = q0 * q1;
    auto q0q2 = q0 * q2;
    auto q0q3 = q0 * q3;
    auto q1q2 = q1 * q2;
    auto q1q3 = q1 * q3;
    auto q2q3 = q2 * q3;

    auto device = rotation_matrix.device();
    auto dtype = rotation_matrix.dtype();
    auto eps = torch::tensor(1e-6, torch::TensorOptions().device(device).dtype(dtype));

    // Case 1: q0 largest
    auto mask_q0_largest = q0 >= torch::max(torch::max(q1, q2), q3);
    auto q0_largest_w = q0;
    auto q0_largest_x = (m21 - m12) / (4 * q0);
    auto q0_largest_y = (m02 - m20) / (4 * q0);
    auto q0_largest_z = (m10 - m01) / (4 * q0);

    // Case 2: q1 largest
    auto mask_q1_largest = (q1 >= q0) & (q1 >= q2) & (q1 >= q3);
    auto q1_largest_w = (m21 - m12) / (4 * q1);
    auto q1_largest_x = q1;
    auto q1_largest_y = (m01 + m10) / (4 * q1);
    auto q1_largest_z = (m02 + m20) / (4 * q1);

    // Case 3: q2 largest
    auto mask_q2_largest = (q2 >= q0) & (q2 >= q1) & (q2 >= q3);
    auto q2_largest_w = (m02 - m20) / (4 * q2);
    auto q2_largest_x = (m01 + m10) / (4 * q2);
    auto q2_largest_y = q2;
    auto q2_largest_z = (m12 + m21) / (4 * q2);

    // Case 4: q3 largest
    auto mask_q3_largest = (q3 >= q0) & (q3 >= q1) & (q3 >= q2);
    auto q3_largest_w = (m10 - m01) / (4 * q3);
    auto q3_largest_x = (m02 + m20) / (4 * q3);
    auto q3_largest_y = (m12 + m21) / (4 * q3);
    auto q3_largest_z = q3;

    // Combine cases
    auto w = torch::zeros_like(q0);
    auto x = torch::zeros_like(q0);
    auto y = torch::zeros_like(q0);
    auto z = torch::zeros_like(q0);

    w = torch::where(mask_q0_largest, q0_largest_w, w);
    x = torch::where(mask_q0_largest, q0_largest_x, x);
    y = torch::where(mask_q0_largest, q0_largest_y, y);
    z = torch::where(mask_q0_largest, q0_largest_z, z);

    w = torch::where(mask_q1_largest, q1_largest_w, w);
    x = torch::where(mask_q1_largest, q1_largest_x, x);
    y = torch::where(mask_q1_largest, q1_largest_y, y);
    z = torch::where(mask_q1_largest, q1_largest_z, z);

    w = torch::where(mask_q2_largest, q2_largest_w, w);
    x = torch::where(mask_q2_largest, q2_largest_x, x);
    y = torch::where(mask_q2_largest, q2_largest_y, y);
    z = torch::where(mask_q2_largest, q2_largest_z, z);

    w = torch::where(mask_q3_largest, q3_largest_w, w);
    x = torch::where(mask_q3_largest, q3_largest_x, x);
    y = torch::where(mask_q3_largest, q3_largest_y, y);
    z = torch::where(mask_q3_largest, q3_largest_z, z);

    // Stack to form quaternion
    auto quaternion = torch::stack({w, x, y, z}, -1);

    // Normalize quaternion
    quaternion = quaternion / torch::norm(quaternion, 2, -1, true);

    return standardize_quaternion(quaternion);
}

torch::Tensor standardize_quaternion(const torch::Tensor& quaternion) {
    // Ensure real part is non-negative
    auto q = quaternion.clone();
    auto mask = q.index({Ellipsis, 0}) < 0;
    q = torch::where(mask.unsqueeze(-1), -q, q);
    return q;
}

} // namespace utils
} // namespace vggt
