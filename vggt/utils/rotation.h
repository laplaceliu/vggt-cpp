// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/torch.h>

namespace vggt {
namespace utils {

// Convert quaternions to rotation matrices.
// Quaternion Order: XYZW (scalar-last).
torch::Tensor quat_to_mat(const torch::Tensor& quaternions);

// Convert rotation matrices to quaternions.
// Quaternion Order: XYZW (scalar-last).
torch::Tensor mat_to_quat(const torch::Tensor& matrix);

// Helper function for sqrt of positive part.
torch::Tensor _sqrt_positive_part(const torch::Tensor& x);

// Standardize quaternion to have non-negative real part.
torch::Tensor standardize_quaternion(const torch::Tensor& quaternions);

} // namespace utils
} // namespace vggt