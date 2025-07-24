#pragma once

#include <torch/torch.h>

namespace vggt {
namespace utils {

torch::Tensor quat_to_mat(const torch::Tensor& quaternions);
torch::Tensor mat_to_quat(const torch::Tensor& matrix);
torch::Tensor _sqrt_positive_part(const torch::Tensor& x);
torch::Tensor standardize_quaternion(const torch::Tensor& quaternions);

} // namespace utils
} // namespace vggt