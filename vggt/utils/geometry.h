#pragma once

#include <torch/torch.h>

namespace vggt {
namespace utils {

// Function declarations
torch::Tensor unproject_depth_map_to_point_map(
    const torch::Tensor& depth_map,
    const torch::Tensor& extrinsics_cam,
    const torch::Tensor& intrinsics_cam);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> depth_to_world_coords_points(
    const torch::Tensor& depth_map,
    const torch::Tensor& extrinsic,
    const torch::Tensor& intrinsic,
    double eps = 1e-8);

torch::Tensor depth_to_cam_coords_points(
    const torch::Tensor& depth_map,
    const torch::Tensor& intrinsic);

torch::Tensor closed_form_inverse_se3(
    const torch::Tensor& se3,
    const torch::Tensor& R = torch::Tensor(),
    const torch::Tensor& T = torch::Tensor());

} // namespace utils
} // namespace vggt