#pragma once

#include <torch/torch.h>
#include <tuple>

namespace vggt {
namespace dependency {

/**
 * Apply intrinsics (and optional radial distortion) to camera-space points.
 * 
 * @param intrinsics (B,3,3) camera matrix K.
 * @param points_cam (B,3,N) homogeneous camera coords (x, y, z)ᵀ.
 * @param extra_params (B, N) or (B, k) distortion params (k = 1,2,4) or None.
 * @param default_val Value used for nan replacement.
 * @return (B,N,2) pixel coordinates.
 */
torch::Tensor img_from_cam(const torch::Tensor& intrinsics, 
                          const torch::Tensor& points_cam, 
                          const torch::Tensor& extra_params = torch::Tensor(), 
                          double default_val = 0.0);

/**
 * Transforms 3D points to 2D using extrinsic and intrinsic parameters.
 * 
 * @param points3D 3D points of shape Px3.
 * @param extrinsics Extrinsic parameters of shape Bx3x4.
 * @param intrinsics Intrinsic parameters of shape Bx3x3.
 * @param extra_params Extra parameters of shape BxN, used for radial distortion.
 * @param default_val Default value to replace NaNs.
 * @param only_points_cam If True, skip the projection and return points2D as None.
 * @return tuple: (points2D, points_cam) where points2D is of shape BxNx2 or empty tensor if only_points_cam=True,
 *         and points_cam is of shape Bx3xN.
 */
std::tuple<torch::Tensor, torch::Tensor> project_3D_points(
    const torch::Tensor& points3D, 
    const torch::Tensor& extrinsics, 
    const torch::Tensor& intrinsics = torch::Tensor(),
    const torch::Tensor& extra_params = torch::Tensor(), 
    double default_val = 0.0, 
    bool only_points_cam = false);

} // namespace dependency
} // namespace vggt