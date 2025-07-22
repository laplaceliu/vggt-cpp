/**
 * @file projection.h
 * @brief Functions for projecting 3D points to 2D using camera parameters
 * 
 * This file contains functions for transforming 3D points to 2D using extrinsic and intrinsic parameters.
 */

#pragma once

#include <torch/torch.h>
#include <c10/util/Optional.h>

namespace vggt {
namespace dependency {

/**
 * @brief Apply intrinsics (and optional radial distortion) to camera-space points.
 * 
 * @param intrinsics (B,3,3) camera matrix K
 * @param points_cam (B,3,N) homogeneous camera coords (x, y, z)ᵀ
 * @param extra_params (B,N) or (B,k) distortion params (k = 1,2,4) or None
 * @param default_val Value used for NaN replacement
 * @return torch::Tensor (B,N,2) pixel coordinates
 */
torch::Tensor img_from_cam(
    const torch::Tensor& intrinsics,
    const torch::Tensor& points_cam,
    const c10::optional<torch::Tensor>& extra_params = c10::nullopt,
    float default_val = 0.0
);

/**
 * @brief Transforms 3D points to 2D using extrinsic and intrinsic parameters.
 * 
 * @param points3D (P,3) 3D points
 * @param extrinsics (B,3,4) Extrinsic parameters [R|t]
 * @param intrinsics (B,3,3) Intrinsic parameters K
 * @param extra_params (B,N) or (B,k) Extra parameters for radial distortion
 * @param default_val Default value to replace NaNs
 * @param only_points_cam If true, skip the projection and return points2D as None
 * @return std::tuple<torch::Tensor, torch::Tensor> (points2D, points_cam) where points2D is (B,N,2) or None,
 *         and points_cam is (B,3,N)
 */
std::tuple<c10::optional<torch::Tensor>, torch::Tensor> project_3D_points(
    const torch::Tensor& points3D,
    const torch::Tensor& extrinsics,
    const c10::optional<torch::Tensor>& intrinsics = c10::nullopt,
    const c10::optional<torch::Tensor>& extra_params = c10::nullopt,
    float default_val = 0.0,
    bool only_points_cam = false
);

} // namespace dependency
} // namespace vggt