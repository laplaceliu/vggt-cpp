/**
 * @file projection.cpp
 * @brief Implementation of functions for projecting 3D points to 2D using camera parameters
 */

#include "projection.h"
#include "distortion.h"

namespace vggt {
namespace dependency {

torch::Tensor img_from_cam(
    const torch::Tensor& intrinsics,
    const torch::Tensor& points_cam,
    const c10::optional<torch::Tensor>& extra_params,
    float default_val
) {
    // 1. perspective divide
    auto z = points_cam.index({"...", torch::indexing::Slice(2, 3), torch::indexing::Slice()});  // (B,1,N)
    auto points_cam_norm = points_cam / z;  // (B,3,N)
    auto uv = points_cam_norm.index({"...", torch::indexing::Slice(0, 2), torch::indexing::Slice()});  // (B,2,N)

    // 2. optional distortion
    if (extra_params.has_value()) {
        auto u = uv.index({"...", 0, torch::indexing::Slice()});
        auto v = uv.index({"...", 1, torch::indexing::Slice()});
        auto distortion_result = apply_distortion(extra_params.value(), u, v);
        auto uu = std::get<0>(distortion_result);
        auto vv = std::get<1>(distortion_result);
        uv = torch::stack({uu, vv}, 1);  // (B,2,N)
    }

    // 3. homogeneous coords then K multiplication
    auto ones = torch::ones_like(uv.index({"...", torch::indexing::Slice(0, 1), torch::indexing::Slice()}));  // (B,1,N)
    auto points_cam_h = torch::cat({uv, ones}, 1);  // (B,3,N)

    // batched mat-mul: K · [u v 1]ᵀ
    auto points2D_h = torch::bmm(intrinsics, points_cam_h);  // (B,3,N)
    auto points2D = points2D_h.index({"...", torch::indexing::Slice(0, 2), torch::indexing::Slice()});  // (B,2,N)
    
    // Replace NaNs with default value
    points2D = torch::nan_to_num(points2D, default_val);

    return points2D.transpose(1, 2);  // (B,N,2)
}

std::tuple<c10::optional<torch::Tensor>, torch::Tensor> project_3D_points(
    const torch::Tensor& points3D,
    const torch::Tensor& extrinsics,
    const c10::optional<torch::Tensor>& intrinsics,
    const c10::optional<torch::Tensor>& extra_params,
    float default_val,
    bool only_points_cam
) {
    // Use double precision for calculations
    torch::NoGradGuard no_grad;
    
    // Get dimensions
    int64_t N = points3D.size(0);  // Number of points
    int64_t B = extrinsics.size(0);  // Batch size, i.e., number of cameras
    
    // Create homogeneous coordinates
    auto points3D_homogeneous = torch::cat({
        points3D, 
        torch::ones({N, 1}, points3D.options())
    }, 1);  // (N,4)
    
    // Reshape for batch processing
    auto points3D_homogeneous_B = points3D_homogeneous.unsqueeze(0).expand({B, -1, -1});  // (B,N,4)
    
    // Step 1: Apply extrinsic parameters
    // Transform 3D points to camera coordinate system for all cameras
    auto points_cam = torch::bmm(extrinsics, points3D_homogeneous_B.transpose(1, 2));  // (B,3,N)
    
    if (only_points_cam) {
        return std::make_tuple(c10::nullopt, points_cam);
    }
    
    // Step 2: Apply intrinsic parameters and (optional) distortion
    if (!intrinsics.has_value()) {
        throw std::invalid_argument("`intrinsics` must be provided unless only_points_cam=True");
    }
    
    auto points2D = img_from_cam(intrinsics.value(), points_cam, extra_params, default_val);
    
    return std::make_tuple(points2D, points_cam);
}

} // namespace dependency
} // namespace vggt