#include "projection.h"
#include "distortion.h"
#include <torch/torch.h>

namespace vggt {
namespace dependency {

torch::Tensor img_from_cam(const torch::Tensor& intrinsics, 
                          const torch::Tensor& points_cam, 
                          const torch::Tensor& extra_params, 
                          double default_val) {
    // Normalize by the third coordinate (homogeneous division)
    torch::Tensor points_cam_norm = points_cam / points_cam.index({torch::indexing::Slice(), 
                                                                  torch::indexing::Slice(2, 3), 
                                                                  torch::indexing::Slice()});
    
    // Extract uv
    torch::Tensor uv = points_cam_norm.index({torch::indexing::Slice(), 
                                             torch::indexing::Slice(0, 2), 
                                             torch::indexing::Slice()});

    // Apply distortion if extra_params are provided
    if (extra_params.defined()) {
        auto [uu, vv] = apply_distortion(extra_params, 
                                         uv.index({torch::indexing::Slice(), 0}), 
                                         uv.index({torch::indexing::Slice(), 1}));
        uv = torch::stack({uu, vv}, 1);
    }

    // Prepare points_cam for batch matrix multiplication
    torch::Tensor ones = torch::ones_like(uv.index({torch::indexing::Slice(), 
                                                   torch::indexing::Slice(0, 1), 
                                                   torch::indexing::Slice()}));
    torch::Tensor points_cam_homo = torch::cat({uv, ones}, 1);  // Bx3xN
    
    // Apply intrinsic parameters using batch matrix multiplication
    torch::Tensor points2D_homo = torch::bmm(intrinsics, points_cam_homo);  // Bx3xN

    // Extract x and y coordinates
    torch::Tensor points2D = points2D_homo.index({torch::indexing::Slice(), 
                                                 torch::indexing::Slice(0, 2), 
                                                 torch::indexing::Slice()});  // Bx2xN

    // Replace NaNs with default value
    points2D = torch::nan_to_num(points2D, default_val);

    return points2D.transpose(1, 2);  // BxNx2
}

std::tuple<torch::Tensor, torch::Tensor> project_3D_points(
    const torch::Tensor& points3D, 
    const torch::Tensor& extrinsics, 
    const torch::Tensor& intrinsics,
    const torch::Tensor& extra_params, 
    double default_val, 
    bool only_points_cam) {
    
    // Use double precision for calculations
    torch::NoGradGuard no_grad;
    
    // Get dimensions
    int64_t N = points3D.size(0);  // Number of points
    int64_t B = extrinsics.size(0);  // Batch size, i.e., number of cameras
    
    // Create homogeneous coordinates
    torch::Tensor points3D_homogeneous = torch::cat({
        points3D, 
        torch::ones_like(points3D.index({torch::indexing::Slice(), 
                                        torch::indexing::Slice(0, 1)}))
    }, 1);  // Nx4
    
    // Reshape for batch processing
    torch::Tensor points3D_homogeneous_B = points3D_homogeneous.unsqueeze(0).expand({B, -1, -1});  // BxNx4
    
    // Step 1: Apply extrinsic parameters
    // Transform 3D points to camera coordinate system for all cameras
    torch::Tensor points_cam = torch::bmm(extrinsics, points3D_homogeneous_B.transpose(1, 2));
    
    if (only_points_cam) {
        return std::make_tuple(torch::Tensor(), points_cam);
    }
    
    // Step 2: Apply intrinsic parameters and (optional) distortion
    if (!intrinsics.defined()) {
        throw std::runtime_error("`intrinsics` must be provided unless only_points_cam=True");
    }
    
    torch::Tensor points2D = img_from_cam(intrinsics, points_cam, extra_params, default_val);
    
    return std::make_tuple(points2D, points_cam);
}

} // namespace dependency
} // namespace vggt