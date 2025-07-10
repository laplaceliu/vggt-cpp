/**
 * @file geometry.cpp
 * @brief Implementation of geometry utility functions for VGGT
 */

#include "geometry.h"
#include <torch/torch.h>

namespace vggt {
namespace utils {

torch::Tensor unproject_depth_map_to_point_map(
    const torch::Tensor& depth_map,
    const torch::Tensor& intrinsics,
    const torch::Tensor& extrinsics) {

    // Get dimensions
    auto batch_size = depth_map.size(0);
    auto height = depth_map.size(2);
    auto width = depth_map.size(3);

    // Create pixel coordinate grid
    auto device = depth_map.device();
    auto y_range = torch::arange(height, torch::TensorOptions().device(device));
    auto x_range = torch::arange(width, torch::TensorOptions().device(device));
    auto y_grid = y_range.view({1, -1, 1}).expand({1, height, width});
    auto x_grid = x_range.view({1, 1, -1}).expand({1, height, width});

    // Stack grid coordinates
    auto pixel_coords = torch::stack({x_grid, y_grid, torch::ones({1, height, width},
                                     torch::TensorOptions().device(device))}, 0);
    pixel_coords = pixel_coords.view({3, -1});  // [3, H*W]

    // Expand for batch dimension
    pixel_coords = pixel_coords.unsqueeze(0).expand({batch_size, -1, -1});  // [B, 3, H*W]

    // Apply inverse intrinsics
    auto inv_intrinsics = torch::inverse(intrinsics);  // [B, 3, 3]
    auto cam_coords = torch::bmm(inv_intrinsics, pixel_coords);  // [B, 3, H*W]

    // Scale by depth
    auto depth_reshaped = depth_map.view({batch_size, 1, -1});  // [B, 1, H*W]
    cam_coords = cam_coords * depth_reshaped;  // [B, 3, H*W]

    // Convert to homogeneous coordinates
    auto ones = torch::ones({batch_size, 1, cam_coords.size(2)},
                           torch::TensorOptions().device(device));
    auto cam_coords_hom = torch::cat({cam_coords, ones}, 1);  // [B, 4, H*W]

    // Apply extrinsics
    auto world_coords_hom = torch::bmm(extrinsics, cam_coords_hom);  // [B, 4, H*W]
    auto world_coords = world_coords_hom.narrow(1, 0, 3);  // [B, 3, H*W]

    // Reshape back to image dimensions
    world_coords = world_coords.view({batch_size, 3, height, width});

    return world_coords;
}

torch::Tensor depth_to_world_coords_points(
    const torch::Tensor& depth_map,
    const torch::Tensor& intrinsics,
    const torch::Tensor& extrinsics) {

    // Get dimensions
    auto batch_size = depth_map.size(0);
    auto height = depth_map.size(2);
    auto width = depth_map.size(3);

    // Create pixel coordinate grid
    auto device = depth_map.device();
    auto y_range = torch::arange(height, torch::TensorOptions().device(device));
    auto x_range = torch::arange(width, torch::TensorOptions().device(device));
    auto y_grid = y_range.view({1, -1, 1}).expand({1, height, width});
    auto x_grid = x_range.view({1, 1, -1}).expand({1, height, width});

    // Stack grid coordinates
    auto pixel_coords = torch::stack({x_grid, y_grid, torch::ones({1, height, width},
                                     torch::TensorOptions().device(device))}, 0);
    pixel_coords = pixel_coords.view({3, -1});  // [3, H*W]

    // Expand for batch dimension
    pixel_coords = pixel_coords.unsqueeze(0).expand({batch_size, -1, -1});  // [B, 3, H*W]

    // Apply inverse intrinsics
    auto inv_intrinsics = torch::inverse(intrinsics);  // [B, 3, 3]
    auto cam_coords = torch::bmm(inv_intrinsics, pixel_coords);  // [B, 3, H*W]

    // Scale by depth
    auto depth_reshaped = depth_map.view({batch_size, 1, -1});  // [B, 1, H*W]
    cam_coords = cam_coords * depth_reshaped;  // [B, 3, H*W]

    // Convert to homogeneous coordinates
    auto ones = torch::ones({batch_size, 1, cam_coords.size(2)},
                           torch::TensorOptions().device(device));
    auto cam_coords_hom = torch::cat({cam_coords, ones}, 1);  // [B, 4, H*W]

    // Apply extrinsics
    auto world_coords_hom = torch::bmm(extrinsics, cam_coords_hom);  // [B, 4, H*W]
    auto world_coords = world_coords_hom.narrow(1, 0, 3);  // [B, 3, H*W]

    // Transpose to get [B, H*W, 3]
    world_coords = world_coords.transpose(1, 2);

    return world_coords;
}

torch::Tensor depth_to_cam_coords_points(
    const torch::Tensor& depth_map,
    const torch::Tensor& intrinsics) {

    // Get dimensions
    auto batch_size = depth_map.size(0);
    auto height = depth_map.size(2);
    auto width = depth_map.size(3);

    // Create pixel coordinate grid
    auto device = depth_map.device();
    auto y_range = torch::arange(height, torch::TensorOptions().device(device));
    auto x_range = torch::arange(width, torch::TensorOptions().device(device));
    auto y_grid = y_range.view({1, -1, 1}).expand({1, height, width});
    auto x_grid = x_range.view({1, 1, -1}).expand({1, height, width});

    // Stack grid coordinates
    auto pixel_coords = torch::stack({x_grid, y_grid, torch::ones({1, height, width},
                                     torch::TensorOptions().device(device))}, 0);
    pixel_coords = pixel_coords.view({3, -1});  // [3, H*W]

    // Expand for batch dimension
    pixel_coords = pixel_coords.unsqueeze(0).expand({batch_size, -1, -1});  // [B, 3, H*W]

    // Apply inverse intrinsics
    auto inv_intrinsics = torch::inverse(intrinsics);  // [B, 3, 3]
    auto cam_coords = torch::bmm(inv_intrinsics, pixel_coords);  // [B, 3, H*W]

    // Scale by depth
    auto depth_reshaped = depth_map.view({batch_size, 1, -1});  // [B, 1, H*W]
    cam_coords = cam_coords * depth_reshaped;  // [B, 3, H*W]

    // Transpose to get [B, H*W, 3]
    cam_coords = cam_coords.transpose(1, 2);

    return cam_coords;
}

torch::Tensor closed_form_inverse_se3(const torch::Tensor& se3_matrix) {
    // Extract rotation and translation components
    auto R = se3_matrix.narrow(1, 0, 3).narrow(2, 0, 3);  // [B, 3, 3]
    auto t = se3_matrix.narrow(1, 0, 3).narrow(2, 3, 1);  // [B, 3, 1]

    // Compute inverse rotation (transpose)
    auto R_inv = R.transpose(1, 2);  // [B, 3, 3]

    // Compute inverse translation
    auto t_inv = -torch::bmm(R_inv, t);  // [B, 3, 1]

    // Create inverse SE3 matrix
    auto batch_size = se3_matrix.size(0);
    auto device = se3_matrix.device();
    auto inv_se3 = torch::zeros_like(se3_matrix);

    // Set rotation part
    inv_se3.narrow(1, 0, 3).narrow(2, 0, 3) = R_inv;

    // Set translation part
    inv_se3.narrow(1, 0, 3).narrow(2, 3, 1) = t_inv;

    // Set bottom row to [0, 0, 0, 1]
    inv_se3.narrow(1, 3, 1).narrow(2, 0, 3).zero_();
    inv_se3.narrow(1, 3, 1).narrow(2, 3, 1).fill_(1.0);

    return inv_se3;
}

} // namespace utils
} // namespace vggt
