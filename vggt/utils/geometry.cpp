#include "geometry.h"

namespace vggt {
namespace utils {

torch::Tensor unproject_depth_map_to_point_map(
    const torch::Tensor& depth_map,
    const torch::Tensor& extrinsics_cam,
    const torch::Tensor& intrinsics_cam) {
    // Check input dimensions
    TORCH_CHECK(depth_map.dim() == 3 || depth_map.dim() == 4, "Depth map must be 3D or 4D");
    TORCH_CHECK(extrinsics_cam.dim() == 3, "Extrinsics must be 3D");
    TORCH_CHECK(intrinsics_cam.dim() == 3, "Intrinsics must be 3D");

    // Squeeze the last dimension if needed
    torch::Tensor depth_map_squeezed = depth_map;
    if (depth_map.dim() == 4) {
        depth_map_squeezed = depth_map.squeeze(-1);
    }

    // Process each frame
    std::vector<torch::Tensor> world_points_list;
    for (int64_t frame_idx = 0; frame_idx < depth_map_squeezed.size(0); ++frame_idx) {
        auto [world_points, cam_points, mask] = depth_to_world_coords_points(
            depth_map_squeezed[frame_idx],
            extrinsics_cam[frame_idx],
            intrinsics_cam[frame_idx]
        );
        world_points_list.push_back(world_points);
    }

    // Stack the results
    return torch::stack(world_points_list, 0);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> depth_to_world_coords_points(
    const torch::Tensor& depth_map,
    const torch::Tensor& extrinsic,
    const torch::Tensor& intrinsic,
    double eps) {
    if (!depth_map.defined()) {
        return {torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }

    // Valid depth mask
    torch::Tensor point_mask = depth_map > eps;

    // Convert depth map to camera coordinates
    torch::Tensor cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic);

    // Compute inverse of extrinsic matrix
    torch::Tensor cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic.unsqueeze(0))[0];

    torch::Tensor R_cam_to_world = cam_to_world_extrinsic.slice(0, 0, 3).slice(1, 0, 3);
    torch::Tensor t_cam_to_world = cam_to_world_extrinsic.slice(0, 0, 3).slice(1, 3, 4);

    // Apply rotation and translation
    torch::Tensor world_coords_points = torch::matmul(cam_coords_points, R_cam_to_world.transpose(0, 1)) + t_cam_to_world.squeeze(-1);

    return {world_coords_points, cam_coords_points, point_mask};
}

torch::Tensor depth_to_cam_coords_points(
    const torch::Tensor& depth_map,
    const torch::Tensor& intrinsic) {
    // Check input dimensions
    TORCH_CHECK(depth_map.dim() == 2, "Depth map must be 2D");
    TORCH_CHECK(intrinsic.dim() == 2 && intrinsic.size(0) == 3 && intrinsic.size(1) == 3,
               "Intrinsic matrix must be 3x3");
    TORCH_CHECK(intrinsic[0][1].item<float>() == 0 && intrinsic[1][0].item<float>() == 0,
               "Intrinsic matrix must have zero skew");

    // Intrinsic parameters
    float fu = intrinsic[0][0].item<float>();
    float fv = intrinsic[1][1].item<float>();
    float cu = intrinsic[0][2].item<float>();
    float cv = intrinsic[1][2].item<float>();

    // Generate grid of pixel coordinates
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(depth_map.device());
    int64_t H = depth_map.size(0);
    int64_t W = depth_map.size(1);
    torch::Tensor u = torch::arange(W, options).view({1, W}).expand({H, W});
    torch::Tensor v = torch::arange(H, options).view({H, 1}).expand({H, W});

    // Unproject to camera coordinates
    torch::Tensor x_cam = (u - cu) * depth_map / fu;
    torch::Tensor y_cam = (v - cv) * depth_map / fv;
    torch::Tensor z_cam = depth_map;

    // Stack to form camera coordinates
    return torch::stack({x_cam, y_cam, z_cam}, -1);
}

torch::Tensor closed_form_inverse_se3(
    const torch::Tensor& se3,
    const torch::Tensor& R,
    const torch::Tensor& T) {
    // Check input dimensions
    TORCH_CHECK(se3.dim() == 3, "se3 must be 3D");
    TORCH_CHECK(se3.size(1) == 4 && se3.size(2) == 4 || se3.size(1) == 3 && se3.size(2) == 4,
               "se3 must be of shape (N,4,4) or (N,3,4)");

    // Extract R and T if not provided
    torch::Tensor R_extracted = !R.defined() ? se3.slice(1, 0, 3).slice(2, 0, 3) : R;
    torch::Tensor T_extracted = !T.defined() ? se3.slice(1, 0, 3).slice(2, 3, 4) : T;

    // Transpose R
    torch::Tensor R_transposed = R_extracted.transpose(1, 2);
    torch::Tensor top_right = -torch::matmul(R_transposed, T_extracted);

    // Construct inverted matrix
    torch::Tensor inverted_matrix = torch::eye(4, torch::TensorOptions().dtype(se3.dtype()).device(se3.device()));
    inverted_matrix = inverted_matrix.unsqueeze(0).expand({se3.size(0), -1, -1});

    inverted_matrix.slice(1, 0, 3).slice(2, 0, 3) = R_transposed;
    inverted_matrix.slice(1, 0, 3).slice(2, 3, 4) = top_right;

    return inverted_matrix;
}

} // namespace utils
} // namespace vggt