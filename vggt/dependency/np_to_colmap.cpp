/**
 * @file np_to_colmap.cpp
 * @brief Implementation of functions to convert camera parameters from torch::Tensor to colmap format
 */

#include "np_to_colmap.h"
#include "../utils/rotation.h"
#include <unordered_map>
#include <stdexcept>

namespace vggt {

// Camera model constants
const std::unordered_map<std::string, int> CAMERA_MODEL_IDS = {
    {"SIMPLE_PINHOLE", 0},
    {"PINHOLE", 1},
    {"SIMPLE_RADIAL", 2},
    {"RADIAL", 3},
    {"OPENCV", 4},
    {"OPENCV_FISHEYE", 5},
    {"FULL_OPENCV", 6},
    {"FOV", 7},
    {"SIMPLE_RADIAL_FISHEYE", 8},
    {"RADIAL_FISHEYE", 9},
    {"THIN_PRISM_FISHEYE", 10}
};

const std::unordered_map<int, std::string> CAMERA_MODEL_NAMES = {
    {0, "SIMPLE_PINHOLE"},
    {1, "PINHOLE"},
    {2, "SIMPLE_RADIAL"},
    {3, "RADIAL"},
    {4, "OPENCV"},
    {5, "OPENCV_FISHEYE"},
    {6, "FULL_OPENCV"},
    {7, "FOV"},
    {8, "SIMPLE_RADIAL_FISHEYE"},
    {9, "RADIAL_FISHEYE"},
    {10, "THIN_PRISM_FISHEYE"}
};

const std::unordered_map<std::string, int> CAMERA_PARAMS_LEN = {
    {"SIMPLE_PINHOLE", 3},
    {"PINHOLE", 4},
    {"SIMPLE_RADIAL", 4},
    {"RADIAL", 5},
    {"OPENCV", 8},
    {"OPENCV_FISHEYE", 8},
    {"FULL_OPENCV", 12},
    {"FOV", 5},
    {"SIMPLE_RADIAL_FISHEYE", 4},
    {"RADIAL_FISHEYE", 5},
    {"THIN_PRISM_FISHEYE", 12}
};

int get_camera_model_id(const std::string& camera_model) {
    auto it = CAMERA_MODEL_IDS.find(camera_model);
    if (it == CAMERA_MODEL_IDS.end()) {
        throw std::invalid_argument("Invalid camera model: " + camera_model);
    }
    return it->second;
}

std::string get_camera_model_name(int camera_model_id) {
    auto it = CAMERA_MODEL_NAMES.find(camera_model_id);
    if (it == CAMERA_MODEL_NAMES.end()) {
        throw std::invalid_argument("Invalid camera model ID: " + std::to_string(camera_model_id));
    }
    return it->second;
}

int get_camera_params_len(const std::string& camera_model) {
    auto it = CAMERA_PARAMS_LEN.find(camera_model);
    if (it == CAMERA_PARAMS_LEN.end()) {
        throw std::invalid_argument("Invalid camera model: " + camera_model);
    }
    return it->second;
}

torch::Tensor rotation_matrix_to_quaternion(torch::Tensor R) {
    // Use the existing mat_to_quat function from rotation.cpp
    return vggt::utils::mat_to_quat(R);
}

std::pair<torch::Tensor, torch::Tensor> convert_pose_to_colmap(torch::Tensor R, torch::Tensor t) {
    // Convert rotation matrix to quaternion (w, x, y, z)
    torch::Tensor qvec = rotation_matrix_to_quaternion(R);

    // In colmap, the camera-to-world rotation is stored
    // We need to invert the rotation and translation
    torch::Tensor R_inv = R.transpose(-1, -2);
    torch::Tensor t_inv = -torch::matmul(R_inv, t.unsqueeze(-1)).squeeze(-1);

    // Convert inverted rotation to quaternion
    torch::Tensor qvec_inv = rotation_matrix_to_quaternion(R_inv);

    return std::make_pair(qvec_inv, t_inv);
}

std::map<std::string, torch::Tensor> torch_to_colmap(
    torch::Tensor poses,
    torch::Tensor intrinsics,
    torch::Tensor image_size,
    const std::string& camera_model
) {
    // Check inputs
    if (poses.dim() != 3 || (poses.size(1) != 3 && poses.size(1) != 4) || (poses.size(2) != 4)) {
        throw std::invalid_argument("poses must have shape [N, 3, 4] or [N, 4, 4]");
    }

    int num_images = poses.size(0);

    // Handle intrinsics
    torch::Tensor params;
    if (intrinsics.dim() == 3) {  // [N, 3, 3]
        if (intrinsics.size(1) != 3 || intrinsics.size(2) != 3) {
            throw std::invalid_argument("intrinsics must have shape [N, 3, 3] or [N, 4]");
        }

        // Extract focal length and principal point
        torch::Tensor fx = intrinsics.index({Ellipsis, 0, 0});
        torch::Tensor fy = intrinsics.index({Ellipsis, 1, 1});
        torch::Tensor cx = intrinsics.index({Ellipsis, 0, 2});
        torch::Tensor cy = intrinsics.index({Ellipsis, 1, 2});

        if (camera_model == "SIMPLE_PINHOLE") {
            // f, cx, cy
            params = torch::stack({fx, cx, cy}, -1);
        } else if (camera_model == "PINHOLE") {
            // fx, fy, cx, cy
            params = torch::stack({fx, fy, cx, cy}, -1);
        } else if (camera_model == "OPENCV") {
            // fx, fy, cx, cy, k1, k2, p1, p2
            torch::Tensor zeros = torch::zeros_like(fx);
            params = torch::stack({fx, fy, cx, cy, zeros, zeros, zeros, zeros}, -1);
        } else {
            throw std::invalid_argument("Unsupported camera model for intrinsics matrix: " + camera_model);
        }
    } else if (intrinsics.dim() == 2) {  // [N, 4]
        if (intrinsics.size(1) != 4) {
            throw std::invalid_argument("intrinsics must have shape [N, 3, 3] or [N, 4]");
        }

        // Extract parameters
        torch::Tensor fx = intrinsics.index({Ellipsis, 0});
        torch::Tensor fy = intrinsics.index({Ellipsis, 1});
        torch::Tensor cx = intrinsics.index({Ellipsis, 2});
        torch::Tensor cy = intrinsics.index({Ellipsis, 3});

        if (camera_model == "SIMPLE_PINHOLE") {
            // f, cx, cy
            params = torch::stack({fx, cx, cy}, -1);
        } else if (camera_model == "PINHOLE") {
            // fx, fy, cx, cy
            params = torch::stack({fx, fy, cx, cy}, -1);
        } else if (camera_model == "OPENCV") {
            // fx, fy, cx, cy, k1, k2, p1, p2
            torch::Tensor zeros = torch::zeros_like(fx);
            params = torch::stack({fx, fy, cx, cy, zeros, zeros, zeros, zeros}, -1);
        } else {
            throw std::invalid_argument("Unsupported camera model for intrinsics vector: " + camera_model);
        }
    } else {
        throw std::invalid_argument("intrinsics must have shape [N, 3, 3] or [N, 4]");
    }

    // Handle image size
    torch::Tensor width, height;
    if (image_size.dim() == 0) {  // scalar
        width = image_size.expand(num_images);
        height = image_size.expand(num_images);
    } else if (image_size.dim() == 1 && image_size.size(0) == 2) {  // [2]
        width = image_size[0].expand(num_images);
        height = image_size[1].expand(num_images);
    } else if (image_size.dim() == 2 && image_size.size(1) == 2) {  // [N, 2]
        width = image_size.index({Ellipsis, 0});
        height = image_size.index({Ellipsis, 1});
    } else {
        throw std::invalid_argument("image_size must be a scalar, [2], or [N, 2]");
    }

    // Create cameras tensor
    int camera_model_id = get_camera_model_id(camera_model);
    torch::Tensor camera_ids = torch::arange(1, num_images + 1, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor model_ids = torch::full_like(camera_ids, camera_model_id);
    torch::Tensor cameras = torch::cat({
        camera_ids.unsqueeze(1),
        model_ids.unsqueeze(1),
        width.unsqueeze(1),
        height.unsqueeze(1),
        params
    }, 1);

    // Create images tensor
    torch::Tensor image_ids = torch::arange(1, num_images + 1, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor qvecs = torch::zeros({num_images, 4}, torch::TensorOptions().dtype(poses.dtype()).device(poses.device()));
    torch::Tensor tvecs = torch::zeros({num_images, 3}, torch::TensorOptions().dtype(poses.dtype()).device(poses.device()));

    // Extract rotation and translation from poses
    for (int i = 0; i < num_images; ++i) {
        torch::Tensor R = poses[i].slice(0, 0, 3).slice(1, 0, 3);
        torch::Tensor t = poses[i].slice(0, 0, 3).slice(1, 3, 4).squeeze(1);

        auto [qvec, tvec] = convert_pose_to_colmap(R, t);
        qvecs[i] = qvec;
        tvecs[i] = tvec;
    }

    // Create images tensor with camera_id, qvec, tvec
    torch::Tensor images = torch::cat({
        image_ids.unsqueeze(1),
        camera_ids.unsqueeze(1),
        qvecs,
        tvecs
    }, 1);

    // Create empty points3D tensor
    torch::Tensor points3D = torch::zeros({0, 7}, torch::TensorOptions().dtype(poses.dtype()).device(poses.device()));

    // Return result
    std::map<std::string, torch::Tensor> result;
    result["cameras"] = cameras;
    result["images"] = images;
    result["points3D"] = points3D;

    return result;
}

} // namespace vggt
