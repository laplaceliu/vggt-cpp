/**
 * @file np_to_colmap.h
 * @brief Functions to convert camera parameters from torch::Tensor to colmap format
 */

#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <map>

namespace vggt {

/**
 * @brief Convert camera parameters from torch::Tensor to colmap format
 *
 * @param poses Camera poses tensor with shape [N, 3, 4] or [N, 4, 4]
 * @param intrinsics Camera intrinsics tensor with shape [N, 3, 3] or [N, 4]
 * @param image_size Image size tensor with shape [N, 2] or scalar value
 * @param camera_model Camera model name, default is "OPENCV"
 * @return std::map containing colmap format camera parameters:
 *   - cameras: Map of camera_id to camera parameters
 *   - images: Map of image_id to image parameters
 *   - points3D: Map of point3D_id to 3D point parameters (empty in this case)
 */
std::map<std::string, torch::Tensor> torch_to_colmap(
    torch::Tensor poses,
    torch::Tensor intrinsics,
    torch::Tensor image_size,
    const std::string& camera_model = "OPENCV"
);

/**
 * @brief Get camera model ID from camera model name
 *
 * @param camera_model Camera model name
 * @return int Camera model ID
 */
int get_camera_model_id(const std::string& camera_model);

/**
 * @brief Get camera model name from camera model ID
 *
 * @param camera_model_id Camera model ID
 * @return std::string Camera model name
 */
std::string get_camera_model_name(int camera_model_id);

/**
 * @brief Get number of parameters for a camera model
 *
 * @param camera_model Camera model name
 * @return int Number of parameters
 */
int get_camera_params_len(const std::string& camera_model);

/**
 * @brief Convert rotation matrix to quaternion
 *
 * @param R Rotation matrix with shape [3, 3]
 * @return torch::Tensor Quaternion with shape [4]
 */
torch::Tensor rotation_matrix_to_quaternion(torch::Tensor R);

/**
 * @brief Convert rotation matrix and translation vector to colmap format
 *
 * @param R Rotation matrix with shape [3, 3]
 * @param t Translation vector with shape [3]
 * @return std::pair<torch::Tensor, torch::Tensor> Quaternion and translation vector
 */
std::pair<torch::Tensor, torch::Tensor> convert_pose_to_colmap(torch::Tensor R, torch::Tensor t);

} // namespace vggt
