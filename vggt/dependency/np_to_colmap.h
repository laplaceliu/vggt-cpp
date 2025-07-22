/**
 * @file np_to_colmap.h
 * @brief Functions for converting NumPy arrays to COLMAP format
 * 
 * This file contains functions for converting batched NumPy arrays to COLMAP format.
 * It is a C++ port of the Python implementation in np_to_pycolmap.py.
 */

#pragma once

#include <torch/torch.h>
#include <c10/util/Optional.h>
#include <Eigen/Dense>
#include <PoseLib/misc/colmap_models.h>
#include <vector>
#include <string>
#include <tuple>
#include <memory>

namespace vggt {
namespace dependency {

/**
 * @brief Forward declaration of COLMAP reconstruction class
 */
class ColmapReconstruction;

/**
 * @brief Forward declaration of COLMAP track class
 */
class ColmapTrack;

/**
 * @brief Forward declaration of COLMAP point2D class
 */
class ColmapPoint2D;

/**
 * @brief Forward declaration of COLMAP image class
 */
class ColmapImage;

/**
 * @brief Forward declaration of COLMAP camera class
 */
class ColmapCamera;

/**
 * @brief Convert batched NumPy arrays to COLMAP format
 * 
 * @param points3d (P,3) 3D points
 * @param extrinsics (N,3,4) Extrinsic parameters [R|t]
 * @param intrinsics (N,3,3) Intrinsic parameters K
 * @param tracks (N,P,2) 2D tracks
 * @param image_size (2) Image size (width, height)
 * @param masks (N,P) Optional masks for valid tracks
 * @param max_reproj_error Optional maximum reprojection error
 * @param max_points3D_val Optional maximum value for 3D points
 * @param shared_camera Whether to share camera parameters
 * @param camera_type Camera model type
 * @param extra_params Optional extra parameters for camera model
 * @param min_inlier_per_frame Minimum number of inliers per frame
 * @param points_rgb Optional RGB colors for 3D points
 * @return std::tuple<std::shared_ptr<ColmapReconstruction>, torch::Tensor> Reconstruction and valid mask
 */
std::tuple<std::shared_ptr<ColmapReconstruction>, torch::Tensor> batch_np_matrix_to_colmap(
    const torch::Tensor& points3d,
    const torch::Tensor& extrinsics,
    const torch::Tensor& intrinsics,
    const torch::Tensor& tracks,
    const torch::Tensor& image_size,
    const c10::optional<torch::Tensor>& masks = c10::nullopt,
    const c10::optional<float>& max_reproj_error = c10::nullopt,
    const c10::optional<float>& max_points3D_val = c10::optional<float>(3000.0),
    bool shared_camera = false,
    const std::string& camera_type = "SIMPLE_PINHOLE",
    const c10::optional<torch::Tensor>& extra_params = c10::nullopt,
    int min_inlier_per_frame = 64,
    const c10::optional<torch::Tensor>& points_rgb = c10::nullopt
);

/**
 * @brief Convert COLMAP reconstruction to batched NumPy arrays
 * 
 * @param reconstruction COLMAP reconstruction object
 * @param camera_type Camera model type
 * @return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>> 
 *         points3D, extrinsics, intrinsics, extra_params
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>> colmap_to_batch_np_matrix(
    const ColmapReconstruction& reconstruction,
    const std::string& camera_type = "SIMPLE_PINHOLE"
);

/**
 * @brief Convert batched NumPy arrays to COLMAP format without tracks
 * 
 * @param points3d (P,3) 3D points
 * @param points_xyf (P,3) 2D points with frame indices
 * @param points_rgb (P,3) RGB colors for 3D points
 * @param extrinsics (N,3,4) Extrinsic parameters [R|t]
 * @param intrinsics (N,3,3) Intrinsic parameters K
 * @param image_size (2) Image size (width, height)
 * @param shared_camera Whether to share camera parameters
 * @param camera_type Camera model type
 * @return std::shared_ptr<ColmapReconstruction> COLMAP reconstruction
 */
std::shared_ptr<ColmapReconstruction> batch_np_matrix_to_colmap_wo_track(
    const torch::Tensor& points3d,
    const torch::Tensor& points_xyf,
    const torch::Tensor& points_rgb,
    const torch::Tensor& extrinsics,
    const torch::Tensor& intrinsics,
    const torch::Tensor& image_size,
    bool shared_camera = false,
    const std::string& camera_type = "SIMPLE_PINHOLE"
);

/**
 * @brief Helper function to build COLMAP intrinsic parameters
 * 
 * @param fidx Frame index
 * @param intrinsics (N,3,3) Intrinsic parameters K
 * @param camera_type Camera model type
 * @param extra_params Optional extra parameters for camera model
 * @return std::vector<double> COLMAP intrinsic parameters
 */
std::vector<double> build_colmap_intri(
    int fidx,
    const torch::Tensor& intrinsics,
    const std::string& camera_type,
    const c10::optional<torch::Tensor>& extra_params = c10::nullopt
);

/**
 * @brief Project 3D points to 2D using NumPy arrays
 * 
 * @param points3d (P,3) 3D points
 * @param extrinsics (N,3,4) Extrinsic parameters [R|t]
 * @param intrinsics (N,3,3) Intrinsic parameters K
 * @return std::tuple<torch::Tensor, torch::Tensor> (points2D, points_cam)
 */
std::tuple<torch::Tensor, torch::Tensor> project_3D_points_np(
    const torch::Tensor& points3d,
    const torch::Tensor& extrinsics,
    const torch::Tensor& intrinsics
);

/**
 * @brief COLMAP reconstruction class
 */
class ColmapReconstruction {
public:
    ColmapReconstruction();
    ~ColmapReconstruction();

    /**
     * @brief Add a 3D point to the reconstruction
     * 
     * @param xyz 3D point coordinates
     * @param track Track object
     * @param rgb RGB color
     * @return int Point ID
     */
    int add_point3D(const Eigen::Vector3d& xyz, const ColmapTrack& track, const Eigen::Vector3i& rgb);

    /**
     * @brief Add a camera to the reconstruction
     * 
     * @param camera Camera object
     * @return int Camera ID
     */
    int add_camera(const ColmapCamera& camera);

    /**
     * @brief Add an image to the reconstruction
     * 
     * @param image Image object
     * @return int Image ID
     */
    int add_image(const ColmapImage& image);

    /**
     * @brief Get all point3D IDs
     * 
     * @return std::vector<int> Point3D IDs
     */
    std::vector<int> point3D_ids() const;

    // Maps to store reconstruction data
    std::map<int, std::shared_ptr<ColmapCamera>> cameras;
    std::map<int, std::shared_ptr<ColmapImage>> images;
    std::map<int, std::shared_ptr<struct Point3D>> points3D;

private:
    int next_point3D_id_;
    int next_camera_id_;
    int next_image_id_;
};

/**
 * @brief COLMAP track class
 */
class ColmapTrack {
public:
    ColmapTrack();
    ~ColmapTrack();

    /**
     * @brief Add an element to the track
     * 
     * @param image_id Image ID
     * @param point2D_idx Point2D index
     */
    void add_element(int image_id, int point2D_idx);

    // Track elements
    std::vector<std::pair<int, int>> elements;
};

/**
 * @brief COLMAP point2D class
 */
class ColmapPoint2D {
public:
    ColmapPoint2D(const Eigen::Vector2d& xy, int point3D_id = -1);
    ~ColmapPoint2D();

    // Point2D data
    Eigen::Vector2d xy;
    int point3D_id;
};

/**
 * @brief COLMAP image class
 */
class ColmapImage {
public:
    ColmapImage(int id, const std::string& name, int camera_id, const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation);
    ~ColmapImage();

    /**
     * @brief Set points2D for the image
     * 
     * @param points2D Vector of Point2D objects
     */
    void set_points2D(const std::vector<ColmapPoint2D>& points2D);

    // Image data
    int id;
    std::string name;
    int camera_id;
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
    std::vector<ColmapPoint2D> points2D;
    bool registered;
};

/**
 * @brief COLMAP camera class
 */
class ColmapCamera {
public:
    ColmapCamera(const std::string& model, int width, int height, const std::vector<double>& params, int camera_id);
    ~ColmapCamera();

    /**
     * @brief Get the calibration matrix
     * 
     * @return Eigen::Matrix3d Calibration matrix
     */
    Eigen::Matrix3d calibration_matrix() const;

    // Camera data
    int camera_id;
    std::string model;
    int width;
    int height;
    std::vector<double> params;
};

/**
 * @brief COLMAP 3D point structure
 */
struct Point3D {
    Eigen::Vector3d xyz;
    Eigen::Vector3i rgb;
    ColmapTrack track;
};

/**
 * @brief COLMAP rigid 3D transformation
 */
class ColmapRigid3d {
public:
    ColmapRigid3d(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation);
    ~ColmapRigid3d();

    /**
     * @brief Get the transformation matrix
     * 
     * @return Eigen::Matrix<double, 3, 4> Transformation matrix [R|t]
     */
    Eigen::Matrix<double, 3, 4> matrix() const;

    // Rigid transformation data
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
};

} // namespace dependency
} // namespace vggt