/**
 * @file np_to_colmap.cpp
 * @brief Implementation of functions for converting NumPy arrays to COLMAP format
 */

#include "np_to_colmap.h"
#include "projection.h"
#include <iostream>
#include <stdexcept>

namespace vggt {
namespace dependency {

// ColmapTrack implementation
ColmapTrack::ColmapTrack() {}

ColmapTrack::~ColmapTrack() {}

void ColmapTrack::add_element(int image_id, int point2D_idx) {
    elements.emplace_back(image_id, point2D_idx);
}

// ColmapPoint2D implementation
ColmapPoint2D::ColmapPoint2D(const Eigen::Vector2d& xy, int point3D_id)
    : xy(xy), point3D_id(point3D_id) {}

ColmapPoint2D::~ColmapPoint2D() {}

// ColmapImage implementation
ColmapImage::ColmapImage(int id, const std::string& name, int camera_id, 
                         const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation)
    : id(id), name(name), camera_id(camera_id), rotation(rotation), 
      translation(translation), registered(false) {}

ColmapImage::~ColmapImage() {}

void ColmapImage::set_points2D(const std::vector<ColmapPoint2D>& points2D) {
    this->points2D = points2D;
}

// ColmapCamera implementation
ColmapCamera::ColmapCamera(const std::string& model, int width, int height, 
                           const std::vector<double>& params, int camera_id)
    : camera_id(camera_id), model(model), width(width), height(height), params(params) {}

ColmapCamera::~ColmapCamera() {}

Eigen::Matrix3d ColmapCamera::calibration_matrix() const {
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    
    if (model == "SIMPLE_PINHOLE") {
        // params: [f, cx, cy]
        K(0, 0) = params[0]; // f
        K(1, 1) = params[0]; // f
        K(0, 2) = params[1]; // cx
        K(1, 2) = params[2]; // cy
    } else if (model == "PINHOLE") {
        // params: [fx, fy, cx, cy]
        K(0, 0) = params[0]; // fx
        K(1, 1) = params[1]; // fy
        K(0, 2) = params[2]; // cx
        K(1, 2) = params[3]; // cy
    } else if (model == "SIMPLE_RADIAL") {
        // params: [f, cx, cy, k]
        K(0, 0) = params[0]; // f
        K(1, 1) = params[0]; // f
        K(0, 2) = params[1]; // cx
        K(1, 2) = params[2]; // cy
        // Note: radial distortion parameter k is not part of the calibration matrix
    } else {
        throw std::runtime_error("Unsupported camera model: " + model);
    }
    
    return K;
}

// ColmapRigid3d implementation
ColmapRigid3d::ColmapRigid3d(const Eigen::Matrix3d& rotation, const Eigen::Vector3d& translation)
    : rotation(rotation), translation(translation) {}

ColmapRigid3d::~ColmapRigid3d() {}

Eigen::Matrix<double, 3, 4> ColmapRigid3d::matrix() const {
    Eigen::Matrix<double, 3, 4> result;
    result.block<3, 3>(0, 0) = rotation;
    result.block<3, 1>(0, 3) = translation;
    return result;
}

// ColmapReconstruction implementation
ColmapReconstruction::ColmapReconstruction()
    : next_point3D_id_(1), next_camera_id_(1), next_image_id_(1) {}

ColmapReconstruction::~ColmapReconstruction() {}

int ColmapReconstruction::add_point3D(const Eigen::Vector3d& xyz, const ColmapTrack& track, const Eigen::Vector3i& rgb) {
    int point3D_id = next_point3D_id_++;
    auto point3D = std::make_shared<Point3D>();
    point3D->xyz = xyz;
    point3D->rgb = rgb;
    point3D->track = track;
    points3D[point3D_id] = point3D;
    return point3D_id;
}

int ColmapReconstruction::add_camera(const ColmapCamera& camera) {
    int camera_id = camera.camera_id;
    if (camera_id <= 0) {
        camera_id = next_camera_id_++;
    }
    cameras[camera_id] = std::make_shared<ColmapCamera>(camera);
    return camera_id;
}

int ColmapReconstruction::add_image(const ColmapImage& image) {
    int image_id = image.id;
    if (image_id <= 0) {
        image_id = next_image_id_++;
    }
    images[image_id] = std::make_shared<ColmapImage>(image);
    return image_id;
}

std::vector<int> ColmapReconstruction::point3D_ids() const {
    std::vector<int> ids;
    ids.reserve(points3D.size());
    for (const auto& pair : points3D) {
        ids.push_back(pair.first);
    }
    return ids;
}

std::vector<double> build_colmap_intri(
    int fidx,
    const torch::Tensor& intrinsics,
    const std::string& camera_type,
    const c10::optional<torch::Tensor>& extra_params
) {
    std::vector<double> colmap_intri;
    
    if (camera_type == "PINHOLE") {
        colmap_intri = {
            intrinsics[fidx][0][0].item<double>(), // fx
            intrinsics[fidx][1][1].item<double>(), // fy
            intrinsics[fidx][0][2].item<double>(), // cx
            intrinsics[fidx][1][2].item<double>()  // cy
        };
    } else if (camera_type == "SIMPLE_PINHOLE") {
        double focal = (intrinsics[fidx][0][0].item<double>() + intrinsics[fidx][1][1].item<double>()) / 2.0;
        colmap_intri = {
            focal,
            intrinsics[fidx][0][2].item<double>(), // cx
            intrinsics[fidx][1][2].item<double>()  // cy
        };
    } else if (camera_type == "SIMPLE_RADIAL") {
        if (!extra_params.has_value()) {
            throw std::runtime_error("extra_params must be provided for SIMPLE_RADIAL camera model");
        }
        double focal = (intrinsics[fidx][0][0].item<double>() + intrinsics[fidx][1][1].item<double>()) / 2.0;
        colmap_intri = {
            focal,
            intrinsics[fidx][0][2].item<double>(), // cx
            intrinsics[fidx][1][2].item<double>(), // cy
            extra_params.value()[fidx][0].item<double>() // k
        };
    } else {
        throw std::runtime_error("Camera type " + camera_type + " is not supported yet");
    }
    
    return colmap_intri;
}

std::tuple<torch::Tensor, torch::Tensor> project_3D_points_np(
    const torch::Tensor& points3d,
    const torch::Tensor& extrinsics,
    const torch::Tensor& intrinsics
) {
    // This is a wrapper around project_3D_points that returns NumPy-compatible tensors
    auto result = project_3D_points(points3d, extrinsics, intrinsics);
    
    if (!std::get<0>(result).has_value()) {
        throw std::runtime_error("project_3D_points returned None for points2D");
    }
    
    return std::make_tuple(std::get<0>(result).value(), std::get<1>(result));
}

std::tuple<std::shared_ptr<ColmapReconstruction>, torch::Tensor> batch_np_matrix_to_colmap(
    const torch::Tensor& points3d,
    const torch::Tensor& extrinsics,
    const torch::Tensor& intrinsics,
    const torch::Tensor& tracks,
    const torch::Tensor& image_size,
    const c10::optional<torch::Tensor>& masks,
    const c10::optional<float>& max_reproj_error,
    const c10::optional<float>& max_points3D_val,
    bool shared_camera,
    const std::string& camera_type,
    const c10::optional<torch::Tensor>& extra_params,
    int min_inlier_per_frame,
    const c10::optional<torch::Tensor>& points_rgb
) {
    // Get dimensions
    int64_t N = tracks.size(0);  // Number of frames
    int64_t P = tracks.size(1);  // Number of tracks
    
    // Validate input dimensions
    TORCH_CHECK(extrinsics.size(0) == N, "extrinsics must have N frames");
    TORCH_CHECK(intrinsics.size(0) == N, "intrinsics must have N frames");
    TORCH_CHECK(points3d.size(0) == P, "points3d must have P points");
    TORCH_CHECK(image_size.size(0) == 2, "image_size must have 2 elements");
    
    // Create masks if needed
    torch::Tensor final_masks;
    if (max_reproj_error.has_value()) {
        auto [projected_points_2d, projected_points_cam] = project_3D_points_np(points3d, extrinsics, intrinsics);
        
        // Calculate reprojection error
        auto projected_diff = torch::norm(projected_points_2d - tracks, 2, -1);
        
        // Set points behind camera to have large error
        auto behind_camera = projected_points_cam.index({Ellipsis, 2}) <= 0;
        projected_points_2d.index_put_({behind_camera}, 1e6);
        
        // Create reprojection mask
        auto reproj_mask = projected_diff < max_reproj_error.value();
        
        if (masks.has_value()) {
            final_masks = masks.value() & reproj_mask;
        } else {
            final_masks = reproj_mask;
        }
    } else if (masks.has_value()) {
        final_masks = masks.value();
    } else {
        throw std::runtime_error("Either masks or max_reproj_error must be provided");
    }
    
    // Check if there are enough inliers per frame
    auto inliers_per_frame = final_masks.sum(1);
    if (inliers_per_frame.min().item<int64_t>() < min_inlier_per_frame) {
        std::cout << "Not enough inliers per frame, skip BA." << std::endl;
        return std::make_tuple(nullptr, torch::Tensor());
    }
    
    // Create reconstruction object
    auto reconstruction = std::make_shared<ColmapReconstruction>();
    
    // Count inliers per track
    auto inlier_num = final_masks.sum(0);
    auto valid_mask = inlier_num >= 2;  // A track is valid if it has at least 2 inliers
    auto valid_idx = torch::nonzero(valid_mask).squeeze();
    
    // Add 3D points with sufficient 2D points
    for (int64_t i = 0; i < valid_idx.size(0); i++) {
        int64_t vidx = valid_idx[i].item<int64_t>();
        
        // Use RGB colors if provided, otherwise use zeros
        Eigen::Vector3i rgb;
        if (points_rgb.has_value()) {
            rgb = Eigen::Vector3i(
                points_rgb.value()[vidx][0].item<int>(),
                points_rgb.value()[vidx][1].item<int>(),
                points_rgb.value()[vidx][2].item<int>()
            );
        } else {
            rgb = Eigen::Vector3i(0, 0, 0);
        }
        
        // Convert points3d to Eigen vector
        Eigen::Vector3d xyz(
            points3d[vidx][0].item<double>(),
            points3d[vidx][1].item<double>(),
            points3d[vidx][2].item<double>()
        );
        
        // Add point to reconstruction
        ColmapTrack track;
        reconstruction->add_point3D(xyz, track, rgb);
    }
    
    int64_t num_points3D = valid_idx.size(0);
    std::shared_ptr<ColmapCamera> camera = nullptr;
    
    // Process each frame
    for (int64_t fidx = 0; fidx < N; fidx++) {
        // Set camera
        if (camera == nullptr || !shared_camera) {
            auto colmap_intri = build_colmap_intri(fidx, intrinsics, camera_type, extra_params);
            
            camera = std::make_shared<ColmapCamera>(
                camera_type,
                image_size[0].item<int>(),
                image_size[1].item<int>(),
                colmap_intri,
                fidx + 1
            );
            
            // Add camera to reconstruction
            reconstruction->add_camera(*camera);
        }
        
        // Set image
        Eigen::Matrix3d rotation;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                rotation(i, j) = extrinsics[fidx][i][j].item<double>();
            }
        }
        
        Eigen::Vector3d translation(
            extrinsics[fidx][0][3].item<double>(),
            extrinsics[fidx][1][3].item<double>(),
            extrinsics[fidx][2][3].item<double>()
        );
        
        ColmapImage image(
            fidx + 1,
            "image_" + std::to_string(fidx + 1),
            camera->camera_id,
            rotation,
            translation
        );
        
        std::vector<ColmapPoint2D> points2D_list;
        int point2D_idx = 0;
        
        // Add points to image
        for (int64_t point3D_id = 1; point3D_id <= num_points3D; point3D_id++) {
            int64_t original_track_idx = valid_idx[point3D_id - 1].item<int64_t>();
            
            // Check if point is valid and visible in this frame
            if (final_masks[fidx][original_track_idx].item<bool>()) {
                // Check if point is within bounds
                if (!max_points3D_val.has_value() || 
                    (std::abs(points3d[original_track_idx][0].item<double>()) < max_points3D_val.value() &&
                     std::abs(points3d[original_track_idx][1].item<double>()) < max_points3D_val.value() &&
                     std::abs(points3d[original_track_idx][2].item<double>()) < max_points3D_val.value())) {
                    
                    // Get 2D point coordinates
                    Eigen::Vector2d point2D_xy(
                        tracks[fidx][original_track_idx][0].item<double>(),
                        tracks[fidx][original_track_idx][1].item<double>()
                    );
                    
                    // Add point to list
                    points2D_list.emplace_back(point2D_xy, point3D_id);
                    
                    // Add element to track
                    auto& track = reconstruction->points3D[point3D_id]->track;
                    track.add_element(fidx + 1, point2D_idx);
                    point2D_idx++;
                }
            }
        }
        
        // Set points2D for image
        try {
            image.set_points2D(points2D_list);
            image.registered = true;
        } catch (const std::exception& e) {
            std::cout << "Frame " << (fidx + 1) << " is out of BA: " << e.what() << std::endl;
            image.registered = false;
        }
        
        // Add image to reconstruction
        reconstruction->add_image(image);
    }
    
    return std::make_tuple(reconstruction, valid_mask);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>> colmap_to_batch_np_matrix(
    const ColmapReconstruction& reconstruction,
    const std::string& camera_type
) {
    // Get dimensions
    int64_t num_images = reconstruction.images.size();
    
    // Find maximum point3D ID
    int max_point3D_id = 0;
    for (const auto& id : reconstruction.point3D_ids()) {
        max_point3D_id = std::max(max_point3D_id, id);
    }
    
    // Create points3D tensor
    auto points3D = torch::zeros({max_point3D_id, 3}, torch::kDouble);
    
    // Fill points3D tensor
    for (const auto& pair : reconstruction.points3D) {
        int point3D_id = pair.first;
        const auto& point = pair.second;
        
        points3D[point3D_id - 1][0] = point->xyz.x();
        points3D[point3D_id - 1][1] = point->xyz.y();
        points3D[point3D_id - 1][2] = point->xyz.z();
    }
    
    // Create extrinsics and intrinsics tensors
    std::vector<torch::Tensor> extrinsics_list;
    std::vector<torch::Tensor> intrinsics_list;
    std::vector<double> extra_params_list;
    bool has_extra_params = (camera_type == "SIMPLE_RADIAL");
    
    // Fill extrinsics and intrinsics tensors
    for (int i = 1; i <= num_images; i++) {
        // Get image and camera
        const auto& image = reconstruction.images.at(i);
        const auto& camera = reconstruction.cameras.at(image->camera_id);
        
        // Create extrinsic matrix
        auto rotation = torch::zeros({3, 3}, torch::kDouble);
        auto translation = torch::zeros({3, 1}, torch::kDouble);
        
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                rotation[r][c] = image->rotation(r, c);
            }
            translation[r][0] = image->translation(r);
        }
        
        auto extrinsic = torch::cat({rotation, translation}, 1);
        extrinsics_list.push_back(extrinsic);
        
        // Create intrinsic matrix
        auto calibration_matrix = camera->calibration_matrix();
        auto intrinsic = torch::zeros({3, 3}, torch::kDouble);
        
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                intrinsic[r][c] = calibration_matrix(r, c);
            }
        }
        
        intrinsics_list.push_back(intrinsic);
        
        // Extract extra parameters if needed
        if (has_extra_params) {
            if (camera->model == "SIMPLE_RADIAL") {
                extra_params_list.push_back(camera->params[3]);  // k
            } else {
                throw std::runtime_error("Camera model " + camera->model + 
                                         " is not compatible with camera_type " + camera_type);
            }
        }
    }
    
    // Stack tensors
    auto extrinsics = torch::stack(extrinsics_list);
    auto intrinsics = torch::stack(intrinsics_list);
    
    // Create extra_params tensor if needed
    c10::optional<torch::Tensor> extra_params = c10::nullopt;
    if (has_extra_params) {
        auto extra_params_tensor = torch::zeros({num_images, 1}, torch::kDouble);
        for (int i = 0; i < num_images; i++) {
            extra_params_tensor[i][0] = extra_params_list[i];
        }
        extra_params = extra_params_tensor;
    }
    
    return std::make_tuple(points3D, extrinsics, intrinsics, extra_params);
}

std::shared_ptr<ColmapReconstruction> batch_np_matrix_to_colmap_wo_track(
    const torch::Tensor& points3d,
    const torch::Tensor& points_xyf,
    const torch::Tensor& points_rgb,
    const torch::Tensor& extrinsics,
    const torch::Tensor& intrinsics,
    const torch::Tensor& image_size,
    bool shared_camera,
    const std::string& camera_type
) {
    // Get dimensions
    int64_t N = extrinsics.size(0);  // Number of frames
    int64_t P = points3d.size(0);    // Number of points
    
    // Create reconstruction object
    auto reconstruction = std::make_shared<ColmapReconstruction>();
    
    // Add all 3D points to reconstruction
    for (int64_t vidx = 0; vidx < P; vidx++) {
        // Convert points3d to Eigen vector
        Eigen::Vector3d xyz(
            points3d[vidx][0].item<double>(),
            points3d[vidx][1].item<double>(),
            points3d[vidx][2].item<double>()
        );
        
        // Convert RGB to Eigen vector
        Eigen::Vector3i rgb(
            points_rgb[vidx][0].item<int>(),
            points_rgb[vidx][1].item<int>(),
            points_rgb[vidx][2].item<int>()
        );
        
        // Add point to reconstruction
        ColmapTrack track;
        reconstruction->add_point3D(xyz, track, rgb);
    }
    
    std::shared_ptr<ColmapCamera> camera = nullptr;
    
    // Process each frame
    for (int64_t fidx = 0; fidx < N; fidx++) {
        // Set camera
        if (camera == nullptr || !shared_camera) {
            auto colmap_intri = build_colmap_intri(fidx, intrinsics, camera_type);
            
            camera = std::make_shared<ColmapCamera>(
                camera_type,
                image_size[0].item<int>(),
                image_size[1].item<int>(),
                colmap_intri,
                fidx + 1
            );
            
            // Add camera to reconstruction
            reconstruction->add_camera(*camera);
        }
        
        // Set image
        Eigen::Matrix3d rotation;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                rotation(i, j) = extrinsics[fidx][i][j].item<double>();
            }
        }
        
        Eigen::Vector3d translation(
            extrinsics[fidx][0][3].item<double>(),
            extrinsics[fidx][1][3].item<double>(),
            extrinsics[fidx][2][3].item<double>()
        );
        
        ColmapImage image(
            fidx + 1,
            "image_" + std::to_string(fidx + 1),
            camera->camera_id,
            rotation,
            translation
        );
        
        std::vector<ColmapPoint2D> points2D_list;
        int point2D_idx = 0;
        
        // Find points that belong to this frame
        auto points_belong_to_fidx = points_xyf.index({Ellipsis, 2}).to(torch::kInt32) == fidx;
        auto points_indices = torch::nonzero(points_belong_to_fidx).squeeze();
        
        // Add points to image
        for (int64_t i = 0; i < points_indices.size(0); i++) {
            int64_t point3D_batch_idx = points_indices[i].item<int64_t>();
            int64_t point3D_id = point3D_batch_idx + 1;
            
            // Get 2D point coordinates
            Eigen::Vector2d point2D_xy(
                points_xyf[point3D_batch_idx][0].item<double>(),
                points_xyf[point3D_batch_idx][1].item<double>()
            );
            
            // Add point to list
            points2D_list.emplace_back(point2D_xy, point3D_id);
            
            // Add element to track
            auto& track = reconstruction->points3D[point3D_id]->track;
            track.add_element(fidx + 1, point2D_idx);
            point2D_idx++;
        }
        
        // Set points2D for image
        try {
            image.set_points2D(points2D_list);
            image.registered = true;
        } catch (const std::exception& e) {
            std::cout << "Frame " << (fidx + 1) << " does not have any points: " << e.what() << std::endl;
            image.registered = false;
        }
        
        // Add image to reconstruction
        reconstruction->add_image(image);
    }
    
    return reconstruction;
}

} // namespace dependency
} // namespace vggt