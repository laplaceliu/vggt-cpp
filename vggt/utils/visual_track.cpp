#include "visual_track.h"
#include <matplot/matplot.h>
#include <filesystem>
#include <iostream>

namespace vggt {
namespace utils {

std::tuple<double, double, double> color_from_xy(double x, double y, int W, int H, const std::string& cmap_name) {
    double x_norm = x / std::max(W - 1, 1);
    double y_norm = y / std::max(H - 1, 1);
    
    // Simple combination:
    double c = (x_norm + y_norm) / 2.0;
    
    // Use matplotplusplus to get color from colormap
    // Get the default colormap
    auto cmap = matplot::colormap();
    
    // Select color based on normalized value c
    int idx = static_cast<int>(c * (cmap.size() - 1));
    idx = std::max(0, std::min(static_cast<int>(cmap.size()) - 1, idx));
    
    // Return RGB values in [0,1]
    return std::make_tuple(cmap[idx][0], cmap[idx][1], cmap[idx][2]);
}

cv::Mat get_track_colors_by_position(
    const torch::Tensor& tracks_b,
    const torch::Tensor& vis_mask_b,
    int image_width,
    int image_height,
    const std::string& cmap_name) {
    
    auto S = tracks_b.size(0);
    auto N = tracks_b.size(1);
    
    // Create output matrix for track colors
    cv::Mat track_colors(N, 3, CV_8UC1, cv::Scalar(0));
    
    // Create visibility mask if not provided
    torch::Tensor vis_mask;
    if (!vis_mask_b.defined()) {
        vis_mask = torch::ones({S, N}, torch::kBool).to(tracks_b.device());
    } else {
        vis_mask = vis_mask_b;
    }
    
    // Process each track
    for (int64_t i = 0; i < N; ++i) {
        // Find first visible frame for track i
        std::vector<int64_t> visible_frames;
        for (int64_t s = 0; s < S; ++s) {
            if (vis_mask[s][i].item<bool>()) {
                visible_frames.push_back(s);
            }
        }
        
        if (visible_frames.empty()) {
            // Track is never visible; assign black
            track_colors.at<uchar>(i, 0) = 0;
            track_colors.at<uchar>(i, 1) = 0;
            track_colors.at<uchar>(i, 2) = 0;
            continue;
        }
        
        int64_t first_s = visible_frames[0];
        // Use that frame's (x,y)
        double x = tracks_b[first_s][i][0].item<double>();
        double y = tracks_b[first_s][i][1].item<double>();
        
        // Map (x,y) -> (R,G,B) in [0,1]
        auto [r, g, b] = color_from_xy(x, y, image_width, image_height, cmap_name);
        
        // Scale to [0,255]
        track_colors.at<uchar>(i, 0) = static_cast<uchar>(r * 255);
        track_colors.at<uchar>(i, 1) = static_cast<uchar>(g * 255);
        track_colors.at<uchar>(i, 2) = static_cast<uchar>(b * 255);
    }
    
    return track_colors;
}

void visualize_tracks_on_images(
    torch::Tensor images,
    torch::Tensor tracks,
    torch::Tensor track_vis_mask,
    const std::string& out_dir,
    const std::string& image_format,
    const std::string& normalize_mode,
    const std::string& cmap_name,
    int frames_per_row,
    bool save_grid) {
    
    // Handle batch dimension if present
    if (tracks.dim() == 4) {
        tracks = tracks.squeeze(0);
        images = images.squeeze(0);
        if (track_vis_mask.defined()) {
            track_vis_mask = track_vis_mask.squeeze(0);
        }
    }
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(out_dir);
    
    int64_t S = images.size(0);
    int64_t N = tracks.size(1);
    
    // Move to CPU
    images = images.cpu().clone();
    tracks = tracks.cpu().clone();
    if (track_vis_mask.defined()) {
        track_vis_mask = track_vis_mask.cpu().clone();
    }
    
    // Infer H, W from images shape
    int64_t H, W;
    if (image_format == "CHW") {
        // e.g. images[s].shape = (3, H, W)
        H = images.size(2);
        W = images.size(3);
    } else {
        // e.g. images[s].shape = (H, W, 3)
        H = images.size(1);
        W = images.size(2);
    }
    
    // Pre-compute the color for each track i based on first visible position
    cv::Mat track_colors_rgb = get_track_colors_by_position(
        tracks,
        track_vis_mask,
        W,
        H,
        cmap_name
    );
    
    // We'll accumulate each frame's drawn image in a vector
    std::vector<cv::Mat> frame_images;
    
    for (int64_t s = 0; s < S; ++s) {
        // Get current frame
        torch::Tensor img = images[s];
        
        // Convert to (H, W, 3)
        if (image_format == "CHW") {
            img = img.permute({1, 2, 0});  // (H, W, 3)
        }
        
        // Convert to numpy-like format
        cv::Mat img_cv(H, W, CV_32FC3);
        std::memcpy(img_cv.data, img.data_ptr<float>(), sizeof(float) * H * W * 3);
        
        // Scale to [0,255] if needed
        if (normalize_mode == "[0,1]") {
            img_cv.convertTo(img_cv, CV_32FC3, 255.0);
            cv::threshold(img_cv, img_cv, 255.0, 255.0, cv::THRESH_TRUNC);
            cv::threshold(img_cv, img_cv, 0.0, 0.0, cv::THRESH_TOZERO);
        } else if (normalize_mode == "[-1,1]") {
            img_cv.convertTo(img_cv, CV_32FC3, 127.5, 127.5);
            cv::threshold(img_cv, img_cv, 255.0, 255.0, cv::THRESH_TRUNC);
            cv::threshold(img_cv, img_cv, 0.0, 0.0, cv::THRESH_TOZERO);
        }
        
        // Convert to uint8
        cv::Mat img_uint8;
        img_cv.convertTo(img_uint8, CV_8UC3);
        
        // For drawing in OpenCV, convert to BGR
        cv::Mat img_bgr;
        cv::cvtColor(img_uint8, img_bgr, cv::COLOR_RGB2BGR);
        
        // Draw each visible track
        torch::Tensor cur_tracks = tracks[s];  // shape (N, 2)
        std::vector<int64_t> valid_indices;
        
        if (track_vis_mask.defined()) {
            for (int64_t i = 0; i < N; ++i) {
                if (track_vis_mask[s][i].item<bool>()) {
                    valid_indices.push_back(i);
                }
            }
        } else {
            for (int64_t i = 0; i < N; ++i) {
                valid_indices.push_back(i);
            }
        }
        
        for (int64_t i : valid_indices) {
            double x = cur_tracks[i][0].item<double>();
            double y = cur_tracks[i][1].item<double>();
            cv::Point pt(static_cast<int>(std::round(x)), static_cast<int>(std::round(y)));
            
            // track_colors_rgb[i] is (R,G,B). For OpenCV circle, we need BGR
            uchar R = track_colors_rgb.at<uchar>(i, 0);
            uchar G = track_colors_rgb.at<uchar>(i, 1);
            uchar B = track_colors_rgb.at<uchar>(i, 2);
            cv::Scalar color_bgr(B, G, R);
            
            cv::circle(img_bgr, pt, 3, color_bgr, -1);
        }
        
        // Convert back to RGB for consistent final saving
        cv::Mat img_rgb;
        cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);
        
        // Save individual frame
        std::string frame_path = out_dir + "/frame_" + 
            std::string(4 - std::to_string(s).length(), '0') + std::to_string(s) + ".png";
        
        // Convert to BGR for OpenCV imwrite
        cv::Mat frame_bgr;
        cv::cvtColor(img_rgb, frame_bgr, cv::COLOR_RGB2BGR);
        cv::imwrite(frame_path, frame_bgr);
        
        frame_images.push_back(img_rgb.clone());
    }
    
    // Only create and save the grid image if save_grid is True
    if (save_grid) {
        // Calculate grid dimensions
        int64_t num_rows = (S + frames_per_row - 1) / frames_per_row;  // Ceiling division
        
        // Create a grid of images
        cv::Mat grid_img;
        
        for (int64_t row = 0; row < num_rows; ++row) {
            int64_t start_idx = row * frames_per_row;
            int64_t end_idx = std::min(start_idx + frames_per_row, S);
            
            // Concatenate this row horizontally
            cv::Mat row_img;
            cv::hconcat(std::vector<cv::Mat>(frame_images.begin() + start_idx, 
                                            frame_images.begin() + end_idx), row_img);
            
            // If this row has fewer than frames_per_row images, pad with black
            if (end_idx - start_idx < frames_per_row) {
                int64_t padding_width = (frames_per_row - (end_idx - start_idx)) * W;
                cv::Mat padding(H, padding_width, CV_8UC3, cv::Scalar(0, 0, 0));
                cv::hconcat(std::vector<cv::Mat>{row_img, padding}, row_img);
            }
            
            // Add this row to the grid
            if (grid_img.empty()) {
                grid_img = row_img;
            } else {
                cv::Mat temp;
                cv::vconcat(std::vector<cv::Mat>{grid_img, row_img}, temp);
                grid_img = temp;
            }
        }
        
        std::string out_path = out_dir + "/tracks_grid.png";
        // Convert back to BGR for OpenCV imwrite
        cv::Mat grid_img_bgr;
        cv::cvtColor(grid_img, grid_img_bgr, cv::COLOR_RGB2BGR);
        cv::imwrite(out_path, grid_img_bgr);
        std::cout << "[INFO] Saved color-by-XY track visualization grid -> " << out_path << std::endl;
    }
    
    std::cout << "[INFO] Saved " << S << " individual frames to " << out_dir << "/frame_*.png" << std::endl;
}

} // namespace utils
} // namespace vggt