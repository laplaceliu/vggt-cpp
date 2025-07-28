#include "load_fn.h"
#include <stdexcept>
#include <iostream>

namespace vggt {
namespace utils {

std::tuple<torch::Tensor, torch::Tensor> load_and_preprocess_images_square(
    const std::vector<std::string>& image_path_list, 
    int target_size) {
    
    // Check for empty list
    if (image_path_list.empty()) {
        throw std::invalid_argument("At least 1 image is required");
    }

    std::vector<torch::Tensor> images;
    std::vector<std::vector<float>> original_coords;

    for (const auto& image_path : image_path_list) {
        // Open image using OpenCV
        cv::Mat img = cv::imread(image_path, cv::IMREAD_UNCHANGED);
        
        // Check if image was loaded successfully
        if (img.empty()) {
            throw std::runtime_error("Failed to load image: " + image_path);
        }

        // If there's an alpha channel, blend onto white background
        if (img.channels() == 4) {
            cv::Mat background(img.size(), CV_8UC4, cv::Scalar(255, 255, 255, 255));
            cv::Mat rgba_img;
            
            // Create a mask from the alpha channel
            std::vector<cv::Mat> channels;
            cv::split(img, channels);
            
            // Apply alpha blending
            double alpha, beta;
            for (int y = 0; y < img.rows; y++) {
                for (int x = 0; x < img.cols; x++) {
                    alpha = channels[3].at<uchar>(y, x) / 255.0;
                    beta = 1.0 - alpha;
                    
                    for (int c = 0; c < 3; c++) {
                        img.at<cv::Vec4b>(y, x)[c] = 
                            alpha * img.at<cv::Vec4b>(y, x)[c] + 
                            beta * background.at<cv::Vec4b>(y, x)[c];
                    }
                }
            }
        }

        // Convert to RGB
        cv::Mat rgb_img;
        if (img.channels() == 4) {
            cv::cvtColor(img, rgb_img, cv::COLOR_BGRA2RGB);
        } else if (img.channels() == 3) {
            cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
        } else if (img.channels() == 1) {
            cv::cvtColor(img, rgb_img, cv::COLOR_GRAY2RGB);
        }

        // Get original dimensions
        int width = rgb_img.cols;
        int height = rgb_img.rows;

        // Make the image square by padding the shorter dimension
        int max_dim = std::max(width, height);

        // Calculate padding
        int left = (max_dim - width) / 2;
        int top = (max_dim - height) / 2;

        // Calculate scale factor for resizing
        float scale = static_cast<float>(target_size) / max_dim;

        // Calculate final coordinates of original image in target space
        float x1 = left * scale;
        float y1 = top * scale;
        float x2 = (left + width) * scale;
        float y2 = (top + height) * scale;

        // Store original image coordinates and scale
        original_coords.push_back({x1, y1, x2, y2, static_cast<float>(width), static_cast<float>(height)});

        // Create a new black square image and paste original
        cv::Mat square_img(max_dim, max_dim, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Rect roi(left, top, width, height);
        rgb_img.copyTo(square_img(roi));

        // Resize to target size
        cv::Mat resized_img;
        cv::resize(square_img, resized_img, cv::Size(target_size, target_size), 0, 0, cv::INTER_CUBIC);

        // Convert to tensor using torch::from_blob
        // OpenCV stores images in HWC format, but PyTorch expects CHW
        // Also need to convert from uint8 [0-255] to float [0-1]
        torch::Tensor img_tensor = torch::from_blob(
            resized_img.data, 
            {resized_img.rows, resized_img.cols, 3}, 
            torch::kByte
        ).permute({2, 0, 1}).to(torch::kFloat32).div(255.0);
        
        images.push_back(img_tensor);
    }

    // Stack all images
    torch::Tensor stacked_images = torch::stack(images);
    
    // Convert original_coords to tensor
    torch::Tensor coords_tensor = torch::zeros({static_cast<int64_t>(original_coords.size()), 6}, torch::kFloat32);
    for (size_t i = 0; i < original_coords.size(); ++i) {
        for (size_t j = 0; j < original_coords[i].size(); ++j) {
            coords_tensor[i][j] = original_coords[i][j];
        }
    }

    // Add additional dimension if single image to ensure correct shape
    if (image_path_list.size() == 1) {
        if (stacked_images.dim() == 3) {
            stacked_images = stacked_images.unsqueeze(0);
            coords_tensor = coords_tensor.unsqueeze(0);
        }
    }

    return std::make_tuple(stacked_images, coords_tensor);
}

torch::Tensor load_and_preprocess_images(
    const std::vector<std::string>& image_path_list, 
    const std::string& mode) {
    
    // Check for empty list
    if (image_path_list.empty()) {
        throw std::invalid_argument("At least 1 image is required");
    }

    // Validate mode
    if (mode != "crop" && mode != "pad") {
        throw std::invalid_argument("Mode must be either 'crop' or 'pad'");
    }

    std::vector<torch::Tensor> images;
    std::set<std::pair<int64_t, int64_t>> shapes;
    int target_size = 518;

    // First process all images and collect their shapes
    for (const auto& image_path : image_path_list) {
        // Open image using OpenCV
        cv::Mat img = cv::imread(image_path, cv::IMREAD_UNCHANGED);
        
        // Check if image was loaded successfully
        if (img.empty()) {
            throw std::runtime_error("Failed to load image: " + image_path);
        }

        // If there's an alpha channel, blend onto white background
        if (img.channels() == 4) {
            cv::Mat background(img.size(), CV_8UC4, cv::Scalar(255, 255, 255, 255));
            cv::Mat rgba_img;
            
            // Create a mask from the alpha channel
            std::vector<cv::Mat> channels;
            cv::split(img, channels);
            
            // Apply alpha blending
            double alpha, beta;
            for (int y = 0; y < img.rows; y++) {
                for (int x = 0; x < img.cols; x++) {
                    alpha = channels[3].at<uchar>(y, x) / 255.0;
                    beta = 1.0 - alpha;
                    
                    for (int c = 0; c < 3; c++) {
                        img.at<cv::Vec4b>(y, x)[c] = 
                            alpha * img.at<cv::Vec4b>(y, x)[c] + 
                            beta * background.at<cv::Vec4b>(y, x)[c];
                    }
                }
            }
        }

        // Convert to RGB
        cv::Mat rgb_img;
        if (img.channels() == 4) {
            cv::cvtColor(img, rgb_img, cv::COLOR_BGRA2RGB);
        } else if (img.channels() == 3) {
            cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
        } else if (img.channels() == 1) {
            cv::cvtColor(img, rgb_img, cv::COLOR_GRAY2RGB);
        }

        int width = rgb_img.cols;
        int height = rgb_img.rows;
        int new_width, new_height;

        if (mode == "pad") {
            // Make the largest dimension 518px while maintaining aspect ratio
            if (width >= height) {
                new_width = target_size;
                new_height = round(height * (static_cast<float>(new_width) / width) / 14) * 14;  // Make divisible by 14
            } else {
                new_height = target_size;
                new_width = round(width * (static_cast<float>(new_height) / height) / 14) * 14;  // Make divisible by 14
            }
        } else {  // mode == "crop"
            // Original behavior: set width to 518px
            new_width = target_size;
            // Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (static_cast<float>(new_width) / width) / 14) * 14;
        }

        // Resize with new dimensions
        cv::Mat resized_img;
        cv::resize(rgb_img, resized_img, cv::Size(new_width, new_height), 0, 0, cv::INTER_CUBIC);

        // Convert to tensor
        torch::Tensor img_tensor = torch::from_blob(
            resized_img.data, 
            {resized_img.rows, resized_img.cols, 3}, 
            torch::kByte
        ).permute({2, 0, 1}).to(torch::kFloat32).div(255.0);

        // Center crop height if it's larger than 518 (only in crop mode)
        if (mode == "crop" && new_height > target_size) {
            int64_t start_y = (new_height - target_size) / 2;
            img_tensor = img_tensor.slice(1, start_y, start_y + target_size);
        }

        // For pad mode, pad to make a square of target_size x target_size
        if (mode == "pad") {
            int64_t h_padding = target_size - img_tensor.size(1);
            int64_t w_padding = target_size - img_tensor.size(2);

            if (h_padding > 0 || w_padding > 0) {
                int64_t pad_top = h_padding / 2;
                int64_t pad_bottom = h_padding - pad_top;
                int64_t pad_left = w_padding / 2;
                int64_t pad_right = w_padding - pad_left;

                // Pad with white (value=1.0)
                img_tensor = torch::nn::functional::pad(
                    img_tensor, 
                    torch::nn::functional::PadFuncOptions({pad_left, pad_right, pad_top, pad_bottom})
                        .mode(torch::kConstant)
                        .value(1.0)
                );
            }
        }

        shapes.insert({img_tensor.size(1), img_tensor.size(2)});
        images.push_back(img_tensor);
    }

    // Check if we have different shapes
    if (shapes.size() > 1) {
        std::cout << "Warning: Found images with different shapes" << std::endl;
        
        // Find maximum dimensions
        int64_t max_height = 0;
        int64_t max_width = 0;
        for (const auto& shape : shapes) {
            max_height = std::max(max_height, shape.first);
            max_width = std::max(max_width, shape.second);
        }

        // Pad images if necessary
        std::vector<torch::Tensor> padded_images;
        for (auto& img : images) {
            int64_t h_padding = max_height - img.size(1);
            int64_t w_padding = max_width - img.size(2);

            if (h_padding > 0 || w_padding > 0) {
                int64_t pad_top = h_padding / 2;
                int64_t pad_bottom = h_padding - pad_top;
                int64_t pad_left = w_padding / 2;
                int64_t pad_right = w_padding - pad_left;

                img = torch::nn::functional::pad(
                    img, 
                    torch::nn::functional::PadFuncOptions({pad_left, pad_right, pad_top, pad_bottom})
                        .mode(torch::kConstant)
                        .value(1.0)
                );
            }
            padded_images.push_back(img);
        }
        images = padded_images;
    }

    torch::Tensor stacked_images = torch::stack(images);  // concatenate images

    // Ensure correct shape when single image
    if (image_path_list.size() == 1) {
        // Verify shape is (1, C, H, W)
        if (stacked_images.dim() == 3) {
            stacked_images = stacked_images.unsqueeze(0);
        }
    }

    return stacked_images;
}

} // namespace utils
} // namespace vggt