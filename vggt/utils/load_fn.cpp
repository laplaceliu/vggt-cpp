/**
 * @file load_fn.cpp
 * @brief Implementation of image loading and preprocessing functions for VGGT
 */

#include "load_fn.h"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace vggt {
namespace utils {

torch::Tensor load_and_preprocess_images_square(
    const std::vector<std::string>& image_paths,
    int target_size,
    bool normalize,
    const std::vector<float>& mean,
    const std::vector<float>& std) {

    int num_images = image_paths.size();
    torch::Tensor batch = torch::zeros({num_images, 3, target_size, target_size});

    for (int i = 0; i < num_images; ++i) {
        // Load image using OpenCV
        cv::Mat img = cv::imread(image_paths[i]);
        if (img.empty()) {
            throw std::runtime_error("Failed to load image: " + image_paths[i]);
        }

        // Convert BGR to RGB
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        // Get original dimensions
        int height = img.rows;
        int width = img.cols;
        int max_dim = std::max(height, width);

        // Create square image with center padding
        cv::Mat square_img = cv::Mat::zeros(max_dim, max_dim, img.type());
        int y_offset = (max_dim - height) / 2;
        int x_offset = (max_dim - width) / 2;

        // Copy original image to center of square image
        img.copyTo(square_img(cv::Rect(x_offset, y_offset, width, height)));

        // Resize to target size
        cv::resize(square_img, square_img, cv::Size(target_size, target_size));

        // Convert to tensor
        torch::Tensor img_tensor = torch::from_blob(
            square_img.data,
            {square_img.rows, square_img.cols, 3},
            torch::kByte
        ).clone();

        // Permute dimensions from HWC to CHW
        img_tensor = img_tensor.permute({2, 0, 1});

        // Convert to float and scale to [0, 1]
        img_tensor = img_tensor.to(torch::kFloat32).div(255.0);

        // Normalize if requested
        if (normalize) {
            // Create normalization tensors
            torch::Tensor mean_tensor = torch::tensor(mean).view({3, 1, 1});
            torch::Tensor std_tensor = torch::tensor(std).view({3, 1, 1});

            // Apply normalization
            img_tensor = img_tensor.sub(mean_tensor).div(std_tensor);
        }

        // Add to batch
        batch[i] = img_tensor;
    }

    return batch;
}

torch::Tensor load_and_preprocess_images(
    const std::vector<std::string>& image_paths,
    int target_size,
    bool normalize,
    const std::vector<float>& mean,
    const std::vector<float>& std) {

    // This is a quick start function that uses the square preprocessing
    return load_and_preprocess_images_square(image_paths, target_size, normalize, mean, std);
}

} // namespace utils
} // namespace vggt
