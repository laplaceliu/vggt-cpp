/**
 * @file visual_track.cpp
 * @brief Implementation of visualization utilities for tracking results
 */

#include "visual_track.h"
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <random>
#include <sstream>
#include <iomanip>

namespace vggt {
namespace utils {

// Helper function to generate random colors for each class
std::vector<cv::Scalar> generate_colors(int num_classes) {
    std::vector<cv::Scalar> colors;
    std::mt19937 rng(12345); // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dist(0, 255);

    for (int i = 0; i < num_classes; ++i) {
        colors.push_back(cv::Scalar(dist(rng), dist(rng), dist(rng)));
    }

    return colors;
}

cv::Mat tensor_to_cv_image(
    const torch::Tensor& tensor_image,
    bool normalize,
    const std::vector<float>& mean,
    const std::vector<float>& std) {

    // Clone and detach tensor to CPU
    auto img = tensor_image.clone().detach().to(torch::kCPU);

    // Denormalize if needed
    if (normalize) {
        // Create normalization tensors
        torch::Tensor mean_tensor = torch::tensor(mean).view({3, 1, 1});
        torch::Tensor std_tensor = torch::tensor(std).view({3, 1, 1});

        // Apply denormalization
        img = img.mul(std_tensor).add(mean_tensor);
    }

    // Clamp values to [0, 1]
    img = torch::clamp(img, 0.0, 1.0);

    // Scale to [0, 255] and convert to uint8
    img = img.mul(255).to(torch::kUInt8);

    // Permute from CHW to HWC
    img = img.permute({1, 2, 0});

    // Convert to OpenCV Mat
    cv::Mat cv_img(img.size(0), img.size(1), CV_8UC3, img.data_ptr<uint8_t>());

    // Convert from RGB to BGR
    cv::cvtColor(cv_img, cv_img, cv::COLOR_RGB2BGR);

    // Create a deep copy to avoid memory issues
    return cv_img.clone();
}

cv::Mat draw_boxes(
    const cv::Mat& image,
    const torch::Tensor& boxes,
    const torch::Tensor& scores,
    const torch::Tensor& labels,
    const std::vector<std::string>& class_names,
    float score_threshold) {

    // Create a copy of the image
    cv::Mat result = image.clone();

    // Generate colors for each class
    std::vector<cv::Scalar> colors = generate_colors(class_names.size());

    // Get image dimensions
    int height = result.rows;
    int width = result.cols;

    // Convert tensors to CPU
    auto boxes_cpu = boxes.to(torch::kCPU);
    auto scores_cpu = scores.to(torch::kCPU);
    auto labels_cpu = labels.to(torch::kCPU);

    // Draw each box
    for (int i = 0; i < boxes_cpu.size(0); ++i) {
        float score = scores_cpu[i].item<float>();

        // Skip low confidence detections
        if (score < score_threshold) {
            continue;
        }

        // Get box coordinates
        int x1 = std::max(0, static_cast<int>(boxes_cpu[i][0].item<float>()));
        int y1 = std::max(0, static_cast<int>(boxes_cpu[i][1].item<float>()));
        int x2 = std::min(width - 1, static_cast<int>(boxes_cpu[i][2].item<float>()));
        int y2 = std::min(height - 1, static_cast<int>(boxes_cpu[i][3].item<float>()));

        // Get class label
        int label_idx = labels_cpu[i].item<int64_t>();
        std::string class_name = (label_idx >= 0 && label_idx < static_cast<int>(class_names.size()))
                                ? class_names[label_idx] : "unknown";

        // Get color for this class
        cv::Scalar color = colors[label_idx % colors.size()];

        // Draw rectangle
        cv::rectangle(result, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

        // Create label text with class name and score
        std::stringstream ss;
        ss << class_name << ": " << std::fixed << std::setprecision(2) << score;
        std::string label_text = ss.str();

        // Get text size
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        // Draw filled rectangle for text background
        cv::rectangle(result,
                     cv::Point(x1, y1 - text_size.height - 5),
                     cv::Point(x1 + text_size.width, y1),
                     color, -1);

        // Draw text
        cv::putText(result, label_text, cv::Point(x1, y1 - 5),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    return result;
}

std::vector<cv::Mat> draw_tracking_results(
    const torch::Tensor& images,
    const torch::Tensor& boxes,
    const torch::Tensor& scores,
    const torch::Tensor& labels,
    const std::vector<std::string>& class_names,
    float score_threshold) {

    std::vector<cv::Mat> result_images;
    int batch_size = images.size(0);

    for (int i = 0; i < batch_size; ++i) {
        // Convert tensor image to OpenCV image
        cv::Mat cv_image = tensor_to_cv_image(images[i]);

        // Draw boxes for this image
        cv::Mat result = draw_boxes(
            cv_image,
            boxes[i],
            scores[i],
            labels[i],
            class_names,
            score_threshold
        );

        result_images.push_back(result);
    }

    return result_images;
}

bool save_tracking_video(
    const std::vector<cv::Mat>& frames,
    const std::string& output_path,
    int fps) {

    if (frames.empty()) {
        return false;
    }

    // Get frame dimensions from the first frame
    int width = frames[0].cols;
    int height = frames[0].rows;

    // Create video writer
    cv::VideoWriter video_writer(
        output_path,
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), // MJPG codec
        fps,
        cv::Size(width, height)
    );

    if (!video_writer.isOpened()) {
        return false;
    }

    // Write frames to video
    for (const auto& frame : frames) {
        video_writer.write(frame);
    }

    // Release video writer
    video_writer.release();

    return true;
}

} // namespace utils
} // namespace vggt
