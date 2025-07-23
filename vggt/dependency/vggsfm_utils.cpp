// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "vggsfm_utils.h"
#include "track_predictor.h"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <torch/script.h>
#include <opencv2/opencv.hpp>

namespace vggt {

std::shared_ptr<TrackerPredictor> buildVggsfmTracker(const std::string& model_path) {
    auto tracker = std::make_shared<TrackerPredictor>();
    
    if (model_path.empty()) {
        // In C++ we would typically download the model using libcurl or similar
        // For simplicity, we'll assume the model is available locally or provide a path
        std::string default_path = "models/vggsfm_v2_tracker.pt";
        try {
            tracker->load(default_path);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to load default model: " + std::string(e.what()));
        }
    } else {
        try {
            tracker->load(model_path);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to load model from " + model_path + ": " + std::string(e.what()));
        }
    }
    
    tracker->eval();
    return tracker;
}

std::vector<int64_t> generateRankByDino(
    const torch::Tensor& images,
    int query_frame_num,
    int image_size,
    const std::string& model_name,
    const std::string& device,
    bool spatial_similarity
) {
    torch::Device torch_device(device);
    
    // Resize images to the target size
    auto resized_images = torch::nn::functional::interpolate(
        images,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{image_size, image_size})
            .mode(torch::kBilinear)
            .align_corners(false)
    );
    
    // Load DINO model
    torch::jit::script::Module dino_v2_model;
    try {
        // In a real implementation, we would need to have the model available locally
        // or implement a download mechanism
        dino_v2_model = torch::jit::load("models/" + model_name + ".pt");
        dino_v2_model.to(torch_device);
        dino_v2_model.eval();
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load DINO model: " + std::string(e.what()));
    }
    
    // Normalize images using ResNet normalization
    auto resnet_mean = torch::tensor(RESNET_MEAN, torch_device).view({1, 3, 1, 1});
    auto resnet_std = torch::tensor(RESNET_STD, torch_device).view({1, 3, 1, 1});
    auto images_resnet_norm = (resized_images - resnet_mean) / resnet_std;
    
    torch::Tensor similarity_matrix;
    torch::NoGradGuard no_grad;
    
    // Forward pass through the model
    auto inputs = std::vector<torch::jit::IValue>{images_resnet_norm};
    auto output = dino_v2_model.forward(inputs);
    
    // Process features based on similarity type
    if (spatial_similarity) {
        // Extract patch tokens
        auto frame_feat = output.toGenericDict().at("x_norm_patchtokens").toTensor();
        auto frame_feat_norm = torch::nn::functional::normalize(frame_feat, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
        
        // Compute the similarity matrix
        frame_feat_norm = frame_feat_norm.permute({1, 0, 2});
        auto batch_size = frame_feat_norm.size(0);
        similarity_matrix = torch::zeros({frame_feat_norm.size(1), frame_feat_norm.size(1)}, torch_device);
        
        for (int i = 0; i < batch_size; ++i) {
            similarity_matrix += torch::mm(frame_feat_norm[i], frame_feat_norm[i].transpose(0, 1));
        }
        similarity_matrix = similarity_matrix / batch_size;
    } else {
        // Extract CLS token
        auto frame_feat = output.toGenericDict().at("x_norm_clstoken").toTensor();
        auto frame_feat_norm = torch::nn::functional::normalize(frame_feat, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
        similarity_matrix = torch::mm(frame_feat_norm, frame_feat_norm.transpose(0, 1));
    }
    
    auto distance_matrix = 100 - similarity_matrix.clone();
    
    // Ignore self-pairing
    auto self_indices = torch::arange(similarity_matrix.size(0), torch::TensorOptions().device(torch_device));
    similarity_matrix.index_put_({self_indices, self_indices}, -100);
    auto similarity_sum = similarity_matrix.sum(1);
    
    // Find the most common frame
    auto most_common_frame_index = torch::argmax(similarity_sum).item<int64_t>();
    
    // Conduct FPS sampling starting from the most common frame
    auto fps_idx = farthestPointSampling(distance_matrix, query_frame_num, most_common_frame_index);
    
    return fps_idx;
}

std::vector<int64_t> farthestPointSampling(
    torch::Tensor distance_matrix,
    int num_samples,
    int most_common_frame_index
) {
    distance_matrix = torch::clamp(distance_matrix, 0);
    int64_t N = distance_matrix.size(0);
    
    // Initialize with the most common frame
    std::vector<int64_t> selected_indices = {most_common_frame_index};
    auto check_distances = distance_matrix[most_common_frame_index];
    
    while (selected_indices.size() < static_cast<size_t>(num_samples)) {
        // Find the farthest point from the current set of selected points
        auto farthest_point = torch::argmax(check_distances).item<int64_t>();
        selected_indices.push_back(farthest_point);
        
        check_distances = distance_matrix[farthest_point];
        // Mark already selected points to avoid selecting them again
        for (const auto& idx : selected_indices) {
            check_distances[idx] = 0;
        }
        
        // Break if all points have been selected
        if (selected_indices.size() == static_cast<size_t>(N)) {
            break;
        }
    }
    
    return selected_indices;
}

torch::Tensor calculateIndexMappings(
    int query_index,
    int S,
    const std::string& device
) {
    auto new_order = torch::arange(S);
    new_order[0] = query_index;
    new_order[query_index] = 0;
    
    if (!device.empty()) {
        torch::Device torch_device(device);
        new_order = new_order.to(torch_device);
    }
    
    return new_order;
}

std::vector<torch::Tensor> switchTensorOrder(
    const std::vector<torch::Tensor>& tensors,
    const torch::Tensor& order,
    int dim
) {
    std::vector<torch::Tensor> result;
    result.reserve(tensors.size());
    
    for (const auto& tensor : tensors) {
        if (tensor.defined()) {
            result.push_back(tensor.index_select(dim, order));
        } else {
            result.push_back(torch::Tensor());
        }
    }
    
    return result;
}

// ALIKED Extractor Implementation
ALIKEDExtractor::ALIKEDExtractor(int max_num_keypoints, float detection_threshold)
    : max_num_keypoints_(max_num_keypoints), detection_threshold_(detection_threshold) {
    try {
        // Load the TorchScript model
        model_ = torch::jit::load("models/aliked.pt");
        model_.eval();
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load ALIKED model: " + std::string(e.what()));
    }
}

torch::Tensor ALIKEDExtractor::extract(const torch::Tensor& image, const torch::Tensor& invalid_mask) {
    torch::NoGradGuard no_grad;
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(image);
    if (invalid_mask.defined()) {
        inputs.push_back(invalid_mask);
    } else {
        inputs.push_back(torch::Tensor());
    }
    
    // Add additional parameters
    inputs.push_back(max_num_keypoints_);
    inputs.push_back(detection_threshold_);
    
    auto output = model_.forward(inputs).toGenericDict();
    return output.at("keypoints").toTensor();
}

// SuperPoint Extractor Implementation
SuperPointExtractor::SuperPointExtractor(int max_num_keypoints, float detection_threshold)
    : max_num_keypoints_(max_num_keypoints), detection_threshold_(detection_threshold) {
    try {
        // Load the TorchScript model
        model_ = torch::jit::load("models/superpoint.pt");
        model_.eval();
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load SuperPoint model: " + std::string(e.what()));
    }
}

torch::Tensor SuperPointExtractor::extract(const torch::Tensor& image, const torch::Tensor& invalid_mask) {
    torch::NoGradGuard no_grad;
    
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(image);
    if (invalid_mask.defined()) {
        inputs.push_back(invalid_mask);
    } else {
        inputs.push_back(torch::Tensor());
    }
    
    // Add additional parameters
    inputs.push_back(max_num_keypoints_);
    inputs.push_back(detection_threshold_);
    
    auto output = model_.forward(inputs).toGenericDict();
    return output.at("keypoints").toTensor();
}

// SIFT Extractor Implementation
SIFTExtractor::SIFTExtractor(int max_num_keypoints)
    : max_num_keypoints_(max_num_keypoints) {
    sift_ = cv::SIFT::create(max_num_keypoints);
}

torch::Tensor SIFTExtractor::extract(const torch::Tensor& image, const torch::Tensor& invalid_mask) {
    // Convert torch tensor to OpenCV Mat
    cv::Mat cv_image;
    if (image.dim() == 3 && image.size(0) == 3) {
        // Convert CHW to HWC and scale to [0, 255]
        auto hwc_image = image.permute({1, 2, 0}).mul(255).to(torch::kUInt8);
        cv_image = cv::Mat(hwc_image.size(0), hwc_image.size(1), CV_8UC3, hwc_image.data_ptr());
    } else {
        throw std::runtime_error("Unsupported image format for SIFT extraction");
    }
    
    // Convert to grayscale if needed
    cv::Mat gray_image;
    if (cv_image.channels() == 3) {
        cv::cvtColor(cv_image, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = cv_image;
    }
    
    // Apply mask if provided
    cv::Mat mask;
    if (invalid_mask.defined()) {
        mask = cv::Mat(invalid_mask.size(0), invalid_mask.size(1), CV_8UC1, invalid_mask.data_ptr());
        cv::bitwise_not(mask, mask); // Invert mask as SIFT expects 1 for valid regions
    }
    
    // Detect keypoints
    std::vector<cv::KeyPoint> keypoints;
    sift_->detect(gray_image, keypoints, mask);
    
    // Limit number of keypoints
    if (keypoints.size() > static_cast<size_t>(max_num_keypoints_)) {
        keypoints.resize(max_num_keypoints_);
    }
    
    // Convert keypoints to tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(image.device());
    auto keypoints_tensor = torch::zeros({1, static_cast<int64_t>(keypoints.size()), 2}, options);
    
    for (size_t i = 0; i < keypoints.size(); ++i) {
        keypoints_tensor[0][i][0] = keypoints[i].pt.x;
        keypoints_tensor[0][i][1] = keypoints[i].pt.y;
    }
    
    return keypoints_tensor;
}

std::unordered_map<std::string, std::shared_ptr<FeatureExtractor>> initializeFeatureExtractors(
    int max_query_num,
    float det_thres,
    const std::string& extractor_method,
    const std::string& device
) {
    std::unordered_map<std::string, std::shared_ptr<FeatureExtractor>> extractors;
    
    // Parse method string
    std::istringstream iss(extractor_method);
    std::string method;
    
    while (std::getline(iss, method, '+')) {
        // Trim whitespace
        method.erase(0, method.find_first_not_of(" \t\n\r\f\v"));
        method.erase(method.find_last_not_of(" \t\n\r\f\v") + 1);
        
        // Convert to lowercase
        std::transform(method.begin(), method.end(), method.begin(),
                      [](unsigned char c) { return std::tolower(c); });
        
        if (method == "aliked") {
            auto aliked_extractor = std::make_shared<ALIKEDExtractor>(max_query_num, det_thres);
            extractors["aliked"] = aliked_extractor;
        } else if (method == "sp") {
            auto sp_extractor = std::make_shared<SuperPointExtractor>(max_query_num, det_thres);
            extractors["sp"] = sp_extractor;
        } else if (method == "sift") {
            auto sift_extractor = std::make_shared<SIFTExtractor>(max_query_num);
            extractors["sift"] = sift_extractor;
        } else {
            std::cerr << "Warning: Unknown feature extractor '" << method << "', ignoring." << std::endl;
        }
    }
    
    // If no valid extractors found, use ALIKED by default
    if (extractors.empty()) {
        std::cerr << "Warning: No valid extractors found in '" << extractor_method 
                  << "'. Using ALIKED by default." << std::endl;
        auto aliked_extractor = std::make_shared<ALIKEDExtractor>(max_query_num, det_thres);
        extractors["aliked"] = aliked_extractor;
    }
    
    return extractors;
}

torch::Tensor extractKeypoints(
    const torch::Tensor& query_image,
    const std::unordered_map<std::string, std::shared_ptr<FeatureExtractor>>& extractors,
    bool round_keypoints
) {
    torch::Tensor query_points;
    
    torch::NoGradGuard no_grad;
    
    for (const auto& [extractor_name, extractor] : extractors) {
        auto extractor_points = extractor->extract(query_image);
        
        if (round_keypoints) {
            extractor_points = extractor_points.round();
        }
        
        if (query_points.defined()) {
            query_points = torch::cat({query_points, extractor_points}, 1);
        } else {
            query_points = extractor_points;
        }
    }
    
    return query_points;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> predictTracksInChunks(
    TrackerPredictor& track_predictor,
    const torch::Tensor& images_feed,
    const std::vector<torch::Tensor>& query_points_list,
    const torch::Tensor& fmaps_feed,
    bool fine_tracking,
    int num_splits,
    int fine_chunk
) {
    std::vector<torch::Tensor> fine_pred_track_list;
    std::vector<torch::Tensor> pred_vis_list;
    std::vector<torch::Tensor> pred_score_list;
    
    // Process each chunk of query points
    for (const auto& split_points : query_points_list) {
        // Feed into track predictor for each split
        auto [fine_pred_track, _, pred_vis, pred_score] = track_predictor.forward(
            images_feed, split_points, fmaps_feed, fine_tracking, fine_chunk
        );
        
        fine_pred_track_list.push_back(fine_pred_track);
        pred_vis_list.push_back(pred_vis);
        
        if (pred_score.defined()) {
            pred_score_list.push_back(pred_score);
        }
    }
    
    // Concatenate the results from all splits
    auto fine_pred_track = torch::cat(fine_pred_track_list, 2);
    auto pred_vis = torch::cat(pred_vis_list, 2);
    
    torch::Tensor pred_score;
    if (!pred_score_list.empty()) {
        pred_score = torch::cat(pred_score_list, 2);
    }
    
    return {fine_pred_track, pred_vis, pred_score};
}

} // namespace vggt