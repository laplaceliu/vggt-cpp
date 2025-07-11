/**
 * @file vggsfm_utils.cpp
 * @brief Implementation of utility functions for VGG Structure from Motion
 */

#include "vggsfm_utils.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <filesystem>

namespace vggt {

torch::jit::Module build_vggsfm_tracker(const std::string& model_path) {
    torch::jit::Module tracker;

    try {
        if (model_path.empty()) {
            // In C++, we can't directly download from HuggingFace like in Python
            // We'll assume the model is already downloaded and provide a default path
            const char* default_path = std::getenv("VGGSFM_MODEL_PATH");
            std::string path = default_path ? default_path : "models/vggsfm_v2_tracker.pt";

            if (!std::filesystem::exists(path)) {
                throw std::runtime_error(
                    "Default model not found at " + path +
                    ". Please set VGGSFM_MODEL_PATH environment variable or provide model_path parameter."
                );
            }

            tracker = torch::jit::load(path);
        } else {
            if (!std::filesystem::exists(model_path)) {
                throw std::runtime_error("Model not found at " + model_path);
            }

            tracker = torch::jit::load(model_path);
        }

        tracker.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        throw;
    }

    return tracker;
}

std::vector<int64_t> generate_rank_by_dino(
    torch::Tensor images,
    int query_frame_num,
    int image_size,
    const std::string& model_name,
    const torch::Device& device,
    bool spatial_similarity
) {
    // Resize images to the target size
    images = torch::nn::functional::interpolate(
        images,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{image_size, image_size})
            .mode(torch::kBilinear)
            .align_corners(false)
    );

    // Load DINO model
    torch::jit::Module dino_v2_model;
    try {
        // In C++, we need to use TorchScript models
        std::string model_path = "models/" + model_name + ".pt";
        if (!std::filesystem::exists(model_path)) {
            throw std::runtime_error("DINO model not found at " + model_path);
        }

        dino_v2_model = torch::jit::load(model_path, device);
        dino_v2_model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the DINO model: " << e.what() << std::endl;
        throw;
    }

    // Normalize images using ResNet normalization
    auto resnet_mean = torch::tensor(RESNET_MEAN, device).view({1, 3, 1, 1});
    auto resnet_std = torch::tensor(RESNET_STD, device).view({1, 3, 1, 1});
    auto images_resnet_norm = (images - resnet_mean) / resnet_std;

    torch::NoGradGuard no_grad;

    // Forward pass through DINO model
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(images_resnet_norm);
    auto output = dino_v2_model.forward(inputs);
    auto output_dict = output.toGenericDict();

    torch::Tensor frame_feat;
    torch::Tensor similarity_matrix;

    if (spatial_similarity) {
        frame_feat = output_dict.at("x_norm_patchtokens").toTensor();
        auto frame_feat_norm = torch::nn::functional::normalize(frame_feat,
            torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

        // Compute the similarity matrix
        frame_feat_norm = frame_feat_norm.permute({1, 0, 2});
        similarity_matrix = torch::bmm(frame_feat_norm, frame_feat_norm.transpose(-1, -2));
        similarity_matrix = similarity_matrix.mean(0);
    } else {
        frame_feat = output_dict.at("x_norm_clstoken").toTensor();
        auto frame_feat_norm = torch::nn::functional::normalize(frame_feat,
            torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
        similarity_matrix = torch::mm(frame_feat_norm, frame_feat_norm.transpose(-1, -2));
    }

    auto distance_matrix = 100 - similarity_matrix.clone();

    // Ignore self-pairing
    auto diag_indices = torch::arange(similarity_matrix.size(0), device);
    similarity_matrix.index_put_({diag_indices, diag_indices}, -100);
    auto similarity_sum = similarity_matrix.sum(1);

    // Find the most common frame
    auto most_common_frame_index = torch::argmax(similarity_sum).item<int>();

    // Conduct FPS sampling starting from the most common frame
    auto fps_idx = farthest_point_sampling(distance_matrix, query_frame_num, most_common_frame_index);

    // Clean up tensors to free memory
    frame_feat = torch::Tensor();
    similarity_matrix = torch::Tensor();
    distance_matrix = torch::Tensor();

    return fps_idx;
}

std::vector<int64_t> farthest_point_sampling(
    torch::Tensor distance_matrix,
    int num_samples,
    int most_common_frame_index
) {
    distance_matrix = torch::clamp(distance_matrix, 0);
    int64_t N = distance_matrix.size(0);

    // Initialize with the most common frame
    std::vector<int64_t> selected_indices = {most_common_frame_index};
    auto check_distances = distance_matrix.index({most_common_frame_index});

    while (static_cast<int>(selected_indices.size()) < num_samples) {
        // Find the farthest point from the current set of selected points
        auto farthest_point = torch::argmax(check_distances).item<int64_t>();
        selected_indices.push_back(farthest_point);

        check_distances = distance_matrix.index({farthest_point});
        // Mark already selected points to avoid selecting them again
        for (const auto& idx : selected_indices) {
            check_distances.index_put_({idx}, 0);
        }

        // Break if all points have been selected
        if (selected_indices.size() == static_cast<size_t>(N)) {
            break;
        }
    }

    return selected_indices;
}

torch::Tensor calculate_index_mappings(
    int query_index,
    int S,
    const torch::Device& device
) {
    auto new_order = torch::arange(S, torch::TensorOptions().device(device));
    new_order.index_put_({0}, query_index);
    new_order.index_put_({query_index}, 0);
    return new_order;
}

std::vector<torch::Tensor> switch_tensor_order(
    const std::vector<torch::Tensor>& tensors,
    torch::Tensor order,
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

std::map<FeatureExtractorType, torch::jit::Module> initialize_feature_extractors(
    int max_query_num,
    float det_thres,
    const std::string& extractor_method,
    const torch::Device& device
) {
    std::map<FeatureExtractorType, torch::jit::Module> extractors;
    std::vector<std::string> methods;

    // Split the extractor_method string by '+'
    size_t start = 0;
    size_t end = extractor_method.find('+');
    while (end != std::string::npos) {
        methods.push_back(extractor_method.substr(start, end - start));
        start = end + 1;
        end = extractor_method.find('+', start);
    }
    methods.push_back(extractor_method.substr(start));

    for (const auto& method : methods) {
        std::string method_trimmed = method;
        // Trim whitespace
        method_trimmed.erase(0, method_trimmed.find_first_not_of(" \t\n\r\f\v"));
        method_trimmed.erase(method_trimmed.find_last_not_of(" \t\n\r\f\v") + 1);

        try {
            if (method_trimmed == "aliked") {
                std::string model_path = "models/aliked_extractor.pt";
                if (!std::filesystem::exists(model_path)) {
                    std::cerr << "Warning: ALIKED model not found at " << model_path << std::endl;
                    continue;
                }

                auto extractor = torch::jit::load(model_path, device);
                // Set properties via TorchScript methods
                std::vector<torch::jit::IValue> args;
                args.push_back(max_query_num);
                args.push_back(det_thres);
                extractor.attr("set_parameters")(args);
                extractor.eval();
                extractors[FeatureExtractorType::ALIKED] = extractor;
            } else if (method_trimmed == "sp") {
                std::string model_path = "models/superpoint_extractor.pt";
                if (!std::filesystem::exists(model_path)) {
                    std::cerr << "Warning: SuperPoint model not found at " << model_path << std::endl;
                    continue;
                }

                auto extractor = torch::jit::load(model_path, device);
                // Set properties via TorchScript methods
                std::vector<torch::jit::IValue> args;
                args.push_back(max_query_num);
                args.push_back(det_thres);
                extractor.attr("set_parameters")(args);
                extractor.eval();
                extractors[FeatureExtractorType::SUPERPOINT] = extractor;
            } else if (method_trimmed == "sift") {
                std::string model_path = "models/sift_extractor.pt";
                if (!std::filesystem::exists(model_path)) {
                    std::cerr << "Warning: SIFT model not found at " << model_path << std::endl;
                    continue;
                }

                auto extractor = torch::jit::load(model_path, device);
                // Set properties via TorchScript methods
                std::vector<torch::jit::IValue> args;
                args.push_back(max_query_num);
                extractor.attr("set_parameters")(args);
                extractor.eval();
                extractors[FeatureExtractorType::SIFT] = extractor;
            } else {
                std::cerr << "Warning: Unknown feature extractor '" << method_trimmed << "', ignoring." << std::endl;
            }
        } catch (const c10::Error& e) {
            std::cerr << "Error loading feature extractor '" << method_trimmed << "': " << e.what() << std::endl;
        }
    }

    if (extractors.empty()) {
        std::cerr << "Warning: No valid extractors found in '" << extractor_method
                  << "'. Using ALIKED by default." << std::endl;

        try {
            std::string model_path = "models/aliked_extractor.pt";
            if (std::filesystem::exists(model_path)) {
                auto extractor = torch::jit::load(model_path, device);
                // Set properties via TorchScript methods
                std::vector<torch::jit::IValue> args;
                args.push_back(max_query_num);
                args.push_back(det_thres);
                extractor.attr("set_parameters")(args);
                extractor.eval();
                extractors[FeatureExtractorType::ALIKED] = extractor;
            } else {
                std::cerr << "Error: Default ALIKED model not found at " << model_path << std::endl;
            }
        } catch (const c10::Error& e) {
            std::cerr << "Error loading default ALIKED extractor: " << e.what() << std::endl;
        }
    }

    return extractors;
}

torch::Tensor extract_keypoints(
    torch::Tensor query_image,
    const std::map<FeatureExtractorType, torch::jit::Module>& extractors,
    bool round_keypoints
) {
    torch::Tensor query_points;

    torch::NoGradGuard no_grad;

    for (const auto& [extractor_type, extractor] : extractors) {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(query_image);
        inputs.push_back(torch::Tensor()); // invalid_mask = None

        auto output = extractor.forward(inputs);
        auto output_dict = output.toGenericDict();
        auto extractor_points = output_dict.at("keypoints").toTensor();

        if (round_keypoints) {
            extractor_points = torch::round(extractor_points);
        }

        if (query_points.defined()) {
            query_points = torch::cat({query_points, extractor_points}, 1);
        } else {
            query_points = extractor_points;
        }
    }

    return query_points;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> predict_tracks_in_chunks(
    torch::jit::Module track_predictor,
    torch::Tensor images_feed,
    const std::vector<torch::Tensor>& query_points_list,
    torch::Tensor fmaps_feed,
    bool fine_tracking,
    int num_splits,
    int fine_chunk
) {
    torch::NoGradGuard no_grad;

    std::vector<torch::Tensor> all_tracks;
    std::vector<torch::Tensor> all_vis;
    std::vector<torch::Tensor> all_scores;

    // Process each chunk of query points
    for (const auto& query_points : query_points_list) {
        if (fine_tracking) {
            // For fine tracking, we need to further split the query points
            // to avoid memory issues
            int N = query_points.size(1);
            int num_fine_chunks = (N + fine_chunk - 1) / fine_chunk;

            for (int i = 0; i < num_fine_chunks; ++i) {
                int start_idx = i * fine_chunk;
                int end_idx = std::min((i + 1) * fine_chunk, N);

                auto query_points_chunk = query_points.index({
                    torch::indexing::Slice(),
                    torch::indexing::Slice(start_idx, end_idx)
                });

                // Forward pass through the tracker
                std::vector<torch::jit::IValue> inputs;
                inputs.push_back(query_points_chunk);
                inputs.push_back(fmaps_feed);

                auto output = track_predictor.forward(inputs);
                auto output_dict = output.toGenericDict();

                auto tracks = output_dict.at("tracks").toTensor();
                auto vis = output_dict.at("visibility").toTensor();
                auto scores = output_dict.at("scores").toTensor();

                all_tracks.push_back(tracks);
                all_vis.push_back(vis);
                all_scores.push_back(scores);
            }
        } else {
            // For regular tracking, process the entire chunk at once
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(query_points);
            inputs.push_back(fmaps_feed);

            auto output = track_predictor.forward(inputs);
            auto output_dict = output.toGenericDict();

            auto tracks = output_dict.at("tracks").toTensor();
            auto vis = output_dict.at("visibility").toTensor();
            auto scores = output_dict.at("scores").toTensor();

            all_tracks.push_back(tracks);
            all_vis.push_back(vis);
            all_scores.push_back(scores);
        }
    }

    // Concatenate all results
    torch::Tensor tracks = torch::cat(all_tracks, 1);
    torch::Tensor visibility = torch::cat(all_vis, 1);
    torch::Tensor scores = torch::cat(all_scores, 1);

    return std::make_tuple(tracks, visibility, scores);
}

} // namespace vggt
