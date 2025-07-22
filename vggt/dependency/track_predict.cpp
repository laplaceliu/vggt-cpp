/**
 * @file track_predict.cpp
 * @brief Implementation of functions for predicting tracks in a sequence of images
 */

#include "track_predict.h"
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <torch/torch.h>
#include <torch/script.h>

namespace vggt {
namespace dependency {

std::tuple<torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>> 
predict_tracks(
    const torch::Tensor& images,
    const c10::optional<torch::Tensor>& conf,
    const c10::optional<torch::Tensor>& points_3d,
    const c10::optional<torch::Tensor>& masks,
    int max_query_pts,
    int query_frame_num,
    const std::string& keypoint_extractor,
    int max_points_num,
    bool fine_tracking,
    bool complete_non_vis
) {
    // Validate input dimensions
    TORCH_CHECK(images.dim() == 4, "images must be 4D tensor [S,3,H,W]");
    
    auto device = images.device();
    auto dtype = images.dtype();
    
    // Build tracker
    auto tracker = build_vggsfm_tracker();
    tracker->to(device, dtype);
    
    // Find query frames
    auto query_frame_indexes = generate_rank_by_dino(images, query_frame_num, device);
    
    // Add the first image to the front if not already present
    if (std::find(query_frame_indexes.begin(), query_frame_indexes.end(), 0) != query_frame_indexes.end()) {
        query_frame_indexes.erase(std::remove(query_frame_indexes.begin(), query_frame_indexes.end(), 0), query_frame_indexes.end());
    }
    query_frame_indexes.insert(query_frame_indexes.begin(), 0);
    
    // Initialize feature extractors
    auto keypoint_extractors = initialize_feature_extractors(max_query_pts, 0.005, keypoint_extractor, device);
    
    std::vector<torch::Tensor> pred_tracks;
    std::vector<torch::Tensor> pred_vis_scores;
    std::vector<c10::optional<torch::Tensor>> pred_confs;
    std::vector<c10::optional<torch::Tensor>> pred_points_3d;
    std::vector<torch::Tensor> pred_colors;
    
    auto fmaps_for_tracker = tracker->process_images_to_fmaps(images);
    
    if (fine_tracking) {
        std::cout << "For faster inference, consider disabling fine_tracking" << std::endl;
    }
    
    for (auto query_index : query_frame_indexes) {
        std::cout << "Predicting tracks for query frame " << query_index << std::endl;
        auto [pred_track, pred_vis, pred_conf, pred_point_3d, pred_color] = _forward_on_query(
            query_index,
            images,
            conf,
            points_3d,
            fmaps_for_tracker,
            keypoint_extractors,
            *tracker,
            max_points_num,
            fine_tracking,
            device
        );
        
        pred_tracks.push_back(pred_track);
        pred_vis_scores.push_back(pred_vis);
        pred_confs.push_back(pred_conf);
        pred_points_3d.push_back(pred_point_3d);
        pred_colors.push_back(pred_color);
    }
    
    if (complete_non_vis) {
        std::tie(pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors) = _augment_non_visible_frames(
            pred_tracks,
            pred_vis_scores,
            pred_confs,
            pred_points_3d,
            pred_colors,
            images,
            conf,
            points_3d,
            fmaps_for_tracker,
            keypoint_extractors,
            *tracker,
            max_points_num,
            fine_tracking,
            500, // min_vis
            0.1, // non_vis_thresh
            device
        );
    }
    
    // Concatenate results
    auto concatenated_tracks = torch::cat(pred_tracks, 1);
    auto concatenated_vis = torch::cat(pred_vis_scores, 1);
    
    c10::optional<torch::Tensor> concatenated_confs = c10::nullopt;
    if (pred_confs[0].has_value()) {
        std::vector<torch::Tensor> confs;
        for (const auto& conf : pred_confs) {
            if (conf.has_value()) {
                confs.push_back(conf.value());
            }
        }
        concatenated_confs = torch::cat(confs, 0);
    }
    
    c10::optional<torch::Tensor> concatenated_points_3d = c10::nullopt;
    if (pred_points_3d[0].has_value()) {
        std::vector<torch::Tensor> points;
        for (const auto& point : pred_points_3d) {
            if (point.has_value()) {
                points.push_back(point.value());
            }
        }
        concatenated_points_3d = torch::cat(points, 0);
    }
    
    c10::optional<torch::Tensor> concatenated_colors = c10::nullopt;
    if (!pred_colors.empty()) {
        concatenated_colors = torch::cat(pred_colors, 0);
    }
    
    return std::make_tuple(
        concatenated_tracks,
        concatenated_vis,
        concatenated_confs,
        concatenated_points_3d,
        concatenated_colors
    );
}

std::tuple<torch::Tensor, torch::Tensor, c10::optional<torch::Tensor>, c10::optional<torch::Tensor>, torch::Tensor> 
_forward_on_query(
    int query_index,
    const torch::Tensor& images,
    const c10::optional<torch::Tensor>& conf,
    const c10::optional<torch::Tensor>& points_3d,
    const torch::Tensor& fmaps_for_tracker,
    const std::unordered_map<std::string, torch::nn::Module>& keypoint_extractors,
    const torch::nn::Module& tracker,
    int max_points_num,
    bool fine_tracking,
    const torch::Device& device
) {
    auto frame_num = images.size(0);
    auto height = images.size(2);
    auto width = images.size(3);
    
    auto query_image = images[query_index];
    auto query_points = extract_keypoints(query_image, keypoint_extractors, false);
    
    // Shuffle query points
    auto perm = torch::randperm(query_points.size(1), torch::TensorOptions().device(device));
    query_points = query_points.index({torch::indexing::Slice(), perm, torch::indexing::Slice()});
    
    // Extract color at keypoint locations
    auto query_points_long = query_points.squeeze(0).round().to(torch::kLong);
    auto pred_color = query_image.index({
        torch::indexing::Slice(),
        query_points_long.index({torch::indexing::Slice(), 1}),
        query_points_long.index({torch::indexing::Slice(), 0})
    });
    pred_color = (pred_color.permute({1, 0}).cpu() * 255).to(torch::kUInt8);
    
    // Query confidence and 3D points if available
    c10::optional<torch::Tensor> pred_conf = c10::nullopt;
    c10::optional<torch::Tensor> pred_point_3d = c10::nullopt;
    
    if (conf.has_value() && points_3d.has_value()) {
        TORCH_CHECK(height == width, "Height and width must be equal when using confidence maps");
        TORCH_CHECK(conf.value().size(-2) == conf.value().size(-1), "Confidence map must be square");
        TORCH_CHECK(conf.value().size(0) == points_3d.value().size(0) && 
                   conf.value().size(1) == points_3d.value().size(1) && 
                   conf.value().size(2) == points_3d.value().size(2), 
                   "Confidence and points_3d must have same dimensions");
        
        auto scale = conf.value().size(-1) / width;
        auto query_points_scaled = (query_points.squeeze(0) * scale).round().to(torch::kLong).cpu();
        
        pred_conf = conf.value().index({
            query_index,
            query_points_scaled.index({torch::indexing::Slice(), 1}),
            query_points_scaled.index({torch::indexing::Slice(), 0})
        });
        
        pred_point_3d = points_3d.value().index({
            query_index,
            query_points_scaled.index({torch::indexing::Slice(), 1}),
            query_points_scaled.index({torch::indexing::Slice(), 0})
        });
        
        // Filter low confidence points
        auto valid_mask = pred_conf.value() > 1.2;
        if (valid_mask.sum().item<int>() > 512) {
            query_points = query_points.index({torch::indexing::Slice(), valid_mask, torch::indexing::Slice()});
            pred_conf = pred_conf.value().index({valid_mask});
            pred_point_3d = pred_point_3d.value().index({valid_mask});
            pred_color = pred_color.index({valid_mask});
        }
    }
    
    auto reorder_index = calculate_index_mappings(query_index, frame_num, device);
    
    auto images_feed = switch_tensor_order({images}, reorder_index, 0)[0].unsqueeze(0);
    auto fmaps_feed = switch_tensor_order({fmaps_for_tracker}, reorder_index, 0)[0].unsqueeze(0);
    
    // Process in chunks if needed
    auto all_points_num = images_feed.size(1) * query_points.size(1);
    std::vector<torch::Tensor> query_points_chunks;
    
    if (all_points_num > max_points_num) {
        auto num_splits = (all_points_num + max_points_num - 1) / max_points_num;
        query_points_chunks = torch::chunk(query_points, num_splits, 1);
    } else {
        query_points_chunks = {query_points};
    }
    
    // Predict tracks
    auto [pred_track, pred_vis, _] = predict_tracks_in_chunks(
        tracker, images_feed, query_points_chunks, fmaps_feed, fine_tracking
    );
    
    // Switch back to original order
    auto [pred_track_ordered, pred_vis_ordered] = switch_tensor_order(
        {pred_track, pred_vis}, reorder_index, 1
    );
    
    return std::make_tuple(
        pred_track_ordered.squeeze(0).cpu().to(torch::kFloat),
        pred_vis_ordered.squeeze(0).cpu().to(torch::kFloat),
        pred_conf,
        pred_point_3d,
        pred_color
    );
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<c10::optional<torch::Tensor>>, 
    std::vector<c10::optional<torch::Tensor>>, std::vector<torch::Tensor>> 
_augment_non_visible_frames(
    const std::vector<torch::Tensor>& pred_tracks,
    const std::vector<torch::Tensor>& pred_vis_scores,
    const std::vector<c10::optional<torch::Tensor>>& pred_confs,
    const std::vector<c10::optional<torch::Tensor>>& pred_points_3d,
    const std::vector<torch::Tensor>& pred_colors,
    const torch::Tensor& images,
    const c10::optional<torch::Tensor>& conf,
    const c10::optional<torch::Tensor>& points_3d,
    const torch::Tensor& fmaps_for_tracker,
    const std::unordered_map<std::string, torch::nn::Module>& keypoint_extractors,
    const torch::nn::Module& tracker,
    int max_points_num,
    bool fine_tracking,
    int min_vis,
    float non_vis_thresh,
    const torch::Device& device
) {
    auto updated_tracks = pred_tracks;
    auto updated_vis = pred_vis_scores;
    auto updated_confs = pred_confs;
    auto updated_points_3d = pred_points_3d;
    auto updated_colors = pred_colors;
    
    int last_query = -1;
    bool final_trial = false;
    auto cur_extractors = keypoint_extractors;
    
    while (true) {
        // Concatenate visibility scores
        auto vis_array = torch::cat(updated_vis, 1);
        
        // Count frames with sufficient visibility
        auto sufficient_vis_count = (vis_array > non_vis_thresh).sum(1);
        auto non_vis_frames = torch::nonzero(sufficient_vis_count < min_vis).squeeze().cpu();
        
        if (non_vis_frames.numel() == 0) {
            break;
        }
        
        if (non_vis_frames.dim() == 0) {
            non_vis_frames = non_vis_frames.unsqueeze(0);
        }
        
        std::cout << "Processing non visible frames: " << non_vis_frames << std::endl;
        
        // Decide frames to process
        std::vector<int> query_frame_list;
        if (non_vis_frames[0].item<int>() == last_query) {
            // Final attempt with all extractors
            final_trial = true;
            cur_extractors = initialize_feature_extractors(2048, 0.005, "sp+sift+aliked", device);
            for (int i = 0; i < non_vis_frames.size(0); i++) {
                query_frame_list.push_back(non_vis_frames[i].item<int>());
            }
        } else {
            query_frame_list = {non_vis_frames[0].item<int>()};
        }
        
        last_query = non_vis_frames[0].item<int>();
        
        // Process each frame
        for (auto query_index : query_frame_list) {
            auto [new_track, new_vis, new_conf, new_point_3d, new_color] = _forward_on_query(
                query_index,
                images,
                conf,
                points_3d,
                fmaps_for_tracker,
                cur_extractors,
                tracker,
                max_points_num,
                fine_tracking,
                device
            );
            
            updated_tracks.push_back(new_track);
            updated_vis.push_back(new_vis);
            updated_confs.push_back(new_conf);
            updated_points_3d.push_back(new_point_3d);
            updated_colors.push_back(new_color);
        }
        
        if (final_trial) {
            break;
        }
    }
    
    return std::make_tuple(updated_tracks, updated_vis, updated_confs, updated_points_3d, updated_colors);
}

std::unordered_map<std::string, torch::nn::Module> 
initialize_feature_extractors(
    int max_query_num, 
    float det_thres, 
    const std::string& extractor_method, 
    const torch::Device& device
) {
    std::unordered_map<std::string, torch::nn::Module> extractors;
    std::vector<std::string> methods;
    
    size_t pos = 0;
    std::string token;
    std::string s = extractor_method;
    while ((pos = s.find('+')) != std::string::npos) {
        token = s.substr(0, pos);
        methods.push_back(token);
        s.erase(0, pos + 1);
    }
    methods.push_back(s);
    
    for (const auto& method : methods) {
        if (method == "aliked") {
            auto aliked_extractor = ALIKED(max_query_num, det_thres);
            extractors["aliked"] = aliked_extractor.to(device).eval();
        } else if (method == "sp") {
            auto sp_extractor = SuperPoint(max_query_num, det_thres);
            extractors["sp"] = sp_extractor.to(device).eval();
        } else if (method == "sift") {
            auto sift_extractor = SIFT(max_query_num);
            extractors["sift"] = sift_extractor.to(device).eval();
        } else {
            std::cerr << "Warning: Unknown feature extractor '" << method << "', ignoring." << std::endl;
        }
    }
    
    if (extractors.empty()) {
        std::cerr << "Warning: No valid extractors found in '" << extractor_method << "'. Using ALIKED by default." << std::endl;
        auto aliked_extractor = ALIKED(max_query_num, det_thres);
        extractors["aliked"] = aliked_extractor.to(device).eval();
    }
    
    return extractors;
}

torch::Tensor 
extract_keypoints(
    const torch::Tensor& query_image, 
    const std::unordered_map<std::string, torch::nn::Module>& extractors, 
    bool round_keypoints
) {
    torch::Tensor query_points;
    
    for (const auto& [name, extractor] : extractors) {
        auto query_points_data = extractor.forward({query_image.unsqueeze(0)}).toTuple();
        auto extractor_points = query_points_data->elements()[0].toTensor();
        
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

} // namespace dependency
} // namespace vggt