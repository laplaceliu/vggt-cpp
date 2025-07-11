/**
 * @file vggsfm_utils.h
 * @brief Utility functions for VGG Structure from Motion
 */

#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>

namespace vggt {

/**
 * @brief Constants for ResNet normalization
 */
const std::vector<float> RESNET_MEAN = {0.485f, 0.456f, 0.406f};
const std::vector<float> RESNET_STD = {0.229f, 0.224f, 0.225f};

/**
 * @brief Build and initialize the VGGSfM tracker
 *
 * @param model_path Path to the model weights file. If empty, weights are downloaded from HuggingFace
 * @return Initialized tracker model in eval mode
 */
torch::jit::Module build_vggsfm_tracker(const std::string& model_path = "");

/**
 * @brief Generate a ranking of frames using DINO ViT features
 *
 * @param images Tensor of shape (S, 3, H, W) with values in range [0, 1]
 * @param query_frame_num Number of frames to select
 * @param image_size Size to resize images to before processing
 * @param model_name Name of the DINO model to use
 * @param device Device to run the model on
 * @param spatial_similarity Whether to use spatial token similarity or CLS token similarity
 * @return std::vector<int64_t> List of frame indices ranked by their representativeness
 */
std::vector<int64_t> generate_rank_by_dino(
    torch::Tensor images,
    int query_frame_num,
    int image_size = 336,
    const std::string& model_name = "dinov2_vitb14_reg",
    const torch::Device& device = torch::Device(torch::kCUDA),
    bool spatial_similarity = false
);

/**
 * @brief Farthest point sampling algorithm to select diverse frames
 *
 * @param distance_matrix Matrix of distances between frames
 * @param num_samples Number of frames to select
 * @param most_common_frame_index Index of the first frame to select
 * @return std::vector<int64_t> List of selected frame indices
 */
std::vector<int64_t> farthest_point_sampling(
    torch::Tensor distance_matrix,
    int num_samples,
    int most_common_frame_index = 0
);

/**
 * @brief Construct an order that switches [query_index] and [0]
 * so that the content of query_index would be placed at [0]
 *
 * @param query_index Index to swap with 0
 * @param S Total number of elements
 * @param device Device to place the tensor on
 * @return torch::Tensor Tensor of indices with the swapped order
 */
torch::Tensor calculate_index_mappings(
    int query_index,
    int S,
    const torch::Device& device = torch::Device(torch::kCPU)
);

/**
 * @brief Reorder tensors along a specific dimension according to the given order
 *
 * @param tensors Vector of tensors to reorder
 * @param order Tensor of indices specifying the new order
 * @param dim Dimension along which to reorder
 * @return std::vector<torch::Tensor> Vector of reordered tensors
 */
std::vector<torch::Tensor> switch_tensor_order(
    const std::vector<torch::Tensor>& tensors,
    torch::Tensor order,
    int dim = 1
);

/**
 * @brief Feature extractor type
 */
enum class FeatureExtractorType {
    ALIKED,
    SUPERPOINT,
    SIFT
};

/**
 * @brief Initialize feature extractors that can be reused based on a method string
 *
 * @param max_query_num Maximum number of keypoints to extract
 * @param det_thres Detection threshold for keypoint extraction
 * @param extractor_method String specifying which extractors to use (e.g., "aliked", "sp+sift", "aliked+sp+sift")
 * @param device Device to run extraction on
 * @return std::map<FeatureExtractorType, torch::jit::Module> Map of initialized extractors
 */
std::map<FeatureExtractorType, torch::jit::Module> initialize_feature_extractors(
    int max_query_num,
    float det_thres = 0.005f,
    const std::string& extractor_method = "aliked",
    const torch::Device& device = torch::Device(torch::kCUDA)
);

/**
 * @brief Extract keypoints using pre-initialized feature extractors
 *
 * @param query_image Input image tensor (3xHxW, range [0, 1])
 * @param extractors Map of initialized extractors
 * @param round_keypoints Whether to round keypoint coordinates
 * @return torch::Tensor Tensor of keypoint coordinates (1xNx2)
 */
torch::Tensor extract_keypoints(
    torch::Tensor query_image,
    const std::map<FeatureExtractorType, torch::jit::Module>& extractors,
    bool round_keypoints = true
);

/**
 * @brief Process a list of query points to avoid memory issues
 *
 * @param track_predictor The track predictor module
 * @param images_feed A tensor of shape (B, T, C, H, W) representing a batch of images
 * @param query_points_list A vector of tensors, each of shape (B, Ni, 2) representing chunks of query points
 * @param fmaps_feed A tensor of feature maps for the tracker
 * @param fine_tracking Whether to perform fine tracking
 * @param num_splits Ignored when query_points_list is provided. Kept for backward compatibility
 * @param fine_chunk Chunk size for fine tracking
 * @return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> A tuple containing the concatenated predicted tracks, visibility, and scores
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> predict_tracks_in_chunks(
    torch::jit::Module track_predictor,
    torch::Tensor images_feed,
    const std::vector<torch::Tensor>& query_points_list,
    torch::Tensor fmaps_feed,
    bool fine_tracking,
    int num_splits = 1,
    int fine_chunk = 40960
);

} // namespace vggt
