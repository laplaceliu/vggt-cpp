

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

namespace vggt {

// Constants
const std::vector<float> RESNET_MEAN = {0.485f, 0.456f, 0.406f};
const std::vector<float> RESNET_STD = {0.229f, 0.224f, 0.225f};

// Forward declarations
class TrackerPredictor;
class FeatureExtractor;

/**
 * Build and initialize the VGGSfM tracker.
 *
 * @param model_path Path to the model weights file. If empty, weights are downloaded from HuggingFace.
 * @return Initialized tracker model in eval mode.
 */
std::shared_ptr<TrackerPredictor> buildVggsfmTracker(const std::string& model_path = "");

/**
 * Generate a ranking of frames using DINO ViT features.
 *
 * @param images Tensor of shape (S, 3, H, W) with values in range [0, 1]
 * @param query_frame_num Number of frames to select
 * @param image_size Size to resize images to before processing
 * @param model_name Name of the DINO model to use
 * @param device Device to run the model on
 * @param spatial_similarity Whether to use spatial token similarity or CLS token similarity
 * @return List of frame indices ranked by their representativeness
 */
std::vector<int64_t> generateRankByDino(
    const torch::Tensor& images,
    int query_frame_num,
    int image_size = 336,
    const std::string& model_name = "dinov2_vitb14_reg",
    const std::string& device = "cuda",
    bool spatial_similarity = false
);

/**
 * Farthest point sampling algorithm to select diverse frames.
 *
 * @param distance_matrix Matrix of distances between frames
 * @param num_samples Number of frames to select
 * @param most_common_frame_index Index of the first frame to select
 * @return List of selected frame indices
 */
std::vector<int64_t> farthestPointSampling(
    torch::Tensor distance_matrix,
    int num_samples,
    int most_common_frame_index = 0
);

/**
 * Construct an order that switches [query_index] and [0]
 * so that the content of query_index would be placed at [0].
 *
 * @param query_index Index to swap with 0
 * @param S Total number of elements
 * @param device Device to place the tensor on
 * @return Tensor of indices with the swapped order
 */
torch::Tensor calculateIndexMappings(
    int query_index,
    int S,
    const std::string& device = ""
);

/**
 * Reorder tensors along a specific dimension according to the given order.
 *
 * @param tensors List of tensors to reorder
 * @param order Tensor of indices specifying the new order
 * @param dim Dimension along which to reorder
 * @return List of reordered tensors
 */
std::vector<torch::Tensor> switchTensorOrder(
    const std::vector<torch::Tensor>& tensors,
    const torch::Tensor& order,
    int dim = 1
);

/**
 * Base class for feature extractors
 */
class FeatureExtractor {
public:
    virtual ~FeatureExtractor() = default;
    virtual torch::Tensor extract(const torch::Tensor& image, const torch::Tensor& invalid_mask = {}) = 0;
};

/**
 * ALIKED feature extractor
 */
class ALIKEDExtractor : public FeatureExtractor {
public:
    ALIKEDExtractor(int max_num_keypoints, float detection_threshold);
    torch::Tensor extract(const torch::Tensor& image, const torch::Tensor& invalid_mask = {}) override;
private:
    torch::jit::script::Module model_;
    int max_num_keypoints_;
    float detection_threshold_;
};

/**
 * SuperPoint feature extractor
 */
class SuperPointExtractor : public FeatureExtractor {
public:
    SuperPointExtractor(int max_num_keypoints, float detection_threshold);
    torch::Tensor extract(const torch::Tensor& image, const torch::Tensor& invalid_mask = {}) override;
private:
    torch::jit::script::Module model_;
    int max_num_keypoints_;
    float detection_threshold_;
};

/**
 * SIFT feature extractor
 */
class SIFTExtractor : public FeatureExtractor {
public:
    explicit SIFTExtractor(int max_num_keypoints);
    torch::Tensor extract(const torch::Tensor& image, const torch::Tensor& invalid_mask = {}) override;
private:
    cv::Ptr<cv::SIFT> sift_;
    int max_num_keypoints_;
};

/**
 * Initialize feature extractors that can be reused based on a method string.
 *
 * @param max_query_num Maximum number of keypoints to extract
 * @param det_thres Detection threshold for keypoint extraction
 * @param extractor_method String specifying which extractors to use (e.g., "aliked", "sp+sift", "aliked+sp+sift")
 * @param device Device to run extraction on
 * @return Dictionary of initialized extractors
 */
std::unordered_map<std::string, std::shared_ptr<FeatureExtractor>> initializeFeatureExtractors(
    int max_query_num,
    float det_thres = 0.005f,
    const std::string& extractor_method = "aliked",
    const std::string& device = "cuda"
);

/**
 * Extract keypoints using pre-initialized feature extractors.
 *
 * @param query_image Input image tensor (3xHxW, range [0, 1])
 * @param extractors Dictionary of initialized extractors
 * @param round_keypoints Whether to round keypoint coordinates to integers
 * @return Tensor of keypoint coordinates (1xNx2)
 */
torch::Tensor extractKeypoints(
    const torch::Tensor& query_image,
    const std::unordered_map<std::string, std::shared_ptr<FeatureExtractor>>& extractors,
    bool round_keypoints = true
);

/**
 * Process a list of query points to avoid memory issues.
 *
 * @param track_predictor The track predictor object used for predicting tracks
 * @param images_feed A tensor of shape (B, T, C, H, W) representing a batch of images
 * @param query_points_list A list of tensors, each of shape (B, Ni, 2) representing chunks of query points
 * @param fmaps_feed A tensor of feature maps for the tracker
 * @param fine_tracking Whether to perform fine tracking
 * @param num_splits Ignored when query_points_list is provided. Kept for backward compatibility
 * @param fine_chunk Chunk size for fine tracking
 * @return A tuple containing the concatenated predicted tracks, visibility, and scores
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> predictTracksInChunks(
    TrackerPredictor& track_predictor,
    const torch::Tensor& images_feed,
    const std::vector<torch::Tensor>& query_points_list,
    const torch::Tensor& fmaps_feed,
    bool fine_tracking,
    int num_splits = 0,
    int fine_chunk = 40960
);

} // namespace vggt
