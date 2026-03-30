#include "track_head.h"
#include "dpt_head.h"

namespace vggt {
namespace heads {

TrackHeadImpl::TrackHeadImpl(
    int64_t dim_in,
    int64_t patch_size,
    int64_t features,
    int64_t iters,
    bool predict_conf,
    int64_t stride,
    int64_t corr_levels,
    int64_t corr_radius,
    int64_t hidden_size
) : patch_size_(patch_size),
    iters_(iters) {

    // Feature extractor based on DPT architecture
    // Processes tokens into feature maps for tracking
    std::vector<int64_t> out_channels = {256, 512, 1024, 1024};
    std::vector<int64_t> intermediate_layer_idx = {4, 11, 17, 23};
    feature_extractor_ = DPTHead(
        dim_in,
        patch_size,
        4,          // output_dim
        "inv_log",  // activation
        "expp1",    // conf_activation
        features,   // features
        out_channels,       // out_channels
        intermediate_layer_idx,  // intermediate_layer_idx
        false,      // pos_embed (disabled for tracking)
        true,       // feature_only (only output features)
        2           // down_ratio
    );
    register_module("feature_extractor", feature_extractor_);

    // Initialize tracker
    tracker_ = dependency::track_modules::BaseTrackerPredictor(
        stride,
        corr_levels,
        corr_radius,
        features,  // latent_dim matches features
        hidden_size,
        true,      // use_spaceatt
        6,         // depth
        false      // fine
    );
    register_module("tracker", tracker_);
}

std::tuple<std::vector<torch::Tensor>, torch::Tensor, torch::Tensor> TrackHeadImpl::forward(
    const std::vector<torch::Tensor>& aggregated_tokens_list,
    torch::Tensor images,
    int64_t patch_start_idx,
    torch::Tensor query_points,
    int64_t iters
) {
    auto sizes = images.sizes();
    int64_t B = sizes[0];
    int64_t S = sizes[1];

    // Extract features from tokens using DPTHead
    // feature_maps has shape (B, S, C, H//2, W//2) due to down_ratio=2
    torch::Tensor feature_maps;
    {
        auto result = feature_extractor_->forward(aggregated_tokens_list, images, patch_start_idx);
        feature_maps = std::get<0>(result);
    }

    // Use default iterations if not specified
    if (iters < 0) {
        iters = iters_;
    }

    // Perform tracking using the extracted features
    auto result = tracker_->forward(query_points, feature_maps, iters, false, 1);

    // Extract the tuple result
    auto result_tuple = std::get<std::tuple<std::vector<torch::Tensor>, torch::Tensor>>(result);

    // TODO: confidence scores are not currently returned by BaseTrackerPredictor
    // For now, return empty tensor for conf_scores
    auto conf_scores = torch::empty({B, S, query_points.size(1)},
        torch::TensorOptions().device(images.device()).dtype(images.dtype()));

    return {std::get<0>(result_tuple), std::get<1>(result_tuple), conf_scores};
}

} // namespace heads
} // namespace vggt
