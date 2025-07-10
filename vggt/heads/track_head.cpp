/**
 * @file track_head.cpp
 * @brief Implementation of track head for point tracking
 */

#include "track_head.h"

namespace vggt {

TrackHeadImpl::TrackHeadImpl(
    int64_t hidden_dim,
    int64_t corr_levels,
    int64_t corr_radius,
    int64_t iters,
    bool multiple_track_feats,
    const std::string& padding_mode) {
    // Initialize parameters
    hidden_dim_ = hidden_dim;
    corr_levels_ = corr_levels;
    corr_radius_ = corr_radius;
    iters_ = iters;
    multiple_track_feats_ = multiple_track_feats;
    padding_mode_ = padding_mode;

    // Initialize network components
    feature_extractor_ = register_module("feature_extractor", DPTHead());

    tracker_predictor_ = register_module("tracker_predictor",
        track_modules::BaseTrackerPredictor(
            hidden_dim, corr_levels, corr_radius, iters, multiple_track_feats, padding_mode));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> TrackHeadImpl::forward(
    const std::vector<torch::Tensor>& x,
    const torch::Tensor& targets,
    const torch::Tensor& coords,
    const torch::Tensor& visibility,
    const torch::Tensor& confidence,
    const torch::Tensor& mask) {
    // Extract features using DPT head
    auto fmaps = feature_extractor_->forward(x);

    // Apply tracker predictor
    auto [coords_pred, vis_pred, conf_pred] = tracker_predictor_->forward(
        fmaps, targets, coords, visibility, confidence, mask);

    return std::make_tuple(coords_pred, vis_pred, conf_pred);
}

} // namespace vggt
