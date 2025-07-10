/**
 * @file base_track_predictor.cpp
 * @brief Implementation of base tracker predictor for point tracking
 */

#include "base_track_predictor.h"
#include <cmath>

namespace vggt {
namespace track_modules {

BaseTrackerPredictorImpl::BaseTrackerPredictorImpl(
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
    int64_t corr_dim = corr_levels * (2 * corr_radius + 1) * (2 * corr_radius + 1);

    corr_embed_ = register_module("corr_embed",
        torch::nn::Linear(torch::nn::LinearOptions(corr_dim, hidden_dim)));

    flow_embed_ = register_module("flow_embed",
        torch::nn::Linear(torch::nn::LinearOptions(2, hidden_dim)));

    vis_embed_ = register_module("vis_embed",
        torch::nn::Linear(torch::nn::LinearOptions(1, hidden_dim)));

    conf_embed_ = register_module("conf_embed",
        torch::nn::Linear(torch::nn::LinearOptions(1, hidden_dim)));

    update_block_ = register_module("update_block",
        EfficientUpdateFormer(hidden_dim, 8, 6, 1024, 0.1, "relu", false, false));

    coord_predictor_ = register_module("coord_predictor",
        torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, 2)));

    vis_predictor_ = register_module("vis_predictor",
        torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, 1)));

    conf_predictor_ = register_module("conf_predictor",
        torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, 1)));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> BaseTrackerPredictorImpl::forward(
    const torch::Tensor& fmaps,
    const torch::Tensor& targets,
    const torch::Tensor& coords,
    const torch::Tensor& visibility,
    const torch::Tensor& confidence,
    const torch::Tensor& mask) {
    auto B = coords.size(0);
    auto S = coords.size(1);
    auto N = coords.size(2);
    auto H = fmaps.size(3);
    auto W = fmaps.size(4);

    // Initialize correlation block
    CorrBlock corr_block(fmaps, corr_levels_, corr_radius_, multiple_track_feats_, padding_mode_);

    // Initialize coordinates, visibility, and confidence
    auto coords_pred = coords.clone();

    torch::Tensor vis_pred;
    if (visibility.defined() && visibility.numel() > 0) {
        vis_pred = visibility.clone();
    } else {
        vis_pred = torch::ones({B, S, N, 1}, coords.options());
    }

    torch::Tensor conf_pred;
    if (confidence.defined() && confidence.numel() > 0) {
        conf_pred = confidence.clone();
    } else {
        conf_pred = torch::ones({B, S, N, 1}, coords.options());
    }

    // Initialize mask if not provided
    torch::Tensor valid_mask;
    if (mask.defined() && mask.numel() > 0) {
        valid_mask = mask;
    } else {
        valid_mask = torch::ones({B, S, N}, coords.options().dtype(torch::kBool));
    }

    // Get position embeddings
    auto pos_embed = get_2d_embedding(coords_pred, H, W);

    // Iterative refinement
    for (int64_t itr = 0; itr < iters_; ++itr) {
        // Sample correlation features
        auto corr_feats = corr_block.corr_sample(targets, coords_pred);

        // Embed correlation features
        auto corr_embed = corr_embed_->forward(corr_feats);

        // Embed flow (coordinates)
        auto flow_embed = flow_embed_->forward(coords_pred);

        // Embed visibility
        auto vis_embed = vis_embed_->forward(vis_pred);

        // Embed confidence
        auto conf_embed = conf_embed_->forward(conf_pred);

        // Combine all features
        auto net_input = corr_embed + flow_embed + vis_embed + conf_embed;

        // Apply transformer update block
        auto net_output = update_block_->forward(net_input, pos_embed, ~valid_mask);

        // Predict delta coordinates
        auto delta_coords = coord_predictor_->forward(net_output);

        // Update coordinates
        coords_pred = coords_pred + delta_coords;

        // Predict visibility and confidence
        vis_pred = vis_predictor_->forward(net_output).sigmoid();
        conf_pred = conf_predictor_->forward(net_output).sigmoid();
    }

    return std::make_tuple(coords_pred, vis_pred, conf_pred);
}

torch::Tensor BaseTrackerPredictorImpl::get_2d_embedding(
    const torch::Tensor& coords,
    int64_t H,
    int64_t W) {
    auto B = coords.size(0);
    auto S = coords.size(1);
    auto N = coords.size(2);

    // Normalize coordinates to [-1, 1]
    auto x_norm = coords.index({"...", 0}) / (W - 1) * 2 - 1;
    auto y_norm = coords.index({"...", 1}) / (H - 1) * 2 - 1;

    // Create position encoding with different frequencies
    std::vector<torch::Tensor> pos_enc;

    for (int64_t i = 0; i < hidden_dim_ / 4; ++i) {
        // Compute frequency for this dimension
        auto freq = std::pow(2.0, static_cast<double>(i) / (hidden_dim_ / 4));

        // Compute sine and cosine embeddings
        pos_enc.push_back(torch::sin(x_norm * M_PI * freq));
        pos_enc.push_back(torch::cos(x_norm * M_PI * freq));
        pos_enc.push_back(torch::sin(y_norm * M_PI * freq));
        pos_enc.push_back(torch::cos(y_norm * M_PI * freq));
    }

    // Concatenate all embeddings
    auto pos_embedding = torch::cat(pos_enc, -1);

    return pos_embedding;
}

} // namespace track_modules
} // namespace vggt
