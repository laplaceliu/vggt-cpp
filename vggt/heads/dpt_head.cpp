/**
 * @file dpt_head.cpp
 * @brief Implementation of DPT head module for dense prediction tasks
 */

#include "dpt_head.h"
#include "head_act.h"
#include "utils.h"
#include <torch/nn/init.h>
#include <stdexcept>
#include <cmath>

namespace vggt {
namespace heads {

DPTHeadImpl::DPTHeadImpl(
    int64_t dim_in,
    int64_t patch_size,
    int64_t output_dim,
    const std::string& activation,
    const std::string& conf_activation,
    int64_t features,
    const std::vector<int64_t>& out_channels,
    const std::vector<int64_t>& intermediate_layer_idx,
    bool pos_embed,
    bool feature_only,
    int64_t down_ratio) {
    // Initialize parameters
    patch_size_ = patch_size;
    activation_ = activation;
    conf_activation_ = conf_activation;
    pos_embed_ = pos_embed;
    feature_only_ = feature_only;
    down_ratio_ = down_ratio;
    intermediate_layer_idx_ = intermediate_layer_idx;

    // Initialize norm layer
    norm_ = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim_in})));

    // Initialize projection layers
    for (size_t i = 0; i < out_channels.size(); ++i) {
        projects_->push_back(register_module(
            "project_" + std::to_string(i),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(dim_in, out_channels[i], 1))));
    }

    // Initialize resize layers
    resize_layers_->push_back(register_module(
        "resize_0",
        torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(
            out_channels[0], out_channels[0], 4).stride(4))));

    resize_layers_->push_back(register_module(
        "resize_1",
        torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(
            out_channels[1], out_channels[1], 2).stride(2))));

    resize_layers_->push_back(register_module("resize_2", torch::nn::Identity()));

    resize_layers_->push_back(register_module(
        "resize_3",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(
            out_channels[3], out_channels[3], 3).stride(2).padding(1))));

    // Initialize scratch
    scratch_ = register_module("scratch", make_scratch(out_channels, features, false));

    // Initialize output conv layers
    if (feature_only) {
        scratch_->output_conv1 = register_module(
            "output_conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(
                features, features, 3).padding(1)));
    } else {
        scratch_->output_conv1 = register_module(
            "output_conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(
                features, features / 2, 3).padding(1)));

        scratch_->output_conv2 = register_module(
            "output_conv2",
            torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(
                    features / 2, 32, 3).padding(1)),
                torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(32, output_dim, 1))));
    }

    // Initialize refinenet blocks
    scratch_->refinenet1 = register_module(
        "refinenet1", make_fusion_block(features));
    scratch_->refinenet2 = register_module(
        "refinenet2", make_fusion_block(features));
    scratch_->refinenet3 = register_module(
        "refinenet3", make_fusion_block(features));
    scratch_->refinenet4 = register_module(
        "refinenet4", make_fusion_block(features, false));

    // Initialize weights
    for (auto& module : modules()) {
        if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
            torch::nn::init::xavier_uniform_(M->weight);
            if (M->bias.defined()) {
                torch::nn::init::zeros_(M->bias);
            }
        }
    }
}

torch::Tensor DPTHeadImpl::forward(
    const std::vector<torch::Tensor>& aggregated_tokens_list,
    const torch::Tensor& images,
    int64_t patch_start_idx,
    int64_t frames_chunk_size) {
    // Check input
    if (aggregated_tokens_list.empty()) {
        throw std::runtime_error("aggregated_tokens_list cannot be empty");
    }

    auto B = images.size(0);
    auto S = images.size(1);
    auto H = images.size(3);
    auto W = images.size(4);

    // If frames_chunk_size is not specified or greater than S, process all frames at once
    if (frames_chunk_size <= 0 || frames_chunk_size >= S) {
        return _forward_impl(aggregated_tokens_list, images, patch_start_idx);
    }

    // Process frames in chunks to manage memory usage
    std::vector<torch::Tensor> all_outputs;
    for (int64_t frames_start_idx = 0; frames_start_idx < S; frames_start_idx += frames_chunk_size) {
        int64_t frames_end_idx = std::min(frames_start_idx + frames_chunk_size, S);

        auto chunk_output = _forward_impl(
            aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx);

        all_outputs.push_back(chunk_output);
    }

    // Concatenate results along the sequence dimension
    return torch::cat(all_outputs, 1);
}

torch::Tensor DPTHeadImpl::_forward_impl(
    const std::vector<torch::Tensor>& aggregated_tokens_list,
    const torch::Tensor& images,
    int64_t patch_start_idx,
    int64_t frames_start_idx,
    int64_t frames_end_idx) {
    auto B = images.size(0);
    auto S = frames_end_idx > 0 ? (frames_end_idx - frames_start_idx) : images.size(1);
    auto H = images.size(3);
    auto W = images.size(4);

    auto patch_h = H / patch_size_;
    auto patch_w = W / patch_size_;

    std::vector<torch::Tensor> out;
    int64_t dpt_idx = 0;

    for (auto layer_idx : intermediate_layer_idx_) {
        auto x = aggregated_tokens_list[layer_idx].slice(2, patch_start_idx, -1);

        // Select frames if processing a chunk
        if (frames_start_idx > 0 && frames_end_idx > 0) {
            x = x.slice(1, frames_start_idx, frames_end_idx);
        }

        x = x.view({B * S, -1, x.size(-1)});
        x = norm_(x);
        x = x.permute({0, 2, 1}).reshape({x.size(0), x.size(-1), patch_h, patch_w});

        x = projects_[dpt_idx]->as<torch::nn::Conv2d>()->forward(x);
        if (pos_embed_) {
            x = _apply_pos_embed(x, W, H);
        }
        x = resize_layers_[dpt_idx]->as<torch::nn::Module>()->forward(x).toTensor();

        out.push_back(x);
        dpt_idx++;
    }

    // Fuse features from multiple layers
    auto fused = scratch_forward(out);

    // Interpolate fused output to match target image resolution
    auto target_h = static_cast<int64_t>(patch_h * patch_size_ / down_ratio_);
    auto target_w = static_cast<int64_t>(patch_w * patch_size_ / down_ratio_);

    fused = custom_interpolate(
        fused, {target_h, target_w}, "bilinear", true);

    if (pos_embed_) {
        fused = _apply_pos_embed(fused, W, H);
    }

    if (feature_only_) {
        return fused.view({B, S, fused.size(1), target_h, target_w});
    }

    auto out_conv = scratch_->output_conv2->as<torch::nn::Sequential>()->forward(fused);
    auto [preds, conf] = activate_head(out_conv, activation_, conf_activation_);

    return {preds.view({B, S, preds.size(1), target_h, target_w}),
            conf.view({B, S, conf.size(1), target_h, target_w})};
}

torch::Tensor DPTHeadImpl::_apply_pos_embed(
    const torch::Tensor& x,
    int64_t W,
    int64_t H,
    float ratio) {
    auto patch_w = x.size(-1);
    auto patch_h = x.size(-2);

    auto pos_embed = create_uv_grid(patch_w, patch_h, static_cast<float>(W) / H, x.dtype(), x.device());
    pos_embed = position_grid_to_embed(pos_embed, x.size(1));
    pos_embed = pos_embed * ratio;
    pos_embed = pos_embed.permute({2, 0, 1}).unsqueeze(0).expand({x.size(0), -1, -1, -1});

    return x + pos_embed;
}

torch::Tensor DPTHeadImpl::scratch_forward(
    const std::vector<torch::Tensor>& features) {
    auto layer_1 = features[0];
    auto layer_2 = features[1];
    auto layer_3 = features[2];
    auto layer_4 = features[3];

    auto layer_1_rn = scratch_->layer1_rn->as<torch::nn::Conv2d>()->forward(layer_1);
    auto layer_2_rn = scratch_->layer2_rn->as<torch::nn::Conv2d>()->forward(layer_2);
    auto layer_3_rn = scratch_->layer3_rn->as<torch::nn::Conv2d>()->forward(layer_3);
    auto layer_4_rn = scratch_->layer4_rn->as<torch::nn::Conv2d>()->forward(layer_4);

    auto out = scratch_->refinenet4->as<FeatureFusionBlock>()->forward(layer_4_rn, {}, layer_3_rn.sizes().slice(2));
    out = scratch_->refinenet3->as<FeatureFusionBlock>()->forward(out, layer_3_rn, layer_2_rn.sizes().slice(2));
    out = scratch_->refinenet2->as<FeatureFusionBlock>()->forward(out, layer_2_rn, layer_1_rn.sizes().slice(2));
    out = scratch_->refinenet1->as<FeatureFusionBlock>()->forward(out, layer_1_rn);

    return scratch_->output_conv1->as<torch::nn::Conv2d>()->forward(out);
}

// Helper class implementations
void Scratch::reset() {
    layer1_rn = nullptr;
    layer2_rn = nullptr;
    layer3_rn = nullptr;
    layer4_rn = nullptr;
    refinenet1 = nullptr;
    refinenet2 = nullptr;
    refinenet3 = nullptr;
    refinenet4 = nullptr;
    output_conv1 = nullptr;
    output_conv2 = nullptr;
    stem_transpose = nullptr;
}

ResidualConvUnitImpl::ResidualConvUnitImpl(
    int64_t features,
    const torch::nn::AnyModule& activation,
    bool bn,
    int64_t groups) {
    conv1_ = register_module("conv1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(features, features, 3).padding(1).groups(groups)));
    conv2_ = register_module("conv2", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(features, features, 3).padding(1).groups(groups)));

    if (bn) {
        norm1_ = register_module("norm1", torch::nn::BatchNorm2d(features));
        norm2_ = register_module("norm2", torch::nn::BatchNorm2d(features));
    }

    activation_ = activation;
    bn_ = bn;
    groups_ = groups;

    // Initialize weights
    torch::nn::init::xavier_uniform_(conv1_->weight);
    torch::nn::init::zeros_(conv1_->bias);
    torch::nn::init::xavier_uniform_(conv2_->weight);
    torch::nn::init::zeros_(conv2_->bias);
}

torch::Tensor ResidualConvUnitImpl::forward(const torch::Tensor& x) {
    auto out = activation_.forward(x).toTensor();
    out = conv1_->forward(out);
    if (norm1_) {
        out = norm1_->forward(out);
    }

    out = activation_.forward(out).toTensor();
    out = conv2_->forward(out);
    if (norm2_) {
        out = norm2_->forward(out);
    }

    return out + x;
}

FeatureFusionBlockImpl::FeatureFusionBlockImpl(
    int64_t features,
    const torch::nn::AnyModule& activation,
    bool deconv,
    bool bn,
    bool expand,
    bool align_corners,
    const std::vector<int64_t>& size,
    bool has_residual,
    int64_t groups) {
    deconv_ = deconv;
    align_corners_ = align_corners;
    groups_ = groups;
    expand_ = expand;
    size_ = size;
    has_residual_ = has_residual;

    int64_t out_features = features;
    if (expand) {
        out_features = features / 2;
    }

    out_conv_ = register_module("out_conv", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(features, out_features, 1).groups(groups)));

    if (has_residual) {
        resConfUnit1_ = register_module("resConfUnit1",
            ResidualConvUnit(features, activation, bn, groups));
    }

    resConfUnit2_ = register_module("resConfUnit2",
        ResidualConvUnit(features, activation, bn, groups));

    // Initialize weights
    torch::nn::init::xavier_uniform_(out_conv_->weight);
    torch::nn::init::zeros_(out_conv_->bias);
}

torch::Tensor FeatureFusionBlockImpl::forward(
    const torch::Tensor& x0,
    const torch::Tensor& x1,
    const std::vector<int64_t>& size) {
    auto output = x0;

    if (has_residual_ && x1.defined()) {
        auto res = resConfUnit1_->forward(x1);
        output = output + res;
    }

    output = resConfUnit2_->forward(output);

    std::vector<int64_t> target_size;
    if (!size.empty()) {
        target_size = size;
    } else if (!size_.empty()) {
        target_size = size_;
    } else {
        target_size = {output.size(2) * 2, output.size(3) * 2};
    }

    output = custom_interpolate(output, target_size, "bilinear", align_corners_);
    return out_conv_->forward(output);
}

// Helper functions
FeatureFusionBlock make_fusion_block(
    int64_t features,
    const std::vector<int64_t>& size,
    bool has_residual,
    int64_t groups) {
    return FeatureFusionBlock(
        features,
        torch::nn::AnyModule(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))),
        false, false, false, true, size, has_residual, groups);
}

Scatch make_scratch(
    const std::vector<int64_t>& in_shape,
    int64_t out_shape,
    bool expand,
    int64_t groups) {
    auto scratch = std::make_shared<Scratch>();

    int64_t out_shape1 = out_shape;
    int64_t out_shape2 = out_shape;
    int64_t out_shape3 = out_shape;
    int64_t out_shape4 = out_shape;

    if (expand) {
        out_shape1 = out_shape;
        out_shape2 = out_shape * 2;
        out_shape3 = out_shape * 4;
        if (in_shape.size() >= 4) {
            out_shape4 = out_shape * 8;
        }
    }

    scratch->layer1_rn = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_shape[0], out_shape1, 3).padding(1).groups(groups));
    scratch->layer2_rn = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_shape[1], out_shape2, 3).padding(1).groups(groups));
    scratch->layer3_rn = torch::nn::Conv2d(
        torch::nn::Conv2dOptions
