#include "dpt_head.h"
#include "head_act.h"
#include "utils.h"

namespace vggt {
namespace heads {

torch::nn::Module _make_scratch(const std::vector<int64_t>& in_shape, int64_t out_shape, int64_t groups, bool expand) {
    torch::nn::Module scratch;
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

    scratch.register_module("layer1_rn", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_shape[0], out_shape1, 3).stride(1).padding(1).bias(false).groups(groups)
    ));

    scratch.register_module("layer2_rn", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_shape[1], out_shape2, 3).stride(1).padding(1).bias(false).groups(groups)
    ));

    scratch.register_module("layer3_rn", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_shape[2], out_shape3, 3).stride(1).padding(1).bias(false).groups(groups)
    ));

    if (in_shape.size() >= 4) {
        scratch.register_module("layer4_rn", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_shape[3], out_shape4, 3).stride(1).padding(1).bias(false).groups(groups)
        ));
    }

    return scratch;
}

torch::Tensor custom_interpolate(
    torch::Tensor x,
    c10::optional<std::pair<int64_t, int64_t>> size,
    c10::optional<double> scale_factor,
    torch::nn::functional::InterpolateFuncOptions::mode_t mode,
    bool align_corners) {
    int64_t target_h, target_w;

    if (size.has_value()) {
        target_h = size.value().first;
        target_w = size.value().second;
    } else if (scale_factor.has_value()) {
        target_h = static_cast<int64_t>(x.size(-2) * scale_factor.value());
        target_w = static_cast<int64_t>(x.size(-1) * scale_factor.value());
    } else {
        throw std::runtime_error("Either size or scale_factor must be provided");
    }

    constexpr int64_t int_max = 1610612736;
    int64_t input_elements = target_h * target_w * x.size(0) * x.size(1);

    if (input_elements > int_max) {
        auto chunks = torch::chunk(x, (input_elements / int_max) + 1, 0);
        std::vector<torch::Tensor> interpolated_chunks;
        for (const auto& chunk : chunks) {
            interpolated_chunks.push_back(
                torch::nn::functional::interpolate(
                    chunk,
                    torch::nn::functional::InterpolateFuncOptions()
                        .size(std::vector<int64_t>({target_h, target_w}))
                        .mode(mode)
                        .align_corners(align_corners)
                )
            );
        }
        return torch::cat(interpolated_chunks, 0).contiguous();
    } else {
        return torch::nn::functional::interpolate(
            x,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>({target_h, target_w}))
                .mode(mode)
                .align_corners(align_corners)
        );
    }
}

ResidualConvUnitImpl::ResidualConvUnitImpl(int64_t features, torch::nn::AnyModule activation, bool bn, int64_t groups)
    : use_bn(bn),
      groups_(groups),
      activation_(std::move(activation)) {
    conv1 = register_module("conv1", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(features, features, 3).stride(1).padding(1).groups(groups).bias(true)
    ));
    conv2 = register_module("conv2", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(features, features, 3).stride(1).padding(1).groups(groups).bias(true)
    ));

    if (bn) {
        norm1 = register_module("norm1", torch::nn::BatchNorm2d(features));
        norm2 = register_module("norm2", torch::nn::BatchNorm2d(features));
    }
    register_module("activation_", activation_.ptr());
}

torch::Tensor ResidualConvUnitImpl::forward(torch::Tensor x) {
    auto out = activation_.forward<torch::Tensor>(x);
    out = conv1->forward(out);
    if (use_bn && !norm1.is_empty()) {
        out = norm1.forward<torch::Tensor>(out);
    }
    out = activation_.forward<torch::Tensor>(out);
    out = conv2->forward(out);
    if (use_bn && !norm2.is_empty()) {
        out = norm2.forward<torch::Tensor>(out);
    }
    return torch::add(out, x);
}

torch::nn::Module _make_fusion_block(
    int64_t features,
    c10::optional<std::pair<int64_t, int64_t>> size,
    bool has_residual,
    int64_t groups) {
    auto fusion_block = torch::nn::Module();
    fusion_block.register_module("resConfUnit", ResidualConvUnit(
        features,
        torch::nn::AnyModule(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))),
        false,
        groups
    ));
    return fusion_block;
}

FeatureFusionBlockImpl::FeatureFusionBlockImpl(
    int64_t features,
    torch::nn::AnyModule activation,
    bool deconv,
    bool bn,
    bool expand,
    bool align_corners,
    c10::optional<std::pair<int64_t, int64_t>> size,
    bool has_residual,
    int64_t groups)
    : deconv_(deconv),
      align_corners_(align_corners),
      groups_(groups),
      expand_(expand),
      has_residual_(has_residual),
      size_(size) {

    int64_t out_features = features;
    if (expand_) {
        out_features = features / 2;
    }

    out_conv = register_module("out_conv", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(features, out_features, 1).stride(1).padding(0).bias(true).groups(groups)
    ));

    if (has_residual) {
        resConfUnit1 = register_module("resConfUnit1", ResidualConvUnit(
            features, activation, bn, groups
        ));
    }

    resConfUnit2 = register_module("resConfUnit2", ResidualConvUnit(
        features, activation, bn, groups
    ));
}

torch::Tensor FeatureFusionBlockImpl::forward(
    const std::vector<torch::Tensor>& xs,
    c10::optional<std::pair<int64_t, int64_t>> size) {

    TORCH_CHECK(xs.size() >= 1, "FeatureFusionBlock requires at least 1 input");
    TORCH_CHECK(!has_residual_ || xs.size() >= 2, "FeatureFusionBlock with has_residual=true requires at least 2 inputs");

    torch::Tensor output = xs[0];

    if (has_residual_ && xs.size() >= 2) {
        torch::Tensor res = resConfUnit1->forward(xs[1]);
        // Interpolate residual to match output resolution if they differ
        if (res.sizes()[2] != output.sizes()[2] || res.sizes()[3] != output.sizes()[3]) {
            res = custom_interpolate(res, std::make_pair(output.size(2), output.size(3)),
                                     c10::nullopt, torch::kBilinear, align_corners_);
        }
        output = torch::add(output, res);
    }

    output = resConfUnit2->forward(output);

    // Determine interpolation parameters
    std::pair<int64_t, int64_t> target_size;
    bool has_size = false;

    if (size.has_value()) {
        target_size = size.value();
        has_size = true;
    } else if (size_.has_value()) {
        target_size = size_.value();
        has_size = true;
    }

    if (has_size) {
        output = custom_interpolate(
            output,
            target_size,
            c10::nullopt,
            torch::kBilinear,
            align_corners_
        );
    } else {
        output = custom_interpolate(
            output,
            c10::nullopt,
            2.0,
            torch::kBilinear,
            align_corners_
        );
    }

    output = out_conv->forward(output);
    return output;
}

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
    int64_t down_ratio)
    : patch_size_(patch_size),
      activation_(activation),
      conf_activation_(conf_activation),
      pos_embed_(pos_embed),
      feature_only_(feature_only),
      down_ratio_(down_ratio),
      intermediate_layer_idx_(intermediate_layer_idx) {

    norm_ = register_module("norm", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim_in})));

    // Create projection layers
    for (size_t i = 0; i < out_channels.size(); i++) {
        projects_.push_back(register_module(
            "projects_" + std::to_string(i),
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(dim_in, out_channels[i], 1).stride(1).padding(0)
            )
        ));
    }

    // Create resize layers
    // resize_layers[0]: ConvTranspose2d 4x upsampling for layer 0 (256 -> 256)
    // resize_layers[1]: ConvTranspose2d 2x upsampling for layer 1 (512 -> 512)
    // resize_layers[2]: Identity for layer 2 (1024)
    // resize_layers[3]: Conv2d 2x downsampling for layer 3 (1024 -> 1024)
    // Note: Must register modules first, then wrap in AnyModule
    auto resize0 = torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(out_channels[0], out_channels[0], 4).stride(4).padding(0)
    );
    register_module("resize0", resize0);
    resize_layers_.push_back(torch::nn::AnyModule(resize0));
    
    auto resize1 = torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(out_channels[1], out_channels[1], 2).stride(2).padding(0)
    );
    register_module("resize1", resize1);
    resize_layers_.push_back(torch::nn::AnyModule(resize1));
    
    resize_layers_.push_back(torch::nn::AnyModule(torch::nn::Identity()));
    
    auto resize3 = torch::nn::Conv2d(
        torch::nn::Conv2dOptions(out_channels[3], out_channels[3], 3).stride(2).padding(1)
    );
    register_module("resize3", resize3);
    resize_layers_.push_back(torch::nn::AnyModule(resize3));

    // Create scratch modules (feature fusion)
    layer1_rn = register_module("layer1_rn", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(out_channels[0], features, 3).stride(1).padding(1).bias(false)
    ));
    layer2_rn = register_module("layer2_rn", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(out_channels[1], features, 3).stride(1).padding(1).bias(false)
    ));
    layer3_rn = register_module("layer3_rn", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(out_channels[2], features, 3).stride(1).padding(1).bias(false)
    ));
    layer4_rn = register_module("layer4_rn", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(out_channels[3], features, 3).stride(1).padding(1).bias(false)
    ));

    // Fusion blocks
    refinenet1 = register_module("refinenet1", FeatureFusionBlock(
        features,
        torch::nn::AnyModule(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))),
        false, false, false, true, c10::nullopt, true, 1
    ));
    refinenet2 = register_module("refinenet2", FeatureFusionBlock(
        features,
        torch::nn::AnyModule(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))),
        false, false, false, true, c10::nullopt, true, 1
    ));
    refinenet3 = register_module("refinenet3", FeatureFusionBlock(
        features,
        torch::nn::AnyModule(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))),
        false, false, false, true, c10::nullopt, true, 1
    ));
    refinenet4 = register_module("refinenet4", FeatureFusionBlock(
        features,
        torch::nn::AnyModule(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true))),
        false, false, false, true, c10::nullopt, false, 1
    ));

    // Output conv1
    if (feature_only) {
        output_conv1 = register_module("output_conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(features, features, 3).stride(1).padding(1)
        ));
    } else {
        output_conv1 = register_module("output_conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(features, features / 2, 3).stride(1).padding(1)
        ));

        output_conv2_ = register_module("output_conv2_", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(features / 2, 32, 3).stride(1).padding(1)),
            torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, output_dim, 1).stride(1).padding(0))
        ));
    }
}

std::tuple<torch::Tensor, torch::Tensor> DPTHeadImpl::forward(
    const std::vector<torch::Tensor>& aggregated_tokens_list,
    torch::Tensor images,
    int64_t patch_start_idx,
    int64_t frames_chunk_size) {

    // If frames_chunk_size is not specified or greater than S, process all frames at once
    if (frames_chunk_size <= 0 || frames_chunk_size >= images.size(1)) {
        return forward_impl(aggregated_tokens_list, images, patch_start_idx);
    }

    // Otherwise, process frames in chunks to manage memory usage
    int64_t S = images.size(1);
    std::vector<torch::Tensor> all_preds;
    std::vector<torch::Tensor> all_conf;

    for (int64_t frames_start_idx = 0; frames_start_idx < S; frames_start_idx += frames_chunk_size) {
        int64_t frames_end_idx = std::min(frames_start_idx + frames_chunk_size, S);

        auto result = forward_impl(
            aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx);

        torch::Tensor chunk_preds = std::get<0>(result);
        torch::Tensor chunk_conf = std::get<1>(result);

        all_preds.push_back(chunk_preds);
        all_conf.push_back(chunk_conf);
    }

    torch::Tensor preds = torch::cat(all_preds, 1);
    torch::Tensor conf = torch::cat(all_conf, 1);
    return {preds, conf};
}

std::tuple<torch::Tensor, torch::Tensor> DPTHeadImpl::forward_impl(
    const std::vector<torch::Tensor>& aggregated_tokens_list,
    torch::Tensor images,
    int64_t patch_start_idx,
    c10::optional<int64_t> frames_start_idx,
    c10::optional<int64_t> frames_end_idx) {

    // Select frames if processing a chunk
    if (frames_start_idx.has_value() && frames_end_idx.has_value()) {
        images = images.index({torch::indexing::Slice(), torch::indexing::Slice(frames_start_idx.value(), frames_end_idx.value())}).contiguous();
    }

    int64_t B = images.size(0);
    int64_t S = images.size(1);
    int64_t C = images.size(2);  // channels = 3
    int64_t H = images.size(3);
    int64_t W = images.size(4);

    int64_t patch_h = H / patch_size_;
    int64_t patch_w = W / patch_size_;

    std::vector<torch::Tensor> out;
    int64_t dpt_idx = 0;

    for (int64_t layer_idx : intermediate_layer_idx_) {
        // Slice token dimension (dim 2) to remove special tokens (camera + register tokens)
        torch::Tensor x = aggregated_tokens_list[layer_idx].slice(2, patch_start_idx);

        // Select frames if processing a chunk
        if (frames_start_idx.has_value() && frames_end_idx.has_value()) {
            x = x.index({torch::indexing::Slice(), torch::indexing::Slice(frames_start_idx.value(), frames_end_idx.value())});
        }

        x = x.view({B * S, -1, x.size(-1)});

        x = norm_->forward(x);

        x = x.permute({0, 2, 1}).reshape({x.size(0), x.size(-1), patch_h, patch_w});

        x = projects_[dpt_idx]->forward(x);

        if (pos_embed_) {
            x = apply_pos_embed(x, W, H);
        }

        x = resize_layers_[dpt_idx].forward<torch::Tensor>(x);

        out.push_back(x);
        dpt_idx++;
    }

    // Fuse features from multiple layers
    torch::Tensor fused = scratch_forward(out);

    // Interpolate to match target image resolution
    int64_t target_h = static_cast<int64_t>(patch_h * patch_size_ / down_ratio_);
    int64_t target_w = static_cast<int64_t>(patch_w * patch_size_ / down_ratio_);

    fused = custom_interpolate(
        fused,
        std::make_pair(target_h, target_w),
        c10::nullopt,
        torch::kBilinear,
        true
    );

    if (pos_embed_) {
        fused = apply_pos_embed(fused, W, H);
    }

    if (feature_only_) {
        fused = fused.view({B, S, fused.size(1), fused.size(2), fused.size(3)});
        return {fused, torch::Tensor()};
    }

    fused = output_conv2_->forward(fused);

    auto result = activate_head(fused, activation_, conf_activation_);
    torch::Tensor preds = std::get<0>(result);
    torch::Tensor conf = std::get<1>(result);

    preds = preds.view({B, S, preds.size(1), preds.size(2), preds.size(3)});
    conf = conf.view({B, S, conf.size(1), conf.size(2), conf.size(3)});

    return {preds, conf};
}

torch::Tensor DPTHeadImpl::scratch_forward(const std::vector<torch::Tensor>& features) {
    TORCH_CHECK(features.size() == 4, "scratch_forward requires 4 features");

    torch::Tensor layer_1_rn_out = layer1_rn->forward(features[0]);
    torch::Tensor layer_2_rn_out = layer2_rn->forward(features[1]);
    torch::Tensor layer_3_rn_out = layer3_rn->forward(features[2]);
    torch::Tensor layer_4_rn_out = layer4_rn->forward(features[3]);

    std::vector<torch::Tensor> refinenet4_inputs = {layer_4_rn_out};
    torch::Tensor out = refinenet4->forward(refinenet4_inputs, std::make_pair(layer_3_rn_out.size(2), layer_3_rn_out.size(3)));

    std::vector<torch::Tensor> refinenet3_inputs = {out, layer_3_rn_out};
    out = refinenet3->forward(refinenet3_inputs, std::make_pair(layer_2_rn_out.size(2), layer_2_rn_out.size(3)));

    std::vector<torch::Tensor> refinenet2_inputs = {out, layer_2_rn_out};
    out = refinenet2->forward(refinenet2_inputs, std::make_pair(layer_1_rn_out.size(2), layer_1_rn_out.size(3)));

    std::vector<torch::Tensor> refinenet1_inputs = {out, layer_1_rn_out};
    out = refinenet1->forward(refinenet1_inputs);

    out = output_conv1->forward(out);
    return out;
}

torch::Tensor DPTHeadImpl::apply_pos_embed(const torch::Tensor& x, int64_t W, int64_t H, float ratio) {
    int64_t patch_w = x.size(-1);
    int64_t patch_h = x.size(-2);

    torch::Tensor pos_embed = create_uv_grid(patch_w, patch_h, static_cast<float>(W) / static_cast<float>(H), c10::make_optional(x.dtype().toScalarType()), c10::make_optional(x.device()));
    pos_embed = position_grid_to_embed(pos_embed, x.size(1));
    pos_embed = pos_embed * ratio;
    pos_embed = pos_embed.permute({2, 0, 1}).unsqueeze(0).expand({x.size(0), -1, -1, -1});

    return x + pos_embed;
}

} // namespace heads
} // namespace vggt
