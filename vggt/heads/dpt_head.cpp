// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "dpt_head.h"

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
    int64_t down_ratio
) {
    this->patch_size = patch_size;
    this->activation = activation;
    this->conf_activation = conf_activation;
    this->pos_embed = pos_embed;
    this->feature_only = feature_only;
    this->down_ratio = down_ratio;
    this->intermediate_layer_idx = intermediate_layer_idx;

    norm = register_module("norm", torch::nn::LayerNorm(dim_in));

    // Projection layers for each output channel from tokens.
    for (size_t i = 0; i < out_channels.size(); ++i) {
        projects->push_back(
            register_module(
                "project_" + std::to_string(i),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(dim_in, out_channels[i], 1).stride(1).padding(0))
            )
        );
    }

    // Resize layers for upsampling feature maps.
    resize_layers->push_back(
        register_module(
            "resize_0",
            torch::nn::ConvTranspose2d(
                torch::nn::ConvTranspose2dOptions(out_channels[0], out_channels[0], 4).stride(4).padding(0)
            )
        )
    );
    resize_layers->push_back(
        register_module(
            "resize_1",
            torch::nn::ConvTranspose2d(
                torch::nn::ConvTranspose2dOptions(out_channels[1], out_channels[1], 2).stride(2).padding(0)
            )
        )
    );
    resize_layers->push_back(register_module("resize_2", torch::nn::Identity()));
    resize_layers->push_back(
        register_module(
            "resize_3",
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(out_channels[3], out_channels[3], 3).stride(2).padding(1)
            )
        )
    );

    // Initialize scratch modules.
    scratch = register_module("scratch", torch::nn::Module());
    scratch->register_module("layer1_rn", torch::nn::Conv2d(out_channels[0], features, 3, 1, 1, false));
    scratch->register_module("layer2_rn", torch::nn::Conv2d(out_channels[1], features, 3, 1, 1, false));
    scratch->register_module("layer3_rn", torch::nn::Conv2d(out_channels[2], features, 3, 1, 1, false));
    scratch->register_module("layer4_rn", torch::nn::Conv2d(out_channels[3], features, 3, 1, 1, false));

    scratch->register_module("refinenet1", _make_fusion_block(features));
    scratch->register_module("refinenet2", _make_fusion_block(features));
    scratch->register_module("refinenet3", _make_fusion_block(features));
    scratch->register_module("refinenet4", _make_fusion_block(features, false));

    if (feature_only) {
        scratch->register_module(
            "output_conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(features, features, 3).stride(1).padding(1))
        );
    } else {
        scratch->register_module(
            "output_conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(features, features / 2, 3).stride(1).padding(1))
        );
        scratch->register_module(
            "output_conv2",
            torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(features / 2, 32, 3).stride(1).padding(1)),
                torch::nn::ReLU(true),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(32, output_dim, 1).stride(1).padding(0))
            )
        );
    }
}

torch::Tensor DPTHeadImpl::forward(
    const std::vector<torch::Tensor>& aggregated_tokens_list,
    const torch::Tensor& images,
    int64_t patch_start_idx,
    int64_t frames_chunk_size
) {
    auto B = images.size(0);
    auto S = images.size(1);

    if (frames_chunk_size <= 0 || frames_chunk_size >= S) {
        return _forward_impl(aggregated_tokens_list, images, patch_start_idx);
    }

    std::vector<torch::Tensor> all_preds;
    std::vector<torch::Tensor> all_conf;

    for (int64_t frames_start_idx = 0; frames_start_idx < S; frames_start_idx += frames_chunk_size) {
        int64_t frames_end_idx = std::min(frames_start_idx + frames_chunk_size, S);

        if (feature_only) {
            all_preds.push_back(
                _forward_impl(aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx)
            );
        } else {
            auto [preds, conf] = _forward_impl(
                aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
            );
            all_preds.push_back(preds);
            all_conf.push_back(conf);
        }
    }

    if (feature_only) {
        return torch::cat(all_preds, 1);
    } else {
        return {torch::cat(all_preds, 1), torch::cat(all_conf, 1)};
    }
}

torch::Tensor DPTHeadImpl::_forward_impl(
    const std::vector<torch::Tensor>& aggregated_tokens_list,
    const torch::Tensor& images,
    int64_t patch_start_idx,
    int64_t frames_start_idx,
    int64_t frames_end_idx
) {
    auto images_chunk = images;
    if (frames_start_idx >= 0 && frames_end_idx >= 0) {
        images_chunk = images.index({torch::indexing::Slice(), torch::indexing::Slice(frames_start_idx, frames_end_idx)}).contiguous();
    }

    auto B = images_chunk.size(0);
    auto S = images_chunk.size(1);
    auto H = images_chunk.size(3);
    auto W = images_chunk.size(4);

    auto patch_h = H / patch_size;
    auto patch_w = W / patch_size;

    std::vector<torch::Tensor> out;
    size_t dpt_idx = 0;

    for (auto layer_idx : intermediate_layer_idx) {
        auto x = aggregated_tokens_list[layer_idx].index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(patch_start_idx, torch::indexing::None)});

        if (frames_start_idx >= 0 && frames_end_idx >= 0) {
            x = x.index({torch::indexing::Slice(), torch::indexing::Slice(frames_start_idx, frames_end_idx)});
        }

        x = x.view({B * S, -1, x.size(-1)});
        x = norm(x);
        x = x.permute({0, 2, 1}).reshape({x.size(0), x.size(-1), patch_h, patch_w});
        x = projects[dpt_idx]->as<torch::nn::Conv2d>()->forward(x);

        if (pos_embed) {
            x = _apply_pos_embed(x, W, H);
        }

        x = resize_layers[dpt_idx]->as<torch::nn::Module>()->forward(x).toTensor();
        out.push_back(x);
        dpt_idx++;
    }

    auto fused = scratch_forward(out);
    auto output = torch::nn::functional::interpolate(
        fused,
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{patch_h * patch_size / down_ratio, patch_w * patch_size / down_ratio})
            .mode(torch::kLinear)
            .align_corners(true)
    );

    if (pos_embed) {
        output = _apply_pos_embed(output, W, H);
    }

    if (feature_only) {
        return output.view({B, S, output.size(1), output.size(2), output.size(3)});
    }

    auto out_conv2 = scratch->named_children()["output_conv2"]->as<torch::nn::Sequential>()->forward(output);
    auto [preds, conf] = activate_head(out_conv2, activation, conf_activation);

    preds = preds.view({B, S, preds.size(1), preds.size(2), preds.size(3)});
    conf = conf.view({B, S, conf.size(1), conf.size(2), conf.size(3)});
    return {preds, conf};
}

torch::Tensor DPTHeadImpl::_apply_pos_embed(const torch::Tensor& x, int64_t W, int64_t H, float ratio) {
    auto patch_w = x.size(-1);
    auto patch_h = x.size(-2);
    auto pos_embed = create_uv_grid(patch_w, patch_h, static_cast<float>(W) / H, x.dtype(), x.device());
    pos_embed = position_grid_to_embed(pos_embed, x.size(1));
    pos_embed = pos_embed * ratio;
    pos_embed = pos_embed.permute({2, 0, 1}).unsqueeze(0).expand({x.size(0), -1, -1, -1});
    return x + pos_embed;
}

torch::Tensor DPTHeadImpl::scratch_forward(const std::vector<torch::Tensor>& features) {
    auto layer_1 = features[0];
    auto layer_2 = features[1];
    auto layer_3 = features[2];
    auto layer_4 = features[3];

    auto layer_1_rn = scratch->named_children()["layer1_rn"]->as<torch::nn::Conv2d>()->forward(layer_1);
    auto layer_2_rn = scratch->named_children()["layer2_rn"]->as<torch::nn::Conv2d>()->forward(layer_2);
    auto layer_3_rn = scratch->named_children()["layer3_rn"]->as<torch::nn::Conv2d>()->forward(layer_3);
    auto layer_4_rn = scratch->named_children()["layer4_rn"]->as<torch::nn::Conv2d>()->forward(layer_4);

    auto out = scratch->named_children()["refinenet4"]->as<torch::nn::Module>()->forward(layer_4_rn);
    out = scratch->named_children()["refinenet3"]->as<torch::nn::Module>()->forward({out, layer_3_rn});
    out = scratch->named_children()["refinenet2"]->as<torch::nn::Module>()->forward({out, layer_2_rn});
    out = scratch->named_children()["refinenet1"]->as<torch::nn::Module>()->forward({out, layer_1_rn});

    return scratch->named_children()["output_conv1"]->as<torch::nn::Conv2d>()->forward(out);
}

} // namespace heads
} // namespace vggt