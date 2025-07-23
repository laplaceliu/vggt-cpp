#include "dpt_head.h"
#include "head_act.h"
#include "utils.h"

namespace vggt {
namespace heads {

DPTHeadImpl::DPTHeadImpl(
    int dim_in,
    int patch_size,
    int output_dim,
    const std::string& activation,
    const std::string& conf_activation,
    int features,
    const std::vector<int>& out_channels,
    const std::vector<int>& intermediate_layer_idx,
    bool pos_embed,
    bool feature_only,
    int down_ratio
) : patch_size(patch_size),
    activation(activation),
    conf_activation(conf_activation),
    pos_embed(pos_embed),
    feature_only(feature_only),
    down_ratio(down_ratio),
    intermediate_layer_idx(intermediate_layer_idx),
    norm(torch::nn::LayerNorm(dim_in)) {
    
    // Initialize projection layers
    for (const auto& oc : out_channels) {
        projects->push_back(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(dim_in, oc, 1).stride(1).padding(0)
        ));
    }

    // Initialize resize layers
    resize_layers->push_back(torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(out_channels[0], out_channels[0], 4).stride(4).padding(0)
    ));
    resize_layers->push_back(torch::nn::ConvTranspose2d(
        torch::nn::ConvTranspose2dOptions(out_channels[1], out_channels[1], 2).stride(2).padding(0)
    ));
    resize_layers->push_back(torch::nn::Identity());
    resize_layers->push_back(torch::nn::Conv2d(
        torch::nn::Conv2dOptions(out_channels[3], out_channels[3], 3).stride(2).padding(1)
    ));

    // Initialize scratch module
    scratch = _make_scratch(out_channels, features, false);

    // Attach additional modules to scratch
    scratch->register_module("stem_transpose", nullptr);
    scratch->register_module("refinenet1", _make_fusion_block(features));
    scratch->register_module("refinenet2", _make_fusion_block(features));
    scratch->register_module("refinenet3", _make_fusion_block(features));
    scratch->register_module("refinenet4", _make_fusion_block(features, false));

    int head_features_1 = features;
    int head_features_2 = 32;

    if (feature_only) {
        scratch->register_module("output_conv1", 
            torch::nn::Conv2d(torch::nn::Conv2dOptions(head_features_1, head_features_1, 3).stride(1).padding(1))
        );
    } else {
        scratch->register_module("output_conv1", 
            torch::nn::Conv2d(torch::nn::Conv2dOptions(head_features_1, head_features_1 / 2, 3).stride(1).padding(1))
        );
        int conv2_in_channels = head_features_1 / 2;

        scratch->register_module("output_conv2", 
            torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(conv2_in_channels, head_features_2, 3).stride(1).padding(1)),
                torch::nn::ReLU(true),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(head_features_2, output_dim, 1).stride(1).padding(0))
            )
        );
    }
}

torch::Tensor DPTHeadImpl::forward(
    const std::vector<torch::Tensor>& aggregated_tokens_list,
    const torch::Tensor& images,
    int patch_start_idx,
    int frames_chunk_size
) {
    int B = images.size(0);
    int S = images.size(1);

    if (frames_chunk_size <= 0 || frames_chunk_size >= S) {
        return _forward_impl(aggregated_tokens_list, images, patch_start_idx);
    }

    std::vector<torch::Tensor> all_preds;
    std::vector<torch::Tensor> all_conf;

    for (int frames_start_idx = 0; frames_start_idx < S; frames_start_idx += frames_chunk_size) {
        int frames_end_idx = std::min(frames_start_idx + frames_chunk_size, S);

        if (feature_only) {
            auto chunk_output = _forward_impl(
                aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
            );
            all_preds.push_back(chunk_output);
        } else {
            auto [chunk_preds, chunk_conf] = _forward_impl(
                aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
            );
            all_preds.push_back(chunk_preds);
            all_conf.push_back(chunk_conf);
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
    int patch_start_idx,
    int frames_start_idx,
    int frames_end_idx
) {
    auto images_chunk = images;
    if (frames_start_idx >= 0 && frames_end_idx >= 0) {
        images_chunk = images.index({torch::indexing::Slice(), torch::indexing::Slice(frames_start_idx, frames_end_idx)}).contiguous();
    }

    int B = images_chunk.size(0);
    int S = images_chunk.size(1);
    int H = images_chunk.size(3);
    int W = images_chunk.size(4);

    int patch_h = H / patch_size;
    int patch_w = W / patch_size;

    std::vector<torch::Tensor> out;
    int dpt_idx = 0;

    for (int layer_idx : intermediate_layer_idx) {
        auto x = aggregated_tokens_list[layer_idx].index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(patch_start_idx, torch::indexing::None)});

        if (frames_start_idx >= 0 && frames_end_idx >= 0) {
            x = x.index({torch::indexing::Slice(), torch::indexing::Slice(frames_start_idx, frames_end_idx)});
        }

        x = x.view({B * S, -1, x.size(-1)});
        x = norm(x);
        x = x.permute({0, 2, 1}).reshape({x.size(0), x.size(1), patch_h, patch_w});
        x = projects[dpt_idx]->as<torch::nn::Conv2d>()->forward(x);

        if (pos_embed) {
            x = _apply_pos_embed(x, W, H);
        }

        x = resize_layers[dpt_idx]->as<torch::nn::Module>()->forward(x).toTensor();
        out.push_back(x);
        dpt_idx++;
    }

    auto fused = scratch_forward(out);
    auto interpolated = custom_interpolate(
        fused,
        {static_cast<int>(patch_h * patch_size / down_ratio), static_cast<int>(patch_w * patch_size / down_ratio)},
        "bilinear",
        true
    );

    if (pos_embed) {
        interpolated = _apply_pos_embed(interpolated, W, H);
    }

    if (feature_only) {
        return interpolated.view({B, S, interpolated.size(1), interpolated.size(2), interpolated.size(3)});
    }

    auto out_conv2 = scratch->named_children()["output_conv2"]->as<torch::nn::Sequential>()->forward(interpolated);
    auto [preds, conf] = activate_head(out_conv2, activation, conf_activation);

    preds = preds.view({B, S, preds.size(1), preds.size(2), preds.size(3)});
    conf = conf.view({B, S, conf.size(1), conf.size(2), conf.size(3)});
    return {preds, conf};
}

torch::Tensor DPTHeadImpl::_apply_pos_embed(
    const torch::Tensor& x,
    int W,
    int H,
    float ratio
) {
    int patch_w = x.size(-1);
    int patch_h = x.size(-2);
    auto pos_embed = create_uv_grid(patch_w, patch_h, static_cast<float>(W) / H, x.dtype(), x.device());
    pos_embed = position_grid_to_embed(pos_embed, x.size(1));
    pos_embed = pos_embed * ratio;
    pos_embed = pos_embed.permute({2, 0, 1}).unsqueeze(0).expand({x.size(0), -1, -1, -1});
    return x + pos_embed;
}

torch::Tensor DPTHeadImpl::scratch_forward(
    const std::vector<torch::Tensor>& features
) {
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

    out = scratch->named_children()["output_conv1"]->as<torch::nn::Conv2d>()->forward(out);
    return out;
}

torch::nn::Module _make_fusion_block(int features, bool has_residual = true) {
    return FeatureFusionBlock(
        features,
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        false,
        false,
        false,
        true,
        has_residual
    );
}

torch::nn::Module _make_scratch(const std::vector<int>& in_shape, int out_shape, bool expand = false) {
    torch::nn::Module scratch;
    int out_shape1 = out_shape;
    int out_shape2 = out_shape;
    int out_shape3 = out_shape;
    int out_shape4 = out_shape;

    if (expand) {
        out_shape1 = out_shape;
        out_shape2 = out_shape * 2;
        out_shape3 = out_shape * 4;
        out_shape4 = out_shape * 8;
    }

    scratch->register_module("layer1_rn", 
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_shape[0], out_shape1, 3).stride(1).padding(1).bias(false))
    );
    scratch->register_module("layer2_rn", 
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_shape[1], out_shape2, 3).stride(1).padding(1).bias(false))
    );
    scratch->register_module("layer3_rn", 
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_shape[2], out_shape3, 3).stride(1).padding(1).bias(false))
    );
    if (in_shape.size() >= 4) {
        scratch->register_module("layer4_rn", 
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_shape[3], out_shape4, 3).stride(1).padding(1).bias(false))
        );
    }
    return scratch;
}

} // namespace heads
} // namespace vggt