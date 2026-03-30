#include "aggregator.h"
#include "../layers/vision_transformer.h"
#include <stdexcept>

namespace vggt {
namespace models {

AggregatorImpl::AggregatorImpl(
    int64_t img_size,
    int64_t patch_size,
    int64_t embed_dim,
    int64_t depth,
    int64_t num_heads,
    double mlp_ratio,
    int64_t num_register_tokens,
    bool qkv_bias,
    bool proj_bias,
    bool ffn_bias,
    const std::string& patch_embed,
    const std::vector<std::string>& aa_order,
    int64_t aa_block_size,
    bool qk_norm,
    double rope_freq,
    double init_values
) : depth_(depth),
    aa_block_size_(aa_block_size),
    aa_order_(aa_order),
    patch_size_(patch_size) {

    // Validate that depth is divisible by aa_block_size
    if (depth_ % aa_block_size_ != 0) {
        throw std::invalid_argument("depth must be divisible by aa_block_size");
    }
    aa_block_num_ = depth_ / aa_block_size_;

    // Build patch embed
    buildPatchEmbed(patch_embed, img_size, patch_size, num_register_tokens,
                    true, 0.0, 0, 1.0, embed_dim);

    // Initialize rotary position embedding if frequency > 0
    if (rope_freq > 0) {
        rope_ = layers::RotaryPositionEmbedding2D(rope_freq);
        register_module("rope", rope_);
        position_getter_ = std::make_shared<layers::PositionGetter>();
    }

    // Create frame blocks
    for (int64_t i = 0; i < depth_; ++i) {
        torch::Tensor init_vals = {};
        if (init_values > 0) {
            init_vals = torch::tensor(init_values);
        }

        torch::nn::AnyModule rope_module;
        if (!rope_.is_empty()) {
            rope_module = torch::nn::AnyModule(rope_);
        }

        auto block = layers::Block(
            embed_dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            proj_bias,
            ffn_bias,
            0.0,  // drop
            0.0,  // attn_drop
            init_vals,
            0.0,  // drop_path
            torch::nn::AnyModule(torch::nn::GELU()),
            torch::nn::AnyModule(torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim}))),
            torch::nn::AnyModule(),  // attn_class
            torch::nn::AnyModule(),  // ffn_layer
            qk_norm,
            true,  // fused_attn
            rope_module
        );

        frame_blocks_.push_back(block);
        register_module("frame_block_" + std::to_string(i), frame_blocks_.back().ptr());
    }

    // Create global blocks
    for (int64_t i = 0; i < depth_; ++i) {
        torch::Tensor init_vals = {};
        if (init_values > 0) {
            init_vals = torch::tensor(init_values);
        }

        torch::nn::AnyModule rope_module;
        if (!rope_.is_empty()) {
            rope_module = torch::nn::AnyModule(rope_);
        }

        auto block = layers::Block(
            embed_dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            proj_bias,
            ffn_bias,
            0.0,  // drop
            0.0,  // attn_drop
            init_vals,
            0.0,  // drop_path
            torch::nn::AnyModule(torch::nn::GELU()),
            torch::nn::AnyModule(torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim}))),
            torch::nn::AnyModule(),  // attn_class
            torch::nn::AnyModule(),  // ffn_layer
            qk_norm,
            true,  // fused_attn
            rope_module
        );

        global_blocks_.push_back(block);
        register_module("global_block_" + std::to_string(i), global_blocks_.back().ptr());
    }

    // Special tokens
    camera_token_ = register_parameter("camera_token", torch::empty({1, 2, 1, embed_dim}));
    torch::nn::init::normal_(camera_token_, 0.0, 1e-6);

    register_token_ = register_parameter("register_token",
        torch::empty({1, 2, num_register_tokens, embed_dim}));
    torch::nn::init::normal_(register_token_, 0.0, 1e-6);

    patch_start_idx_ = 1 + num_register_tokens;

    // Normalization constants
    auto mean_tensor = torch::tensor({0.485, 0.456, 0.406});
    auto std_tensor = torch::tensor({0.229, 0.224, 0.225});
    _resnet_mean = register_buffer("_resnet_mean",
        mean_tensor.view({1, 1, 3, 1, 1}).clone());
    _resnet_std = register_buffer("_resnet_std",
        std_tensor.view({1, 1, 3, 1, 1}).clone());
}

void AggregatorImpl::buildPatchEmbed(
    const std::string& patch_embed,
    int64_t img_size,
    int64_t patch_size,
    int64_t num_register_tokens,
    bool interpolate_antialias,
    double interpolate_offset,
    int64_t block_chunks,
    double init_values,
    int64_t embed_dim
) {
    if (patch_embed.find("conv") != std::string::npos) {
        auto pe = layers::PatchEmbed(img_size, patch_size, 3, embed_dim,
                                     torch::nn::AnyModule(), true);
        patch_embed_ = torch::nn::AnyModule(pe);
        register_module("patch_embed", patch_embed_.ptr());
    } else {
        // DINOv2 vision transformer models
        torch::nn::AnyModule vit_model;
        if (patch_embed.find("vitl14") != std::string::npos) {
            vit_model = torch::nn::AnyModule(layers::vit_large(patch_size, num_register_tokens));
        } else if (patch_embed.find("vitb14") != std::string::npos) {
            vit_model = torch::nn::AnyModule(layers::vit_base(patch_size, num_register_tokens));
        } else if (patch_embed.find("vits14") != std::string::npos) {
            vit_model = torch::nn::AnyModule(layers::vit_small(patch_size, num_register_tokens));
        } else if (patch_embed.find("vitg2") != std::string::npos) {
            vit_model = torch::nn::AnyModule(layers::vit_giant2(patch_size, num_register_tokens));
        } else {
            throw std::runtime_error("Unknown patch_embed type: " + patch_embed);
        }
        patch_embed_ = vit_model;
        register_module("patch_embed", patch_embed_.ptr());
    }
}

std::pair<std::vector<torch::Tensor>, int64_t> AggregatorImpl::forward(torch::Tensor images) {
    auto sizes = images.sizes();
    int64_t B = sizes[0];
    int64_t S = sizes[1];
    int64_t C_in = sizes[2];
    int64_t H = sizes[3];
    int64_t W = sizes[4];

    if (C_in != 3) {
        throw std::invalid_argument("Expected 3 input channels");
    }

    // Normalize images
    images = (images - _resnet_mean) / _resnet_std;

    // Reshape to [B*S, C, H, W] for patch embedding
    images = images.view({B * S, C_in, H, W});

    // Extract patch tokens
    torch::Tensor patch_tokens = patch_embed_.forward<torch::Tensor>(images);

    int64_t P = patch_tokens.size(1);
    int64_t C = patch_tokens.size(2);

    // Expand camera and register tokens
    auto camera_token = sliceExpandAndFlatten(camera_token_, B, S);
    auto register_token = sliceExpandAndFlatten(register_token_, B, S);

    // Concatenate special tokens with patch tokens
    auto tokens = torch::cat({camera_token, register_token, patch_tokens}, 1);

    // Get position embeddings
    torch::Tensor pos;
    if (rope_ && position_getter_) {
        pos = (*position_getter_)(B * S, H / patch_size_, W / patch_size_, images.device());
    }

    if (patch_start_idx_ > 0 && pos.defined()) {
        pos = pos + 1;
        auto pos_special = torch::zeros({B * S, patch_start_idx_, 2},
            torch::TensorOptions().device(images.device()).dtype(pos.dtype()));
        pos = torch::cat({pos_special, pos}, 1);
    }

    P = tokens.size(1);

    int64_t frame_idx = 0;
    int64_t global_idx = 0;
    std::vector<torch::Tensor> output_list;

    for (int64_t block_idx = 0; block_idx < aa_block_num_; ++block_idx) {
        std::vector<torch::Tensor> frame_intermediates;
        std::vector<torch::Tensor> global_intermediates;

        for (const auto& attn_type : aa_order_) {
            if (attn_type == "frame") {
                auto result = processFrameAttention(tokens, B, S, P, C, frame_idx, pos);
                tokens = std::get<0>(result);
                frame_idx = std::get<1>(result);
                frame_intermediates = std::get<2>(result);
            } else if (attn_type == "global") {
                auto result = processGlobalAttention(tokens, B, S, P, C, global_idx, pos);
                tokens = std::get<0>(result);
                global_idx = std::get<1>(result);
                global_intermediates = std::get<2>(result);
            }
        }

        for (size_t i = 0; i < frame_intermediates.size(); ++i) {
            auto concat_inter = torch::cat({frame_intermediates[i], global_intermediates[i]}, -1);
            output_list.push_back(concat_inter);
        }
    }

    return {output_list, patch_start_idx_};
}

std::tuple<torch::Tensor, int64_t, std::vector<torch::Tensor>>
AggregatorImpl::processFrameAttention(
    torch::Tensor tokens,
    int64_t B, int64_t S, int64_t P, int64_t C,
    int64_t frame_idx, torch::Tensor pos) {

    if (tokens.size(0) != B * S || tokens.size(1) != P) {
        tokens = tokens.view({B, S, P, C}).view({B * S, P, C});
    }

    if (pos.defined() && (pos.size(0) != B * S || pos.size(1) != P)) {
        pos = pos.view({B, S, P, 2}).view({B * S, P, 2});
    }

    std::vector<torch::Tensor> intermediates;

    for (int64_t i = 0; i < aa_block_size_; ++i) {
        auto& block = frame_blocks_[frame_idx];

        if (pos.defined()) {
            tokens = block->forward(tokens, pos);
        } else {
            tokens = block->forward(tokens, {});
        }

        frame_idx++;
        intermediates.push_back(tokens.view({B, S, P, C}));
    }

    return {tokens, frame_idx, intermediates};
}

std::tuple<torch::Tensor, int64_t, std::vector<torch::Tensor>>
AggregatorImpl::processGlobalAttention(
    torch::Tensor tokens,
    int64_t B, int64_t S, int64_t P, int64_t C,
    int64_t global_idx, torch::Tensor pos) {

    if (tokens.size(0) != B || tokens.size(1) != S * P) {
        tokens = tokens.view({B, S, P, C}).view({B, S * P, C});
    }

    if (pos.defined() && (pos.size(0) != B || pos.size(1) != S * P)) {
        pos = pos.view({B, S, P, 2}).view({B, S * P, 2});
    }

    std::vector<torch::Tensor> intermediates;

    for (int64_t i = 0; i < aa_block_size_; ++i) {
        auto& block = global_blocks_[global_idx];

        if (pos.defined()) {
            tokens = block->forward(tokens, pos);
        } else {
            tokens = block->forward(tokens, {});
        }

        global_idx++;
        intermediates.push_back(tokens.view({B, S, P, C}));
    }

    return {tokens, global_idx, intermediates};
}

torch::Tensor sliceExpandAndFlatten(const torch::Tensor& token_tensor, int64_t B, int64_t S) {
    auto query = token_tensor.index({0, torch::indexing::Slice(0, 1)})
                      .expand({B, 1, token_tensor.size(2), token_tensor.size(3)});

    auto others = token_tensor.index({0, torch::indexing::Slice(1, 2)})
                       .expand({B, S - 1, token_tensor.size(2), token_tensor.size(3)});

    auto combined = torch::cat({query, others}, 1);
    combined = combined.view({B * S, token_tensor.size(2), token_tensor.size(3)});

    return combined;
}

} // namespace models
} // namespace vggt
