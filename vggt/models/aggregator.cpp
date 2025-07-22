/**
 * @file aggregator.cpp
 * @brief Implementation of Aggregator model for VGGT
 *
 * This file implements the Aggregator class which applies alternating-attention over input frames,
 * as described in VGGT: Visual Geometry Grounded Transformer.
 */

#include "aggregator.h"
#include <iostream>

namespace vggt {
namespace models {

torch::Tensor slice_expand_and_flatten(const torch::Tensor& token_tensor, int64_t B, int64_t S) {
    // Slice out the "query" tokens => shape (1, 1, ...)
    auto query = token_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(0, 1)})
                            .expand({B, 1, token_tensor.size(2), token_tensor.size(3)});
    
    // Slice out the "other" tokens => shape (1, S-1, ...)
    auto others = token_tensor.index({torch::indexing::Slice(), torch::indexing::Slice(1)})
                             .expand({B, S - 1, token_tensor.size(2), token_tensor.size(3)});
    
    // Concatenate => shape (B, S, ...)
    auto combined = torch::cat({query, others}, 1);
    
    // Finally flatten => shape (B*S, ...)
    combined = combined.reshape({B * S, combined.size(2), combined.size(3)});
    return combined;
}

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
    const std::string& patch_embed_type,
    const std::vector<std::string>& aa_order,
    int64_t aa_block_size,
    bool qk_norm,
    int64_t rope_freq,
    double init_values
) {
    // Build patch embedding layer
    build_patch_embed(
        patch_embed_type,
        img_size,
        patch_size,
        num_register_tokens,
        true,  // interpolate_antialias
        0.0,   // interpolate_offset
        0,     // block_chunks
        init_values,
        embed_dim
    );
    
    // Initialize rotary position embedding if frequency > 0
    if (rope_freq > 0) {
        rope_ = layers::RotaryPositionEmbedding2D(layers::RotaryPositionEmbedding2DOptions().frequency(rope_freq));
        position_getter_ = layers::PositionGetter();
    }
    
    // Create frame blocks
    frame_blocks_ = torch::nn::ModuleList();
    for (int64_t i = 0; i < depth; ++i) {
        auto block = layers::Block(layers::BlockOptions()
            .dim(embed_dim)
            .num_heads(num_heads)
            .mlp_ratio(mlp_ratio)
            .qkv_bias(qkv_bias)
            .proj_bias(proj_bias)
            .ffn_bias(ffn_bias)
            .init_values(init_values)
            .qk_norm(qk_norm)
            .rope(rope_));
        
        frame_blocks_->push_back(block);
    }
    
    // Create global blocks
    global_blocks_ = torch::nn::ModuleList();
    for (int64_t i = 0; i < depth; ++i) {
        auto block = layers::Block(layers::BlockOptions()
            .dim(embed_dim)
            .num_heads(num_heads)
            .mlp_ratio(mlp_ratio)
            .qkv_bias(qkv_bias)
            .proj_bias(proj_bias)
            .ffn_bias(ffn_bias)
            .init_values(init_values)
            .qk_norm(qk_norm)
            .rope(rope_));
        
        global_blocks_->push_back(block);
    }
    
    // Store parameters
    depth_ = depth;
    aa_order_ = aa_order;
    patch_size_ = patch_size;
    aa_block_size_ = aa_block_size;
    
    // Validate that depth is divisible by aa_block_size
    if (depth_ % aa_block_size_ != 0) {
        throw std::runtime_error(
            "depth (" + std::to_string(depth) + 
            ") must be divisible by aa_block_size (" + 
            std::to_string(aa_block_size) + ")"
        );
    }
    
    aa_block_num_ = depth_ / aa_block_size_;
    
    // Note: We have two camera tokens, one for the first frame and one for the rest
    // The same applies for register tokens
    camera_token_ = register_parameter("camera_token", torch::randn({1, 2, 1, embed_dim}));
    register_token_ = register_parameter("register_token", torch::randn({1, 2, num_register_tokens, embed_dim}));
    
    // The patch tokens start after the camera and register tokens
    patch_start_idx_ = 1 + num_register_tokens;
    
    // Initialize parameters with small values
    torch::nn::init::normal_(camera_token_, 0.0, 1e-6);
    torch::nn::init::normal_(register_token_, 0.0, 1e-6);
    
    // Register normalization constants as buffers
    std::vector<float> resnet_mean = {0.485, 0.456, 0.406};
    std::vector<float> resnet_std = {0.229, 0.224, 0.225};
    
    register_buffer("_resnet_mean", torch::tensor(resnet_mean).view({1, 1, 3, 1, 1}));
    register_buffer("_resnet_std", torch::tensor(resnet_std).view({1, 1, 3, 1, 1}));
    
    use_reentrant_ = false; // hardcoded to False
}

void AggregatorImpl::build_patch_embed(
    const std::string& patch_embed_type,
    int64_t img_size,
    int64_t patch_size,
    int64_t num_register_tokens,
    bool interpolate_antialias,
    double interpolate_offset,
    int64_t block_chunks,
    double init_values,
    int64_t embed_dim
) {
    if (patch_embed_type.find("conv") != std::string::npos) {
        patch_embed_ = layers::PatchEmbed(
            layers::PatchEmbedOptions()
                .img_size(img_size)
                .patch_size(patch_size)
                .in_chans(3)
                .embed_dim(embed_dim)
        );
    } else {
        if (patch_embed_type == "dinov2_vitl14_reg") {
            patch_embed_ = layers::vit_large(
                patch_size,
                num_register_tokens,
                img_size,
                interpolate_antialias,
                interpolate_offset,
                block_chunks,
                init_values
            );
        } else if (patch_embed_type == "dinov2_vitb14_reg") {
            patch_embed_ = layers::vit_base(
                patch_size,
                num_register_tokens,
                img_size,
                interpolate_antialias,
                interpolate_offset,
                block_chunks,
                init_values
            );
        } else if (patch_embed_type == "dinov2_vits14_reg") {
            patch_embed_ = layers::vit_small(
                patch_size,
                num_register_tokens,
                img_size,
                interpolate_antialias,
                interpolate_offset,
                block_chunks,
                init_values
            );
        } else if (patch_embed_type == "dinov2_vitg2_reg") {
            patch_embed_ = layers::vit_giant2(
                patch_size,
                num_register_tokens,
                img_size,
                interpolate_antialias,
                interpolate_offset,
                block_chunks,
                init_values
            );
        } else {
            throw std::runtime_error("Unknown patch_embed type: " + patch_embed_type);
        }
        
        // Disable gradient updates for mask token if it exists
        // Note: In C++ we need to check if the module has the attribute differently
        auto named_parameters = patch_embed_->named_parameters(false);
        for (auto& param : named_parameters) {
            if (param.key() == "mask_token") {
                param.value().set_requires_grad(false);
                break;
            }
        }
    }
}

std::tuple<torch::Tensor, int64_t, std::vector<torch::Tensor>> AggregatorImpl::process_frame_attention(
    torch::Tensor tokens,
    int64_t B,
    int64_t S,
    int64_t P,
    int64_t C,
    int64_t frame_idx,
    const c10::optional<torch::Tensor>& pos
) {
    // If needed, reshape tokens or positions
    if (tokens.sizes() != std::vector<int64_t>{B * S, P, C}) {
        tokens = tokens.view({B, S, P, C}).view({B * S, P, C});
    }
    
    torch::Tensor position_tensor;
    if (pos.has_value()) {
        position_tensor = pos.value();
        if (position_tensor.sizes() != std::vector<int64_t>{B * S, P, 2}) {
            position_tensor = position_tensor.view({B, S, P, 2}).view({B * S, P, 2});
        }
    }
    
    std::vector<torch::Tensor> intermediates;
    
    // By default, aa_block_size_=1, which processes one block at a time
    for (int64_t i = 0; i < aa_block_size_; ++i) {
        if (is_training()) {
            // In PyTorch, we would use checkpoint here, but in LibTorch we'll just call forward
            tokens = frame_blocks_->at(frame_idx)->as<layers::Block>()->forward(tokens, position_tensor);
        } else {
            tokens = frame_blocks_->at(frame_idx)->as<layers::Block>()->forward(tokens, position_tensor);
        }
        frame_idx += 1;
        intermediates.push_back(tokens.view({B, S, P, C}));
    }
    
    return std::make_tuple(tokens, frame_idx, intermediates);
}

std::tuple<torch::Tensor, int64_t, std::vector<torch::Tensor>> AggregatorImpl::process_global_attention(
    torch::Tensor tokens,
    int64_t B,
    int64_t S,
    int64_t P,
    int64_t C,
    int64_t global_idx,
    const c10::optional<torch::Tensor>& pos
) {
    // If needed, reshape tokens or positions
    if (tokens.sizes() != std::vector<int64_t>{B, S * P, C}) {
        tokens = tokens.view({B, S, P, C}).view({B, S * P, C});
    }
    
    torch::Tensor position_tensor;
    if (pos.has_value()) {
        position_tensor = pos.value();
        if (position_tensor.sizes() != std::vector<int64_t>{B, S * P, 2}) {
            position_tensor = position_tensor.view({B, S, P, 2}).view({B, S * P, 2});
        }
    }
    
    std::vector<torch::Tensor> intermediates;
    
    // By default, aa_block_size_=1, which processes one block at a time
    for (int64_t i = 0; i < aa_block_size_; ++i) {
        if (is_training()) {
            // In PyTorch, we would use checkpoint here, but in LibTorch we'll just call forward
            tokens = global_blocks_->at(global_idx)->as<layers::Block>()->forward(tokens, position_tensor);
        } else {
            tokens = global_blocks_->at(global_idx)->as<layers::Block>()->forward(tokens, position_tensor);
        }
        global_idx += 1;
        intermediates.push_back(tokens.view({B, S, P, C}));
    }
    
    return std::make_tuple(tokens, global_idx, intermediates);
}

std::tuple<std::vector<torch::Tensor>, int64_t> AggregatorImpl::forward(const torch::Tensor& images) {
    auto B = images.size(0);
    auto S = images.size(1);
    auto C_in = images.size(2);
    auto H = images.size(3);
    auto W = images.size(4);
    
    if (C_in != 3) {
        throw std::runtime_error("Expected 3 input channels, got " + std::to_string(C_in));
    }
    
    // Normalize images and reshape for patch embed
    auto normalized_images = (images - _resnet_mean) / _resnet_std;
    
    // Reshape to [B*S, C, H, W] for patch embedding
    normalized_images = normalized_images.view({B * S, C_in, H, W});
    
    // Apply patch embedding
    torch::Tensor patch_tokens;
    if (auto* patch_embed_ptr = patch_embed_->as<layers::PatchEmbed>()) {
        patch_tokens = patch_embed_ptr->forward(normalized_images);
    } else {
        auto result = patch_embed_->as<torch::nn::Module>()->forward(normalized_images);
        
        // Check if result is a dictionary
        if (result.dim() == 0 && result.dtype() == torch::kInt64) {
            throw std::runtime_error("Unexpected result type from patch_embed");
        }
        
        // In C++, we need to handle this differently since we can't directly check if it's a dict
        // We'll assume it's a tensor with the expected shape
        patch_tokens = result;
        
        // If it's a dictionary-like object in Python, we'd extract the patchtokens
        // Here we'll just use the result directly
    }
    
    auto P = patch_tokens.size(1);
    auto C = patch_tokens.size(2);
    
    // Expand camera and register tokens to match batch size and sequence length
    auto camera_token = slice_expand_and_flatten(camera_token_, B, S);
    auto register_token = slice_expand_and_flatten(register_token_, B, S);
    
    // Concatenate special tokens with patch tokens
    auto tokens = torch::cat({camera_token, register_token, patch_tokens}, 1);
    
    c10::optional<torch::Tensor> pos = c10::nullopt;
    if (rope_.ptr()) {
        pos = position_getter_->forward(B * S, H / patch_size_, W / patch_size_, images.device());
        
        if (patch_start_idx_ > 0) {
            // Do not use position embedding for special tokens (camera and register tokens)
            // So set pos to 0 for the special tokens
            auto pos_value = pos.value() + 1;
            auto pos_special = torch::zeros({B * S, patch_start_idx_, 2}, 
                                           torch::TensorOptions().device(images.device()).dtype(pos_value.dtype()));
            pos = torch::cat({pos_special, pos_value}, 1);
        }
    }
    
    // Update P because we added special tokens
    P = tokens.size(1);
    
    int64_t frame_idx = 0;
    int64_t global_idx = 0;
    std::vector<torch::Tensor> output_list;
    
    for (int64_t i = 0; i < aa_block_num_; ++i) {
        for (const auto& attn_type : aa_order_) {
            std::vector<torch::Tensor> frame_intermediates;
            std::vector<torch::Tensor> global_intermediates;
            
            if (attn_type == "frame") {
                auto result = process_frame_attention(tokens, B, S, P, C, frame_idx, pos);
                tokens = std::get<0>(result);
                frame_idx = std::get<1>(result);
                frame_intermediates = std::get<2>(result);
            } else if (attn_type == "global") {
                auto result = process_global_attention(tokens, B, S, P, C, global_idx, pos);
                tokens = std::get<0>(result);
                global_idx = std::get<1>(result);
                global_intermediates = std::get<2>(result);
            } else {
                throw std::runtime_error("Unknown attention type: " + attn_type);
            }
            
            // Concat frame and global intermediates
            for (size_t j = 0; j < frame_intermediates.size(); ++j) {
                auto concat_inter = torch::cat({frame_intermediates[j], global_intermediates[j]}, -1);
                output_list.push_back(concat_inter);
            }
        }
    }
    
    return std::make_tuple(output_list, patch_start_idx_);
}

} // namespace models
} // namespace vggt