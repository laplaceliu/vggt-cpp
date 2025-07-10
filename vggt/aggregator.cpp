/**
 * @file aggregator.cpp
 * @brief Implementation of Aggregator module for VGGT library
 */

#include "aggregator.h"
#include "layers/attention.h"
#include "layers/block.h"
#include "layers/drop_path.h"
#include "layers/layer_scale.h"
#include "layers/mlp.h"
#include "layers/patch_embed.h"
#include "layers/rope.h"
#include "layers/swiglu_ffn.h"
#include "layers/vision_transformer.h"

#include <torch/torch.h>
#include <vector>
#include <string>

namespace vggt {

AggregatorImpl::AggregatorImpl(
    int img_size,
    int patch_size,
    int embed_dim,
    int depth,
    int num_heads,
    float mlp_ratio,
    int num_register_tokens,
    bool qkv_bias,
    bool proj_bias,
    bool ffn_bias,
    const std::string& patch_embed,
    const std::vector<std::string>& aa_order,
    int aa_block_size,
    bool qk_norm,
    int rope_freq,
    float init_values
) : depth(depth),
    aa_order(aa_order),
    patch_size(patch_size),
    aa_block_size(aa_block_size),
    aa_block_num(0),
    patch_start_idx(0),
    use_reentrant(false) {

    // Initialize mean and std for normalization
    _resnet_mean = torch::tensor({0.485, 0.456, 0.406}).view({1, 3, 1, 1});
    _resnet_std = torch::tensor({0.229, 0.224, 0.225}).view({1, 3, 1, 1});

    // Build patch embedding
    build_patch_embed(patch_embed, img_size, patch_size, num_register_tokens,
                     true, 0.0f, 0, init_values, embed_dim);

    // Initialize camera token and register tokens
    camera_token = torch::zeros({1, 1, embed_dim});
    register_token = torch::zeros({1, num_register_tokens, embed_dim});

    // Initialize rope if needed
    if (rope_freq > 0) {
        rope = RotaryPositionEmbedding2D(embed_dim, rope_freq);
    }

    // Initialize blocks
    for (int i = 0; i < depth; ++i) {
        std::string block_type = aa_order[(i / aa_block_size) % aa_order.size()];

        if (block_type == "frame") {
            frame_blocks->push_back(Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias, proj_bias, ffn_bias,
                qk_norm, rope_freq > 0, init_values
            ));
        } else if (block_type == "global") {
            global_blocks->push_back(Block(
                embed_dim, num_heads, mlp_ratio, qkv_bias, proj_bias, ffn_bias,
                qk_norm, rope_freq > 0, init_values
            ));
        }
    }

    // Register all parameters and submodules
    register_parameter("camera_token", this->camera_token);
    register_parameter("register_token", this->register_token);
    register_module("patch_embed", this->patch_embed);
    register_module("rope", this->rope);
    register_module("frame_blocks", this->frame_blocks);
    register_module("global_blocks", this->global_blocks);
}

void AggregatorImpl::build_patch_embed(
    const std::string& patch_embed,
    int img_size,
    int patch_size,
    int num_register_tokens,
    bool interpolate_antialias,
    float interpolate_offset,
    int block_chunks,
    float init_values,
    int embed_dim
) {
    if (patch_embed == "dinov2_vitl14_reg") {
        this->patch_embed = PatchEmbed(
            img_size, patch_size, embed_dim, num_register_tokens,
            interpolate_antialias, interpolate_offset, block_chunks, init_values
        );
        this->patch_start_idx = num_register_tokens + 1; // +1 for camera token
    } else {
        throw std::runtime_error("Unsupported patch_embed type: " + patch_embed);
    }
}

std::tuple<std::vector<torch::Tensor>, int> AggregatorImpl::forward(torch::Tensor images) {
    // Normalize images
    images = (images - _resnet_mean.to(images.device())) / _resnet_std.to(images.device());

    // Get batch and sequence dimensions
    int B = images.size(0);
    int S = images.size(1);

    // Process through patch embed
    auto [tokens, pos] = patch_embed->forward(images);
    int P = tokens.size(1) - patch_start_idx; // Number of patch tokens

    // Expand camera and register tokens
    auto camera_tokens = camera_token.expand({B, S, -1, -1});
    auto register_tokens = register_token.expand({B, S, -1, -1});

    // Concatenate tokens: [camera, register, patch]
    tokens = torch::cat({camera_tokens, register_tokens, tokens}, 2);

    // Process through blocks
    std::vector<torch::Tensor> outputs;
    for (int i = 0; i < depth; ++i) {
        std::string block_type = aa_order[(i / aa_block_size) % aa_order.size()];

        if (block_type == "frame") {
            auto [new_tokens, new_pos, block_outputs] = process_frame_attention(
                tokens, B, S, P, tokens.size(-1), i, pos
            );
            tokens = new_tokens;
            pos = new_pos;
            outputs.insert(outputs.end(), block_outputs.begin(), block_outputs.end());
        } else if (block_type == "global") {
            auto [new_tokens, new_pos, block_outputs] = process_global_attention(
                tokens, B, S, P, tokens.size(-1), i, pos
            );
            tokens = new_tokens;
            pos = new_pos;
            outputs.insert(outputs.end(), block_outputs.begin(), block_outputs.end());
        }
    }

    return {outputs, patch_start_idx};
}

std::tuple<torch::Tensor, int, std::vector<torch::Tensor>> AggregatorImpl::process_frame_attention(
    torch::Tensor tokens,
    int B,
    int S,
    int P,
    int C,
    int frame_idx,
    torch::Tensor pos
) {
    // Reshape tokens for frame attention: [B*S, 1+R+P, C]
    auto reshaped_tokens = tokens.reshape({B * S, 1 + register_token.size(1) + P, C});

    // Apply frame attention
    auto block = frame_blocks[frame_idx];
    auto [new_tokens, outputs] = block->forward(reshaped_tokens);

    // Reshape back to [B, S, 1+R+P, C]
    new_tokens = new_tokens.reshape({B, S, 1 + register_token.size(1) + P, C});

    return {new_tokens, pos, outputs};
}

std::tuple<torch::Tensor, int, std::vector<torch::Tensor>> AggregatorImpl::process_global_attention(
    torch::Tensor tokens,
    int B,
    int S,
    int P,
    int C,
    int global_idx,
    torch::Tensor pos
) {
    // Reshape tokens for global attention: [B, S*(1+R+P), C]
    auto reshaped_tokens = tokens.reshape({B, S * (1 + register_token.size(1) + P), C});

    // Apply global attention
    auto block = global_blocks[global_idx];
    auto [new_tokens, outputs] = block->forward(reshaped_tokens);

    // Reshape back to [B, S, 1+R+P, C]
    new_tokens = new_tokens.reshape({B, S, 1 + register_token.size(1) + P, C});

    return {new_tokens, pos, outputs};
}

} // namespace vggt
