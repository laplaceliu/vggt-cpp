#include "aggregator.h"

namespace vggt {
namespace models {

AggregatorImpl::AggregatorImpl(
    int64_t img_size,
    int64_t patch_size,
    int64_t embed_dim,
    int64_t num_heads,
    double mlp_ratio,
    bool qkv_bias,
    bool proj_bias,
    bool ffn_bias,
    int64_t depth,
    double init_values,
    const std::string& patch_embed,
    int64_t num_register_tokens,
    bool interpolate_antialias,
    double interpolate_offset,
    int64_t block_chunks,
    bool qk_norm,
    bool use_flex_attn
) : img_size_(img_size),
    patch_size_(patch_size),
    embed_dim_(embed_dim),
    depth_(depth),
    num_register_tokens_(num_register_tokens),
    patch_start_idx_(1 + num_register_tokens) {  // 1 camera token per frame + register tokens per frame

    // Build the patch embedding module
    buildPatchEmbed(
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias,
        interpolate_offset,
        block_chunks,
        init_values,
        embed_dim
    );

    // Create learnable tokens for camera and registers
    // In Python: self.camera_token = nn.Parameter(torch.zeros(1, 2, 1, embed_dim))
    camera_token_ = register_parameter("camera_token", torch::zeros({1, 2, 1, embed_dim}));

    // Register tokens
    // In Python: self.register_token = nn.Parameter(torch.zeros(1, 2, num_register_tokens, embed_dim))
    if (num_register_tokens > 0) {
        register_token_ = register_parameter("register_token", torch::zeros({1, 2, num_register_tokens, embed_dim}));
    }

    // Alternating attention configuration
    // AA_ORDER = ("frame", "global") in the Python code
    aa_order_ = {"frame", "global"};
    aa_block_size_ = 1;  // Number of blocks per attention type
    aa_block_num_ = depth / aa_block_size_;  // 24 / 1 = 24 (matching Python: depth // aa_block_size)

    // Initialize RoPE (optional)
    // rope_ = torch::nn::AnyModule(rope::RoPE())  // If using RoPE

    // Create position getter (if not using RoPE)
    position_getter_ = std::make_shared<layers::PositionGetter>();

    // Create frame and global blocks
    for (int64_t i = 0; i < depth_; ++i) {
        torch::Tensor init_vals = {};
        if (init_values > 0) {
            init_vals = torch::tensor(init_values);
        }

        torch::nn::AnyModule rope_module;
        if (!rope_.is_empty()) {
            rope_module = torch::nn::AnyModule(rope_);
        }

        // Use shared_ptr to avoid copying issues
        auto block = std::make_shared<layers::BlockImpl>(
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
        register_module("frame_blocks_" + std::to_string(i), block);
    }

    for (int64_t i = 0; i < depth_; ++i) {
        torch::Tensor init_vals = {};
        if (init_values > 0) {
            init_vals = torch::tensor(init_values);
        }

        torch::nn::AnyModule rope_module;
        if (!rope_.is_empty()) {
            rope_module = torch::nn::AnyModule(rope_);
        }

        // Use shared_ptr to avoid copying issues
        auto block = std::make_shared<layers::BlockImpl>(
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
        register_module("global_blocks_" + std::to_string(i), block);
    }

    // Initialize ResNet normalization constants
    _resnet_mean = register_buffer("_resnet_mean", torch::tensor({0.485f, 0.456f, 0.406f}).view({1, 1, 3, 1, 1}));
    _resnet_std = register_buffer("_resnet_std", torch::tensor({0.229f, 0.224f, 0.225f}).view({1, 1, 3, 1, 1}));
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
    // Always create a simple PatchEmbed (Conv2d)
    // Note: DINOv2 vision transformer embedding is not fully implemented yet
    auto pe = std::make_shared<layers::PatchEmbedImpl>(img_size, patch_size, 3, embed_dim,
                                 torch::nn::AnyModule(), true);
    register_module("patch_embed", pe);
    // Store as AnyModule for forward compatibility
    patch_embed_ = torch::nn::AnyModule(layers::PatchEmbed(pe));
}

torch::Tensor AggregatorImpl::sliceExpandAndFlatten(torch::Tensor token, int64_t B, int64_t S) {
    // token: [1, 2, N, C] where dim=1 has 2 variants:
    // - variant 0 (index 0): for the first frame
    // - variant 1 (index 1): for remaining frames (S-1 frames)
    // Result: [B*S, N, C]
    
    int64_t N = token.size(2);
    int64_t C = token.size(3);
    
    // Slice out query tokens for first frame: [1, 1, N, C] -> expand to [B, 1, N, C]
    auto query = token.slice(1, 0, 1).expand({B, 1, N, C});
    
    // Slice out others tokens for remaining frames: [1, 1, N, C] -> expand to [B, S-1, N, C]
    auto others = token.slice(1, 1, 2).expand({B, S - 1, N, C});
    
    // Concatenate: [B, 1, N, C] + [B, S-1, N, C] -> [B, S, N, C]
    auto combined = torch::cat({query, others}, 1);
    
    // Flatten to [B*S, N, C]
    return combined.reshape({B * S, N, C});
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
            // Clone tensors to avoid potential memory issues with views
            auto frame_clone = frame_intermediates[i].clone();
            auto global_clone = global_intermediates[i].clone();
            auto concat_inter = torch::cat({frame_clone, global_clone}, -1);
            output_list.emplace_back(std::move(concat_inter));
        }
    }

    return {output_list, patch_start_idx_};
}

std::tuple<torch::Tensor, int64_t, std::vector<torch::Tensor>> AggregatorImpl::processFrameAttention(
    torch::Tensor tokens, int64_t B, int64_t S, int64_t P, int64_t C, int64_t start_idx, torch::Tensor pos) {

    std::vector<torch::Tensor> intermediates;

    for (int64_t i = 0; i < aa_block_size_; ++i) {
        auto block = frame_blocks_[start_idx];

        if (pos.defined()) {
            tokens = block->forward(tokens, pos);
        } else {
            tokens = block->forward(tokens, {});
        }

        start_idx++;
        intermediates.push_back(tokens.view({B, S, P, C}));
    }

    return {tokens, start_idx, intermediates};
}

std::tuple<torch::Tensor, int64_t, std::vector<torch::Tensor>> AggregatorImpl::processGlobalAttention(
    torch::Tensor tokens, int64_t B, int64_t S, int64_t P, int64_t C, int64_t start_idx, torch::Tensor pos) {

    // Reshape tokens for global attention: [B*S, P, C] -> [B, S*P, C]
    tokens = tokens.view({B, S * P, C});

    std::vector<torch::Tensor> intermediates;

    for (int64_t i = 0; i < aa_block_size_; ++i) {
        auto block = global_blocks_[start_idx];

        if (pos.defined()) {
            // For global attention, we need to handle position embeddings differently
            // The pos tensor has shape [B*S, P, 2], we need to reshape it to [B, S*P, 2]
            auto pos_reshaped = pos.view({B, S * P, pos.size(-1)});
            tokens = block->forward(tokens, pos_reshaped);
        } else {
            tokens = block->forward(tokens, {});
        }

        start_idx++;
        intermediates.push_back(tokens.view({B, S, P, C}));
    }

    // Reshape back to [B*S, P, C]
    tokens = tokens.view({B * S, P, C});

    return {tokens, start_idx, intermediates};
}

} // namespace models
} // namespace vggt
