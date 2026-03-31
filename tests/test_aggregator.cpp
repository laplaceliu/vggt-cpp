#include <gtest/gtest.h>
#include <torch/torch.h>
#include "vggt/models/aggregator.h"

namespace vggt {
namespace models {
namespace {

TEST(AggregatorTest, ConstructorDefault) {
    torch::manual_seed(42);

    auto aggregator = Aggregator();

    EXPECT_TRUE(aggregator);
}

TEST(AggregatorTest, ConstructorWithCustomParams) {
    torch::manual_seed(42);

    auto aggregator = Aggregator(
        518,    // img_size
        14,     // patch_size
        1024,   // embed_dim
        16,     // num_heads
        4.0,    // mlp_ratio
        true,   // qkv_bias
        true,   // proj_bias
        true,   // ffn_bias
        24,     // depth
        0.01,   // init_values
        "dinov2_vitl14_reg",  // patch_embed
        4,      // num_register_tokens
        true,   // interpolate_antialias
        0.0,    // interpolate_offset
        0,      // block_chunks
        true,   // qk_norm
        false   // use_flex_attn
    );

    EXPECT_TRUE(aggregator);
}

TEST(AggregatorTest, ConstructorWithDifferentEmbedDim) {
    torch::manual_seed(42);

    auto aggregator = Aggregator(
        224,    // img_size
        16,     // patch_size
        768     // embed_dim
    );

    EXPECT_TRUE(aggregator);
}

TEST(AggregatorTest, ConstructorWithSmallImage) {
    torch::manual_seed(42);

    auto aggregator = Aggregator(
        112,    // img_size
        14,     // patch_size
        256     // embed_dim
    );

    EXPECT_TRUE(aggregator);
}

TEST(AggregatorTest, ConstructorNoRegisterTokens) {
    torch::manual_seed(42);

    auto aggregator = Aggregator(
        224,    // img_size
        16,     // patch_size
        512,    // embed_dim
        8,      // num_heads
        4.0,    // mlp_ratio
        true,   // qkv_bias
        true,   // proj_bias
        true,   // ffn_bias
        12,     // depth
        0.01,   // init_values
        "simple", // patch_embed
        0       // num_register_tokens
    );

    EXPECT_TRUE(aggregator);
}

TEST(AggregatorTest, ForwardBasic) {
    torch::manual_seed(42);

    auto aggregator = Aggregator(
        112,    // img_size
        14,     // patch_size
        256,    // embed_dim
        8,      // num_heads
        4.0,    // mlp_ratio
        true,   // qkv_bias
        true,   // proj_bias
        true,   // ffn_bias
        2,      // depth (reduced for faster test)
        0.01,   // init_values
        "simple", // patch_embed
        2       // num_register_tokens
    );

    // Input: [B, S, 3, H, W] = [1, 2, 3, 112, 112]
    torch::Tensor images = torch::rand({1, 2, 3, 112, 112});

    auto [output_list, patch_start_idx] = aggregator->forward(images);

    // Check output list is not empty
    EXPECT_FALSE(output_list.empty());
    
    // Check patch_start_idx (should be 2 + num_register_tokens = 4)
    EXPECT_EQ(patch_start_idx, 4);
}

TEST(AggregatorTest, ForwardMultipleFrames) {
    torch::manual_seed(42);

    auto aggregator = Aggregator(
        112,    // img_size
        14,     // patch_size
        256,    // embed_dim
        8,      // num_heads
        4.0,    // mlp_ratio
        true,   // qkv_bias
        true,   // proj_bias
        true,   // ffn_bias
        2,      // depth
        0.01,   // init_values
        "simple", // patch_embed
        2       // num_register_tokens
    );

    // Input: [B, S, 3, H, W] = [1, 5, 3, 112, 112]
    torch::Tensor images = torch::rand({1, 5, 3, 112, 112});

    auto [output_list, patch_start_idx] = aggregator->forward(images);

    EXPECT_FALSE(output_list.empty());
}

TEST(AggregatorTest, ForwardBatchSize) {
    torch::manual_seed(42);

    auto aggregator = Aggregator(
        112,    // img_size
        14,     // patch_size
        256,    // embed_dim
        8,      // num_heads
        4.0,    // mlp_ratio
        true,   // qkv_bias
        true,   // proj_bias
        true,   // ffn_bias
        2,      // depth
        0.01,   // init_values
        "simple", // patch_embed
        2       // num_register_tokens
    );

    // Input: [B, S, 3, H, W] = [3, 2, 3, 112, 112]
    torch::Tensor images = torch::rand({3, 2, 3, 112, 112});

    auto [output_list, patch_start_idx] = aggregator->forward(images);

    EXPECT_FALSE(output_list.empty());
}

TEST(AggregatorTest, ForwardOutputStructure) {
    torch::manual_seed(42);

    auto aggregator = Aggregator(
        112,    // img_size
        14,     // patch_size
        256,    // embed_dim
        8,      // num_heads
        4.0,    // mlp_ratio
        true,   // qkv_bias
        true,   // proj_bias
        true,   // ffn_bias
        2,      // depth (aa_block_num = 2)
        0.01,   // init_values
        "simple", // patch_embed
        2       // num_register_tokens
    );

    torch::Tensor images = torch::rand({1, 2, 3, 112, 112});

    auto [output_list, patch_start_idx] = aggregator->forward(images);

    // Each output should be a 4D tensor [B, S, P, C]
    for (const auto& out : output_list) {
        EXPECT_EQ(out.dim(), 4);
        EXPECT_EQ(out.size(0), 1);  // B
        EXPECT_EQ(out.size(1), 2);  // S
        // P = patch_start_idx + num_patches
        // C = 2 * embed_dim (concatenated frame and global)
        EXPECT_EQ(out.size(3), 512);  // 2 * embed_dim = 512
    }
}

TEST(AggregatorTest, ForwardPreservesGrad) {
    torch::manual_seed(42);

    auto aggregator = Aggregator(
        112,    // img_size
        14,     // patch_size
        128,    // embed_dim (reduced for faster test)
        4,      // num_heads
        4.0,    // mlp_ratio
        true,   // qkv_bias
        true,   // proj_bias
        true,   // ffn_bias
        2,      // depth
        0.01,   // init_values
        "simple", // patch_embed
        2       // num_register_tokens
    );

    torch::Tensor images = torch::rand({1, 2, 3, 112, 112}, torch::requires_grad());

    auto [output_list, patch_start_idx] = aggregator->forward(images);

    // Backward through the first output
    output_list[0].sum().backward();

    EXPECT_TRUE(images.grad().defined());
}

} // namespace
} // namespace models
} // namespace vggt
