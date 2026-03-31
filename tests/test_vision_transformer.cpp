#include <gtest/gtest.h>
#include <torch/torch.h>
#include "vggt/layers/vision_transformer.h"

namespace vggt {
namespace layers {
namespace {

TEST(VisionTransformerTest, VitSmallConstructor) {
    torch::manual_seed(42);

    auto vit = vit_small();

    EXPECT_TRUE(vit);
    EXPECT_EQ(vit->num_features(), 384);
    EXPECT_EQ(vit->embed_dim(), 384);
    EXPECT_EQ(vit->patch_size_val(), 16);
}

TEST(VisionTransformerTest, VitBaseConstructor) {
    torch::manual_seed(42);

    auto vit = vit_base();

    EXPECT_TRUE(vit);
    EXPECT_EQ(vit->num_features(), 768);
    EXPECT_EQ(vit->embed_dim(), 768);
    EXPECT_EQ(vit->patch_size_val(), 16);
}

TEST(VisionTransformerTest, VitLargeConstructor) {
    torch::manual_seed(42);

    auto vit = vit_large();

    EXPECT_TRUE(vit);
    EXPECT_EQ(vit->num_features(), 1024);
    EXPECT_EQ(vit->embed_dim(), 1024);
    EXPECT_EQ(vit->patch_size_val(), 16);
}

TEST(VisionTransformerTest, VitWithCustomPatchSize) {
    torch::manual_seed(42);

    auto vit = vit_small(14);

    EXPECT_TRUE(vit);
    EXPECT_EQ(vit->patch_size_val(), 14);
}

TEST(VisionTransformerTest, VitWithRegisterTokens) {
    torch::manual_seed(42);

    auto vit = vit_small(16, 4);

    EXPECT_TRUE(vit);
}

TEST(VisionTransformerTest, ForwardFeaturesBasic) {
    torch::manual_seed(42);

    auto vit = vit_small();
    torch::Tensor x = torch::randn({2, 3, 224, 224});

    torch::Tensor features = vit->forward_features(x, torch::Tensor());

    // Output should be [B, num_patches+1, embed_dim]
    EXPECT_EQ(features.dim(), 3);
    EXPECT_EQ(features.size(0), 2);
    EXPECT_EQ(features.size(1), 197);  // 196 patches + 1 cls_token
    EXPECT_EQ(features.size(2), 384);  // embed_dim
}

TEST(VisionTransformerTest, ForwardBasic) {
    torch::manual_seed(42);

    auto vit = vit_small();
    torch::Tensor x = torch::randn({2, 3, 224, 224});

    torch::Tensor out = vit->forward(x, torch::Tensor());

    // Output should be [B, embed_dim] (class token output)
    EXPECT_EQ(out.dim(), 2);
    EXPECT_EQ(out.size(0), 2);
    EXPECT_EQ(out.size(1), 384);
}

TEST(VisionTransformerTest, ForwardWithMasks) {
    torch::manual_seed(42);

    auto vit = vit_small();
    torch::Tensor x = torch::randn({2, 3, 224, 224});

    // Create mask with spatial dimensions matching num_patches (14x14 for 224x224 with patch_size=16)
    torch::Tensor masks = torch::zeros({2, 14, 14}, torch::kBool);
    masks[0][0][0] = true;
    masks[1][7][7] = true;

    torch::Tensor out = vit->forward(x, masks);

    EXPECT_TRUE(out.defined());
    EXPECT_EQ(out.dim(), 2);
    EXPECT_EQ(out.size(0), 2);  // B
}

TEST(VisionTransformerTest, ForwardPreservesGrad) {
    torch::manual_seed(42);

    auto vit = vit_small();
    torch::Tensor x = torch::randn({1, 3, 224, 224}, torch::requires_grad());

    torch::Tensor out = vit->forward(x, torch::Tensor());
    out.sum().backward();

    EXPECT_TRUE(x.grad().defined());
}

TEST(VisionTransformerTest, GetIntermediateLayers) {
    torch::manual_seed(42);

    auto vit = vit_small();
    torch::Tensor x = torch::randn({1, 3, 224, 224});

    auto outputs = vit->get_intermediate_layers(x, 4);

    EXPECT_EQ(outputs.size(), 4);
    for (const auto& out : outputs) {
        EXPECT_EQ(out.dim(), 3);
        EXPECT_EQ(out.size(0), 1);
        EXPECT_EQ(out.size(2), 384);
    }
}

TEST(VisionTransformerTest, GetIntermediateLayersWithReshape) {
    torch::manual_seed(42);

    auto vit = vit_small();
    torch::Tensor x = torch::randn({1, 3, 224, 224});

    auto outputs = vit->get_intermediate_layers(x, 2, true);

    EXPECT_EQ(outputs.size(), 2);
    for (const auto& out : outputs) {
        // With reshape=true, output should be [B, C, H, W]
        EXPECT_EQ(out.dim(), 4);
        EXPECT_EQ(out.size(0), 1);
        EXPECT_EQ(out.size(1), 384);  // embed_dim
        EXPECT_EQ(out.size(2), 14);   // H / patch_size
        EXPECT_EQ(out.size(3), 14);   // W / patch_size
    }
}

TEST(VisionTransformerTest, InterpolatePosEncoding) {
    torch::manual_seed(42);

    auto vit = vit_small();
    // Create tensor with different spatial dimensions
    torch::Tensor x = torch::randn({1, 197, 384});  // B, num_patches+1, embed_dim

    torch::Tensor pos_embed = vit->interpolate_pos_encoding(x, 256, 256);

    // Interpolated pos_embed should match input length
    EXPECT_EQ(pos_embed.size(1), x.size(1));
    EXPECT_EQ(pos_embed.size(2), 384);
}

TEST(VisionTransformerTest, PrepareTokensWithMasks) {
    torch::manual_seed(42);

    auto vit = vit_small();
    torch::Tensor x = torch::randn({1, 3, 224, 224});

    // Create mask with spatial dimensions (14x14 patches for 224x224 image)
    torch::Tensor masks = torch::zeros({1, 14, 14}, torch::kBool);
    masks[0][0][0] = true;  // Mask first patch

    torch::Tensor tokens = vit->prepare_tokens_with_masks(x, masks);

    EXPECT_TRUE(tokens.defined());
    EXPECT_EQ(tokens.dim(), 3);
    EXPECT_EQ(tokens.size(0), 1);  // B
    EXPECT_EQ(tokens.size(2), 384);  // embed_dim
}

TEST(VisionTransformerTest, PrepareTokensWithoutMasks) {
    torch::manual_seed(42);

    auto vit = vit_small();
    torch::Tensor x = torch::randn({1, 3, 224, 224});

    torch::Tensor tokens = vit->prepare_tokens_with_masks(x, torch::Tensor());

    EXPECT_EQ(tokens.dim(), 3);
    EXPECT_EQ(tokens.size(0), 1);
    EXPECT_EQ(tokens.size(1), 197);
    EXPECT_EQ(tokens.size(2), 384);
}

TEST(VisionTransformerTest, ForwardFeaturesList) {
    torch::manual_seed(42);

    auto vit = vit_small();
    std::vector<torch::Tensor> x_list = {
        torch::randn({1, 3, 224, 224}),
        torch::randn({1, 3, 224, 224})
    };
    std::vector<torch::Tensor> masks_list = {
        torch::Tensor(),
        torch::Tensor()
    };

    auto outputs = vit->forward_features_list(x_list, masks_list);

    EXPECT_EQ(outputs.size(), 2);
    for (const auto& out : outputs) {
        EXPECT_EQ(out.dim(), 3);
        EXPECT_EQ(out.size(0), 1);
        EXPECT_EQ(out.size(2), 384);
    }
}

TEST(VisionTransformerTest, NumPatchesCalculation) {
    torch::manual_seed(42);

    auto vit = vit_small();
    int64_t expected_patches = (224 / 16) * (224 / 16);  // 14 * 14 = 196

    EXPECT_EQ(vit->num_patches(), expected_patches);
}

TEST(VisionTransformerTest, DifferentImageSizes) {
    torch::manual_seed(42);

    auto vit = vit_small(16, 0, 256);

    EXPECT_TRUE(vit);
    torch::Tensor x = torch::randn({1, 3, 256, 256});

    torch::Tensor out = vit->forward(x, torch::Tensor());

    EXPECT_EQ(out.dim(), 2);
    EXPECT_EQ(out.size(0), 1);
    EXPECT_EQ(out.size(1), 384);
}

} // namespace
} // namespace layers
} // namespace vggt
