#include <gtest/gtest.h>
#include <torch/torch.h>
#include "layers/patch_embed.h"

namespace vggt {
namespace layers {
namespace {

TEST(PatchEmbedTest, ForwardOutputShape) {
    torch::manual_seed(42);
    
    int64_t img_size = 32;
    int64_t patch_size = 8;
    int64_t in_chans = 3;
    int64_t embed_dim = 64;
    
    PatchEmbed patch_embed(img_size, patch_size, in_chans, embed_dim);
    patch_embed->eval();
    
    // Input: [B, C, H, W] = [2, 3, 32, 32]
    torch::Tensor x = torch::randn({2, 3, 32, 32});
    
    torch::Tensor output = patch_embed->forward(x);
    
    // Output: [B, num_patches, embed_dim] where num_patches = (32/8)*(32/8) = 16
    EXPECT_EQ(output.dim(), 3);
    EXPECT_EQ(output.size(0), 2);   // batch
    EXPECT_EQ(output.size(1), 16);  // num_patches
    EXPECT_EQ(output.size(2), embed_dim); // embed_dim
}

TEST(PatchEmbedTest, DefaultConstruction) {
    torch::manual_seed(42);
    
    // Test with default parameters
    PatchEmbed patch_embed;
    patch_embed->eval();
    
    // Input: [B, C, H, W] = [1, 3, 224, 224]
    torch::Tensor x = torch::randn({1, 3, 224, 224});
    
    torch::Tensor output = patch_embed->forward(x);
    
    // Default: img_size=224, patch_size=16, embed_dim=768
    // num_patches = (224/16)*(224/16) = 14*14 = 196
    EXPECT_EQ(output.dim(), 3);
    EXPECT_EQ(output.size(0), 1);   // batch
    EXPECT_EQ(output.size(1), 196);  // num_patches
    EXPECT_EQ(output.size(2), 768); // embed_dim
}

TEST(PatchEmbedTest, NoFlattenEmbedding) {
    torch::manual_seed(42);
    
    PatchEmbed patch_embed(32, 8, 3, 64, torch::nn::AnyModule(), false);
    patch_embed->eval();
    
    torch::Tensor x = torch::randn({2, 3, 32, 32});
    
    torch::Tensor output = patch_embed->forward(x);
    
    // Output when flatten=false: [B, H, W, C] = [2, 4, 4, 64]
    EXPECT_EQ(output.dim(), 4);
    EXPECT_EQ(output.size(0), 2);   // batch
    EXPECT_EQ(output.size(1), 4);   // H
    EXPECT_EQ(output.size(2), 4);   // W
    EXPECT_EQ(output.size(3), 64); // embed_dim
}

TEST(PatchEmbedTest, WithNormLayer) {
    torch::manual_seed(42);
    
    torch::nn::LayerNorm norm_layer(torch::nn::LayerNormOptions({64}));
    PatchEmbed patch_embed(32, 8, 3, 64, torch::nn::AnyModule(norm_layer));
    patch_embed->eval();
    
    torch::Tensor x = torch::randn({1, 3, 32, 32});
    
    torch::Tensor output = patch_embed->forward(x);
    
    EXPECT_EQ(output.dim(), 3);
    EXPECT_EQ(output.size(0), 1);
    EXPECT_EQ(output.size(1), 16);
    EXPECT_EQ(output.size(2), 64);
}

TEST(PatchEmbedTest, GradientFlow) {
    torch::manual_seed(42);
    
    PatchEmbed patch_embed(32, 8, 3, 64);
    patch_embed->train();
    
    torch::Tensor x = torch::randn({2, 3, 32, 32});
    x.set_requires_grad(true);
    
    torch::Tensor output = patch_embed->forward(x);
    torch::Tensor loss = output.sum();
    loss.backward();
    
    EXPECT_TRUE(x.grad().defined());
    EXPECT_EQ(x.grad().size(0), 2);
    EXPECT_EQ(x.grad().size(1), 3);
    EXPECT_EQ(x.grad().size(2), 32);
    EXPECT_EQ(x.grad().size(3), 32);
}

TEST(PatchEmbedTest, TrainEvalModes) {
    torch::manual_seed(42);
    
    PatchEmbed patch_embed(32, 8, 3, 64);
    
    torch::Tensor x = torch::randn({1, 3, 32, 32});
    
    patch_embed->train();
    EXPECT_TRUE(patch_embed->is_training());
    
    patch_embed->eval();
    EXPECT_FALSE(patch_embed->is_training());
    
    torch::Tensor output = patch_embed->forward(x);
    EXPECT_EQ(output.size(1), 16);
}

TEST(PatchEmbedTest, ModuleRegistration) {
    torch::manual_seed(42);
    
    PatchEmbed patch_embed(32, 8, 3, 64);
    
    // Should have proj module
    auto named_modules = patch_embed->named_modules();
    EXPECT_GT(named_modules.size(), 0);
    
    bool has_proj = false;
    for (const auto& item : named_modules) {
        if (item.key() == "proj") {
            has_proj = true;
            break;
        }
    }
    EXPECT_TRUE(has_proj);
}

TEST(PatchEmbedTest, DifferentPatchSizes) {
    torch::manual_seed(42);
    
    std::vector<int64_t> patch_sizes = {4, 8, 16};
    
    for (int64_t patch_size : patch_sizes) {
        int64_t img_size = 32;
        if (32 % patch_size == 0) {
            PatchEmbed patch_embed(img_size, patch_size, 3, 64);
            patch_embed->eval();
            
            torch::Tensor x = torch::randn({1, 3, img_size, img_size});
            torch::Tensor output = patch_embed->forward(x);
            
            int64_t num_patches = (img_size / patch_size) * (img_size / patch_size);
            EXPECT_EQ(output.size(1), num_patches);
        }
    }
}

TEST(PatchEmbedTest, DifferentEmbedDims) {
    torch::manual_seed(42);
    
    std::vector<int64_t> embed_dims = {64, 128, 256, 512};
    
    for (int64_t embed_dim : embed_dims) {
        PatchEmbed patch_embed(32, 8, 3, embed_dim);
        patch_embed->eval();
        
        torch::Tensor x = torch::randn({1, 3, 32, 32});
        torch::Tensor output = patch_embed->forward(x);
        
        EXPECT_EQ(output.size(2), embed_dim);
    }
}

TEST(PatchEmbedTest, DifferentInputChannels) {
    torch::manual_seed(42);
    
    std::vector<int64_t> in_chans = {1, 3, 4};
    
    for (int64_t in_chans : in_chans) {
        PatchEmbed patch_embed(32, 8, in_chans, 64);
        patch_embed->eval();
        
        torch::Tensor x = torch::randn({1, in_chans, 32, 32});
        torch::Tensor output = patch_embed->forward(x);
        
        EXPECT_EQ(output.size(2), 64);
    }
}

TEST(PatchEmbedTest, SquareImageAndPatch) {
    torch::manual_seed(42);
    
    PatchEmbed patch_embed(64, 16, 3, 128);
    patch_embed->eval();
    
    torch::Tensor x = torch::randn({4, 3, 64, 64});
    torch::Tensor output = patch_embed->forward(x);
    
    // 64/16 = 4, so 4*4 = 16 patches
    EXPECT_EQ(output.size(0), 4);   // batch
    EXPECT_EQ(output.size(1), 16);  // num_patches
    EXPECT_EQ(output.size(2), 128); // embed_dim
}

TEST(PatchEmbedTest, OutputFiniteValues) {
    torch::manual_seed(42);
    
    PatchEmbed patch_embed(32, 8, 3, 64);
    patch_embed->eval();
    
    torch::Tensor x = torch::randn({1, 3, 32, 32});
    torch::Tensor output = patch_embed->forward(x);
    
    EXPECT_TRUE(torch::isfinite(output).all().item<bool>());
}

TEST(PatchEmbedTest, NonSquarePatchNotSupported) {
    // Note: PatchEmbed constructor takes single int64_t patch_size,
    // and internally uses make_2tuple to convert to (H, W).
    // So non-square patches need to be handled differently.
    // This test documents that the current implementation only supports square patches.
    GTEST_SKIP() << "PatchEmbed only supports square patch sizes";
}

} // namespace
} // namespace layers
} // namespace vggt
