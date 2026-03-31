#include <gtest/gtest.h>
#include <torch/torch.h>
#include "vggt/heads/dpt_head.h"

namespace vggt {
namespace heads {
namespace {

// ==================== custom_interpolate Tests ====================

TEST(CustomInterpolateTest, BasicBilinearWithSize) {
    torch::Tensor input = torch::randn({1, 3, 32, 32});
    auto output = custom_interpolate(input, std::make_pair(64, 64), c10::nullopt, torch::kBilinear, true);
    EXPECT_EQ(output.sizes(), std::vector<int64_t>({1, 3, 64, 64}));
}

TEST(CustomInterpolateTest, BasicBilinearWithScaleFactor) {
    torch::Tensor input = torch::randn({1, 3, 32, 32});
    auto output = custom_interpolate(input, c10::nullopt, 2.0, torch::kBilinear, true);
    EXPECT_EQ(output.sizes(), std::vector<int64_t>({1, 3, 64, 64}));
}

TEST(CustomInterpolateTest, Downsample) {
    torch::Tensor input = torch::randn({1, 3, 64, 64});
    auto output = custom_interpolate(input, std::make_pair(32, 32), c10::nullopt, torch::kBilinear, true);
    EXPECT_EQ(output.sizes(), std::vector<int64_t>({1, 3, 32, 32}));
}

TEST(CustomInterpolateTest, NoResize) {
    torch::Tensor input = torch::randn({1, 3, 64, 64});
    auto output = custom_interpolate(input, std::make_pair(64, 64), c10::nullopt, torch::kBilinear, true);
    EXPECT_EQ(output.sizes(), std::vector<int64_t>({1, 3, 64, 64}));
}

TEST(CustomInterpolateTest, MultiChannel) {
    torch::Tensor input = torch::randn({2, 16, 32, 32});
    auto output = custom_interpolate(input, std::make_pair(64, 64), c10::nullopt, torch::kBilinear, true);
    EXPECT_EQ(output.sizes(), std::vector<int64_t>({2, 16, 64, 64}));
}

TEST(CustomInterpolateTest, OutputIsContinuous) {
    torch::Tensor input = torch::randn({1, 3, 32, 32});
    auto output = custom_interpolate(input, std::make_pair(64, 64), c10::nullopt, torch::kBilinear, true);
    EXPECT_TRUE(output.is_contiguous());
}

TEST(CustomInterpolateTest, OutputIsFinite) {
    torch::Tensor input = torch::randn({1, 3, 32, 32});
    auto output = custom_interpolate(input, std::make_pair(64, 64), c10::nullopt, torch::kBilinear, true);
    EXPECT_TRUE(torch::isfinite(output).all().item<bool>());
}

TEST(CustomInterpolateTest, NeitherSizeNorScaleProvided) {
    torch::Tensor input = torch::randn({1, 3, 32, 32});
    EXPECT_THROW(
        custom_interpolate(input, c10::nullopt, c10::nullopt, torch::kBilinear, true),
        std::runtime_error
    );
}

// ==================== _make_scratch Tests ====================

TEST(MakeScratchTest, BasicConstruction) {
    std::vector<int64_t> in_shape = {256, 512, 1024, 1024};
    auto scratch = _make_scratch(in_shape, 256, 1, false);
    // Just verify it was created without throwing
    EXPECT_TRUE(true);
}

TEST(MakeScratchTest, WithExpansion) {
    std::vector<int64_t> in_shape = {256, 512, 1024, 1024};
    auto scratch = _make_scratch(in_shape, 256, 1, true);
    EXPECT_TRUE(true);
}

TEST(MakeScratchTest, ThreeInputChannels) {
    std::vector<int64_t> in_shape = {256, 512, 1024};
    auto scratch = _make_scratch(in_shape, 256, 1, false);
    EXPECT_TRUE(true);
}

TEST(MakeScratchTest, ScratchRegisteredModules) {
    std::vector<int64_t> in_shape = {256, 512, 1024, 1024};
    auto scratch = _make_scratch(in_shape, 256, 1, false);
    
    // Use children() instead of named_modules() to avoid libtorch API issues
    // children() returns immediate submodules
    std::vector<std::string> module_names;
    for (const auto& child : scratch.children()) {
        module_names.push_back(child->name());
    }
    
    // Should have 4 submodules (layer1_rn, layer2_rn, layer3_rn, layer4_rn)
    EXPECT_EQ(module_names.size(), 4);
}

// ==================== _make_fusion_block Tests ====================

TEST(MakeFusionBlockTest, BasicConstruction) {
    auto fusion_block = _make_fusion_block(256, c10::nullopt, true, 1);
    EXPECT_TRUE(true);
}

TEST(MakeFusionBlockTest, HasResidualConvUnit) {
    auto fusion_block = _make_fusion_block(256, c10::nullopt, true, 1);
    
    // Use children() instead of named_modules() to avoid libtorch API issues
    std::vector<std::string> module_names;
    for (const auto& child : fusion_block.children()) {
        module_names.push_back(child->name());
    }
    
    // Should have 1 submodule (resConfUnit)
    EXPECT_EQ(module_names.size(), 1);
}

TEST(MakeFusionBlockTest, WithoutResidual) {
    auto fusion_block = _make_fusion_block(256, c10::nullopt, false, 1);
    EXPECT_TRUE(true);
}

// ==================== ResidualConvUnit Tests ====================

TEST(ResidualConvUnitTest, ConstructorWithReLU) {
    EXPECT_NO_THROW({
        ResidualConvUnit unit(256, torch::nn::AnyModule(torch::nn::ReLU()), true);
    });
}

TEST(ResidualConvUnitTest, ConstructorWithGELU) {
    EXPECT_NO_THROW({
        ResidualConvUnit unit(256, torch::nn::AnyModule(torch::nn::GELU()), true);
    });
}

TEST(ResidualConvUnitTest, ConstructorWithoutBN) {
    EXPECT_NO_THROW({
        ResidualConvUnit unit(256, torch::nn::AnyModule(torch::nn::ReLU()), false);
    });
}

TEST(ResidualConvUnitTest, ConstructorWithGroups) {
    EXPECT_NO_THROW({
        ResidualConvUnit unit(256, torch::nn::AnyModule(torch::nn::ReLU()), true, 4);
    });
}

TEST(ResidualConvUnitTest, ForwardBasic) {
    ResidualConvUnit unit(256, torch::nn::AnyModule(torch::nn::ReLU()), true);
    torch::Tensor input = torch::randn({1, 256, 32, 32});
    torch::Tensor output = unit->forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST(ResidualConvUnitTest, ForwardMultiBatch) {
    ResidualConvUnit unit(128, torch::nn::AnyModule(torch::nn::ReLU()), true);
    torch::Tensor input = torch::randn({4, 128, 16, 16});
    torch::Tensor output = unit->forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST(ResidualConvUnitTest, ForwardOutputIsFinite) {
    ResidualConvUnit unit(128, torch::nn::AnyModule(torch::nn::ReLU()), true);
    torch::Tensor input = torch::randn({1, 128, 16, 16}) * 0.1;
    torch::Tensor output = unit->forward(input);
    EXPECT_TRUE(torch::isfinite(output).all().item<bool>());
}

TEST(ResidualConvUnitTest, ForwardResidualConnection) {
    ResidualConvUnit unit(64, torch::nn::AnyModule(torch::nn::ReLU()), false);
    torch::Tensor input = torch::ones({1, 64, 8, 8});
    torch::Tensor output = unit->forward(input);
    // Check that output is different from input (conv transforms it)
    EXPECT_FALSE(torch::allclose(output, input));
}

// ==================== FeatureFusionBlock Tests ====================

TEST(FeatureFusionBlockTest, ConstructorBasic) {
    EXPECT_NO_THROW({
        FeatureFusionBlock block(
            256,
            torch::nn::AnyModule(torch::nn::ReLU()),
            false, false, false, true,
            c10::nullopt, true, 1
        );
    });
}

TEST(FeatureFusionBlockTest, ConstructorWithSize) {
    EXPECT_NO_THROW({
        FeatureFusionBlock block(
            256,
            torch::nn::AnyModule(torch::nn::ReLU()),
            false, false, false, true,
            std::make_pair(64, 64), true, 1
        );
    });
}

TEST(FeatureFusionBlockTest, ConstructorWithExpand) {
    EXPECT_NO_THROW({
        FeatureFusionBlock block(
            256,
            torch::nn::AnyModule(torch::nn::ReLU()),
            false, false, true, true,
            c10::nullopt, true, 1
        );
    });
}

TEST(FeatureFusionBlockTest, ConstructorWithoutResidual) {
    EXPECT_NO_THROW({
        FeatureFusionBlock block(
            256,
            torch::nn::AnyModule(torch::nn::ReLU()),
            false, false, false, true,
            c10::nullopt, false, 1
        );
    });
}

TEST(FeatureFusionBlockTest, ForwardSingleInput) {
    FeatureFusionBlock block(
        256,
        torch::nn::AnyModule(torch::nn::ReLU()),
        false, false, false, true,
        c10::nullopt, false, 1
    );
    std::vector<torch::Tensor> inputs = {torch::randn({1, 256, 32, 32})};
    torch::Tensor output = block->forward(inputs);
    EXPECT_EQ(output.dim(), 4);
}

TEST(FeatureFusionBlockTest, ForwardTwoInputsWithResidual) {
    // Test with two inputs of different resolutions - residual is now interpolated to match
    FeatureFusionBlock block(
        256,
        torch::nn::AnyModule(torch::nn::ReLU()),
        false, false, false, true,
        c10::nullopt, true, 1
    );
    std::vector<torch::Tensor> inputs = {torch::randn({1, 256, 32, 32}), torch::randn({1, 256, 64, 64})};
    torch::Tensor output = block->forward(inputs);
    EXPECT_EQ(output.dim(), 4);
    EXPECT_TRUE(torch::isfinite(output).all().item<bool>());
}

TEST(FeatureFusionBlockTest, ForwardDifferentResolutions) {
    // Test with inputs at different resolutions - residual is interpolated
    FeatureFusionBlock block(
        128,
        torch::nn::AnyModule(torch::nn::ReLU()),
        false, false, false, true,
        std::make_pair(32, 32), true, 1
    );
    std::vector<torch::Tensor> inputs = {torch::randn({1, 128, 16, 16}), torch::randn({1, 128, 32, 32})};
    torch::Tensor output = block->forward(inputs);
    EXPECT_EQ(output.dim(), 4);
}

TEST(FeatureFusionBlockTest, ForwardMultiBatch) {
    // Test with batch size > 1 and different resolutions
    FeatureFusionBlock block(
        256,
        torch::nn::AnyModule(torch::nn::ReLU()),
        false, false, false, true,
        c10::nullopt, true, 1
    );
    std::vector<torch::Tensor> inputs = {torch::randn({4, 256, 32, 32}), torch::randn({4, 256, 64, 64})};
    torch::Tensor output = block->forward(inputs);
    EXPECT_EQ(output.size(0), 4);
}

TEST(FeatureFusionBlockTest, ForwardOutputIsFinite) {
    // Test with different resolutions and small input values
    FeatureFusionBlock block(
        256,
        torch::nn::AnyModule(torch::nn::ReLU()),
        false, false, false, true,
        c10::nullopt, true, 1
    );
    std::vector<torch::Tensor> inputs = {torch::randn({1, 256, 32, 32}) * 0.1, torch::randn({1, 256, 64, 64}) * 0.1};
    torch::Tensor output = block->forward(inputs);
    EXPECT_TRUE(torch::isfinite(output).all().item<bool>());
}

TEST(FeatureFusionBlockTest, ForwardSizeOverride) {
    // Test with size override and two inputs of different resolutions
    FeatureFusionBlock block(
        256,
        torch::nn::AnyModule(torch::nn::ReLU()),
        false, false, false, true,
        c10::nullopt, true, 1
    );
    std::vector<torch::Tensor> inputs = {torch::randn({1, 256, 32, 32}), torch::randn({1, 256, 64, 64})};
    torch::Tensor output = block->forward(inputs, std::make_pair(16, 16));
    EXPECT_EQ(output.dim(), 4);
}

TEST(FeatureFusionBlockTest, RequiresAtLeastOneInput) {
    FeatureFusionBlock block(
        256,
        torch::nn::AnyModule(torch::nn::ReLU()),
        false, false, false, true,
        c10::nullopt, false, 1
    );
    std::vector<torch::Tensor> inputs = {};
    EXPECT_THROW(block->forward(inputs), c10::Error);
}

TEST(FeatureFusionBlockTest, RequiresTwoInputsWithResidual) {
    FeatureFusionBlock block(
        256,
        torch::nn::AnyModule(torch::nn::ReLU()),
        false, false, false, true,
        c10::nullopt, true, 1
    );
    std::vector<torch::Tensor> inputs = {torch::randn({1, 256, 32, 32})};
    EXPECT_THROW(block->forward(inputs), c10::Error);
}

} // namespace
} // namespace heads
} // namespace vggt
