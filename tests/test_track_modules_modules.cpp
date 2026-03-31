#include <gtest/gtest.h>
#include <torch/torch.h>
#include "vggt/dependency/track_modules/modules.h"

namespace vggt {
namespace dependency {
namespace track_modules {
namespace {

// ==================== Helper Functions Tests ====================

TEST(NTupleTest, TwoTupleWithOne) {
    auto result = _ntuple(2, 1);
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0], 1);
    EXPECT_EQ(result[1], 1);
}

TEST(NTupleTest, TwoTupleWithFour) {
    auto result = _ntuple(2, 4);
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0], 4);
    EXPECT_EQ(result[1], 4);
}

TEST(ExistsTest, DefinedTensor) {
    torch::Tensor t = torch::randn({3, 3});
    EXPECT_TRUE(exists(t));
}

TEST(ExistsTest, UndefinedTensor) {
    torch::Tensor t;
    EXPECT_FALSE(exists(t));
}

TEST(DefaultValTest, TensorDefined) {
    torch::Tensor val = torch::tensor({1.0});
    torch::Tensor default_tensor = torch::tensor({2.0});
    auto result = default_val(val, default_tensor);
    EXPECT_EQ(result.item<float>(), 1.0);
}

TEST(DefaultValTest, TensorUndefined) {
    torch::Tensor val;
    torch::Tensor default_tensor = torch::tensor({2.0});
    auto result = default_val(val, default_tensor);
    EXPECT_EQ(result.item<float>(), 2.0);
}

TEST(To2TupleTest, SingleValue) {
    auto result = to_2tuple(3);
    EXPECT_EQ(result.size(), 2);
    EXPECT_EQ(result[0], 3);
    EXPECT_EQ(result[1], 3);
}

TEST(To2TupleTest, PairValue) {
    auto result = to_2tuple(4);
    EXPECT_EQ(result.size(), 2);
}

// ==================== ResidualBlock Tests ====================

TEST(ResidualBlockTest, ConstructorWithGroupNorm) {
    // This should not throw
    EXPECT_NO_THROW({
        ResidualBlock block(64, 128, "group", 1, 3);
    });
}

TEST(ResidualBlockTest, ConstructorWithBatchNorm) {
    EXPECT_NO_THROW({
        ResidualBlock block(64, 128, "batch", 1, 3);
    });
}

TEST(ResidualBlockTest, ConstructorWithInstanceNorm) {
    EXPECT_NO_THROW({
        ResidualBlock block(64, 128, "instance", 1, 3);
    });
}

TEST(ResidualBlockTest, ConstructorWithNoNorm) {
    EXPECT_NO_THROW({
        ResidualBlock block(64, 128, "none", 1, 3);
    });
}

TEST(ResidualBlockTest, ConstructorInvalidNorm) {
    EXPECT_THROW({
        ResidualBlock block(64, 128, "invalid", 1, 3);
    }, std::runtime_error);
}

TEST(ResidualBlockTest, ForwardSameChannels) {
    ResidualBlock block(64, 64, "instance", 1, 3);
    torch::Tensor input = torch::randn({1, 64, 32, 32});
    torch::Tensor output = block->forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST(ResidualBlockTest, ForwardChannelExpansion) {
    // Test channel expansion with stride=1 - downsample conv should be created
    ResidualBlock block(64, 128, "instance", 1, 3);
    torch::Tensor input = torch::randn({2, 64, 32, 32});
    torch::Tensor output = block->forward(input);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 128);
    EXPECT_EQ(output.size(2), 32);
    EXPECT_EQ(output.size(3), 32);
}

TEST(ResidualBlockTest, ForwardWithStride2) {
    ResidualBlock block(64, 64, "instance", 2, 3);
    torch::Tensor input = torch::randn({1, 64, 32, 32});
    torch::Tensor output = block->forward(input);
    EXPECT_EQ(output.size(2), 16);
    EXPECT_EQ(output.size(3), 16);
}

TEST(ResidualBlockTest, ForwardOutputIsFinite) {
    ResidualBlock block(64, 64, "group", 1, 3);
    torch::Tensor input = torch::randn({1, 64, 16, 16});
    torch::Tensor output = block->forward(input);
    EXPECT_TRUE(torch::isfinite(output).all().item<bool>());
}

// ==================== Mlp Tests ====================

TEST(MlpTest, ConstructorDefault) {
    EXPECT_NO_THROW({
        Mlp mlp(128);
    });
}

TEST(MlpTest, ConstructorFullParams) {
    EXPECT_NO_THROW({
        Mlp mlp(128, 256, 64, torch::nn::AnyModule(torch::nn::GELU()),
                torch::nn::AnyModule(), true, 0.1);
    });
}

TEST(MlpTest, ForwardBasic) {
    Mlp mlp(128);
    torch::Tensor input = torch::randn({4, 128});
    torch::Tensor output = mlp->forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST(MlpTest, ForwardHiddenFeatures) {
    Mlp mlp(128, 512);
    torch::Tensor input = torch::randn({2, 128});
    torch::Tensor output = mlp->forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST(MlpTest, ForwardOutputFeatures) {
    Mlp mlp(128, 256, 64);
    torch::Tensor input = torch::randn({1, 128});
    torch::Tensor output = mlp->forward(input);
    EXPECT_EQ(output.size(1), 64);
}

TEST(MlpTest, ForwardWithDropout) {
    Mlp mlp(128, 256, 128, torch::nn::AnyModule(torch::nn::GELU()),
            torch::nn::AnyModule(), true, 0.5);
    torch::Tensor input = torch::randn({4, 128});
    torch::Tensor output = mlp->forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST(MlpTest, ForwardOutputIsFinite) {
    Mlp mlp(128);
    torch::Tensor input = torch::randn({2, 128}) * 0.1;  // small input
    torch::Tensor output = mlp->forward(input);
    EXPECT_TRUE(torch::isfinite(output).all().item<bool>());
}

// ==================== AttnBlock Tests ====================

TEST(AttnBlockTest, Constructor) {
    EXPECT_NO_THROW({
        AttnBlock block(128, 4, torch::nn::AnyModule(torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(128, 4))));
    });
}

TEST(AttnBlockTest, ForwardBasic) {
    AttnBlock block(128, 4, torch::nn::AnyModule(torch::nn::MultiheadAttention(
        torch::nn::MultiheadAttentionOptions(128, 4))));
    // Input: [batch, seq, embed]
    torch::Tensor input = torch::randn({2, 10, 128});
    torch::Tensor output = block->forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST(AttnBlockTest, ForwardSingleBatch) {
    AttnBlock block(128, 4, torch::nn::AnyModule(torch::nn::MultiheadAttention(
        torch::nn::MultiheadAttentionOptions(128, 4))));
    torch::Tensor input = torch::randn({1, 8, 128});
    torch::Tensor output = block->forward(input);
    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST(AttnBlockTest, ForwardOutputIsFinite) {
    AttnBlock block(128, 4, torch::nn::AnyModule(torch::nn::MultiheadAttention(
        torch::nn::MultiheadAttentionOptions(128, 4))));
    torch::Tensor input = torch::randn({1, 8, 128}) * 0.1;
    torch::Tensor output = block->forward(input);
    EXPECT_TRUE(torch::isfinite(output).all().item<bool>());
}

TEST(AttnBlockTest, ForwardWithEmptyMask) {
    AttnBlock block(128, 4, torch::nn::AnyModule(torch::nn::MultiheadAttention(
        torch::nn::MultiheadAttentionOptions(128, 4))));
    torch::Tensor input = torch::randn({1, 8, 128});
    torch::Tensor mask = torch::tensor(0.0f);  // empty mask
    torch::Tensor output = block->forward(input, mask);
    EXPECT_EQ(output.sizes(), input.sizes());
}

// ==================== CrossAttnBlock Tests ====================

TEST(CrossAttnBlockTest, Constructor) {
    EXPECT_NO_THROW({
        CrossAttnBlock block(128, 128, 4);
    });
}

TEST(CrossAttnBlockTest, ConstructorDefaultNumHeads) {
    EXPECT_NO_THROW({
        CrossAttnBlock block(128, 128, 1);
    });
}

TEST(CrossAttnBlockTest, ForwardBasic) {
    // Note: In actual usage, hidden_size == context_dim
    CrossAttnBlock block(128, 128, 4);
    torch::Tensor x = torch::randn({2, 10, 128});
    torch::Tensor context = torch::randn({2, 20, 128});
    torch::Tensor output = block->forward(x, context);
    EXPECT_EQ(output.sizes(), x.sizes());
}

TEST(CrossAttnBlockTest, ForwardSingleBatch) {
    CrossAttnBlock block(128, 128, 4);
    torch::Tensor x = torch::randn({1, 8, 128});
    torch::Tensor context = torch::randn({1, 16, 128});
    torch::Tensor output = block->forward(x, context);
    EXPECT_EQ(output.sizes(), x.sizes());
}

TEST(CrossAttnBlockTest, ForwardOutputIsFinite) {
    CrossAttnBlock block(128, 128, 4);
    torch::Tensor x = torch::randn({1, 8, 128}) * 0.1;
    torch::Tensor context = torch::randn({1, 16, 128}) * 0.1;
    torch::Tensor output = block->forward(x, context);
    EXPECT_TRUE(torch::isfinite(output).all().item<bool>());
}

TEST(CrossAttnBlockTest, ForwardWithEmptyMask) {
    CrossAttnBlock block(128, 128, 4);
    torch::Tensor x = torch::randn({1, 8, 128});
    torch::Tensor context = torch::randn({1, 16, 128});
    torch::Tensor mask;  // empty/undefined mask (default)
    torch::Tensor output = block->forward(x, context, mask);
    EXPECT_EQ(output.sizes(), x.sizes());
}

} // namespace
} // namespace track_modules
} // namespace dependency
} // namespace vggt
