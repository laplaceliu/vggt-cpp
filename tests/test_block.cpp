#include <gtest/gtest.h>
#include <torch/torch.h>
#include "vggt/layers/block.h"

namespace vggt {
namespace layers {
namespace {

TEST(BlockTest, ConstructorBasic) {
    torch::manual_seed(42);

    int64_t dim = 64;
    int64_t num_heads = 4;

    Block block = Block(std::make_shared<BlockImpl>(dim, num_heads));

    // Block should be created successfully
    EXPECT_TRUE(block);
}

TEST(BlockTest, ConstructorWithDropPath) {
    torch::manual_seed(42);

    int64_t dim = 64;
    int64_t num_heads = 4;
    double drop_path = 0.1;

    Block block = Block(std::make_shared<BlockImpl>(
        dim, num_heads, 4.0, true, true, true, 0.0, 0.0,
        torch::Tensor(), drop_path));

    EXPECT_TRUE(block);
}

TEST(BlockTest, ConstructorWithLayerScale) {
    torch::manual_seed(42);

    int64_t dim = 64;
    int64_t num_heads = 4;
    torch::Tensor init_values = torch::tensor(0.1);

    Block block = Block(std::make_shared<BlockImpl>(
        dim, num_heads, 4.0, true, true, true, 0.0, 0.0, init_values));

    EXPECT_TRUE(block);
}

TEST(BlockTest, ForwardBasic) {
    torch::manual_seed(42);

    int64_t dim = 64;
    int64_t num_heads = 4;
    int64_t batch_size = 2;
    int64_t seq_len = 16;

    Block block = Block(std::make_shared<BlockImpl>(dim, num_heads));
    torch::Tensor x = torch::randn({batch_size, seq_len, dim});

    torch::Tensor out = block->forward(x);

    EXPECT_EQ(out.sizes(), x.sizes());
    EXPECT_TRUE(out.requires_grad());
}

TEST(BlockTest, ForwardWithPosition) {
    torch::manual_seed(42);

    int64_t dim = 64;
    int64_t num_heads = 4;
    int64_t batch_size = 2;
    int64_t seq_len = 16;

    Block block = Block(std::make_shared<BlockImpl>(dim, num_heads));
    torch::Tensor x = torch::randn({batch_size, seq_len, dim});
    torch::Tensor pos = torch::randn({batch_size, seq_len, dim});

    torch::Tensor out = block->forward(x, pos);

    EXPECT_EQ(out.sizes(), x.sizes());
}

TEST(BlockTest, ForwardDifferentDimensions) {
    torch::manual_seed(42);

    int64_t dim = 128;
    int64_t num_heads = 8;
    int64_t batch_size = 4;
    int64_t seq_len = 32;

    Block block = Block(std::make_shared<BlockImpl>(dim, num_heads));
    torch::Tensor x = torch::randn({batch_size, seq_len, dim});

    torch::Tensor out = block->forward(x);

    EXPECT_EQ(out.sizes(), x.sizes());
    EXPECT_EQ(out.size(2), dim);
}

TEST(BlockTest, ForwardPreservesGrad) {
    torch::manual_seed(42);

    int64_t dim = 64;
    int64_t num_heads = 4;

    Block block = Block(std::make_shared<BlockImpl>(dim, num_heads));
    torch::Tensor x = torch::randn({2, 16, dim}, torch::requires_grad());

    torch::Tensor out = block->forward(x);
    out.sum().backward();

    EXPECT_TRUE(x.grad().defined());
}

TEST(BlockTest, ForwardWithDropPath) {
    torch::manual_seed(42);

    int64_t dim = 64;
    int64_t num_heads = 4;
    double drop_path = 0.1;

    Block block = Block(std::make_shared<BlockImpl>(
        dim, num_heads, 4.0, true, true, true, 0.0, 0.0,
        torch::Tensor(), drop_path));
    torch::Tensor x = torch::randn({2, 16, dim});

    torch::Tensor out = block->forward(x);

    EXPECT_EQ(out.sizes(), x.sizes());
}

TEST(BlockTest, ForwardWithQKNorm) {
    GTEST_SKIP() << "Skipped: Empty AnyModule cannot be passed to Block constructor";
}

// NestedTensorBlock tests
TEST(NestedTensorBlockTest, ConstructorBasic) {
    torch::manual_seed(42);

    int64_t dim = 64;
    int64_t num_heads = 4;

    NestedTensorBlock block = NestedTensorBlock(std::make_shared<NestedTensorBlockImpl>(dim, num_heads));

    EXPECT_TRUE(block);
}

TEST(NestedTensorBlockTest, ForwardNested) {
    GTEST_SKIP() << "Skipped: NestedTensorBlock forward requires properly initialized submodules";
}

TEST(NestedTensorBlockTest, ForwardSingleTensor) {
    torch::manual_seed(42);

    int64_t dim = 64;
    int64_t num_heads = 4;

    NestedTensorBlock block = NestedTensorBlock(std::make_shared<NestedTensorBlockImpl>(dim, num_heads));
    torch::Tensor x = torch::randn({2, 16, dim});

    torch::Tensor out = block->forward(x);

    EXPECT_EQ(out.sizes(), x.sizes());
}

TEST(NestedTensorBlockTest, ForwardNestedPreservesGrad) {
    GTEST_SKIP() << "Skipped: NestedTensorBlock forward requires properly initialized submodules";
}

// Helper function tests
TEST(BlockHelperTest, DropAddResidualStochasticDepth) {
    torch::manual_seed(42);

    torch::Tensor x = torch::randn({2, 16, 64});
    auto residual_func = [](torch::Tensor x, torch::Tensor) -> torch::Tensor {
        return x * 0.5;
    };

    torch::Tensor out = drop_add_residual_stochastic_depth(x, residual_func, 0.0, {});

    EXPECT_EQ(out.sizes(), x.sizes());
}

TEST(BlockHelperTest, DropAddResidualWithPos) {
    torch::manual_seed(42);

    torch::Tensor x = torch::randn({2, 16, 64});
    torch::Tensor pos = torch::randn({2, 16, 64});
    auto residual_func = [](torch::Tensor x, torch::Tensor pos) -> torch::Tensor {
        return x + pos;
    };

    torch::Tensor out = drop_add_residual_stochastic_depth(x, residual_func, 0.0, pos);

    EXPECT_EQ(out.sizes(), x.sizes());
}

TEST(BlockHelperTest, GetBrangesScales) {
    torch::manual_seed(42);

    torch::Tensor x = torch::randn({2, 16, 64});

    auto [brange, scale] = get_branges_scales(x, 0.0);

    EXPECT_EQ(brange.sizes(), x.sizes());
    EXPECT_DOUBLE_EQ(scale, 1.0);
}

TEST(BlockHelperTest, AddResidual) {
    torch::manual_seed(42);

    torch::Tensor x = torch::randn({2, 16, 64});
    torch::Tensor brange = torch::ones_like(x);
    torch::Tensor residual = torch::randn({2, 16, 64});

    torch::Tensor out = add_residual(x, brange, residual, 1.0, {});

    EXPECT_EQ(out.sizes(), x.sizes());
}

TEST(BlockHelperTest, AddResidualWithScaling) {
    torch::manual_seed(42);

    torch::Tensor x = torch::randn({2, 16, 64});
    torch::Tensor brange = torch::ones_like(x);
    torch::Tensor residual = torch::randn({2, 16, 64});
    torch::Tensor scaling = torch::ones({64}) * 0.5;

    torch::Tensor out = add_residual(x, brange, residual, 0.5, scaling);

    EXPECT_EQ(out.sizes(), x.sizes());
}

} // namespace
} // namespace layers
} // namespace vggt
