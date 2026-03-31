#include <gtest/gtest.h>
#include <torch/torch.h>
#include "vggt/heads/utils.h"

namespace vggt {
namespace heads {
namespace {

TEST(HeadsUtilsTest, MakeSinCosPosEmbedBasic) {
    torch::manual_seed(42);

    int64_t embed_dim = 32;
    torch::Tensor pos = torch::tensor({0.0f, 1.0f, 2.0f});

    torch::Tensor embed = make_sincos_pos_embed(embed_dim, pos);

    EXPECT_EQ(embed.dim(), 2);
    EXPECT_EQ(embed.size(0), 3);   // M positions
    EXPECT_EQ(embed.size(1), embed_dim);  // embed_dim
}

TEST(HeadsUtilsTest, MakeSinCosPosEmbedEvenDim) {
    torch::manual_seed(42);

    // embed_dim must be even
    int64_t embed_dim = 64;
    torch::Tensor pos = torch::tensor({0.0f, 1.0f});

    torch::Tensor embed = make_sincos_pos_embed(embed_dim, pos);

    EXPECT_EQ(embed.size(1), embed_dim);
}

TEST(HeadsUtilsTest, MakeSinCosPosEmbedOddDim) {
    torch::manual_seed(42);

    // embed_dim must be even - should throw c10::Error
    int64_t embed_dim = 33;
    torch::Tensor pos = torch::tensor({0.0f, 1.0f});

    EXPECT_THROW(make_sincos_pos_embed(embed_dim, pos), c10::Error);
}

TEST(HeadsUtilsTest, MakeSinCosPosEmbedValueRange) {
    torch::manual_seed(42);

    int64_t embed_dim = 32;
    torch::Tensor pos = torch::tensor({0.0f, 0.5f, 1.0f});

    torch::Tensor embed = make_sincos_pos_embed(embed_dim, pos);

    // sin/cos values should be in [-1, 1]
    EXPECT_TRUE((embed >= -1.0).all().item<bool>());
    EXPECT_TRUE((embed <= 1.0).all().item<bool>());
}

TEST(HeadsUtilsTest, PositionGridToEmbedBasic) {
    torch::manual_seed(42);

    // Position grid [H=2, W=3, 2] where 2 is (x, y)
    torch::Tensor pos_grid = torch::randn({2, 3, 2});
    int64_t embed_dim = 32;

    torch::Tensor embed = position_grid_to_embed(pos_grid, embed_dim);

    EXPECT_EQ(embed.dim(), 3);
    EXPECT_EQ(embed.size(0), 2);   // H
    EXPECT_EQ(embed.size(1), 3);   // W
    EXPECT_EQ(embed.size(2), embed_dim);  // embed_dim
}

TEST(HeadsUtilsTest, PositionGridToEmbedWrongDim) {
    torch::manual_seed(42);

    // Position grid must have last dim = 2
    torch::Tensor pos_grid = torch::randn({2, 3, 3});  // wrong dim
    int64_t embed_dim = 32;

    EXPECT_THROW(position_grid_to_embed(pos_grid, embed_dim), c10::Error);
}

TEST(HeadsUtilsTest, CreateUVGridBasic) {
    torch::manual_seed(42);

    int64_t width = 8;
    int64_t height = 8;

    torch::Tensor uv_grid = create_uv_grid(width, height);

    EXPECT_EQ(uv_grid.dim(), 3);  // [H, W, 2]
    EXPECT_EQ(uv_grid.size(0), height);
    EXPECT_EQ(uv_grid.size(1), width);
    EXPECT_EQ(uv_grid.size(2), 2);  // (u, v)
}

TEST(HeadsUtilsTest, CreateUVGridNonSquare) {
    torch::manual_seed(42);

    int64_t width = 16;
    int64_t height = 8;

    torch::Tensor uv_grid = create_uv_grid(width, height);

    EXPECT_EQ(uv_grid.size(0), height);
    EXPECT_EQ(uv_grid.size(1), width);
}

TEST(HeadsUtilsTest, CreateUVGridWithAspectRatio) {
    torch::manual_seed(42);

    int64_t width = 16;
    int64_t height = 8;
    float aspect_ratio = 2.0f;

    torch::Tensor uv_grid = create_uv_grid(width, height, aspect_ratio);

    EXPECT_EQ(uv_grid.sizes(), torch::IntArrayRef({height, width, 2}));
}

TEST(HeadsUtilsTest, CreateUVGridAutoAspectRatio) {
    torch::manual_seed(42);

    int64_t width = 16;
    int64_t height = 8;

    torch::Tensor uv_grid = create_uv_grid(width, height, -1.0f);

    // Should compute aspect_ratio = width/height = 2
    EXPECT_EQ(uv_grid.sizes(), torch::IntArrayRef({height, width, 2}));
}

TEST(HeadsUtilsTest, CreateUVGridValueRange) {
    torch::manual_seed(42);

    int64_t width = 8;
    int64_t height = 8;

    torch::Tensor uv_grid = create_uv_grid(width, height);

    // UV values should be in reasonable range
    EXPECT_TRUE(torch::isfinite(uv_grid).all().item<bool>());
}

TEST(HeadsUtilsTest, CreateUVGridDevice) {
    torch::manual_seed(42);

    int64_t width = 4;
    int64_t height = 4;

    // Create on CPU (default)
    torch::Tensor uv_grid = create_uv_grid(width, height);

    EXPECT_EQ(uv_grid.device().type(), torch::kCPU);
}

TEST(HeadsUtilsTest, PositionGridToEmbedLarge) {
    torch::manual_seed(42);

    // Test with larger grid
    torch::Tensor pos_grid = torch::randn({16, 16, 2});
    int64_t embed_dim = 128;

    torch::Tensor embed = position_grid_to_embed(pos_grid, embed_dim);

    EXPECT_EQ(embed.sizes(), torch::IntArrayRef({16, 16, 128}));
}

} // namespace
} // namespace heads
} // namespace vggt
