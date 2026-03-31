#include <gtest/gtest.h>
#include <torch/torch.h>
#include "vggt/models/vggt.h"

namespace vggt {
namespace models {
namespace {

TEST(VGGTTest, ConstructorDefault) {
    torch::manual_seed(42);

    auto model = VGGT();

    EXPECT_TRUE(model);
}

TEST(VGGTTest, ConstructorWithCustomParams) {
    torch::manual_seed(42);

    auto model = VGGT(
        518,    // img_size
        14,     // patch_size
        1024    // embed_dim
    );

    EXPECT_TRUE(model);
}

TEST(VGGTTest, ConstructorWithSmallImage) {
    torch::manual_seed(42);

    auto model = VGGT(
        224,    // img_size
        14,     // patch_size
        768     // embed_dim
    );

    EXPECT_TRUE(model);
}

TEST(VGGTTest, ConstructorMinimal) {
    torch::manual_seed(42);

    auto model = VGGT(
        112,    // img_size
        14,     // patch_size
        256     // embed_dim
    );

    EXPECT_TRUE(model);
}

TEST(VGGTTest, ForwardBasic) {
    torch::manual_seed(42);

    auto model = VGGT(
        112,    // img_size
        14,     // patch_size
        256     // embed_dim
    );

    // Input: [S, 3, H, W] - will be expanded to [B, S, 3, H, W]
    torch::Tensor images = torch::rand({2, 3, 112, 112});

    auto predictions = model->forward(images);

    // Check that we have predictions
    EXPECT_TRUE(predictions.find("pose_enc") != predictions.end());
    EXPECT_TRUE(predictions.find("depth") != predictions.end());
    EXPECT_TRUE(predictions.find("depth_conf") != predictions.end());
    EXPECT_TRUE(predictions.find("world_points") != predictions.end());
    EXPECT_TRUE(predictions.find("world_points_conf") != predictions.end());
    EXPECT_TRUE(predictions.find("images") != predictions.end());
}

TEST(VGGTTest, ForwardWithBatch) {
    torch::manual_seed(42);

    auto model = VGGT(
        112,    // img_size
        14,     // patch_size
        256     // embed_dim
    );

    // Input: [B, S, 3, H, W]
    torch::Tensor images = torch::rand({1, 2, 3, 112, 112});

    auto predictions = model->forward(images);

    EXPECT_TRUE(predictions.find("pose_enc") != predictions.end());
    EXPECT_TRUE(predictions.find("depth") != predictions.end());
    EXPECT_TRUE(predictions.find("world_points") != predictions.end());
}

TEST(VGGTTest, ForwardWithQueryPoints) {
    torch::manual_seed(42);

    auto model = VGGT(
        224,    // img_size - larger to avoid pool2d size issues
        14,     // patch_size
        256     // embed_dim
    );

    torch::Tensor images = torch::rand({2, 3, 224, 224});
    // Query points: [N, 2] - normalized coordinates [0, 1]
    torch::Tensor query_points = torch::tensor({{0.5, 0.5}, {0.25, 0.75}});

    auto predictions = model->forward(images, query_points);

    // Check tracking outputs are present
    EXPECT_TRUE(predictions.find("track") != predictions.end());
    EXPECT_TRUE(predictions.find("vis") != predictions.end());
    EXPECT_TRUE(predictions.find("conf") != predictions.end());
    
    // Check track shape: [B, S, N, 2]
    auto track = predictions["track"];
    EXPECT_EQ(track.dim(), 4);
    EXPECT_EQ(track.size(0), 1);  // B
    EXPECT_EQ(track.size(1), 2);  // S
    EXPECT_EQ(track.size(2), 2);  // N
}

TEST(VGGTTest, ForwardOutputShapes) {
    torch::manual_seed(42);

    auto model = VGGT(
        112,    // img_size
        14,     // patch_size
        256     // embed_dim
    );

    int64_t B = 1;
    int64_t S = 2;
    int64_t H = 112;
    int64_t W = 112;

    torch::Tensor images = torch::rand({B, S, 3, H, W});

    auto predictions = model->forward(images);

    // Check output shapes
    auto pose_enc = predictions["pose_enc"];
    EXPECT_EQ(pose_enc.dim(), 3);
    EXPECT_EQ(pose_enc.size(0), B);
    EXPECT_EQ(pose_enc.size(1), S);
    EXPECT_EQ(pose_enc.size(2), 9);  // Camera pose encoding dimension

    auto depth = predictions["depth"];
    EXPECT_EQ(depth.dim(), 5);
    EXPECT_EQ(depth.size(0), B);
    EXPECT_EQ(depth.size(1), S);
    EXPECT_EQ(depth.size(4), 1);  // Single channel for depth

    auto world_points = predictions["world_points"];
    EXPECT_EQ(world_points.dim(), 5);
    EXPECT_EQ(world_points.size(0), B);
    EXPECT_EQ(world_points.size(1), S);
    EXPECT_EQ(world_points.size(4), 3);  // XYZ coordinates
}

TEST(VGGTTest, ForwardSingleFrame) {
    torch::manual_seed(42);

    auto model = VGGT(
        112,    // img_size
        14,     // patch_size
        256     // embed_dim
    );

    // Single frame: [1, 3, H, W]
    torch::Tensor images = torch::rand({1, 3, 112, 112});

    auto predictions = model->forward(images);

    EXPECT_TRUE(predictions.find("pose_enc") != predictions.end());
    EXPECT_EQ(predictions["pose_enc"].size(0), 1);  // B = 1
    EXPECT_EQ(predictions["pose_enc"].size(1), 1);  // S = 1
}

TEST(VGGTTest, ForwardMultipleFrames) {
    torch::manual_seed(42);

    auto model = VGGT(
        112,    // img_size
        14,     // patch_size
        256     // embed_dim
    );

    // Multiple frames: [S, 3, H, W]
    torch::Tensor images = torch::rand({5, 3, 112, 112});

    auto predictions = model->forward(images);

    EXPECT_EQ(predictions["pose_enc"].size(0), 1);  // B = 1
    EXPECT_EQ(predictions["pose_enc"].size(1), 5);  // S = 5
}

TEST(VGGTTest, ForwardPreservesGrad) {
    torch::manual_seed(42);

    auto model = VGGT(
        112,    // img_size
        14,     // patch_size
        128     // embed_dim (reduced for speed)
    );

    torch::Tensor images = torch::rand({1, 2, 3, 112, 112}, torch::requires_grad());

    auto predictions = model->forward(images);

    // Backward through pose_enc
    predictions["pose_enc"].sum().backward();

    EXPECT_TRUE(images.grad().defined());
}

TEST(VGGTTest, ForwardWithBatchQueryPoints) {
    torch::manual_seed(42);

    auto model = VGGT(
        224,    // img_size - larger to avoid pool2d issues
        14,     // patch_size
        256     // embed_dim
    );

    torch::Tensor images = torch::rand({1, 2, 3, 224, 224});
    // Query points with batch dimension: [B, N, 2]
    torch::Tensor query_points = torch::rand({1, 5, 2});

    auto predictions = model->forward(images, query_points);

    EXPECT_TRUE(predictions.find("track") != predictions.end());
    EXPECT_EQ(predictions["track"].size(0), 1);  // B
    EXPECT_EQ(predictions["track"].size(1), 2);  // S
    EXPECT_EQ(predictions["track"].size(2), 5);  // N
}

} // namespace
} // namespace models
} // namespace vggt
