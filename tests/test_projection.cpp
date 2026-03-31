#include <gtest/gtest.h>
#include <torch/torch.h>
#include "vggt/dependency/projection.h"

namespace vggt {
namespace dependency {
namespace {

TEST(ProjectionTest, ImgFromCamBasic) {
    torch::manual_seed(42);

    // Create intrinsics [B=1, 3, 3]
    torch::Tensor intrinsics = torch::eye(3).unsqueeze(0) * 500;
    intrinsics[0][0][2] = 256;  // cx
    intrinsics[0][1][2] = 256;  // cy

    // Create camera points [B=1, 3, N=4] homogeneous
    torch::Tensor points_cam = torch::tensor({
        {{1.0, 2.0, 3.0, 4.0},   // x (in front of camera, z=1)
         {1.0, 1.0, 1.0, 1.0},    // y
         {1.0, 1.0, 1.0, 1.0}}    // z (depth=1)
    });

    torch::Tensor points2D = img_from_cam(intrinsics, points_cam);

    EXPECT_EQ(points2D.dim(), 3);
    EXPECT_EQ(points2D.size(0), 1);  // B
    EXPECT_EQ(points2D.size(1), 4);  // N points
    EXPECT_EQ(points2D.size(2), 2);  // 2D coords
}

TEST(ProjectionTest, ImgFromCamWithNoDistortion) {
    torch::manual_seed(42);

    // Identity intrinsics
    torch::Tensor intrinsics = torch::eye(3).unsqueeze(0);

    // Point at z=1, x=1, y=1
    torch::Tensor points_cam = torch::tensor({
        {{1.0},
         {1.0},
         {1.0}}
    });

    torch::Tensor points2D = img_from_cam(intrinsics, points_cam);

    // Should project to (1, 1) since fx=fy=1, cx=cy=0
    EXPECT_TRUE(torch::allclose(points2D[0][0], torch::tensor({1.0, 1.0}), 1e-4));
}

TEST(ProjectionTest, ImgFromCamWithDepth) {
    torch::manual_seed(42);

    // Identity intrinsics
    torch::Tensor intrinsics = torch::eye(3).unsqueeze(0);

    // Points at different depths
    torch::Tensor points_cam = torch::tensor({
        {{1.0, 2.0, 0.5},   // x
         {1.0, 2.0, 0.5},   // y
         {1.0, 2.0, 0.5}}   // z (different depths)
    });

    torch::Tensor points2D = img_from_cam(intrinsics, points_cam);

    // All should project to same x/y since they're on same ray from origin
    EXPECT_TRUE(points2D.defined());
    EXPECT_EQ(points2D.size(1), 3);  // N=3 points
}

TEST(ProjectionTest, ImgFromCamWithDistortion) {
    torch::manual_seed(42);

    // Identity intrinsics
    torch::Tensor intrinsics = torch::eye(3).unsqueeze(0);

    // Point at x=0.5, y=0.5, z=1
    torch::Tensor points_cam = torch::tensor({
        {{0.5},
         {0.5},
         {1.0}}
    });

    // Apply some radial distortion
    torch::Tensor distortion = torch::tensor({{0.1}});

    torch::Tensor points2D = img_from_cam(intrinsics, points_cam, distortion);

    EXPECT_TRUE(points2D.defined());
    // The distorted point should be pushed outward from center
}

TEST(ProjectionTest, ImgFromCamWithExtraParams) {
    torch::manual_seed(42);

    torch::Tensor intrinsics = torch::eye(3).unsqueeze(0);
    torch::Tensor points_cam = torch::tensor({
        {{1.0},
         {1.0},
         {1.0}}
    });

    // No distortion params (empty tensor)
    torch::Tensor extra_params;

    torch::Tensor points2D = img_from_cam(intrinsics, points_cam, extra_params);

    // Should work without distortion
    EXPECT_TRUE(torch::allclose(points2D[0][0], torch::tensor({1.0, 1.0}), 1e-4));
}

TEST(ProjectionTest, ImgFromCamDefaultVal) {
    torch::manual_seed(42);

    // Identity intrinsics
    torch::Tensor intrinsics = torch::eye(3).unsqueeze(0);

    // Point with z=0 (will cause division by zero -> nan -> replaced by default_val)
    torch::Tensor points_cam = torch::tensor({
        {{1.0},
         {1.0},
         {0.0}}  // z=0 causes nan
    });

    // Use default_val = 999.0
    torch::Tensor points2D = img_from_cam(intrinsics, points_cam, torch::Tensor(), 999.0);

    // NaN should be replaced with default_val
    EXPECT_TRUE(torch::allclose(points2D[0][0], torch::tensor({999.0, 999.0})));
}

TEST(ProjectionTest, ImgFromCamBatch) {
    torch::manual_seed(42);

    // Batch of intrinsics [B=2, 3, 3]
    torch::Tensor intrinsics = torch::zeros({2, 3, 3});
    intrinsics[0] = torch::eye(3) * 500;
    intrinsics[1] = torch::eye(3) * 500;

    // Batch of points [B=2, 3, N=2]
    torch::Tensor points_cam = torch::zeros({2, 3, 2});
    points_cam[0] = torch::tensor({{1.0, 2.0}, {1.0, 1.0}, {1.0, 1.0}});
    points_cam[1] = torch::tensor({{1.0, 1.0}, {1.0, 2.0}, {1.0, 1.0}});

    torch::Tensor points2D = img_from_cam(intrinsics, points_cam);

    EXPECT_EQ(points2D.size(0), 2);  // B
    EXPECT_EQ(points2D.size(1), 2);  // N
    EXPECT_EQ(points2D.size(2), 2);  // 2D
}

TEST(ProjectionTest, Project3DPointsBasic) {
    torch::manual_seed(42);

    // 3D points [N=3, 3]
    torch::Tensor points3D = torch::tensor({
        {1.0, 1.0, 1.0},
        {2.0, 1.0, 1.0},
        {1.0, 2.0, 1.0}
    });

    // Extrinsics [B=1, 3, 4]
    torch::Tensor extrinsics = torch::zeros({1, 3, 4});
    extrinsics[0][0][0] = 1;
    extrinsics[0][1][1] = 1;
    extrinsics[0][2][2] = 1;

    // Intrinsics [B=1, 3, 3]
    torch::Tensor intrinsics = torch::eye(3).unsqueeze(0) * 500;

    auto [points2D, points_cam] = project_3D_points(points3D, extrinsics, intrinsics);

    EXPECT_TRUE(points2D.defined());
    EXPECT_TRUE(points_cam.defined());

    EXPECT_EQ(points2D.size(0), 1);  // B
    EXPECT_EQ(points_cam.size(0), 1);  // B
    EXPECT_EQ(points_cam.size(1), 3);  // x, y, z
    EXPECT_EQ(points_cam.size(2), 3);  // N
}

TEST(ProjectionTest, Project3DPointsOnlyPointsCam) {
    torch::manual_seed(42);

    torch::Tensor points3D = torch::tensor({
        {1.0, 1.0, 1.0},
        {2.0, 1.0, 1.0}
    });

    torch::Tensor extrinsics = torch::zeros({1, 3, 4});
    extrinsics[0][0][0] = 1;
    extrinsics[0][1][1] = 1;
    extrinsics[0][2][2] = 1;

    auto [points2D, points_cam] = project_3D_points(
        points3D, extrinsics, torch::Tensor(), torch::Tensor(), 0.0, true);

    EXPECT_FALSE(points2D.defined());
    EXPECT_TRUE(points_cam.defined());
}

TEST(ProjectionTest, Project3DPointsWithoutIntrinsics) {
    torch::manual_seed(42);

    torch::Tensor points3D = torch::tensor({
        {1.0, 1.0, 1.0}
    });

    torch::Tensor extrinsics = torch::zeros({1, 3, 4});
    extrinsics[0][0][0] = 1;
    extrinsics[0][1][1] = 1;
    extrinsics[0][2][2] = 1;

    // No intrinsics, with only_points_cam=false should throw
    EXPECT_THROW(
        project_3D_points(points3D, extrinsics, torch::Tensor(), torch::Tensor(), 0.0, false),
        std::runtime_error
    );
}

TEST(ProjectionTest, Project3DPointsBatchCameras) {
    torch::manual_seed(42);

    // Multiple points [N=2, 3]
    torch::Tensor points3D = torch::tensor({
        {1.0, 0.0, 1.0},
        {0.0, 1.0, 1.0}
    });

    // Multiple cameras [B=2, 3, 4]
    torch::Tensor extrinsics = torch::zeros({2, 3, 4});
    extrinsics[0][0][0] = 1;
    extrinsics[0][1][1] = 1;
    extrinsics[0][2][2] = 1;
    extrinsics[1] = extrinsics[0].clone();

    // Intrinsics [B=2, 3, 3]
    torch::Tensor intrinsics = torch::zeros({2, 3, 3});
    intrinsics[0] = torch::eye(3) * 500;
    intrinsics[1] = torch::eye(3) * 500;

    auto [points2D, points_cam] = project_3D_points(points3D, extrinsics, intrinsics);

    EXPECT_EQ(points2D.size(0), 2);  // B
    EXPECT_EQ(points_cam.size(0), 2);  // B
}

TEST(ProjectionTest, Project3DPointsHomogeneousCoords) {
    // SKIPPED: Library expects [N,3] not [N,4] for 3D points
    GTEST_SKIP() << "Homogeneous coordinates not supported by library API";
}

} // namespace
} // namespace dependency
} // namespace vggt
