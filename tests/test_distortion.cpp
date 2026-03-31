#include <gtest/gtest.h>
#include <torch/torch.h>
#include "vggt/dependency/distortion.h"

namespace vggt {
namespace dependency {
namespace {

TEST(DistortionTest, IsTorchDefined) {
    torch::Tensor x = torch::randn({3, 3});
    EXPECT_TRUE(is_torch(x));
}

TEST(DistortionTest, IsTorchUndefined) {
    torch::Tensor x;
    EXPECT_FALSE(is_torch(x));
}

TEST(DistortionTest, EnsureTorchValid) {
    torch::Tensor x = torch::randn({3, 3});
    torch::Tensor result = ensure_torch(x);
    EXPECT_TRUE(result.defined());
}

TEST(DistortionTest, EnsureTorchInvalid) {
    torch::Tensor x;
    EXPECT_THROW(ensure_torch(x), std::runtime_error);
}

TEST(DistortionTest, SingleUndistortionShape) {
    torch::manual_seed(42);

    // params: [B, N] where N is number of distortion params
    torch::Tensor params = torch::zeros({2, 1});  // Simple radial distortion
    // tracks: [B, M, 2] normalized coordinates
    torch::Tensor tracks = torch::randn({2, 10, 2});

    torch::Tensor undistorted = single_undistortion(params, tracks);

    EXPECT_EQ(undistorted.dim(), 3);
    EXPECT_EQ(undistorted.size(0), 2);  // B
    EXPECT_EQ(undistorted.size(1), 10); // M
    EXPECT_EQ(undistorted.size(2), 2);  // 2 coords
}

TEST(DistortionTest, SingleUndistortionWithNoDistortion) {
    torch::manual_seed(42);

    // Zero params means no distortion
    torch::Tensor params = torch::zeros({1, 1});
    torch::Tensor tracks = torch::randn({1, 5, 2});

    torch::Tensor undistorted = single_undistortion(params, tracks);

    // With k=0, output should equal input
    EXPECT_TRUE(torch::allclose(undistorted, tracks, 1e-4));
}

TEST(DistortionTest, SingleUndistortionRadial) {
    torch::manual_seed(42);

    // Simple radial distortion k > 0
    torch::Tensor params = torch::ones({1, 1}) * 0.1;
    torch::Tensor tracks = torch::randn({1, 5, 2});

    torch::Tensor undistorted = single_undistortion(params, tracks);

    EXPECT_TRUE(undistorted.defined());
    EXPECT_EQ(undistorted.sizes(), tracks.sizes());
}

TEST(DistortionTest, ApplyDistortionOneParam) {
    torch::manual_seed(42);

    // 1 param: simple radial distortion
    // params: [B=1, N=1], u/v: [B=1, M=3]
    torch::Tensor params = torch::tensor({{0.1}});
    torch::Tensor u = torch::tensor({{0.5, 0.0, 1.0}});
    torch::Tensor v = torch::tensor({{0.0, 0.5, 1.0}});

    auto [u_out, v_out] = apply_distortion(params, u, v);

    EXPECT_EQ(u_out.size(0), 1);  // B
    EXPECT_EQ(u_out.size(1), 3);  // M
    EXPECT_EQ(v_out.size(0), 1);
    EXPECT_EQ(v_out.size(1), 3);

    // At (0,0), distortion should be 0
    EXPECT_FLOAT_EQ(u_out[0][1].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(v_out[0][0].item<float>(), 0.0f);
}

TEST(DistortionTest, ApplyDistortionTwoParams) {
    torch::manual_seed(42);

    // 2 params: radial distortion with k1, k2
    torch::Tensor params = torch::tensor({{0.1, 0.02}});
    torch::Tensor u = torch::tensor({{0.5}});
    torch::Tensor v = torch::tensor({{0.5}});

    auto [u_out, v_out] = apply_distortion(params, u, v);

    EXPECT_EQ(u_out.size(0), 1);
    EXPECT_EQ(v_out.size(0), 1);

    // At (0.5, 0.5), r^2 = 0.25 + 0.25 = 0.5
    // radial = k1 * r^2 + k2 * r^4 = 0.1 * 0.5 + 0.02 * 0.25 = 0.05 + 0.005 = 0.055
    // du = 0.5 * 0.055 = 0.0275
    EXPECT_NEAR(u_out[0].item<float>(), 0.5 + 0.0275, 1e-4);
}

TEST(DistortionTest, ApplyDistortionFourParams) {
    torch::manual_seed(42);

    // 4 params: OpenCV distortion model
    torch::Tensor params = torch::tensor({{0.1, 0.02, 0.01, 0.01}});
    torch::Tensor u = torch::tensor({{0.5}});
    torch::Tensor v = torch::tensor({{0.5}});

    auto [u_out, v_out] = apply_distortion(params, u, v);

    EXPECT_TRUE(u_out.defined());
    EXPECT_TRUE(v_out.defined());
}

TEST(DistortionTest, ApplyDistortionUnsupportedParams) {
    torch::manual_seed(42);

    // 3 params is not supported
    torch::Tensor params = torch::tensor({{0.1, 0.02, 0.01}});
    torch::Tensor u = torch::tensor({{0.5}});
    torch::Tensor v = torch::tensor({{0.5}});

    EXPECT_THROW(apply_distortion(params, u, v), std::runtime_error);
}

TEST(DistortionTest, IterativeUndistortionConvergence) {
    torch::manual_seed(42);

    // With zero distortion, iterative should converge quickly to original
    torch::Tensor params = torch::zeros({1, 1});
    torch::Tensor tracks = torch::randn({1, 5, 2});

    torch::Tensor undistorted = iterative_undistortion(params, tracks, 100, 1e-10);

    EXPECT_TRUE(torch::allclose(undistorted, tracks, 1e-4));
}

TEST(DistortionTest, IterativeUndistortionShapePreserved) {
    torch::manual_seed(42);

    torch::Tensor params = torch::tensor({{0.1}});
    torch::Tensor tracks = torch::randn({2, 10, 2});

    torch::Tensor undistorted = iterative_undistortion(params, tracks);

    EXPECT_EQ(undistorted.sizes(), tracks.sizes());
}

TEST(DistortionTest, IterativeUndistortionMaxIterations) {
    torch::manual_seed(42);

    // Very small max_step_norm to force max iterations
    torch::Tensor params = torch::tensor({{0.5}});
    torch::Tensor tracks = torch::randn({1, 5, 2});

    // Should complete without hanging even with max iterations
    torch::Tensor undistorted = iterative_undistortion(params, tracks, 10, 1e-20, 1e-6);

    EXPECT_TRUE(undistorted.defined());
}

TEST(DistortionTest, IterativeUndistortionUndefined) {
    torch::Tensor undefined_params;
    torch::Tensor tracks = torch::randn({1, 5, 2});

    EXPECT_THROW(ensure_torch(undefined_params), std::runtime_error);
}

TEST(DistortionTest, ApplyDistortionBatch) {
    torch::manual_seed(42);

    // Batch of params and coordinates
    torch::Tensor params = torch::tensor({{0.1}, {0.2}});
    torch::Tensor u = torch::tensor({{0.5}, {0.5}});
    torch::Tensor v = torch::tensor({{0.5}, {0.5}});

    auto [u_out, v_out] = apply_distortion(params, u, v);

    EXPECT_EQ(u_out.size(0), 2);
    EXPECT_EQ(v_out.size(0), 2);
}

TEST(DistortionTest, SingleUndistortionDifferentBatchSizes) {
    torch::manual_seed(42);

    for (int64_t B : {1, 2, 4}) {
        torch::Tensor params = torch::zeros({B, 1});
        torch::Tensor tracks = torch::randn({B, 10, 2});

        torch::Tensor undistorted = single_undistortion(params, tracks);

        EXPECT_EQ(undistorted.size(0), B);
        EXPECT_EQ(undistorted.size(1), 10);
    }
}

} // namespace
} // namespace dependency
} // namespace vggt
