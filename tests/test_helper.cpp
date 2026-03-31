#include <gtest/gtest.h>
#include <torch/torch.h>
#include "vggt/utils/helper.h"

namespace vggt {
namespace utils {
namespace {

TEST(HelperTest, RandomlyLimitTruesWithinBudget) {
    torch::manual_seed(42);

    // Create a mask with 5 True values
    torch::Tensor mask = torch::tensor({true, false, true, false, true, false, true, false, true, false});
    int64_t max_trues = 10;

    torch::Tensor limited = randomly_limit_trues(mask, max_trues);

    // Should return the same mask since we're within budget
    EXPECT_EQ(limited.sizes(), mask.sizes());
    EXPECT_TRUE(torch::equal(limited, mask));
}

TEST(HelperTest, RandomlyLimitTruesExceedsBudget) {
    torch::manual_seed(42);

    // Create a mask with 5 True values
    torch::Tensor mask = torch::tensor({true, true, true, true, true, false, false, false, false, false});
    int64_t max_trues = 3;

    torch::Tensor limited = randomly_limit_trues(mask, max_trues);

    // Should reduce to exactly 3 True values
    EXPECT_EQ(limited.sizes(), mask.sizes());
    int64_t num_trues = limited.sum().item<int64_t>();
    EXPECT_EQ(num_trues, max_trues);
}

TEST(HelperTest, RandomlyLimitTruesAllFalse) {
    torch::manual_seed(42);

    // Create a mask with all False values
    torch::Tensor mask = torch::zeros({10}, torch::kBool);
    int64_t max_trues = 5;

    torch::Tensor limited = randomly_limit_trues(mask, max_trues);

    // Should remain all False
    EXPECT_EQ(limited.sizes(), mask.sizes());
    EXPECT_EQ(limited.sum().item<int64_t>(), 0);
}

TEST(HelperTest, RandomlyLimitTrues2DMask) {
    torch::manual_seed(42);

    // Create a 2D mask
    torch::Tensor mask = torch::tensor({
        {true, true, false},
        {true, false, true},
        {false, true, true}
    });
    int64_t max_trues = 4;

    torch::Tensor limited = randomly_limit_trues(mask, max_trues);

    // Should reduce to exactly 4 True values
    EXPECT_EQ(limited.sizes(), mask.sizes());
    int64_t num_trues = limited.sum().item<int64_t>();
    EXPECT_EQ(num_trues, max_trues);
}

TEST(HelperTest, RandomlyLimitTruesDeterministicWithSeed) {
    // Test that with the same seed, we get consistent results
    torch::manual_seed(123);

    torch::Tensor mask = torch::tensor({true, true, true, true, true, true, true, true, true, true});
    int64_t max_trues = 5;

    torch::Tensor limited1 = randomly_limit_trues(mask, max_trues);

    torch::manual_seed(123);
    torch::Tensor limited2 = randomly_limit_trues(mask, max_trues);

    // With same seed, should get the same result
    EXPECT_TRUE(torch::equal(limited1, limited2));
}

TEST(HelperTest, CreatePixelCoordinateGridBasic) {
    int64_t num_frames = 2;
    int64_t height = 3;
    int64_t width = 4;

    auto [points_xyf, y_coords, x_coords, f_coords] = create_pixel_coordinate_grid(num_frames, height, width);

    // Check shapes
    EXPECT_EQ(points_xyf.sizes(), std::vector<int64_t>({num_frames, height, width, 3}));
    EXPECT_EQ(y_coords.sizes(), std::vector<int64_t>({num_frames, height, width}));
    EXPECT_EQ(x_coords.sizes(), std::vector<int64_t>({num_frames, height, width}));
    EXPECT_EQ(f_coords.sizes(), std::vector<int64_t>({num_frames, height, width}));
}

TEST(HelperTest, CreatePixelCoordinateGridValues) {
    int64_t num_frames = 1;
    int64_t height = 2;
    int64_t width = 3;

    auto [points_xyf, y_coords, x_coords, f_coords] = create_pixel_coordinate_grid(num_frames, height, width);

    // Check x coordinates
    EXPECT_FLOAT_EQ(x_coords[0][0][0].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(x_coords[0][0][1].item<float>(), 1.0f);
    EXPECT_FLOAT_EQ(x_coords[0][0][2].item<float>(), 2.0f);

    // Check y coordinates
    EXPECT_FLOAT_EQ(y_coords[0][0][0].item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(y_coords[0][1][0].item<float>(), 1.0f);

    // Check frame indices
    EXPECT_FLOAT_EQ(f_coords[0][0][0].item<float>(), 0.0f);
}

TEST(HelperTest, CreatePixelCoordinateGridMultipleFrames) {
    int64_t num_frames = 3;
    int64_t height = 2;
    int64_t width = 2;

    auto [points_xyf, y_coords, x_coords, f_coords] = create_pixel_coordinate_grid(num_frames, height, width);

    // Check frame indices for each frame
    for (int64_t f = 0; f < num_frames; ++f) {
        EXPECT_FLOAT_EQ(f_coords[f][0][0].item<float>(), static_cast<float>(f));
        EXPECT_FLOAT_EQ(f_coords[f][1][1].item<float>(), static_cast<float>(f));
    }
}

TEST(HelperTest, CreatePixelCoordinateGridPointsXYF) {
    int64_t num_frames = 1;
    int64_t height = 2;
    int64_t width = 2;

    auto [points_xyf, y_coords, x_coords, f_coords] = create_pixel_coordinate_grid(num_frames, height, width);

    // Check points_xyf structure: [x, y, f] at each position
    // At position [0, 0]: x=0, y=0, f=0
    EXPECT_FLOAT_EQ(points_xyf[0][0][0][0].item<float>(), 0.0f);  // x
    EXPECT_FLOAT_EQ(points_xyf[0][0][0][1].item<float>(), 0.0f);  // y
    EXPECT_FLOAT_EQ(points_xyf[0][0][0][2].item<float>(), 0.0f);  // f

    // At position [0, 1]: x=1, y=0, f=0
    EXPECT_FLOAT_EQ(points_xyf[0][0][1][0].item<float>(), 1.0f);  // x
    EXPECT_FLOAT_EQ(points_xyf[0][0][1][1].item<float>(), 0.0f);  // y

    // At position [1, 0]: x=0, y=1, f=0
    EXPECT_FLOAT_EQ(points_xyf[0][1][0][0].item<float>(), 0.0f);  // x
    EXPECT_FLOAT_EQ(points_xyf[0][1][0][1].item<float>(), 1.0f);  // y
}

} // namespace
} // namespace utils
} // namespace vggt
