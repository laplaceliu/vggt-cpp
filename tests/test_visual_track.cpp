#include <gtest/gtest.h>
#include <torch/torch.h>
#include "vggt/utils/visual_track.h"
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace vggt {
namespace utils {
namespace {

TEST(VisualTrackTest, ColorFromXYBasic) {
    auto [r, g, b] = color_from_xy(50, 50, 100, 100, "hsv");
    
    // Colors should be in [0,1]
    EXPECT_GE(r, 0.0);
    EXPECT_LE(r, 1.0);
    EXPECT_GE(g, 0.0);
    EXPECT_LE(g, 1.0);
    EXPECT_GE(b, 0.0);
    EXPECT_LE(b, 1.0);
}

TEST(VisualTrackTest, ColorFromXYCorner) {
    // Test top-left corner
    auto [r1, g1, b1] = color_from_xy(0, 0, 100, 100, "hsv");
    EXPECT_GE(r1, 0.0);
    EXPECT_LE(r1, 1.0);
    
    // Test bottom-right corner
    auto [r2, g2, b2] = color_from_xy(99, 99, 100, 100, "hsv");
    EXPECT_GE(r2, 0.0);
    EXPECT_LE(r2, 1.0);
}

TEST(VisualTrackTest, ColorFromXYDifferentCmaps) {
    // Test different colormap names
    auto [r1, g1, b1] = color_from_xy(50, 50, 100, 100, "hsv");
    auto [r2, g2, b2] = color_from_xy(50, 50, 100, 100, "jet");
    auto [r3, g3, b3] = color_from_xy(50, 50, 100, 100, "rainbow");
    
    // All should return valid colors
    EXPECT_GE(r1, 0.0); EXPECT_LE(r1, 1.0);
    EXPECT_GE(r2, 0.0); EXPECT_LE(r2, 1.0);
    EXPECT_GE(r3, 0.0); EXPECT_LE(r3, 1.0);
}

TEST(VisualTrackTest, GetTrackColorsByPositionBasic) {
    torch::Tensor tracks = torch::tensor({
        {{10.0, 10.0}, {50.0, 50.0}},
        {{15.0, 15.0}, {55.0, 55.0}}
    });  // [S=2, N=2, 2]
    
    cv::Mat colors = get_track_colors_by_position(tracks, torch::Tensor(), 100, 100, "hsv");
    
    EXPECT_EQ(colors.rows, 2);  // N tracks
    EXPECT_EQ(colors.cols, 3);  // RGB
    EXPECT_EQ(colors.type(), CV_8UC1);
}

TEST(VisualTrackTest, GetTrackColorsWithVisibilityMask) {
    torch::Tensor tracks = torch::tensor({
        {{10.0, 10.0}, {50.0, 50.0}},
        {{15.0, 15.0}, {55.0, 55.0}}
    });
    
    // Second track is not visible in frame 0
    torch::Tensor vis_mask = torch::tensor({
        {true, false},
        {true, true}
    }, torch::kBool);
    
    cv::Mat colors = get_track_colors_by_position(tracks, vis_mask, 100, 100, "hsv");
    
    EXPECT_EQ(colors.rows, 2);
    EXPECT_EQ(colors.cols, 3);
}

TEST(VisualTrackTest, GetTrackColorsAllInvisible) {
    torch::Tensor tracks = torch::tensor({
        {{10.0, 10.0}, {50.0, 50.0}},
        {{15.0, 15.0}, {55.0, 55.0}}
    });
    
    // All tracks invisible
    torch::Tensor vis_mask = torch::tensor({
        {false, false},
        {false, false}
    }, torch::kBool);
    
    cv::Mat colors = get_track_colors_by_position(tracks, vis_mask, 100, 100, "hsv");
    
    EXPECT_EQ(colors.rows, 2);
    
    // All should be black (0, 0, 0)
    for (int i = 0; i < colors.rows; ++i) {
        EXPECT_EQ(colors.at<uchar>(i, 0), 0);
        EXPECT_EQ(colors.at<uchar>(i, 1), 0);
        EXPECT_EQ(colors.at<uchar>(i, 2), 0);
    }
}

TEST(VisualTrackTest, GetTrackColorsSingleTrack) {
    torch::Tensor tracks = torch::tensor({
        {{50.0, 50.0}}
    });  // [S=1, N=1, 2]
    
    cv::Mat colors = get_track_colors_by_position(tracks, torch::Tensor(), 100, 100, "hsv");
    
    EXPECT_EQ(colors.rows, 1);
    EXPECT_EQ(colors.cols, 3);
}

TEST(VisualTrackTest, VisualizeTracksBasic) {
    // Create dummy images and tracks
    torch::Tensor images = torch::rand({4, 3, 64, 64});  // [S=4, C=3, H=64, W=64]
    torch::Tensor tracks = torch::tensor({
        {{10.0, 10.0}, {30.0, 30.0}},
        {{15.0, 15.0}, {35.0, 35.0}},
        {{20.0, 20.0}, {40.0, 40.0}},
        {{25.0, 25.0}, {45.0, 45.0}}
    });  // [S=4, N=2, 2]
    
    std::string out_dir = "/tmp/test_visual_track_basic";
    
    // Clean up if exists
    std::filesystem::remove_all(out_dir);
    
    visualize_tracks_on_images(images, tracks, torch::Tensor(), out_dir, "CHW", "[0,1]", "hsv", 2, true);
    
    // Check output files exist
    EXPECT_TRUE(std::filesystem::exists(out_dir));
    EXPECT_TRUE(std::filesystem::exists(out_dir + "/tracks_grid.png"));
    EXPECT_TRUE(std::filesystem::exists(out_dir + "/frame_0000.png"));
    
    // Cleanup
    std::filesystem::remove_all(out_dir);
}

TEST(VisualTrackTest, VisualizeTracksWithVisibilityMask) {
    torch::Tensor images = torch::rand({2, 3, 64, 64});
    torch::Tensor tracks = torch::tensor({
        {{10.0, 10.0}, {30.0, 30.0}},
        {{15.0, 15.0}, {35.0, 35.0}}
    });
    
    // First track visible in both frames, second only in frame 1
    torch::Tensor vis_mask = torch::tensor({
        {true, false},
        {true, true}
    }, torch::kBool);
    
    std::string out_dir = "/tmp/test_visual_track_vis";
    std::filesystem::remove_all(out_dir);
    
    visualize_tracks_on_images(images, tracks, vis_mask, out_dir, "CHW", "[0,1]", "hsv", 2, true);
    
    EXPECT_TRUE(std::filesystem::exists(out_dir + "/tracks_grid.png"));
    
    std::filesystem::remove_all(out_dir);
}

TEST(VisualTrackTest, VisualizeTracksNoGrid) {
    torch::Tensor images = torch::rand({2, 3, 64, 64});
    torch::Tensor tracks = torch::tensor({
        {{10.0, 10.0}},
        {{15.0, 15.0}}
    });
    
    std::string out_dir = "/tmp/test_visual_track_nogrid";
    std::filesystem::remove_all(out_dir);
    
    // save_grid = false
    visualize_tracks_on_images(images, tracks, torch::Tensor(), out_dir, "CHW", "[0,1]", "hsv", 2, false);
    
    // Only individual frames should exist
    EXPECT_FALSE(std::filesystem::exists(out_dir + "/tracks_grid.png"));
    EXPECT_TRUE(std::filesystem::exists(out_dir + "/frame_0000.png"));
    
    std::filesystem::remove_all(out_dir);
}

TEST(VisualTrackTest, VisualizeTracksWithBatchDim) {
    // Input with batch dimension [B=1, S=2, C, H, W]
    torch::Tensor images = torch::rand({1, 2, 3, 64, 64});
    torch::Tensor tracks = torch::rand({1, 2, 2, 2});  // [B=1, S=2, N=2, 2]
    
    std::string out_dir = "/tmp/test_visual_track_batch";
    std::filesystem::remove_all(out_dir);
    
    visualize_tracks_on_images(images, tracks, torch::Tensor(), out_dir, "CHW", "[0,1]", "hsv", 2, false);
    
    EXPECT_TRUE(std::filesystem::exists(out_dir + "/frame_0000.png"));
    
    std::filesystem::remove_all(out_dir);
}

TEST(VisualTrackTest, VisualizeTracksNormalizeModes) {
    torch::Tensor images = torch::rand({2, 3, 64, 64});
    torch::Tensor tracks = torch::tensor({
        {{10.0, 10.0}},
        {{15.0, 15.0}}
    });
    
    // Test [0,1] normalization
    std::string out_dir1 = "/tmp/test_visual_track_norm1";
    std::filesystem::remove_all(out_dir1);
    visualize_tracks_on_images(images, tracks, torch::Tensor(), out_dir1, "CHW", "[0,1]", "hsv", 2, false);
    EXPECT_TRUE(std::filesystem::exists(out_dir1 + "/frame_0000.png"));
    std::filesystem::remove_all(out_dir1);
    
    // Test [-1,1] normalization
    torch::Tensor images_norm = images * 2 - 1;  // Scale to [-1, 1]
    std::string out_dir2 = "/tmp/test_visual_track_norm2";
    std::filesystem::remove_all(out_dir2);
    visualize_tracks_on_images(images_norm, tracks, torch::Tensor(), out_dir2, "CHW", "[-1,1]", "hsv", 2, false);
    EXPECT_TRUE(std::filesystem::exists(out_dir2 + "/frame_0000.png"));
    std::filesystem::remove_all(out_dir2);
}

} // namespace
} // namespace utils
} // namespace vggt
