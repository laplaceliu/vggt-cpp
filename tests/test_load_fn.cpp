#include <gtest/gtest.h>
#include <torch/torch.h>
#include "vggt/utils/load_fn.h"
#include <fstream>

namespace vggt {
namespace utils {
namespace {

// Helper function to create a simple test image
void createTestImage(const std::string& path, int width, int height, bool withAlpha = false) {
    // Create a simple image using OpenCV would require cv::imwrite
    // For simplicity, we'll copy an existing test image
}

TEST(LoadFnTest, LoadAndPreprocessImagesSquareEmptyList) {
    std::vector<std::string> empty_list;
    
    EXPECT_THROW(
        load_and_preprocess_images_square(empty_list, 1024),
        std::invalid_argument
    );
}

TEST(LoadFnTest, LoadAndPreprocessImagesEmptyList) {
    std::vector<std::string> empty_list;
    
    EXPECT_THROW(
        load_and_preprocess_images(empty_list, "crop"),
        std::invalid_argument
    );
}

TEST(LoadFnTest, LoadAndPreprocessImagesInvalidMode) {
    std::vector<std::string> image_list = {"dummy.jpg"};
    
    EXPECT_THROW(
        load_and_preprocess_images(image_list, "invalid_mode"),
        std::invalid_argument
    );
}

TEST(LoadFnTest, LoadAndPreprocessImagesNonExistentFile) {
    std::vector<std::string> image_list = {"/non/existent/path/image.jpg"};
    
    EXPECT_THROW(
        load_and_preprocess_images(image_list, "crop"),
        std::runtime_error
    );
}

TEST(LoadFnTest, LoadAndPreprocessImagesSquareNonExistentFile) {
    std::vector<std::string> image_list = {"/non/existent/path/image.jpg"};
    
    EXPECT_THROW(
        load_and_preprocess_images_square(image_list, 1024),
        std::runtime_error
    );
}

TEST(LoadFnTest, LoadAndPreprocessImagesCropMode) {
    // Use existing test images from misc directory
    std::vector<std::string> image_list = {
        "../misc/leopard-shaped-stone-paperweight/01.jpg",
        "../misc/leopard-shaped-stone-paperweight/02.jpg"
    };
    
    torch::Tensor images = load_and_preprocess_images(image_list, "crop");
    
    EXPECT_EQ(images.dim(), 4);  // [N, C, H, W]
    EXPECT_EQ(images.size(0), 2);  // 2 images
    EXPECT_EQ(images.size(1), 3);  // 3 channels (RGB)
    // Height and width should be processed according to crop mode
    EXPECT_EQ(images.size(3), 518);  // width is fixed to 518 in crop mode
}

TEST(LoadFnTest, LoadAndPreprocessImagesPadMode) {
    std::vector<std::string> image_list = {
        "../misc/leopard-shaped-stone-paperweight/01.jpg"
    };
    
    torch::Tensor images = load_and_preprocess_images(image_list, "pad");
    
    EXPECT_EQ(images.dim(), 4);  // [N, C, H, W]
    EXPECT_EQ(images.size(0), 1);  // 1 image
    EXPECT_EQ(images.size(1), 3);  // 3 channels
}

TEST(LoadFnTest, LoadAndPreprocessImagesSingleImage) {
    std::vector<std::string> image_list = {
        "../misc/leopard-shaped-stone-paperweight/01.jpg"
    };
    
    torch::Tensor images = load_and_preprocess_images(image_list, "crop");
    
    EXPECT_EQ(images.dim(), 4);  // Should be [1, C, H, W], not [C, H, W]
    EXPECT_EQ(images.size(0), 1);
}

TEST(LoadFnTest, LoadAndPreprocessImagesSquareBasic) {
    std::vector<std::string> image_list = {
        "../misc/leopard-shaped-stone-paperweight/01.jpg",
        "../misc/leopard-shaped-stone-paperweight/02.jpg"
    };
    int target_size = 512;
    
    auto [images, coords] = load_and_preprocess_images_square(image_list, target_size);
    
    EXPECT_EQ(images.dim(), 4);  // [N, C, H, W]
    EXPECT_EQ(images.size(0), 2);  // 2 images
    EXPECT_EQ(images.size(1), 3);  // 3 channels
    EXPECT_EQ(images.size(2), target_size);  // target height
    EXPECT_EQ(images.size(3), target_size);  // target width
    
    // Check coordinates tensor
    EXPECT_EQ(coords.dim(), 2);  // [N, 6]
    EXPECT_EQ(coords.size(0), 2);  // 2 images
    EXPECT_EQ(coords.size(1), 6);  // x1, y1, x2, y2, orig_width, orig_height
}

TEST(LoadFnTest, LoadAndPreprocessImagesSquareSingleImage) {
    std::vector<std::string> image_list = {
        "../misc/leopard-shaped-stone-paperweight/01.jpg"
    };
    
    auto [images, coords] = load_and_preprocess_images_square(image_list, 512);
    
    EXPECT_EQ(images.dim(), 4);  // Should be [1, C, H, W]
    EXPECT_EQ(images.size(0), 1);
    EXPECT_EQ(coords.dim(), 2);  // Should be [1, 6]
    EXPECT_EQ(coords.size(0), 1);
}

TEST(LoadFnTest, LoadAndPreprocessImagesSquareDifferentTargetSizes) {
    std::vector<std::string> image_list = {
        "../misc/leopard-shaped-stone-paperweight/01.jpg"
    };
    
    auto [images_256, coords_256] = load_and_preprocess_images_square(image_list, 256);
    EXPECT_EQ(images_256.size(2), 256);
    EXPECT_EQ(images_256.size(3), 256);
    
    auto [images_1024, coords_1024] = load_and_preprocess_images_square(image_list, 1024);
    EXPECT_EQ(images_1024.size(2), 1024);
    EXPECT_EQ(images_1024.size(3), 1024);
}

TEST(LoadFnTest, LoadAndPreprocessImagesValuesInRange) {
    std::vector<std::string> image_list = {
        "../misc/leopard-shaped-stone-paperweight/01.jpg"
    };
    
    torch::Tensor images = load_and_preprocess_images(image_list, "crop");
    
    // Values should be in [0, 1] after normalization
    EXPECT_TRUE((images >= 0.0).all().item<bool>());
    EXPECT_TRUE((images <= 1.0).all().item<bool>());
}

TEST(LoadFnTest, LoadAndPreprocessImagesSquareValuesInRange) {
    std::vector<std::string> image_list = {
        "../misc/leopard-shaped-stone-paperweight/01.jpg"
    };
    
    auto [images, coords] = load_and_preprocess_images_square(image_list, 512);
    
    // Values should be in [0, 1] after normalization
    EXPECT_TRUE((images >= 0.0).all().item<bool>());
    EXPECT_TRUE((images <= 1.0).all().item<bool>());
}

} // namespace
} // namespace utils
} // namespace vggt
