#include <gtest/gtest.h>
#include <torch/torch.h>
#include "vggt/utils/rotation.h"
#include "vggt/utils/pose_enc.h"

TEST(RotationTest, QuatMatConversion) {
    torch::manual_seed(42);
    
    // Create a valid rotation matrix [1, 3, 3]
    torch::Tensor R = torch::eye(3).unsqueeze(0);
    
    // Convert to quaternion
    torch::Tensor quat = vggt::utils::mat_to_quat(R);
    
    // Should be [1, 4]
    EXPECT_EQ(quat.dim(), 2);
    EXPECT_EQ(quat.size(-1), 4);
    
    // Convert back to matrix
    torch::Tensor R_back = vggt::utils::quat_to_mat(quat);
    
    // Should be close to original
    EXPECT_TRUE(torch::allclose(R, R_back.reshape({1, 3, 3}), 1e-4));
}

TEST(PoseEncTest, SimpleEncode) {
    torch::manual_seed(42);
    
    // Create identity extrinsics [1, 1, 3, 4]
    torch::Tensor ext = torch::zeros({1, 1, 3, 4});
    ext[0][0][0][0] = 1;
    ext[0][0][1][1] = 1;
    ext[0][0][2][2] = 1;
    
    // Create intrinsics [1, 1, 3, 3]
    torch::Tensor intri = torch::zeros({1, 1, 3, 3});
    intri[0][0][0][0] = 500;
    intri[0][0][1][1] = 500;
    intri[0][0][0][2] = 256;
    intri[0][0][1][2] = 256;
    intri[0][0][2][2] = 1;
    
    std::pair<int64_t, int64_t> img_size = {512, 512};
    
    torch::Tensor encoding = vggt::utils::extri_intri_to_pose_encoding(ext, intri, img_size);
    
    // Should be [1, 1, 9]
    EXPECT_EQ(encoding.size(0), 1);
    EXPECT_EQ(encoding.size(1), 1);
    EXPECT_EQ(encoding.size(2), 9);
}

TEST(PoseEncTest, SimpleDecode) {
    torch::manual_seed(42);
    
    // Create identity extrinsics [1, 1, 3, 4]
    torch::Tensor ext = torch::zeros({1, 1, 3, 4});
    ext[0][0][0][0] = 1;
    ext[0][0][1][1] = 1;
    ext[0][0][2][2] = 1;
    
    // Create intrinsics [1, 1, 3, 3]
    torch::Tensor intri = torch::zeros({1, 1, 3, 3});
    intri[0][0][0][0] = 500;
    intri[0][0][1][1] = 500;
    intri[0][0][0][2] = 256;
    intri[0][0][1][2] = 256;
    intri[0][0][2][2] = 1;
    
    std::pair<int64_t, int64_t> img_size = {512, 512};
    
    // Encode
    torch::Tensor encoding = vggt::utils::extri_intri_to_pose_encoding(ext, intri, img_size);
    
    // Decode
    auto [ext_dec, intri_dec] = vggt::utils::pose_encoding_to_extri_intri(encoding, img_size);
    
    EXPECT_TRUE(ext_dec.defined());
    EXPECT_TRUE(intri_dec.defined());
}
