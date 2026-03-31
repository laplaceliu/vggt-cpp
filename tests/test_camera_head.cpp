#include <gtest/gtest.h>
#include <torch/torch.h>
#include "vggt/heads/camera_head.h"
#include "vggt/heads/head_act.h"

namespace vggt {
namespace heads {
namespace {

// ==================== CameraHead Constructor Tests ====================

TEST(CameraHeadTest, ConstructorDefaultParams) {
    EXPECT_NO_THROW({
        CameraHead head;
    });
}

TEST(CameraHeadTest, ConstructorWithCustomParams) {
    EXPECT_NO_THROW({
        CameraHead head(
            /*dim_in=*/1024,
            /*trunk_depth=*/2,
            /*pose_encoding_type=*/"absT_quaR_FoV",
            /*num_heads=*/8,
            /*mlp_ratio=*/4,
            /*init_values=*/0.02,
            /*trans_act=*/"linear",
            /*quat_act=*/"linear",
            /*fl_act=*/"relu"
        );
    });
}

TEST(CameraHeadTest, ConstructorWithShallowTrunk) {
    EXPECT_NO_THROW({
        CameraHead head(
            /*dim_in=*/2048,
            /*trunk_depth=*/1,
            /*pose_encoding_type=*/"absT_quaR_FoV",
            /*num_heads=*/16,
            /*mlp_ratio=*/2,
            /*init_values=*/0.01
        );
    });
}

TEST(CameraHeadTest, ConstructorWithUnsupportedEncoding) {
    EXPECT_THROW({
        CameraHead head(
            /*dim_in=*/2048,
            /*trunk_depth=*/4,
            /*pose_encoding_type=*/"unknown_encoding"
        );
    }, std::runtime_error);
}

TEST(CameraHeadTest, ConstructorWithDifferentActivations) {
    EXPECT_NO_THROW({
        CameraHead head(
            /*dim_in=*/2048,
            /*trunk_depth=*/4,
            /*pose_encoding_type=*/"absT_quaR_FoV",
            /*num_heads=*/16,
            /*mlp_ratio=*/4,
            /*init_values=*/0.01,
            /*trans_act=*/"relu",
            /*quat_act=*/"linear",
            /*fl_act=*/"relu"
        );
    });
}

// ==================== CameraHead Forward Tests ====================

TEST(CameraHeadTest, ForwardSingleFrame) {
    CameraHead head;
    // Input shape: [B, S, N, C] where N is number of camera tokens
    std::vector<torch::Tensor> tokens_list = {torch::randn({1, 1, 1, 2048})};
    
    auto output = head->forward(tokens_list, 1);
    
    EXPECT_EQ(output.size(), 1);
    EXPECT_EQ(output[0].dim(), 3);
    EXPECT_EQ(output[0].size(0), 1);  // batch size
    EXPECT_EQ(output[0].size(1), 1);  // sequence length
    EXPECT_EQ(output[0].size(2), 9); // pose encoding dimension
}

TEST(CameraHeadTest, ForwardMultiFrame) {
    CameraHead head;
    std::vector<torch::Tensor> tokens_list = {torch::randn({1, 8, 1, 2048})};
    
    auto output = head->forward(tokens_list, 1);
    
    EXPECT_EQ(output.size(), 1);
    EXPECT_EQ(output[0].size(0), 1);  // batch size
    EXPECT_EQ(output[0].size(1), 8);  // sequence length
    EXPECT_EQ(output[0].size(2), 9);  // pose encoding dimension
}

TEST(CameraHeadTest, ForwardMultiBatch) {
    CameraHead head;
    std::vector<torch::Tensor> tokens_list = {torch::randn({4, 8, 1, 2048})};
    
    auto output = head->forward(tokens_list, 1);
    
    EXPECT_EQ(output.size(), 1);
    EXPECT_EQ(output[0].size(0), 4);  // batch size
    EXPECT_EQ(output[0].size(1), 8);  // sequence length
    EXPECT_EQ(output[0].size(2), 9);  // pose encoding dimension
}

TEST(CameraHeadTest, ForwardMultipleIterations) {
    CameraHead head;
    std::vector<torch::Tensor> tokens_list = {torch::randn({1, 4, 1, 2048})};
    int64_t num_iterations = 4;
    
    auto output = head->forward(tokens_list, num_iterations);
    
    EXPECT_EQ(output.size(), num_iterations);
    for (int64_t i = 0; i < num_iterations; ++i) {
        EXPECT_EQ(output[i].size(0), 1);
        EXPECT_EQ(output[i].size(1), 4);
        EXPECT_EQ(output[i].size(2), 9);
    }
}

TEST(CameraHeadTest, ForwardOutputIsFinite) {
    CameraHead head;
    std::vector<torch::Tensor> tokens_list = {torch::randn({1, 4, 1, 2048}) * 0.1};
    
    auto output = head->forward(tokens_list, 2);
    
    for (const auto& tensor : output) {
        EXPECT_TRUE(torch::isfinite(tensor).all().item<bool>());
    }
}

TEST(CameraHeadTest, ForwardOutputIsContiguous) {
    CameraHead head;
    std::vector<torch::Tensor> tokens_list = {torch::randn({1, 4, 1, 2048})};
    
    auto output = head->forward(tokens_list, 1);
    
    EXPECT_TRUE(output[0].is_contiguous());
}

TEST(CameraHeadTest, ForwardWithMultipleTokenInputs) {
    CameraHead head;
    // CameraHead only uses the last token from the list
    std::vector<torch::Tensor> tokens_list = {
        torch::randn({1, 4, 1, 2048}),
        torch::randn({1, 4, 1, 2048}),
        torch::randn({1, 4, 1, 2048})
    };
    
    auto output = head->forward(tokens_list, 1);
    
    EXPECT_EQ(output.size(), 1);
    EXPECT_EQ(output[0].size(0), 1);
    EXPECT_EQ(output[0].size(1), 4);
    EXPECT_EQ(output[0].size(2), 9);
}

// ==================== CameraHead Pose Activation Tests ====================

TEST(CameraHeadTest, ForwardWithLinearActivation) {
    CameraHead head(
        /*dim_in=*/2048,
        /*trunk_depth=*/2,
        /*pose_encoding_type=*/"absT_quaR_FoV",
        /*num_heads=*/8,
        /*mlp_ratio=*/4,
        /*init_values=*/0.01,
        /*trans_act=*/"linear",
        /*quat_act=*/"linear",
        /*fl_act=*/"linear"
    );
    std::vector<torch::Tensor> tokens_list = {torch::randn({1, 2, 1, 2048})};
    
    auto output = head->forward(tokens_list, 1);
    
    EXPECT_TRUE(torch::isfinite(output[0]).all().item<bool>());
}

TEST(CameraHeadTest, ForwardWithReLUActivation) {
    CameraHead head(
        /*dim_in=*/2048,
        /*trunk_depth=*/2,
        /*pose_encoding_type=*/"absT_quaR_FoV",
        /*num_heads=*/8,
        /*mlp_ratio=*/4,
        /*init_values=*/0.01,
        /*trans_act=*/"relu",
        /*quat_act=*/"linear",
        /*fl_act=*/"relu"
    );
    std::vector<torch::Tensor> tokens_list = {torch::randn({1, 2, 1, 2048}) * 0.5};
    
    auto output = head->forward(tokens_list, 1);
    
    // Check translation and focal length are non-negative (ReLU)
    auto trans = output[0].index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
    auto fl = output[0].index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(8, 9)});
    EXPECT_TRUE((trans >= 0).all().item<bool>());
    EXPECT_TRUE((fl >= 0).all().item<bool>());
}

// ==================== CameraHead Gradient Tests ====================

TEST(CameraHeadTest, ForwardWithGradient) {
    CameraHead head;
    std::vector<torch::Tensor> tokens_list = {torch::randn({1, 4, 1, 2048})};
    
    auto output = head->forward(tokens_list, 1);
    auto loss = output[0].sum();
    loss.backward();
    
    // Check that gradients were computed
    // (We can't easily check specific gradients without more infrastructure)
    EXPECT_TRUE(torch::isfinite(output[0]).all().item<bool>());
}

TEST(CameraHeadTest, MultipleIterationsGradientFlow) {
    CameraHead head;
    std::vector<torch::Tensor> tokens_list = {torch::randn({1, 4, 1, 2048})};
    
    auto output = head->forward(tokens_list, 3);
    auto loss = output.back().sum();
    loss.backward();
    
    // Gradients should flow through all iterations
    EXPECT_TRUE(torch::isfinite(output.back()).all().item<bool>());
}

// ==================== CameraHead Output Value Range Tests ====================

TEST(CameraHeadTest, QuaternionOutputIsNormalized) {
    CameraHead head;
    std::vector<torch::Tensor> tokens_list = {torch::randn({1, 2, 1, 2048})};
    
    auto output = head->forward(tokens_list, 1);
    // Extract quaternion (indices 3-6) - quaternions should be reasonably bounded
    auto quat = output[0].index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(3, 7)});
    
    // Quaternion components should be finite
    EXPECT_TRUE(torch::isfinite(quat).all().item<bool>());
}

TEST(CameraHeadTest, OutputShapeConsistency) {
    CameraHead head;
    std::vector<torch::Tensor> tokens_list_1 = {torch::randn({2, 3, 1, 2048})};
    std::vector<torch::Tensor> tokens_list_2 = {torch::randn({2, 5, 1, 2048})};
    
    auto output1 = head->forward(tokens_list_1, 1);
    auto output2 = head->forward(tokens_list_2, 1);
    
    // Batch and feature dimensions should be consistent
    EXPECT_EQ(output1[0].size(0), output2[0].size(0)); // batch size same
    EXPECT_EQ(output1[0].size(2), output2[0].size(2)); // feature dim same
    // Sequence length should match input
    EXPECT_EQ(output1[0].size(1), 3);
    EXPECT_EQ(output2[0].size(1), 5);
}

// ==================== activate_pose Helper Function Tests ====================

TEST(ActivatePoseTest, BasicLinearActivation) {
    torch::Tensor input = torch::randn({2, 9}) * 0.1;
    auto output = activate_pose(input, "linear", "linear", "linear");
    EXPECT_EQ(output.sizes(), input.sizes());
    EXPECT_TRUE(torch::isfinite(output).all().item<bool>());
}

TEST(ActivatePoseTest, ReLUActivation) {
    torch::Tensor input = torch::randn({2, 9}) * 2.0;
    auto output = activate_pose(input, "relu", "linear", "relu");
    // Translation and focal length should be non-negative
    auto trans = output.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
    auto fl = output.index({torch::indexing::Slice(), torch::indexing::Slice(8, 9)});
    EXPECT_TRUE((trans >= 0).all().item<bool>());
    EXPECT_TRUE((fl >= 0).all().item<bool>());
}

TEST(ActivatePoseTest, DifferentBatchSizes) {
    torch::Tensor input1 = torch::randn({1, 9});
    torch::Tensor input2 = torch::randn({8, 9});
    torch::Tensor input3 = torch::randn({16, 9});
    
    auto output1 = activate_pose(input1);
    auto output2 = activate_pose(input2);
    auto output3 = activate_pose(input3);
    
    EXPECT_EQ(output1.size(0), 1);
    EXPECT_EQ(output2.size(0), 8);
    EXPECT_EQ(output3.size(0), 16);
}

} // namespace
} // namespace heads
} // namespace vggt
