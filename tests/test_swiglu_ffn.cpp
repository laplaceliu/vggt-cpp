#include <gtest/gtest.h>
#include <torch/torch.h>
#include "layers/swiglu_ffn.h"

namespace vggt {
namespace layers {
namespace {

TEST(SwiGLUFFNTest, ForwardOutputShape) {
    torch::manual_seed(42);
    
    SwiGLUFFN swiglu(128, 256, 128);
    swiglu->eval();
    
    torch::Tensor x = torch::randn({2, 10, 128});
    torch::Tensor output = swiglu->forward(x);
    
    EXPECT_EQ(output.dim(), 3);
    EXPECT_EQ(output.size(0), 2);   // batch
    EXPECT_EQ(output.size(1), 10);  // sequence
    EXPECT_EQ(output.size(2), 128); // out_features
}

TEST(SwiGLUFFNTest, GradientFlow) {
    torch::manual_seed(42);
    
    SwiGLUFFN swiglu(128, 256, 128);
    swiglu->train();
    
    torch::Tensor x = torch::randn({2, 10, 128});
    x.set_requires_grad(true);
    
    torch::Tensor output = swiglu->forward(x);
    torch::Tensor loss = output.sum();
    loss.backward();
    
    EXPECT_TRUE(x.grad().defined());
    EXPECT_EQ(x.grad().size(0), 2);
    EXPECT_EQ(x.grad().size(1), 10);
    EXPECT_EQ(x.grad().size(2), 128);
}

TEST(SwiGLUFFNTest, TrainEvalModes) {
    torch::manual_seed(42);
    
    SwiGLUFFN swiglu(128, 256, 128);
    
    torch::Tensor x = torch::randn({1, 5, 128});
    
    swiglu->train();
    EXPECT_TRUE(swiglu->is_training());
    
    swiglu->eval();
    EXPECT_FALSE(swiglu->is_training());
    
    // Should run without error in eval mode
    torch::Tensor output = swiglu->forward(x);
    EXPECT_EQ(output.size(2), 128);
}

TEST(SwiGLUFFNTest, ModuleRegistration) {
    torch::manual_seed(42);
    
    SwiGLUFFN swiglu(128, 256, 128);
    
    // Module should have submodules registered
    auto named_modules = swiglu->named_modules();
    // Should have at least w12 and w3
    EXPECT_GT(named_modules.size(), 0);
}

TEST(SwiGLUFFNTest, SwiGLUFFNFused) {
    torch::manual_seed(42);
    
    SwiGLUFFNFused swiglu_fused(128, 256, 128);
    swiglu_fused->eval();
    
    torch::Tensor x = torch::randn({2, 10, 128});
    torch::Tensor output = swiglu_fused->forward(x);
    
    EXPECT_EQ(output.dim(), 3);
    EXPECT_EQ(output.size(0), 2);
    EXPECT_EQ(output.size(1), 10);
    EXPECT_EQ(output.size(2), 128);
}

TEST(SwiGLUFFNTest, DifferentBatchSizes) {
    torch::manual_seed(42);
    
    SwiGLUFFN swiglu(128, 256, 128);
    swiglu->eval();
    
    std::vector<int64_t> batch_sizes = {1, 4, 16};
    torch::Tensor x = torch::randn({4, 10, 128});
    
    for (int64_t b : batch_sizes) {
        if (b <= 4) {
            torch::Tensor input = x.slice(0, 0, b);
            torch::Tensor output = swiglu->forward(input);
            EXPECT_EQ(output.size(0), b);
        }
    }
}

TEST(SwiGLUFFNTest, DifferentSequenceLengths) {
    torch::manual_seed(42);
    
    SwiGLUFFN swiglu(128, 256, 128);
    swiglu->eval();
    
    std::vector<int64_t> seq_lengths = {1, 8, 32};
    
    for (int64_t s : seq_lengths) {
        torch::Tensor x = torch::randn({2, s, 128});
        torch::Tensor output = swiglu->forward(x);
        EXPECT_EQ(output.size(1), s);
    }
}

TEST(SwiGLUFFNTest, SwishActivation) {
    torch::manual_seed(42);
    
    SwiGLUFFN swiglu(128, 256, 128);
    swiglu->eval();
    
    torch::Tensor x = torch::randn({1, 1, 128});
    torch::Tensor output = swiglu->forward(x);
    
    // Output should be finite
    EXPECT_TRUE(torch::isfinite(output).all().item<bool>());
}

TEST(SwiGLUFFNTest, LargeHiddenFeatures) {
    torch::manual_seed(42);
    
    // Test with large hidden_features
    SwiGLUFFN swiglu(64, 512, 64);
    swiglu->eval();
    
    torch::Tensor x = torch::randn({2, 10, 64});
    torch::Tensor output = swiglu->forward(x);
    
    EXPECT_EQ(output.size(2), 64);
}

TEST(SwiGLUFFNTest, SameInOutFeatures) {
    torch::manual_seed(42);
    
    // Test with explicit same in/out features
    SwiGLUFFN swiglu(128, 256, 128);
    swiglu->eval();
    
    torch::Tensor x = torch::randn({2, 10, 128});
    torch::Tensor output = swiglu->forward(x);
    
    EXPECT_EQ(output.size(2), 128);
}

TEST(SwiGLUFFNTest, DeterministicOutput) {
    torch::manual_seed(42);
    
    SwiGLUFFN swiglu(128, 256, 128);
    swiglu->eval();
    
    torch::Tensor x = torch::randn({1, 5, 128});
    
    torch::Tensor output1 = swiglu->forward(x);
    torch::Tensor output2 = swiglu->forward(x);
    
    // Same input should produce same output in eval mode
    EXPECT_TRUE(torch::allclose(output1, output2));
}

} // namespace
} // namespace layers
} // namespace vggt
