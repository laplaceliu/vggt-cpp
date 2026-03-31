#include <gtest/gtest.h>
#include <torch/torch.h>
#include "vggt/layers/attention.h"

TEST(AttentionTest, BasicForward) {
    torch::manual_seed(42);
    
    int64_t dim = 128;
    int64_t num_heads = 4;
    
    vggt::layers::Attention attention(dim, num_heads);
    
    // Input: [B, N, C] = [2, 16, 128]
    torch::Tensor x = torch::randn({2, 16, dim});
    
    torch::Tensor output = attention->forward(x);
    
    EXPECT_EQ(output.sizes(), torch::IntArrayRef({2, 16, dim}));
}

TEST(AttentionTest, DifferentShapes) {
    torch::manual_seed(42);
    
    int64_t dim = 64;
    int64_t num_heads = 4;
    
    vggt::layers::Attention attention(dim, num_heads);
    
    // Single batch
    torch::Tensor x1 = torch::randn({1, 32, dim});
    torch::Tensor output1 = attention->forward(x1);
    EXPECT_EQ(output1.sizes(), torch::IntArrayRef({1, 32, dim}));
    
    // Larger sequence
    torch::Tensor x2 = torch::randn({1, 128, dim});
    torch::Tensor output2 = attention->forward(x2);
    EXPECT_EQ(output2.sizes(), torch::IntArrayRef({1, 128, dim}));
}

TEST(AttentionTest, TrainEvalMode) {
    torch::manual_seed(42);
    
    vggt::layers::Attention attention(64, 4);
    
    attention->train();
    EXPECT_TRUE(attention->is_training());
    
    attention->eval();
    EXPECT_FALSE(attention->is_training());
}

TEST(AttentionTest, GradientFlow) {
    torch::manual_seed(42);
    
    vggt::layers::Attention attention(64, 4);
    
    torch::Tensor x = torch::randn({2, 16, 64}, torch::requires_grad(true));
    torch::Tensor output = attention->forward(x);
    
    output.sum().backward();
    
    EXPECT_TRUE(x.grad().defined());
}

TEST(AttentionTest, WithQKNorm) {
    torch::manual_seed(42);
    
    int64_t dim = 128;
    int64_t num_heads = 8;
    
    // qk_norm = true
    vggt::layers::Attention attention(dim, num_heads, true, true, 0.0, 0.0, 
                                       torch::nn::AnyModule(torch::nn::LayerNorm(torch::nn::LayerNormOptions({}))),
                                       true,  // qk_norm
                                       false, // fused_attn
                                       torch::nn::AnyModule()); // no rope
    
    torch::Tensor x = torch::randn({2, 16, dim});
    torch::Tensor output = attention->forward(x);
    
    EXPECT_EQ(output.sizes(), torch::IntArrayRef({2, 16, dim}));
}
