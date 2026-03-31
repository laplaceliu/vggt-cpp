#include <gtest/gtest.h>
#include <torch/torch.h>
#include "vggt/layers/mlp.h"

TEST(MLPTest, BasicForward) {
    torch::manual_seed(42);
    
    vggt::layers::Mlp mlp(128, 256, 64);
    
    torch::Tensor input = torch::randn({4, 128});
    torch::Tensor output = mlp->forward(input);
    
    EXPECT_EQ(output.sizes(), torch::IntArrayRef({4, 64}));
}

TEST(MLPTest, DefaultHiddenDimension) {
    torch::manual_seed(42);
    
    vggt::layers::Mlp mlp(128);  // hidden_features = in_features
    
    torch::Tensor input = torch::randn({2, 128});
    torch::Tensor output = mlp->forward(input);
    
    EXPECT_EQ(output.sizes(), torch::IntArrayRef({2, 128}));
}

TEST(MLPTest, DifferentInputShapes) {
    torch::manual_seed(42);
    
    vggt::layers::Mlp mlp(512, 1024, 256);
    
    torch::Tensor input = torch::randn({1, 512});
    torch::Tensor output = mlp->forward(input);
    
    EXPECT_EQ(output.sizes(), torch::IntArrayRef({1, 256}));
}

TEST(MLPTest, TrainEvalMode) {
    vggt::layers::Mlp mlp(64, 128, 32);
    
    mlp->train();
    EXPECT_TRUE(mlp->is_training());
    
    mlp->eval();
    EXPECT_FALSE(mlp->is_training());
}

TEST(MLPTest, GradientComputation) {
    torch::manual_seed(42);
    
    vggt::layers::Mlp mlp(64, 128, 32);
    
    torch::Tensor input = torch::randn({2, 64}, torch::requires_grad(true));
    torch::Tensor output = mlp->forward(input);
    
    EXPECT_TRUE(output.requires_grad());
    
    output.backward(torch::ones_like(output));
    EXPECT_TRUE(input.grad().defined());
}

TEST(MLPTest, SingleInput) {
    torch::manual_seed(42);
    
    vggt::layers::Mlp mlp(256, 512, 128);
    
    torch::Tensor input = torch::randn({256});
    torch::Tensor output = mlp->forward(input);
    
    EXPECT_EQ(output.sizes(), torch::IntArrayRef({128}));
}
