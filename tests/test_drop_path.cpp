#include <gtest/gtest.h>
#include <torch/torch.h>
#include "vggt/layers/drop_path.h"

TEST(DropPathTest, NoDropWhenProbZero) {
    torch::manual_seed(42);
    
    torch::Tensor input = torch::randn({4, 128});
    torch::Tensor output = vggt::layers::drop_path(input, 0.0f, true);
    
    EXPECT_TRUE(torch::allclose(input, output));
}

TEST(DropPathTest, NoDropWhenNotTraining) {
    torch::manual_seed(42);
    
    torch::Tensor input = torch::randn({4, 128});
    torch::Tensor output = vggt::layers::drop_path(input, 0.5f, false);
    
    EXPECT_TRUE(torch::allclose(input, output));
}

TEST(DropPathTest, DropPathModule) {
    torch::manual_seed(42);
    
    vggt::layers::DropPath drop_path(0.0f);
    
    torch::Tensor input = torch::randn({4, 128});
    torch::Tensor output = drop_path->forward(input);
    
    EXPECT_TRUE(torch::allclose(input, output));
}

TEST(DropPathTest, DropPathModuleEval) {
    torch::manual_seed(42);
    
    vggt::layers::DropPath drop_path(0.5f);
    drop_path->eval();
    
    torch::Tensor input = torch::randn({4, 128});
    torch::Tensor output = drop_path->forward(input);
    
    EXPECT_TRUE(torch::allclose(input, output));
}

TEST(DropPathTest, DropPathShapePreserved) {
    torch::manual_seed(42);
    
    vggt::layers::DropPath drop_path(0.5f);
    torch::Tensor input = torch::randn({2, 3, 64, 64});
    torch::Tensor output = drop_path->forward(input);
    
    EXPECT_EQ(output.sizes(), input.sizes());
}

TEST(DropPathTest, DropPathStochastic) {
    torch::manual_seed(42);
    
    // With 0.9 drop_prob, most paths should be dropped
    vggt::layers::DropPath drop_path(0.9f);
    drop_path->train();
    
    int num_zeros = 0;
    const int num_runs = 10;
    
    for (int i = 0; i < num_runs; ++i) {
        torch::Tensor input = torch::ones({1, 1});
        torch::Tensor output = drop_path->forward(input);
        if (output.item<float>() == 0.0f) {
            num_zeros++;
        }
    }
    
    // At least some paths should be dropped with 0.9 probability
    EXPECT_GT(num_zeros, 0);
}
