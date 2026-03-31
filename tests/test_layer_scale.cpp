#include <gtest/gtest.h>
#include <torch/torch.h>
#include "vggt/layers/layer_scale.h"

TEST(LayerScaleTest, BasicInitialization) {
    torch::manual_seed(42);
    
    vggt::layers::LayerScale layer_scale(64, 1e-5);
    
    torch::Tensor input = torch::randn({2, 64});
    torch::Tensor output = layer_scale->forward(input);
    
    EXPECT_EQ(output.sizes(), torch::IntArrayRef({2, 64}));
}

TEST(LayerScaleTest, WithInitValues) {
    torch::manual_seed(42);
    
    torch::Tensor init_values = torch::ones({64}) * 0.1;
    vggt::layers::LayerScale layer_scale(64, init_values);
    
    torch::Tensor input = torch::randn({2, 64});
    torch::Tensor output = layer_scale->forward(input);
    
    // output should be input * 0.1
    EXPECT_TRUE(torch::allclose(output, input * 0.1, 1e-6));
}

TEST(LayerScaleTest, DefaultInitValue) {
    torch::manual_seed(42);
    
    vggt::layers::LayerScale layer_scale(128, 1e-5);
    
    torch::Tensor input = torch::randn({1, 128});
    torch::Tensor output = layer_scale->forward(input);
    
    // Check gamma parameter is initialized correctly
    auto gamma = layer_scale->named_parameters().find("gamma");
    ASSERT_NE(gamma, nullptr);
    EXPECT_TRUE(torch::allclose(*gamma, torch::ones({128}) * 1e-5));
}

TEST(LayerScaleTest, InplaceMode) {
    torch::manual_seed(42);
    
    vggt::layers::LayerScale layer_scale(32, 0.5, true);
    
    torch::Tensor input = torch::randn({1, 32});
    torch::Tensor output = layer_scale->forward(input);
    
    // Inplace should modify input in place and share memory
    EXPECT_EQ(input.data_ptr(), output.data_ptr());
    // And produce correct values
    EXPECT_TRUE(torch::allclose(input, output, 1e-6));
}

TEST(LayerScaleTest, NonInplaceMode) {
    torch::manual_seed(42);
    
    vggt::layers::LayerScale layer_scale(32, 0.5, false);
    
    torch::Tensor input = torch::randn({1, 32});
    torch::Tensor input_copy = input.clone();
    torch::Tensor output = layer_scale->forward(input);
    
    // Non-inplace should not modify input
    EXPECT_TRUE(torch::allclose(input, input_copy));
    EXPECT_TRUE(torch::allclose(output, input * 0.5));
}

TEST(LayerScaleTest, GradientFlow) {
    torch::manual_seed(42);
    
    vggt::layers::LayerScale layer_scale(64, 0.1);
    
    torch::Tensor input = torch::randn({2, 64}, torch::requires_grad(true));
    torch::Tensor output = layer_scale->forward(input);
    
    output.sum().backward();
    
    EXPECT_TRUE(input.grad().defined());
}

TEST(LayerScaleTest, DifferentDimensions) {
    torch::manual_seed(42);
    
    vggt::layers::LayerScale layer_scale(256, 0.01);
    
    torch::Tensor input = torch::randn({4, 8, 256});
    torch::Tensor output = layer_scale->forward(input);
    
    EXPECT_EQ(output.sizes(), torch::IntArrayRef({4, 8, 256}));
}
