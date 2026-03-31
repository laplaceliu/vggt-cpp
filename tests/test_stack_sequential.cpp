#include <gtest/gtest.h>
#include <torch/torch.h>
#include "vggt/utils/stack_sequential.h"

namespace vggt {
namespace utils {
namespace {

TEST(StackSequentialTest, BasicForward) {
    torch::manual_seed(42);

    // Create a simple StackSequential with linear layers
    StackSequential sequential(
        torch::nn::Linear(10, 20),
        torch::nn::ReLU(),
        torch::nn::Linear(20, 5)
    );

    torch::Tensor input = torch::randn({4, 10});
    torch::Tensor output = sequential->forward(input);

    EXPECT_EQ(output.dim(), 2);
    EXPECT_EQ(output.size(0), 4);  // batch
    EXPECT_EQ(output.size(1), 5);  // output features
}

TEST(StackSequentialTest, SingleLayer) {
    torch::manual_seed(42);

    StackSequential sequential(
        torch::nn::Linear(8, 4)
    );

    torch::Tensor input = torch::randn({2, 8});
    torch::Tensor output = sequential->forward(input);

    EXPECT_EQ(output.sizes(), torch::IntArrayRef({2, 4}));
}

TEST(StackSequentialTest, GradientFlow) {
    torch::manual_seed(42);

    StackSequential sequential(
        torch::nn::Linear(10, 20),
        torch::nn::ReLU(),
        torch::nn::Linear(20, 5)
    );

    torch::Tensor input = torch::randn({4, 10}, torch::requires_grad(true));
    torch::Tensor output = sequential->forward(input);
    torch::Tensor loss = output.sum();
    loss.backward();

    EXPECT_TRUE(input.grad().defined());
    EXPECT_EQ(input.grad().size(0), 4);
}

TEST(StackSequentialTest, TrainEvalModes) {
    torch::manual_seed(42);

    StackSequential sequential(
        torch::nn::Linear(10, 20),
        torch::nn::Dropout(0.5),
        torch::nn::Linear(20, 5)
    );

    torch::Tensor input = torch::randn({2, 10});

    sequential->train();
    EXPECT_TRUE(sequential->is_training());

    sequential->eval();
    EXPECT_FALSE(sequential->is_training());

    // Should run without error in eval mode
    torch::Tensor output = sequential->forward(input);
    EXPECT_EQ(output.size(0), 2);
}

TEST(StackSequentialTest, ReturnsTensorNotVector) {
    torch::manual_seed(42);

    // The key feature of StackSequential is that forward returns a Tensor
    // (not std::vector<torch::Tensor> like torch::nn::Sequential)
    StackSequential sequential(
        torch::nn::Linear(16, 8),
        torch::nn::ReLU()
    );

    torch::Tensor input = torch::randn({3, 16});
    torch::Tensor output = sequential->forward(input);

    // Should return a single Tensor, not a vector
    EXPECT_EQ(output.dim(), 2);
    EXPECT_EQ(output.sizes(), torch::IntArrayRef({3, 8}));
}

TEST(StackSequentialTest, DifferentInputShapes) {
    torch::manual_seed(42);

    StackSequential sequential(
        torch::nn::Linear(128, 64),
        torch::nn::ReLU(),
        torch::nn::Linear(64, 32)
    );

    // Test with different batch sizes
    for (int64_t batch_size : {1, 4, 16}) {
        torch::Tensor input = torch::randn({batch_size, 128});
        torch::Tensor output = sequential->forward(input);
        EXPECT_EQ(output.size(0), batch_size);
        EXPECT_EQ(output.size(1), 32);
    }
}

} // namespace
} // namespace utils
} // namespace vggt
