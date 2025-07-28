#pragma once

#include <torch/torch.h>

namespace vggt {
namespace utils {

struct StackSequentialImpl : torch::nn::SequentialImpl {
    using SequentialImpl::SequentialImpl;
    torch::Tensor forward(torch::Tensor x) {
        return torch::nn::SequentialImpl::forward(x);
    }
};
TORCH_MODULE(StackSequential);

} // namespace utils
} // namespace vggt
