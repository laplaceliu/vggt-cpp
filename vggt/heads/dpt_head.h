#pragma once

#include <torch/torch.h>
#include "../utils/stack_sequential.h"

namespace vggt {
namespace heads {

// Function declarations
utils::StackSequential _make_fusion_block(int64_t features, int64_t size = -1, bool has_residual = true, int64_t groups = 1);

torch::nn::Module _make_scratch(const std::vector<int64_t>& in_shape, int64_t out_shape, int64_t groups = 1, bool expand = false);

torch::Tensor custom_interpolate(const torch::Tensor& input, const std::pair<int64_t, int64_t>& size, const torch::nn::functional::InterpolateFuncOptions::mode_t& mode = torch::kBilinear, bool align_corners = true);

} // namespace heads
} // namespace vggt
