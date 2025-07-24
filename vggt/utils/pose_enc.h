#pragma once

#include <torch/torch.h>

namespace vggt {
namespace utils {

torch::Tensor extri_intri_to_pose_encoding(
    const torch::Tensor& extrinsics,
    const torch::Tensor& intrinsics,
    const std::pair<int64_t, int64_t>& image_size_hw = {},
    const std::string& pose_encoding_type = "absT_quaR_FoV"
);

std::pair<torch::Tensor, torch::Tensor> pose_encoding_to_extri_intri(
    const torch::Tensor& pose_encoding,
    const std::pair<int64_t, int64_t>& image_size_hw = {},
    const std::string& pose_encoding_type = "absT_quaR_FoV",
    bool build_intrinsics = true
);

} // namespace utils
} // namespace vggt