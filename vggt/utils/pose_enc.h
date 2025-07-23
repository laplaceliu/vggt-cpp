// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/torch.h>

namespace vggt {
namespace utils {

/**
 * @brief Convert camera extrinsics and intrinsics to a compact pose encoding.
 * 
 * @param extrinsics Camera extrinsic parameters with shape BxSx3x4.
 * @param intrinsics Camera intrinsic parameters with shape BxSx3x3.
 * @param image_size_hw Tuple of (height, width) of the image in pixels.
 * @param pose_encoding_type Type of pose encoding to use.
 * @return torch::Tensor Encoded camera pose parameters with shape BxSx9.
 */
torch::Tensor extri_intri_to_pose_encoding(
    const torch::Tensor& extrinsics,
    const torch::Tensor& intrinsics,
    const std::pair<int, int>& image_size_hw = {},
    const std::string& pose_encoding_type = "absT_quaR_FoV"
);

/**
 * @brief Convert a pose encoding back to camera extrinsics and intrinsics.
 * 
 * @param pose_encoding Encoded camera pose parameters with shape BxSx9.
 * @param image_size_hw Tuple of (height, width) of the image in pixels.
 * @param pose_encoding_type Type of pose encoding used.
 * @param build_intrinsics Whether to reconstruct the intrinsics matrix.
 * @return std::tuple<torch::Tensor, torch::Tensor> (extrinsics, intrinsics)
 */
std::tuple<torch::Tensor, torch::Tensor> pose_encoding_to_extri_intri(
    const torch::Tensor& pose_encoding,
    const std::pair<int, int>& image_size_hw = {},
    const std::string& pose_encoding_type = "absT_quaR_FoV",
    bool build_intrinsics = true
);

} // namespace utils
} // namespace vggt