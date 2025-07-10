/**
 * @file pose_enc.cpp
 * @brief Implementation of camera pose encoding utility functions for VGGT
 */

#include "pose_enc.h"
#include <torch/torch.h>

namespace vggt {
namespace utils {

torch::Tensor extri_intri_to_pose_encoding(
    const torch::Tensor& extrinsics,
    const torch::Tensor& intrinsics) {

    // Extract rotation and translation from extrinsics
    auto R = extrinsics.narrow(1, 0, 3).narrow(2, 0, 3);  // [B, 3, 3]
    auto t = extrinsics.narrow(1, 0, 3).narrow(2, 3, 1);  // [B, 3, 1]

    // Extract focal length and principal point from intrinsics
    auto fx = intrinsics.index({Ellipsis, 0, 0}).unsqueeze(-1);  // [B, 1]
    auto fy = intrinsics.index({Ellipsis, 1, 1}).unsqueeze(-1);  // [B, 1]
    auto cx = intrinsics.index({Ellipsis, 0, 2}).unsqueeze(-1);  // [B, 1]
    auto cy = intrinsics.index({Ellipsis, 1, 2}).unsqueeze(-1);  // [B, 1]

    // Flatten rotation matrix
    auto R_flat = R.reshape({R.size(0), 9});  // [B, 9]

    // Flatten translation vector
    auto t_flat = t.reshape({t.size(0), 3});  // [B, 3]

    // Concatenate all components to form pose encoding
    auto pose_encoding = torch::cat({R_flat, t_flat, fx, fy, cx, cy}, 1);  // [B, 16]

    return pose_encoding;
}

std::tuple<torch::Tensor, torch::Tensor> pose_encoding_to_extri_intri(
    const torch::Tensor& pose_encoding) {

    auto batch_size = pose_encoding.size(0);
    auto device = pose_encoding.device();

    // Extract components from pose encoding
    auto R_flat = pose_encoding.narrow(1, 0, 9);  // [B, 9]
    auto t_flat = pose_encoding.narrow(1, 9, 3);  // [B, 3]
    auto fx = pose_encoding.narrow(1, 12, 1);  // [B, 1]
    auto fy = pose_encoding.narrow(1, 13, 1);  // [B, 1]
    auto cx = pose_encoding.narrow(1, 14, 1);  // [B, 1]
    auto cy = pose_encoding.narrow(1, 15, 1);  // [B, 1]

    // Reshape rotation matrix
    auto R = R_flat.reshape({batch_size, 3, 3});  // [B, 3, 3]

    // Reshape translation vector
    auto t = t_flat.reshape({batch_size, 3, 1});  // [B, 3, 1]

    // Create extrinsics matrix
    auto extrinsics = torch::zeros({batch_size, 4, 4},
                                  torch::TensorOptions().device(device).dtype(pose_encoding.dtype()));

    // Set rotation part
    extrinsics.narrow(1, 0, 3).narrow(2, 0, 3) = R;

    // Set translation part
    extrinsics.narrow(1, 0, 3).narrow(2, 3, 1) = t;

    // Set bottom row to [0, 0, 0, 1]
    extrinsics.narrow(1, 3, 1).narrow(2, 0, 3).zero_();
    extrinsics.narrow(1, 3, 1).narrow(2, 3, 1).fill_(1.0);

    // Create intrinsics matrix
    auto intrinsics = torch::zeros({batch_size, 3, 3},
                                  torch::TensorOptions().device(device).dtype(pose_encoding.dtype()));

    // Set focal length
    intrinsics.index_put_({torch::arange(batch_size), 0, 0}, fx.squeeze(-1));
    intrinsics.index_put_({torch::arange(batch_size), 1, 1}, fy.squeeze(-1));

    // Set principal point
    intrinsics.index_put_({torch::arange(batch_size), 0, 2}, cx.squeeze(-1));
    intrinsics.index_put_({torch::arange(batch_size), 1, 2}, cy.squeeze(-1));

    // Set [0, 0, 1] in bottom row
    intrinsics.index_put_({torch::arange(batch_size), 2, 2}, torch::ones({batch_size},
                         torch::TensorOptions().device(device).dtype(pose_encoding.dtype())));

    return std::make_tuple(extrinsics, intrinsics);
}

} // namespace utils
} // namespace vggt
