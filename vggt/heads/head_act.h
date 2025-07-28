#pragma once

#include <torch/torch.h>

namespace vggt {
namespace heads {

torch::Tensor activate_pose(
    torch::Tensor pred_pose_enc,
    const std::string& trans_act = "linear",
    const std::string& quat_act = "linear",
    const std::string& fl_act = "linear");

torch::Tensor base_pose_act(
    torch::Tensor pose_enc,
    const std::string& act_type = "linear");

std::tuple<torch::Tensor, torch::Tensor> activate_head(
    torch::Tensor out,
    const std::string& activation = "norm_exp",
    const std::string& conf_activation = "expp1");

torch::Tensor inverse_log_transform(torch::Tensor y);

} // namespace heads
} // namespace vggt