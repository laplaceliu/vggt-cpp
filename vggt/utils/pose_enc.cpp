

#include "pose_enc.h"
#include "rotation.h"

namespace vggt {
namespace utils {

torch::Tensor extri_intri_to_pose_encoding(
    const torch::Tensor& extrinsics,
    const torch::Tensor& intrinsics,
    const std::pair<int, int>& image_size_hw,
    const std::string& pose_encoding_type
) {
    if (pose_encoding_type == "absT_quaR_FoV") {
        auto R = extrinsics.slice(-1, 0, 3).slice(-2, 0, 3); // BxSx3x3
        auto T = extrinsics.slice(-1, 3).slice(-2, 0, 3);     // BxSx3

        auto quat = mat_to_quat(R);
        auto H = image_size_hw.first;
        auto W = image_size_hw.second;
        auto fov_h = 2 * torch::atan((H / 2.0) / intrinsics.index({torch::indexing::Ellipsis, 1, 1}));
        auto fov_w = 2 * torch::atan((W / 2.0) / intrinsics.index({torch::indexing::Ellipsis, 0, 0}));
        auto pose_encoding = torch::cat({T, quat, fov_h.unsqueeze(-1), fov_w.unsqueeze(-1)}, -1).to(torch::kFloat);
        return pose_encoding;
    } else {
        throw std::runtime_error("Unsupported pose encoding type: " + pose_encoding_type);
    }
}

std::tuple<torch::Tensor, torch::Tensor> pose_encoding_to_extri_intri(
    const torch::Tensor& pose_encoding,
    const std::pair<int, int>& image_size_hw,
    const std::string& pose_encoding_type,
    bool build_intrinsics
) {
    torch::Tensor intrinsics;

    if (pose_encoding_type == "absT_quaR_FoV") {
        auto T = pose_encoding.slice(-1, 0, 3);
        auto quat = pose_encoding.slice(-1, 3, 7);
        auto fov_h = pose_encoding.slice(-1, 7);
        auto fov_w = pose_encoding.slice(-1, 8);

        auto R = quat_to_mat(quat);
        auto extrinsics = torch::cat({R, T.unsqueeze(-1)}, -1);

        if (build_intrinsics) {
            auto H = image_size_hw.first;
            auto W = image_size_hw.second;
            auto fy = (H / 2.0) / torch::tan(fov_h / 2.0);
            auto fx = (W / 2.0) / torch::tan(fov_w / 2.0);
            intrinsics = torch::zeros(
                torch::IntArrayRef(pose_encoding.sizes().begin(), pose_encoding.sizes().end() - 1) + std::initializer_list<int64_t>{3, 3},
                pose_encoding.options()
            );
            intrinsics.index_put_({torch::indexing::Ellipsis, 0, 0}, fx);
            intrinsics.index_put_({torch::indexing::Ellipsis, 1, 1}, fy);
            intrinsics.index_put_({torch::indexing::Ellipsis, 0, 2}, W / 2.0);
            intrinsics.index_put_({torch::indexing::Ellipsis, 1, 2}, H / 2.0);
            intrinsics.index_put_({torch::indexing::Ellipsis, 2, 2}, 1.0);
        }
        return {extrinsics, intrinsics};
    } else {
        throw std::runtime_error("Unsupported pose encoding type: " + pose_encoding_type);
    }
}

} // namespace utils
} // namespace vggt
