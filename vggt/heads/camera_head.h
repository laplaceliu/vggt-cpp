/**
 * @file camera_head.h
 * @brief Camera head module for VGGT
 */

#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>

namespace vggt {
namespace heads {

class CameraHeadImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a new CameraHeadImpl object
     *
     * @param dim_in Input dimension (default=2048)
     * @param trunk_depth Number of transformer blocks in trunk (default=4)
     * @param pose_encoding_type Type of pose encoding (default="absT_quaR_FoV")
     * @param num_heads Number of attention heads (default=16)
     * @param mlp_ratio MLP expansion ratio (default=4)
     * @param init_values Initialization value for blocks (default=0.01)
     * @param trans_act Activation type for translation (default="linear")
     * @param quat_act Activation type for quaternion (default="linear")
     * @param fl_act Activation type for focal length (default="relu")
     */
    CameraHeadImpl(
        int64_t dim_in = 2048,
        int64_t trunk_depth = 4,
        const std::string& pose_encoding_type = "absT_quaR_FoV",
        int64_t num_heads = 16,
        int64_t mlp_ratio = 4,
        float init_values = 0.01f,
        const std::string& trans_act = "linear",
        const std::string& quat_act = "linear",
        const std::string& fl_act = "relu");

    /**
     * @brief Forward pass for camera head
     *
     * @param aggregated_tokens_list List of token tensors from network
     * @param num_iterations Number of refinement iterations (default=4)
     * @return std::vector<torch::Tensor> List of predicted camera encodings
     */
    std::vector<torch::Tensor> forward(
        const std::vector<torch::Tensor>& aggregated_tokens_list,
        int64_t num_iterations = 4);

private:
    int64_t target_dim_;
    std::string trans_act_;
    std::string quat_act_;
    std::string fl_act_;
    int64_t trunk_depth_;

    // Submodules
    torch::nn::Sequential trunk_;
    torch::nn::LayerNorm token_norm_;
    torch::nn::LayerNorm trunk_norm_;
    torch::Tensor empty_pose_tokens_;
    torch::nn::Linear embed_pose_;
    torch::nn::Sequential poseLN_modulation_;
    torch::nn::LayerNorm adaln_norm_;
    torch::nn::Sequential pose_branch_;

    /**
     * @brief Trunk function for iterative refinement
     *
     * @param pose_tokens Input pose tokens
     * @param num_iterations Number of refinement iterations
     * @return std::vector<torch::Tensor> List of predicted camera encodings
     */
    std::vector<torch::Tensor> trunk_fn(
        const torch::Tensor& pose_tokens,
        int64_t num_iterations);
};

TORCH_MODULE(CameraHead);

} // namespace heads
} // namespace vggt
