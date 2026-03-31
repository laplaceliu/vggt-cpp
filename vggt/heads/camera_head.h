#pragma once
/**
 * @file camera_head.h
 * @brief Camera pose prediction head for VGGT
 *
 * Predicts camera poses (translation, rotation as quaternion, focal length)
 * from aggregated visual tokens using an iterative refinement trunk.
 */

#include <torch/torch.h>
#include "utils/stack_sequential.h"

namespace vggt {
namespace heads {

/**
 * @brief Camera pose prediction head
 *
 * Takes aggregated visual tokens from the encoder and predicts camera poses
 * through iterative refinement. The output is a pose encoding containing:
 * - Translation (3D): absT
 * - Rotation (quaternion, 4D): quaR
 * - Field of view/focal length (1D): FoV
 *
 * Uses AdaLN (Adaptive Layer Normalization) modulation for conditioning
 * and a trunk of transformer blocks for iterative refinement.
 *
 * @note Currently only supports "absT_quaR_FoV" pose encoding type
 */
class CameraHeadImpl : public torch::nn::Module {
public:
    /**
     * @brief Construct a CameraHead
     *
     * @param dim_in Input feature dimension (default: 2048)
     * @param trunk_depth Number of transformer blocks in refinement trunk (default: 4)
     * @param pose_encoding_type Pose encoding format (default: "absT_quaR_FoV")
     * @param num_heads Number of attention heads in transformer blocks (default: 16)
     * @param mlp_ratio MLP hidden dim expansion ratio (default: 4)
     * @param init_values LayerScale initial values (default: 0.01)
     * @param trans_act Activation type for translation (default: "linear")
     * @param quat_act Activation type for quaternion (default: "linear")
     * @param fl_act Activation type for focal length (default: "relu")
     */
    CameraHeadImpl(
        int64_t dim_in = 2048,
        int64_t trunk_depth = 4,
        const std::string& pose_encoding_type = "absT_quaR_FoV",
        int64_t num_heads = 16,
        int64_t mlp_ratio = 4,
        double init_values = 0.01,
        const std::string& trans_act = "linear",
        const std::string& quat_act = "linear",
        const std::string& fl_act = "relu"
    );

    /**
     * @brief Forward pass through camera head
     *
     * @param aggregated_tokens_list List of token tensors from aggregator
     *        Each tensor should have shape [B, S, N, C] where:
     *        - B: batch size
     *        - S: sequence length (number of frames)
     *        - N: number of camera tokens
     *        - C: feature dimension (dim_in)
     * @param num_iterations Number of refinement iterations (default: 4)
     * @return Vector of pose encoding tensors, one per iteration
     *         Each tensor has shape [B, S, 9] containing:
     *         - indices 0-2: translation (3D)
     *         - indices 3-6: rotation quaternion (4D)
     *         - index 8: focal length (1D)
     */
    std::vector<torch::Tensor> forward(const std::vector<torch::Tensor>& aggregated_tokens_list, int64_t num_iterations = 4);

private:
    /**
     * @brief Internal trunk function for iterative refinement
     *
     * @param pose_tokens Input pose tokens [B, S, C]
     * @param num_iterations Number of refinement iterations
     * @return Vector of activated pose encodings per iteration
     */
    std::vector<torch::Tensor> trunk_fn(const torch::Tensor& pose_tokens, int64_t num_iterations);

    int64_t target_dim;           ///< Target pose encoding dimension (9 = 3 + 4 + 1 + 1)
    std::string trans_act;        ///< Translation activation type
    std::string quat_act;        ///< Quaternion activation type
    std::string fl_act;          ///< Focal length activation type
    int64_t trunk_depth;         ///< Number of refinement blocks

    utils::StackSequential trunk;           ///< Refinement trunk (StackSequential of Blocks)
    torch::nn::LayerNorm token_norm{nullptr};     ///< Token normalization layer
    torch::nn::LayerNorm trunk_norm{nullptr};     ///< Trunk normalization layer
    torch::Tensor empty_pose_tokens;        ///< Learnable empty pose token [1, 1, 9]
    torch::nn::Linear embed_pose{nullptr};  ///< Pose embedding linear layer [9 -> dim_in]
    utils::StackSequential poseLN_modulation;     ///< AdaLN modulation network
    torch::nn::LayerNorm adaln_norm{nullptr};      ///< AdaLN normalization
    torch::nn::AnyModule pose_branch;       ///< Final pose prediction MLP
};

/**
 * @brief CameraHead module wrapper with TORCH_MODULE
 */
TORCH_MODULE(CameraHead);

} // namespace heads
} // namespace vggt
