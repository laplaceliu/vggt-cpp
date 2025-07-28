#pragma once

#include <torch/torch.h>

namespace vggt {
namespace dependency {

/**
 * Check if a tensor is a torch tensor
 * @param x Input tensor
 * @return True if x is a torch tensor
 */
bool is_torch(const torch::Tensor& x);

/**
 * Ensure input is a torch tensor
 * @param x Input tensor
 * @return Torch tensor
 */
torch::Tensor ensure_torch(const torch::Tensor& x);

/**
 * Apply undistortion to the normalized tracks using the given distortion parameters once.
 * @param params Distortion parameters of shape BxN.
 * @param tracks_normalized Normalized tracks tensor of shape [batch_size, num_tracks, 2].
 * @return Undistorted normalized tracks tensor.
 */
torch::Tensor single_undistortion(const torch::Tensor& params, const torch::Tensor& tracks_normalized);

/**
 * Iteratively undistort the normalized tracks using the given distortion parameters.
 * @param params Distortion parameters of shape BxN.
 * @param tracks_normalized Normalized tracks tensor of shape [batch_size, num_tracks, 2].
 * @param max_iterations Maximum number of iterations for the undistortion process.
 * @param max_step_norm Maximum step norm for convergence.
 * @param rel_step_size Relative step size for numerical differentiation.
 * @return Undistorted normalized tracks tensor.
 */
torch::Tensor iterative_undistortion(const torch::Tensor& params, const torch::Tensor& tracks_normalized, 
                                    int64_t max_iterations = 100, double max_step_norm = 1e-10, 
                                    double rel_step_size = 1e-6);

/**
 * Applies radial or OpenCV distortion to the given 2D points.
 * @param extra_params Distortion parameters of shape BxN, where N can be 1, 2, or 4.
 * @param u Normalized x coordinates of shape Bxnum_tracks.
 * @param v Normalized y coordinates of shape Bxnum_tracks.
 * @return std::pair of distorted u and v coordinates
 */
std::pair<torch::Tensor, torch::Tensor> apply_distortion(const torch::Tensor& extra_params, 
                                                        const torch::Tensor& u, 
                                                        const torch::Tensor& v);

} // namespace dependency
} // namespace vggt