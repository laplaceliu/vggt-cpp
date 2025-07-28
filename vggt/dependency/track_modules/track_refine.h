#pragma once

#include <torch/torch.h>
#include <einops.hpp>
#include <tuple>
#include <vector>

namespace vggt {
namespace dependency {
namespace track_modules {

/**
 * Refines the tracking of images using a fine track predictor and a fine feature network.
 * Check https://arxiv.org/abs/2312.04563 for more details.
 *
 * @param images The images to be tracked.
 * @param fine_fnet The fine feature network.
 * @param fine_tracker The fine track predictor.
 * @param coarse_pred The coarse predictions of tracks.
 * @param compute_score Whether to compute the score. Defaults to false.
 * @param pradius The radius of a patch. Defaults to 15.
 * @param sradius The search radius. Defaults to 2.
 * @param fine_iters Number of iterations for fine tracking. Defaults to 6.
 * @param chunk Chunk size for processing patches. Defaults to 40960.
 * @return A tuple containing the refined tracks and optionally the score.
 */
std::tuple<torch::Tensor, torch::Tensor> refine_track(
    torch::Tensor images, 
    torch::nn::AnyModule& fine_fnet, 
    torch::nn::AnyModule& fine_tracker, 
    torch::Tensor coarse_pred, 
    bool compute_score = false, 
    int64_t pradius = 15, 
    int64_t sradius = 2, 
    int64_t fine_iters = 6,
    int64_t chunk = 40960
);

/**
 * COPIED FROM VGGSfM
 * Refines the tracking of images using a fine track predictor and a fine feature network.
 * Check https://arxiv.org/abs/2312.04563 for more details.
 *
 * @param images The images to be tracked.
 * @param fine_fnet The fine feature network.
 * @param fine_tracker The fine track predictor.
 * @param coarse_pred The coarse predictions of tracks.
 * @param compute_score Whether to compute the score. Defaults to false.
 * @param pradius The radius of a patch. Defaults to 15.
 * @param sradius The search radius. Defaults to 2.
 * @param fine_iters Number of iterations for fine tracking. Defaults to 6.
 * @return A tuple containing the refined tracks and optionally the score.
 */
std::tuple<torch::Tensor, torch::Tensor> refine_track_v0(
    torch::Tensor images, 
    torch::nn::AnyModule& fine_fnet, 
    torch::nn::AnyModule& fine_tracker, 
    torch::Tensor coarse_pred, 
    bool compute_score = false, 
    int64_t pradius = 15, 
    int64_t sradius = 2, 
    int64_t fine_iters = 6
);

/**
 * Compute the scores, i.e., the standard deviation of the 2D similarity heatmaps,
 * given the query point features and reference frame feature maps
 */
torch::Tensor compute_score_fn(
    torch::Tensor query_point_feat, 
    torch::Tensor patch_feat, 
    torch::Tensor fine_pred_track, 
    int64_t sradius, 
    int64_t psize, 
    int64_t B, 
    int64_t N, 
    int64_t S, 
    int64_t C_out
);

} // namespace track_modules
} // namespace dependency
} // namespace vggt