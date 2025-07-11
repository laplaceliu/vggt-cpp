/**
 * @file track_refine.h
 * @brief Functions for refining tracks using fine-grained features
 */

#pragma once

#include <torch/torch.h>
#include <tuple>
#include <vector>

namespace vggt {

/**
 * @brief Refines the tracking of images using a fine track predictor and a fine feature network.
 *
 * @param images The images to be tracked (B, S, 3, H, W)
 * @param fine_fnet The fine feature network
 * @param fine_tracker The fine track predictor
 * @param coarse_pred The coarse predictions of tracks (B, S, N, 2)
 * @param compute_score Whether to compute the score
 * @param pradius The radius of a patch
 * @param sradius The search radius
 * @param fine_iters Number of iterations for fine tracking
 * @param chunk Chunk size for processing patches (negative for no chunking)
 * @return std::tuple<torch::Tensor, torch::Tensor> Refined tracks and scores (if compute_score is true)
 */
std::tuple<torch::Tensor, torch::Tensor> refine_track(
    torch::Tensor images,
    torch::jit::Module fine_fnet,
    torch::jit::Module fine_tracker,
    torch::Tensor coarse_pred,
    bool compute_score = false,
    int pradius = 15,
    int sradius = 2,
    int fine_iters = 6,
    int chunk = 40960
);

/**
 * @brief Original version of refine_track from VGGSfM
 *
 * @param images The images to be tracked (B, S, 3, H, W)
 * @param fine_fnet The fine feature network
 * @param fine_tracker The fine track predictor
 * @param coarse_pred The coarse predictions of tracks (B, S, N, 2)
 * @param compute_score Whether to compute the score
 * @param pradius The radius of a patch
 * @param sradius The search radius
 * @param fine_iters Number of iterations for fine tracking
 * @return std::tuple<torch::Tensor, torch::Tensor> Refined tracks and scores (if compute_score is true)
 */
std::tuple<torch::Tensor, torch::Tensor> refine_track_v0(
    torch::Tensor images,
    torch::jit::Module fine_fnet,
    torch::jit::Module fine_tracker,
    torch::Tensor coarse_pred,
    bool compute_score = false,
    int pradius = 15,
    int sradius = 2,
    int fine_iters = 6
);

/**
 * @brief Compute the scores, i.e., the standard deviation of the 2D similarity heatmaps
 *
 * @param query_point_feat Query point features
 * @param patch_feat Patch features
 * @param fine_pred_track Fine prediction tracks
 * @param sradius Search radius
 * @param psize Patch size
 * @param B Batch size
 * @param N Number of tracks
 * @param S Sequence length
 * @param C_out Output channel dimension
 * @return torch::Tensor Computed scores
 */
torch::Tensor compute_score_fn(
    torch::Tensor query_point_feat,
    torch::Tensor patch_feat,
    torch::Tensor fine_pred_track,
    int sradius,
    int psize,
    int B,
    int N,
    int S,
    int C_out
);

/**
 * @brief Extract glimpses (patches) from a tensor at specified offsets
 *
 * @param tensor Input tensor (B, C, W, H)
 * @param size Size of the glimpse (h, w)
 * @param offsets Offsets where to extract glimpses (B, N, 2)
 * @param mode Interpolation mode ("bilinear" or "nearest")
 * @param padding_mode Padding mode ("zeros", "border", or "reflection")
 * @param debug Whether to enable debug mode
 * @param orib Original batch size (optional)
 * @return torch::Tensor Extracted glimpses (B, N, C, h, w)
 */
torch::Tensor extract_glimpse(
    torch::Tensor tensor,
    std::tuple<int, int> size,
    torch::Tensor offsets,
    const std::string& mode = "bilinear",
    const std::string& padding_mode = "zeros",
    bool debug = false,
    int orib = -1
);

} // namespace vggt
