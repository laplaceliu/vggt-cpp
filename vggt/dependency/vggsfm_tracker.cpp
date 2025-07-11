/**
 * @file vggsfm_tracker.cpp
 * @brief Implementation of VGG-SFM tracker
 */

#include "vggsfm_tracker.h"
#include "../track_modules/base_track_predictor.h"
#include "../track_modules/blocks.h"
#include "../track_modules/track_refine.h"
#include <torch/torch.h>
#include <stdexcept>

namespace vggt {

TrackerPredictor::TrackerPredictor() {
    // Initialize coarse networks
    int coarse_stride = 4;
    coarse_down_ratio = 2;

    // Create coarse feature network
    coarse_fnet = register_module("coarse_fnet",
        BasicEncoder(coarse_stride));

    // Create coarse predictor
    coarse_predictor = register_module("coarse_predictor",
        BaseTrackerPredictor(coarse_stride));

    // Create fine feature network
    fine_fnet = register_module("fine_fnet",
        ShallowEncoder(/*stride=*/1));

    // Create fine predictor
    fine_predictor = register_module("fine_predictor",
        BaseTrackerPredictor(
            /*stride=*/1,
            /*depth=*/4,
            /*corr_levels=*/3,
            /*corr_radius=*/3,
            /*latent_dim=*/32,
            /*hidden_size=*/256,
            /*fine=*/true,
            /*use_spaceatt=*/false
        ));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
TrackerPredictor::forward(
    torch::Tensor images,
    torch::Tensor query_points,
    torch::Tensor fmaps,
    int coarse_iters,
    bool inference,
    bool fine_tracking,
    int fine_chunk
) {
    // Validate input tensors
    if (images.dim() != 5 || images.size(2) != 3) {
        throw std::invalid_argument("images must have shape [B, S, 3, H, W]");
    }
    if (query_points.dim() != 3 || query_points.size(2) != 2) {
        throw std::invalid_argument("query_points must have shape [B, N, 2]");
    }

    // Process images to feature maps if not provided
    if (!fmaps.defined()) {
        auto batch_num = images.size(0);
        auto frame_num = images.size(1);
        auto image_dim = images.size(2);
        auto height = images.size(3);
        auto width = images.size(4);

        // Reshape images for processing
        auto reshaped_image = images.reshape({batch_num * frame_num, image_dim, height, width});
        fmaps = process_images_to_fmaps(reshaped_image);
        fmaps = fmaps.reshape({batch_num, frame_num, -1, fmaps.size(-2), fmaps.size(-1)});

        if (inference) {
            torch::cuda::empty_cache();
        }
    }

    // Coarse prediction
    auto [coarse_pred_track_lists, pred_vis] = coarse_predictor.forward<BaseTrackerPredictor>(
        query_points, fmaps, coarse_iters, coarse_down_ratio);
    auto coarse_pred_track = coarse_pred_track_lists[-1];

    if (inference) {
        torch::cuda::empty_cache();
    }

    torch::Tensor fine_pred_track;
    torch::Tensor pred_score;

    if (fine_tracking) {
        // Refine the coarse prediction
        std::tie(fine_pred_track, pred_score) = refine_track(
            images,
            fine_fnet,
            fine_predictor,
            coarse_pred_track,
            /*compute_score=*/false,
            fine_chunk
        );

        if (inference) {
            torch::cuda::empty_cache();
        }
    } else {
        fine_pred_track = coarse_pred_track;
        pred_score = torch::ones_like(pred_vis);
    }

    return std::make_tuple(fine_pred_track, coarse_pred_track, pred_vis, pred_score);
}

torch::Tensor TrackerPredictor::process_images_to_fmaps(torch::Tensor images) {
    // Validate input tensor
    if (images.dim() != 4 || images.size(1) != 3) {
        throw std::invalid_argument("images must have shape [S, 3, H, W]");
    }

    if (coarse_down_ratio > 1) {
        // Scale down images to save memory
        auto scaled_images = torch::nn::functional::interpolate(
            images,
            torch::nn::functional::InterpolateFuncOptions()
                .scale_factor(1.0 / coarse_down_ratio)
                .mode(torch::kBilinear)
                .align_corners(true)
        );
        return coarse_fnet.forward<BasicEncoder>(scaled_images);
    } else {
        return coarse_fnet.forward<BasicEncoder>(images);
    }
}

} // namespace vggt
