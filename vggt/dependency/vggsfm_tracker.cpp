#include "vggsfm_tracker.h"
#include "track_modules/blocks.h"
#include "track_modules/base_track_predictor.h"
#include "track_modules/track_refine.h"

namespace vggt {
namespace dependency {

TrackerPredictorImpl::TrackerPredictorImpl() {
    // Define coarse predictor configuration
    int64_t coarse_stride = 4;
    coarse_down_ratio = 2;

    // Create networks
    coarse_fnet = register_module("coarse_fnet", track_modules::BasicEncoder(coarse_stride));
    coarse_predictor = register_module("coarse_predictor", track_modules::BaseTrackerPredictor(coarse_stride));

    // Create fine predictor with stride = 1
    fine_fnet = register_module("fine_fnet", track_modules::ShallowEncoder(1));
    fine_predictor = register_module("fine_predictor", track_modules::BaseTrackerPredictor(
        1, // stride
        4, // depth
        3, // corr_levels
        3, // corr_radius
        32, // latent_dim
        256, // hidden_size
        true, // fine
        false // use_spaceatt
    ));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> TrackerPredictorImpl::forward(
    torch::Tensor images,
    torch::Tensor query_points,
    torch::Tensor fmaps,
    int64_t coarse_iters,
    bool inference,
    bool fine_tracking,
    int64_t fine_chunk
) {
    if (!fmaps.defined()) {
        auto batch_num = images.size(0);
        auto frame_num = images.size(1);
        auto image_dim = images.size(2);
        auto height = images.size(3);
        auto width = images.size(4);

        auto reshaped_image = images.reshape({batch_num * frame_num, image_dim, height, width});
        fmaps = process_images_to_fmaps(reshaped_image);
        fmaps = fmaps.reshape({batch_num, frame_num, -1, fmaps.size(-2), fmaps.size(-1)});

        if (inference) {
            // No need to call empty_cache in C++
        }
    }

    // Coarse prediction
    auto result = coarse_predictor.forward(
        query_points, fmaps, coarse_iters, coarse_down_ratio
    );
    auto coarse_pred_track_lists = result[0];
    auto pred_vis = result[1];
    auto coarse_pred_track = coarse_pred_track_lists[-1];

    if (inference) {
        // No need to call empty_cache in C++
    }

    torch::Tensor fine_pred_track, pred_score;
    if (fine_tracking) {
        // Refine the coarse prediction
        std::tie(fine_pred_track, pred_score) = track_modules::refine_track(
            images, fine_fnet, fine_predictor, coarse_pred_track, false, fine_chunk
        );

        if (inference) {
            // No need to call empty_cache in C++
        }
    } else {
        fine_pred_track = coarse_pred_track;
        pred_score = torch::ones_like(pred_vis);
    }

    return std::make_tuple(fine_pred_track, coarse_pred_track, pred_vis, pred_score);
}

torch::Tensor TrackerPredictorImpl::process_images_to_fmaps(torch::Tensor images) {
    if (coarse_down_ratio > 1) {
        auto scaled_images = torch::nn::functional::interpolate(
            images,
            torch::nn::functional::InterpolateFuncOptions()
                .scale_factor(std::vector<double>{1.0 / coarse_down_ratio})
                .mode(torch::kBilinear)
                .align_corners(true)
        );
        return coarse_fnet.forward(scaled_images);
    } else {
        return coarse_fnet.forward(images);
    }
}

} // namespace dependency
} // namespace vggt
