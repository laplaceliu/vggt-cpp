#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace vggt {
namespace utils {

/**
 * Map (x, y) -> color in (R, G, B).
 * 1) Normalize x,y to [0,1].
 * 2) Combine them into a single scalar c in [0,1].
 * 3) Use matplot's colormap to convert c -> (R,G,B).
 *
 * @param x X coordinate
 * @param y Y coordinate
 * @param W Image width
 * @param H Image height
 * @param cmap_name Colormap name (e.g., "hsv", "rainbow", "jet")
 * @return RGB color tuple in [0,1]
 */
std::tuple<double, double, double> color_from_xy(double x, double y, int W, int H, const std::string& cmap_name = "hsv");

/**
 * Given all tracks in one sample (b), compute a (N,3) array of RGB color values
 * in [0,255]. The color is determined by the (x,y) position in the first
 * visible frame for each track.
 *
 * @param tracks_b Tensor of shape (S, N, 2). (x,y) for each track in each frame.
 * @param vis_mask_b (S, N) boolean mask; if None, assume all are visible.
 * @param image_width Used for normalizing x.
 * @param image_height Used for normalizing y.
 * @param cmap_name For matplot (e.g., 'hsv', 'rainbow', 'jet').
 * @return cv::Mat of shape (N, 3), each row is (R,G,B) in [0,255].
 */
cv::Mat get_track_colors_by_position(
    const torch::Tensor& tracks_b,
    const torch::Tensor& vis_mask_b = torch::Tensor(),
    int image_width = 0,
    int image_height = 0,
    const std::string& cmap_name = "hsv");

/**
 * Visualizes frames in a grid layout with specified frames per row.
 * Each track's color is determined by its (x,y) position
 * in the first visible frame (or frame 0 if always visible).
 * Finally convert the BGR result to RGB before saving.
 * Also saves each individual frame as a separate PNG file.
 *
 * @param images torch::Tensor (S, 3, H, W) if CHW or (S, H, W, 3) if HWC.
 * @param tracks torch::Tensor (S, N, 2), last dim = (x, y).
 * @param track_vis_mask torch::Tensor (S, N) or empty tensor.
 * @param out_dir Folder to save visualizations.
 * @param image_format "CHW" or "HWC".
 * @param normalize_mode "[0,1]", "[-1,1]", or empty string for direct raw -> 0..255
 * @param cmap_name A matplot colormap name for color_from_xy.
 * @param frames_per_row Number of frames to display in each row of the grid.
 * @param save_grid Whether to save all frames in one grid image.
 */
void visualize_tracks_on_images(
    torch::Tensor images,
    torch::Tensor tracks,
    torch::Tensor track_vis_mask = torch::Tensor(),
    const std::string& out_dir = "track_visuals_concat_by_xy",
    const std::string& image_format = "CHW",
    const std::string& normalize_mode = "[0,1]",
    const std::string& cmap_name = "hsv",
    int frames_per_row = 4,
    bool save_grid = true);

} // namespace utils
} // namespace vggt