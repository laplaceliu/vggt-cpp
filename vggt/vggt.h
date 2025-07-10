/**
 * @file vggt.h
 * @brief Main header file for VGGT (Visual Geometry and Graph Tracking) library
 */

#pragma once

#include <torch/torch.h>

// Include all utility headers
#include "utils/geometry.h"
#include "utils/helper.h"
#include "utils/load_fn.h"
#include "utils/pose_enc.h"
#include "utils/rotation.h"
#include "utils/visual_track.h"

// Include layer headers
#include "layers/attention.h"
#include "layers/block.h"
#include "layers/drop_path.h"
#include "layers/layer_scale.h"
#include "layers/mlp.h"
#include "layers/patch_embed.h"
#include "layers/rope.h"
#include "layers/swiglu_ffn.h"
#include "layers/vision_transformer.h"

/**
 * @namespace vggt
 * @brief Main namespace for VGGT (Visual Geometry and Graph Tracking) library
 */
namespace vggt {

/**
 * @brief Get the version of VGGT library
 * @return std::string Version string
 */
inline std::string version() {
    return "0.1.0";
}

} // namespace vggt
