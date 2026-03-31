/**
 * @file test_all.cpp
 * @brief Unified test suite for all VGGT-CPP modules
 *
 * This file includes all test files to create a single unified test executable.
 * Run with: ./test_all
 */

// Core Layers
#include "test_mlp.cpp"
#include "test_drop_path.cpp"
#include "test_layer_scale.cpp"
#include "test_attention.cpp"
#include "test_swiglu_ffn.cpp"
#include "test_patch_embed.cpp"
#include "test_rope.cpp"
#include "test_block.cpp"
#include "test_vision_transformer.cpp"

// Utility Functions
#include "test_geometry.cpp"
#include "test_distortion.cpp"
#include "test_projection.cpp"
#include "test_helper.cpp"
#include "test_load_fn.cpp"
#include "test_visual_track.cpp"
#include "test_weight_loader.cpp"
#include "test_pose_enc.cpp"

// Head Components
#include "test_head_act.cpp"
#include "test_heads_utils.cpp"
#include "test_stack_sequential.cpp"
#include "test_dpt_head.cpp"
#include "test_camera_head.cpp"

// Track Modules
#include "test_track_modules_utils.cpp"
#include "test_track_modules_modules.cpp"
#include "test_track_modules_blocks.cpp"
#include "test_vggsfm_tracker.cpp"
#include "test_base_track_predictor.cpp"
#include "test_track_refine.cpp"

// Core Models
#include "test_aggregator.cpp"
#include "test_vggt.cpp"
