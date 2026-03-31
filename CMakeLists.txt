cmake_minimum_required(VERSION 3.15)
project(vggt-cpp)

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DEINOPS_TORCH_BACKEND")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(DEPENDENCIES ${CMAKE_CURRENT_SOURCE_DIR}/dependencies)

set(glad_DIR ${DEPENDENCIES}/lib/cmake/glad)
set(Matplot++_DIR ${DEPENDENCIES}/lib/cmake/Matplot++)
set(OpenCV_DIR ${DEPENDENCIES}/lib/cmake/opencv4)
set(spdlog_DIR ${DEPENDENCIES}/lib/cmake/spdlog)
set(Torch_DIR ${DEPENDENCIES}/share/cmake/Torch)
set(Eigen3_DIR ${DEPENDENCIES}/share/eigen3/cmake)

find_package(glad REQUIRED)
find_package(vpi REQUIRED)
find_package(spdlog REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Matplot++ REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(GTest REQUIRED)

include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${DEPENDENCIES}/include
    ${DEPENDENCIES}/include/opencv4
    ${DEPENDENCIES}/include/eigen3
    ${DEPENDENCIES}/include/einops
)

################################################################################
# 添加vggt子目录
################################################################################
add_subdirectory(vggt)
target_compile_definitions(vggt PUBLIC EINOPS_TORCH_BACKEND)

################################################################################
# ${PROJECT_NAME}
################################################################################
# 源文件
add_executable(demo_vggt
    demo_vggt.cpp
)

# 链接库
target_link_libraries(demo_vggt
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    -Wl,--allow-multiple-definition
)

################################################################################
# Unit Tests
################################################################################
enable_testing()

# MLP Test
add_executable(test_mlp
    tests/test_mlp.cpp
)
target_link_libraries(test_mlp
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME mlp_test COMMAND test_mlp)

# DropPath Test
add_executable(test_drop_path
    tests/test_drop_path.cpp
)
target_link_libraries(test_drop_path
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME drop_path_test COMMAND test_drop_path)

# LayerScale Test
add_executable(test_layer_scale
    tests/test_layer_scale.cpp
)
target_link_libraries(test_layer_scale
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME layer_scale_test COMMAND test_layer_scale)

# Attention Test
add_executable(test_attention
    tests/test_attention.cpp
)
target_link_libraries(test_attention
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME attention_test COMMAND test_attention)

# SwiGLUFFN Test
add_executable(test_swiglu_ffn
    tests/test_swiglu_ffn.cpp
)
target_link_libraries(test_swiglu_ffn
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME swiglu_ffn_test COMMAND test_swiglu_ffn)

# PatchEmbed Test
add_executable(test_patch_embed
    tests/test_patch_embed.cpp
)
target_link_libraries(test_patch_embed
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME patch_embed_test COMMAND test_patch_embed)

# RoPE Test
add_executable(test_rope
    tests/test_rope.cpp
)
target_link_libraries(test_rope
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME rope_test COMMAND test_rope)

# Block Test - SKIPPED: Block/NestedTensorBlock have design issues with
# libtorch ModuleHolder (no default constructor, constructor forwarding issues)

# Geometry Test
add_executable(test_geometry
    tests/test_geometry.cpp
)
target_link_libraries(test_geometry
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME geometry_test COMMAND test_geometry)

# Distortion Test
add_executable(test_distortion
    tests/test_distortion.cpp
)
target_link_libraries(test_distortion
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME distortion_test COMMAND test_distortion)

# Projection Test
add_executable(test_projection
    tests/test_projection.cpp
)
target_link_libraries(test_projection
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME projection_test COMMAND test_projection)

# Head Activation Functions Test
add_executable(test_head_act
    tests/test_head_act.cpp
)
target_link_libraries(test_head_act
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME head_act_test COMMAND test_head_act)

# Heads Utils Test
add_executable(test_heads_utils
    tests/test_heads_utils.cpp
)
target_link_libraries(test_heads_utils
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME heads_utils_test COMMAND test_heads_utils)

# Stack Sequential Test
add_executable(test_stack_sequential
    tests/test_stack_sequential.cpp
)
target_link_libraries(test_stack_sequential
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME stack_sequential_test COMMAND test_stack_sequential)

# Track Modules Utils Test
add_executable(test_track_modules_utils
    tests/test_track_modules_utils.cpp
)
target_link_libraries(test_track_modules_utils
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME track_modules_utils_test COMMAND test_track_modules_utils)

# Track Modules Modules Test
add_executable(test_track_modules_modules
    tests/test_track_modules_modules.cpp
)
target_link_libraries(test_track_modules_modules
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME track_modules_modules_test COMMAND test_track_modules_modules)

# DPT Head Test
add_executable(test_dpt_head
    tests/test_dpt_head.cpp
)
target_link_libraries(test_dpt_head
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME dpt_head_test COMMAND test_dpt_head)

# Track Modules Blocks Test
add_executable(test_track_modules_blocks
    tests/test_track_modules_blocks.cpp
)
target_link_libraries(test_track_modules_blocks
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME track_modules_blocks_test COMMAND test_track_modules_blocks)

# Camera Head Test
add_executable(test_camera_head
    tests/test_camera_head.cpp
)
target_link_libraries(test_camera_head
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME camera_head_test COMMAND test_camera_head)

# Block Test
add_executable(test_block
    tests/test_block.cpp
)
target_link_libraries(test_block
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME block_test COMMAND test_block)

# Vision Transformer Test
add_executable(test_vision_transformer
    tests/test_vision_transformer.cpp
)
target_link_libraries(test_vision_transformer
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME vision_transformer_test COMMAND test_vision_transformer)

# Load Function Test
add_executable(test_load_fn
    tests/test_load_fn.cpp
)
target_link_libraries(test_load_fn
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME load_fn_test COMMAND test_load_fn)

# VGGSFM Tracker Test
add_executable(test_vggsfm_tracker
    tests/test_vggsfm_tracker.cpp
)
target_link_libraries(test_vggsfm_tracker
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME vggsfm_tracker_test COMMAND test_vggsfm_tracker)

# Visual Track Test
add_executable(test_visual_track
    tests/test_visual_track.cpp
)
target_link_libraries(test_visual_track
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME visual_track_test COMMAND test_visual_track)

# Weight Loader Test
add_executable(test_weight_loader
    tests/test_weight_loader.cpp
)
target_link_libraries(test_weight_loader
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME weight_loader_test COMMAND test_weight_loader)

# Base Track Predictor Test
add_executable(test_base_track_predictor
    tests/test_base_track_predictor.cpp
)
target_link_libraries(test_base_track_predictor
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME base_track_predictor_test COMMAND test_base_track_predictor)

# Helper Test
add_executable(test_helper
    tests/test_helper.cpp
)
target_link_libraries(test_helper
    PRIVATE
    vggt
    pthread
    ${TORCH_LIBRARIES}
    ${DEPENDENCIES}/lib/libgtest.a
    ${DEPENDENCIES}/lib/libgtest_main.a
    -Wl,--allow-multiple-definition
)
add_test(NAME helper_test COMMAND test_helper)
