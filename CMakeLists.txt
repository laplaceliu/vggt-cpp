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

# Block Test - SKIPPED: Block/NestedTensorBlock have design issues with 
# libtorch ModuleHolder (no default constructor, constructor forwarding issues)
