cmake_minimum_required(VERSION 3.15)
project(vggt)

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-fPIC")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(DEPENDENCIES ${CMAKE_CURRENT_SOURCE_DIR}/dependencies)

set(spdlog_DIR ${DEPENDENCIES}/lib/cmake/spdlog)
set(OpenCV_DIR ${DEPENDENCIES}/lib/cmake/opencv4)
set(Torch_DIR ${DEPENDENCIES}/share/cmake/Torch)
set(Eigen3_DIR ${DEPENDENCIES}/share/eigen3/cmake)

find_package(vpi REQUIRED)
find_package(spdlog REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

################################################################################
# ${PROJECT_NAME}
################################################################################
# 源文件
add_executable(${PROJECT_NAME}
    demo_vggt.cpp
)

# 链接库
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    pthread
    ${TORCH_LIBRARIES}
)
