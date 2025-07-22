cmake_minimum_required(VERSION 3.15)

# 收集所有源文件
file(GLOB_RECURSE VGGT_SOURCES
    "*.cpp"
)

# 收集所有头文件
file(GLOB_RECURSE VGGT_HEADERS
    "*.h"
)

# 创建静态库
add_library(vggt STATIC ${VGGT_SOURCES} ${VGGT_HEADERS})

# 设置包含目录
target_include_directories(vggt
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/heads
    ${CMAKE_CURRENT_SOURCE_DIR}/layers
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/dependency
)

# 链接依赖库
target_link_libraries(vggt
    PUBLIC
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    Eigen3::Eigen
    spdlog::spdlog
    ${CUDA_LIBRARIES}
    vpi
)

# 设置编译选项
target_compile_features(vggt PUBLIC cxx_std_17)