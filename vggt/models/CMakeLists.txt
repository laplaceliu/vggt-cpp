cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# Add source files
target_sources(vggt
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/aggregator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/vggt.cpp
)

# Include directories
target_include_directories(vggt
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)