cmake_minimum_required(VERSION 3.10)
project(sim3D)

set(CMAKE_CXX_FLAGS "-std=c++17")

find_package(Eigen3 3.3.7 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters segmentation search)

set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")

include_directories(${Pangolin_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIRS} ${Sophus_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library(ortho_synth STATIC
    src/ortho_synth.cpp
)

add_executable(sim3D ${SRC_DIR}/sim3D.cpp)

target_link_libraries(sim3D ortho_synth ${Pangolin_LIBRARIES} ${YAML_CPP_LIBRARIES} ${PCL_LIBRARIES})