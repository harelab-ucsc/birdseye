cmake_minimum_required(VERSION 3.10)
project(Birdseye_Benchmark)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3.7 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
)
include_directories("/usr/local/include/apriltag")

# Add executable
add_executable(Benchmark src/Birdseye_benchmark.cpp)
add_executable(Calibrate src/Birdseye_calibrate.cpp)

# Link libraries
target_link_libraries(Benchmark
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    ${YAML_CPP_LIBRARIES}
    nlohmann_json::nlohmann_json
)

target_link_libraries(Calibrate 
    ${OpenCV_LIBS}
    ${YAML_CPP_LIBRARIES}
    apriltag
)
