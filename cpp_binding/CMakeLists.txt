cmake_minimum_required(VERSION 3.1)
project(cpytorch_testing)
set(CMAKE_PREFIX_PATH /home/lamlam/downloads/libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(test test.cpp keypoint_block.cpp)
target_link_libraries(test "${TORCH_LIBRARIES}")
set_property(TARGET test PROPERTY CXX_STANDARD 14)
