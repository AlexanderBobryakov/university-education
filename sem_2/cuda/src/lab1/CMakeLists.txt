# cmake_minimum_required(VERSION <specify CMake version here>)
cmake_minimum_required(VERSION 3.16)

project(untitled3 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(untitled3 main.cu)

set_target_properties(
        untitled3
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)