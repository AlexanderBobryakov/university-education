cmake_minimum_required(VERSION 3.16)
project(lab2 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(lab2 main.cu)

set_target_properties(
        lab2
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)