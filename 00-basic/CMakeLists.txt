cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(basic LANGUAGES CXX CUDA)

add_executable(add_cpu add_cpu.cpp)
add_executable(add_gpu_v1 add_gpu_v1.cu)
add_executable(add_gpu_v2 add_gpu_v2.cu)
add_executable(add_gpu_v3 add_gpu_v3.cu)
add_executable(add_gpu_v4 add_gpu_v4.cu)