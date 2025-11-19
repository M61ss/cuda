cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(basic LANGUAGES CXX CUDA)

add_executable(static static.cu)
add_executable(dynamic dynamic.cu)