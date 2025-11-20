cmake_minimum_required(VERSION 3.20)

project(vectorAdd LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 100 110 120)

add_executable(vectorAddGPU vectorAdd.cu)
add_executable(vectorAddCPU vectorAdd.cpp)
