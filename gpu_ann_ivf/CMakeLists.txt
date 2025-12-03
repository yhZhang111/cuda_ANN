cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(ann_gpu_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# 指定编译架构 — 你的 GPU 是 8.6 → sm_86
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to build for" FORCE)

add_executable(brute_nn src/main.cu)
