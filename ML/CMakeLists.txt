cmake_minimum_required(VERSION 3.18)
project(NeuralForge LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 70 75 80)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    src/cpp/include
)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;--use_fast_math)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

add_library(neuralforge_core SHARED
    src/cpp/extension.cpp
    src/cpp/operators.cpp
    src/cuda/kernels.cu
    src/cuda/matmul.cu
    src/cuda/activations.cu
    src/cuda/optimizers.cu
)

set_target_properties(neuralforge_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(neuralforge_core
    ${CUDA_LIBRARIES}
    ${Python3_LIBRARIES}
)