
cmake_minimum_required(VERSION 3.20)

project(
    work3
    LANGUAGES
        CXX
        CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)


add_library(
    matrix_core
    INTERFACE
)

target_include_directories(
    matrix_core
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/include
)

target_link_libraries(
    matrix_core
    INTERFACE
        CUDA::cudart
)


add_executable(
    matrix_demo
    demo/demo.cu
)

target_link_libraries(
    matrix_demo
    PRIVATE
        matrix_core
)


add_executable(
    matrix_tests
    tests/test_matrix.cu
)

target_link_libraries(
    matrix_tests
    PRIVATE
        matrix_core
        GTest::gtest
        GTest::gtest_main
        Eigen3::Eigen
)


add_executable(
    matrix_benchmark
    benchmarks/benchmark_matrix.cu
)

target_link_libraries(
    matrix_benchmark
    PRIVATE
        matrix_core
        benchmark::benchmark
)


set_target_properties(
    matrix_demo
    matrix_tests
    matrix_benchmark

    PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
