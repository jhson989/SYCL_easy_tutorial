# CMake bianry version
cmake_minimum_required(VERSION 3.10)

set(PATH_SYCL_BUILD /data/share/oneapi/llvm/build)
set(CMAKE_CXX_COMPILER ${PATH_SYCL_BUILD}/bin/clang++)
set(PATH_SYCL_INC ${PATH_SYCL_BUILD}/include/sycl)
set(PATH_SYCL_LIB ${PATH_SYCL_BUILD}/lib)
set(SYCL_COMPILE_OPTION -fsycl -fsycl-targets=nvptx64-nvidia-cuda)

# Proejct info
project(
    SYCL_EASY_EXAMPLE
    VERSION 0.1
    DESCRIPTION "sycl easy example"
)

add_subdirectory(test)