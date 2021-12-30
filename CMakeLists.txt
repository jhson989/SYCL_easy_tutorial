# CMake bianry version
cmake_minimum_required(VERSION 3.10)

set(PATH_SYCL_BUILD /data/share/oneapi/llvm/build)
set(CMAKE_CXX_COMPILER ${PATH_SYCL_BUILD}/bin/clang++)
set(PATH_SYCL_INC ${PATH_SYCL_BUILD}/include/sycl)
set(PATH_SYCL_LIB ${PATH_SYCL_BUILD}/lib)

# Proejct info
project(
    SYCL_VECTOR_ADDITION
    VERSION 0.1
    DESCRIPTION "sycl vector addition example"
)

add_subdirectory(test)