# CMake bianry version
cmake_minimum_required(VERSION 3.10)

set(APP app.out)
set(PATH_BUILD /data/share/oneapi/llvm/build)
set(CMAKE_CXX_COMPILER ${PATH_BUILD}/bin/clang++)
set(PATH_SYCL_INC ${PATH_BUILD}/include/sycl)
set(PATH_SYCL_LIB ${PATH_BUILD}/lib)

set(MAIN vectorAdd.cpp)
set(SRC_FILES 
    src/deviceProperties.cpp
)


# Proejct info
project(
    SYCL_VECTOR_ADDITION
    VERSION 0.1
    DESCRIPTION "sycl vector addition example"
)

add_executable(${APP} ${MAIN} ${SRC_FILES})

target_include_directories(${APP} PUBLIC ${PATH_SYCL_INC} include/)
target_compile_options(${APP} PUBLIC -fsycl -fsycl-targets=nvptx64-nvidia-cuda)

target_link_libraries(${APP} PUBLIC sycl)
target_link_directories(${APP} PUBLIC ${PATH_SYCL_LIB})
target_link_options(${APP}  PUBLIC -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
