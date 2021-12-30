# CMake bianry version
cmake_minimum_required(VERSION 3.10)

set(APP VectorAddition.out)


set(MAIN main.cpp)
set(SRC_FILES 
    ../../src/deviceProperties.cpp
)


# Proejct info
project(
    SYCL_VECTOR_ADDITION
    VERSION 0.1
    DESCRIPTION "sycl vector addition example"
)

add_executable(${APP} ${MAIN} ${SRC_FILES})

target_include_directories(${APP} PUBLIC ${PATH_SYCL_INC} ../../include/)
target_compile_options(${APP} PUBLIC -fsycl -fsycl-targets=nvptx64-nvidia-cuda)

target_link_libraries(${APP} PUBLIC sycl)
target_link_directories(${APP} PUBLIC ${PATH_SYCL_LIB})
target_link_options(${APP}  PUBLIC -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
