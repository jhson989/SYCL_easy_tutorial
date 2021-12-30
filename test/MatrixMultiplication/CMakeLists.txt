set(APP MatrixMultiplication.out)

set(MAIN main.cpp)
set(SRC_FILES 
    ../../src/deviceProperties.cpp
)

add_executable(${APP} ${MAIN} ${SRC_FILES})

target_include_directories(${APP} PUBLIC ${PATH_SYCL_INC} ../../include/)
target_compile_options(${APP} PUBLIC -fsycl -fsycl-targets=nvptx64-nvidia-cuda)

target_link_libraries(${APP} PUBLIC sycl)
target_link_directories(${APP} PUBLIC ${PATH_SYCL_LIB})
target_link_options(${APP}  PUBLIC -fsycl -fsycl-targets=nvptx64-nvidia-cuda)
