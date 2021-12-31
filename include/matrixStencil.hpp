#ifndef __JH_MATRIX_STENCIL__
#define __JH_MATRIX_STENCIL__

#include <CL/sycl.hpp>
namespace sycl = cl::sycl;



template <typename T>
void parallel_matrix_stencil(
    sycl::queue queue, std::vector<T>& input, std::vector<T>& output, const T* kernel, 
    const size_t* size_input, const size_t* size_output, const size_t* size_kernel, const size_t* offset, const size_t* pad) {

    sycl::buffer<T, 1> buf_input(input.data(), input.size());
    sycl::buffer<T, 1> buf_output(output.data(), output.size());
    sycl::buffer<T, 1> buf_kernel(kernel, size_kernel[0]*size_kernel[1]);

    size_t size_input_y = size_input[0], size_input_x = size_input[1];
    size_t size_output_y = size_output[0], size_output_x = size_output[1];
    size_t size_kernel_y = size_kernel[0], size_kernel_x = size_kernel[1];
    size_t offset_y = offset[0], offset_x = offset[1];
    size_t pad_y = pad[0], pad_x = pad[1];


    queue.submit([&] (sycl::handler& cgh) {

        auto acc_input = buf_input.template get_access<sycl::access::mode::read>(cgh);
        auto acc_kernel = buf_kernel.template get_access<sycl::access::mode::read>(cgh);
        auto acc_output = buf_output.template get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for(sycl::range<1>(output.size()), [=] (sycl::id<1> idx) {

            int i = idx/size_output_x;
            int j = idx%size_output_x;

            float sum = 0.0f;
            int y = i*offset_y-pad_y;
            int x = j*offset_x-pad_x;

            for (int ky=0; ky<size_kernel_y; ky++) {
                for (int kx=0; kx<size_kernel_x; kx++) {
                    if (0 <= y+ky && y+ky < size_input_y && 0 <= x+kx && x+kx < size_input_x) {
                        sum += acc_input[(y+ky)*size_input_x+(x+kx)]*acc_kernel[ky*size_kernel_x+ky];
                    }
                }
            }
            acc_output[idx] = sum;


        });

    });

}

#endif