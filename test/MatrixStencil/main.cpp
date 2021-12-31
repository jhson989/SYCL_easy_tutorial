#include <vector>
#include <cstdlib>
#include <algorithm>
#include <sys/time.h>

#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#include <matrixStencil.hpp>
#include <devceiProperties.hpp>
bool check_result(std::vector<float>& input, std::vector<float>& output, float* kernel);
float float_rand() {return (rand()%100-50)/100.0f;}

const size_t size_input[2] = {10*1024, 10*1024};
const size_t size_kernel[2] = {3, 3};
const size_t offset[2] = {1,1};
const size_t pad[2] = {1,1};
const size_t size_output[2] = {(size_input[0]-size_kernel[0]+2*pad[0])/offset[0] + 1, (size_input[1]-size_kernel[1]+2*pad[1])/offset[1] + 1};



int main(void) {

    // Explicitly context selection
    sycl::platform platform(sycl::gpu_selector{});
    sycl::device device = platform.get_devices(sycl::info::device_type::gpu)[0];
    sycl::context context(device);
    sycl::queue queue(context, device);

    // Querying device info.
    print_properties(queue);

    // Data initialization
    std::vector<float> input(size_input[0]*size_input[1]);
    std::generate(input.begin(), input.end(), float_rand);
    std::vector<float> output(size_output[0]*size_output[1], 0.0f);
    float kernel[] = {-1, -1, -1,
                      -1, +9, -1,
                      -1, -1, -1};
    
    // Run the kernel
    printf("==============================================================\n");
    printf("Matrix Stencil Operation\n");

    timeval st, ed;
    gettimeofday(&st, NULL);
    parallel_matrix_stencil(queue, input, output, kernel, size_input, size_output, size_kernel, offset, pad);
    gettimeofday(&ed, NULL);

    // Print the performance
    float time = (ed.tv_sec - st.tv_sec) + ((ed.tv_usec-st.tv_usec)*1e-6);
    printf("    -- Kernel run finished\n");
    printf("    -- Elapsed time: %.3f s\n", time);

    // Check the result correctness
    bool correctness = check_result(input, output, kernel);
    if (correctness == true) {
        printf("    -- result test succeeded\n");
    } else {
        printf("    [[ERR]] result check failed!\n");
    }

    return 0;
}

bool in_range(float pred, float gt) {
    return (gt-(1e-4) <= pred && pred <= gt+(1e-4));
}

bool check_result(std::vector<float>& input, std::vector<float>& output, float* kernel) {

    for (int i=0; i<size_output[0]; i++) {
        for (int j=0; j<size_output[1]; j++) {

            float sum = 0.0f;
            int y = i*offset[0]-pad[0];
            int x = j*offset[1]-pad[1];

            for (int ky=0; ky<size_kernel[0]; ky++) {
                for (int kx=0; kx<size_kernel[1]; kx++) {
                    if (0 <= y+ky && y+ky < size_input[0] && 0 <= x+kx && x+kx < size_input[1]) {
                        sum += input[(y+ky)*size_input[1]+(x+kx)]*kernel[ky*size_kernel[1]+ky];
                    }
                }
            }
            if (in_range(output[i*size_output[1]+j],sum) == false) {
                printf("    [[ERR]] output[%d,%d] is %f, but correct vaule is %f\n", i, j, output[i*size_output[1]+j],sum);
                return false;
            }
        }
    }

    return true;
}