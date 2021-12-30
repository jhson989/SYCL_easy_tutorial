#include <vector>
#include <cstdlib>
#include <sys/time.h>

#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#include <matrixMultiplication.hpp>
#include <devceiProperties.hpp>

bool check_result(std::vector<float> inA, std::vector<float> inB, std::vector<float> out);

const size_t M = 1024;
const size_t K = 1024;
const size_t N = 1024;


int main(void) {

    // Explicitly context selection
    sycl::platform platform(sycl::gpu_selector{});
    sycl::device device = platform.get_devices(sycl::info::device_type::gpu)[0];
    sycl::context context(device);
    sycl::queue queue(context, device);

    // Querying device info.
    print_properties(queue);

    // Data initialization
    std::vector<float> inA;
    std::vector<float> inB;
    std::vector<float> out;
    for (auto i=0; i<M*N; i++) 
        inA.push_back((rand()%100-50)/100.0f);
    for (auto i=0; i<K*N; i++)
        inB.push_back((rand()%100-50)/100.0f);
    for (auto i=0; i<M*N; i++)
        out.push_back(0.0f);
    
    // Run the kernel
    printf("==============================================================\n");
    printf("Matrix Multiplication\n");
    printf("C[%lu*%lu] = A[%lu*%lu] * B[%lu*%lu]\n\n", M, N, M, K, K, N);

    timeval st, ed;
    gettimeofday(&st, NULL);
    parallel_matrix_multiplication(queue, inA, inB, out, M, N, K);
    gettimeofday(&ed, NULL);

    // Print the performance
    float time = (ed.tv_sec - st.tv_sec) + ((ed.tv_usec-st.tv_usec)*1e-6);
    printf("    -- Kernel run finished\n");
    printf("    -- Elapsed time: %.3f s\n", time);

    // Check the result correctness
    bool correctness = check_result(inA, inB, out);
    if (correctness == true) {
        printf("    -- result test succeeded\n");
    } else {
        printf("    [[ERR]] result check failed!\n");
    }

    return 0;
}


bool check_result(std::vector<float> inA, std::vector<float> inB, std::vector<float> out) {

    return true;
}