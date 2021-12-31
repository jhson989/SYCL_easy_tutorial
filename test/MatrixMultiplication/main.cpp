#include <vector>
#include <cstdlib>
#include <algorithm>
#include <sys/time.h>

#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#include <matrixMultiplication.hpp>
#include <deviceProperties.hpp>

bool check_result(std::vector<float>& inA, std::vector<float>& inB, std::vector<float>& out);

const size_t M = 5*1024;
const size_t K = 8*1024;
const size_t N = 7*1024;


int main(void) {

    // Explicitly context selection
    sycl::platform platform(sycl::gpu_selector{});
    sycl::device device = platform.get_devices(sycl::info::device_type::gpu)[0];
    sycl::context context(device);
    sycl::queue queue(context, device);

    // Querying device info.
    print_properties(queue);

    // Data initialization
    std::vector<float> inA(M*K);
    std::vector<float> inB(K*N);
    std::vector<float> out(M*N, 0.0f);
    for (auto i=0; i<M*K; i++) 
        inA.push_back((rand()%100-50)/100.0f);
    for (auto i=0; i<K*N; i++)
        inB.push_back((rand()%100-50)/100.0f);
    
    // Run the kernel
    printf("==============================================================\n");
    printf("Matrix Multiplication\n");
    printf("C[%lu,%lu] = A[%lu,%lu] * B[%lu,%lu]\n", M, N, M, K, K, N);

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

inline bool in_range(float pred, float gt) {
    return (gt-(1e-4) <= pred && pred <= gt+(1e-4));
}

bool check_result(std::vector<float>& inA, std::vector<float>& inB, std::vector<float>& out) {
    for (size_t m=0; m<M; m++) {
        for (size_t n=0; n<N; n++) {
            float sum = 0.0f;
            for (size_t k=0; k<K; k++) {
                sum += inA[m*K+k] * inB[k*N+n];
            }
            if (in_range(out[m*N+n], sum) == false) {
                printf("    [[ERR]] out[%lu,%lu] is %f, but correct vaule is %f\n", m, n, out[m*N+n], sum);
                return false;
            }
                
        }
    }

    return true;
}