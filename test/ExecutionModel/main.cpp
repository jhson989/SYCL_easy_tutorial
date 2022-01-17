#include <vector>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <algorithm>

#include <CL/sycl.hpp>
namespace sycl = cl::sycl;
#include <deviceProperties.hpp>
#include "./include/basicKernel.hpp"
#include "./include/NDRangeKernel.hpp"

//#define DEBUG_MODE
#define DTYPE float
timeval start, end;
#define ELAPSED_TIME(st, ed) ((ed.tv_sec - st.tv_sec) + ((ed.tv_usec-st.tv_usec)*1e-6))

const size_t M=1024, N=1024, K=1024;

void check_result(const std::vector<DTYPE>& A, const std::vector<DTYPE>& B, const std::vector<DTYPE>& C);

int main(void) {

    // Run the kernel
    std::cout << "==============================================================\n";
    std::cout << "Execution Model Performance Test with a Signle GPU Device\n";
    std::cout << "Matrix Multiplication: C[M,N] = A[M,k] * B[K,N], where M=" << M << " N=" << N << " K=" << K << "\n";
    std::cout << "Total Memory Size : " << sizeof(DTYPE)*(M*N+M*K+K*N)/1024.0f/1024/1024 << " GB\n";
    
    sycl::gpu_selector selector;
    sycl::queue queue(selector);
    // Querying device info.
    print_properties(queue);

    // Data Initialization
    std::vector<DTYPE> A(M*K);
    std::vector<DTYPE> B(K*N);
    std::vector<DTYPE> C(M*N);
    std::generate(A.begin(), A.end(), [](){return (rand()%100-50)/10;});
    std::generate(B.begin(), B.end(), [](){return (rand()%100-50)/10;});
    
    /**************************************************************************************
     * Basic kernel
     ***/
    std::cout << "Basic kernel launched\n";
    gettimeofday(&start, NULL);
    // TODO
    gettimeofday(&end, NULL);
    std::cout << "--- Elapsed time : " << ELAPSED_TIME(start, end) << "s\n\n";

    #ifdef DEBUG_MODE
    check_result(A, B, C);
    #endif

    /**************************************************************************************
     * ND range kernel
     ***/
    std::cout << "ND range kernel launched\n";
    gettimeofday(&start, NULL);
    // TODO
    gettimeofday(&end, NULL);
    std::cout << "--- Elapsed time : " << ELAPSED_TIME(start, end) << "s\n\n";

    #ifdef DEBUG_MODE
    check_result(A, B, C);
    #endif

    // Hierarchy kernel
    // TODO


    return 0;
}


void check_result(const std::vector<DTYPE>& A, const std::vector<DTYPE>& B, const std::vector<DTYPE>& C) {

    for (size_t m=0; m<M; m++) {
        for (size_t n=0; n<N; n++) {
            DTYPE sum = 0;
            for (size_t k=0; k<K; k++) {
                sum += A[m*K+k] * B[k*N+n];
            }
            if (sum != C[m*N+n]) {
                std::cout << "[[[ERROR]]] Checking the result fails at [" << m << "," << n << "]\n";
                std::cout << "--- Expected : " << sum << ", but result : " << C[m*N+n] << std::endl;
                return;
            }
        }
    }
    std::cout << "--- Checking the result succeed!!! \n";
}