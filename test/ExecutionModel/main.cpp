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


const size_t M=1024*18+13, N=1024*19+1, K=1024*18+7;

void check_result(const std::vector<DTYPE>& A, const std::vector<DTYPE>& B, const std::vector<DTYPE>& C);

int main(void) {
    srand(time(NULL));
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
    std::vector<DTYPE> C(M*N, 0);
    std::generate(A.begin(), A.end(), [](){return (rand()%100-50);});
    std::generate(B.begin(), B.end(), [](){return (rand()%100-50);});
    
    /**************************************************************************************
     * Basic kernel
     ***/
    std::cout << "\nBasic kernel launched\n";
    gettimeofday(&start, NULL);
    basic::matrix_multiplication(queue, A, B, C, M, N, K);
    gettimeofday(&end, NULL);
    std::cout << "--- Elapsed time : " << ELAPSED_TIME(start, end) << "s\n";

    #ifdef DEBUG_MODE
    check_result(A, B, C);
    #endif

    
    /**************************************************************************************
     * ND range kernel
     ***/
    std::cout << "\nND range kernel launched\n";
    gettimeofday(&start, NULL);
    NDRange::matrix_multiplication(queue, A, B, C, M, N, K);
    gettimeofday(&end, NULL);
    std::cout << "--- Elapsed time : " << ELAPSED_TIME(start, end) << "s\n";

    #ifdef DEBUG_MODE
    check_result(A, B, C);
    #endif


    /**************************************************************************************
     * ND range kernel - GPU optimized version
     ***/
    std::cout << "\nND range kernel launched - GPU optimized version\n";
    gettimeofday(&start, NULL);
    NDRange::matrix_multiplication_gpu_optimized(queue, A, B, C, M, N, K, 16);
    gettimeofday(&end, NULL);
    std::cout << "--- Elapsed time : " << ELAPSED_TIME(start, end) << "s\n";

    #ifdef DEBUG_MODE
    check_result(A, B, C);
    #endif


    /**************************************************************************************
     * Hierarchical kernel
     ***/
    // TODO


    return 0;
}


inline bool IN_RANGE(DTYPE gt, DTYPE t) {return( gt-(0.001)<=t && t <= gt+(0.001) );}

void check_result(const std::vector<DTYPE>& A, const std::vector<DTYPE>& B, const std::vector<DTYPE>& C) {
    bool correct = true;
    #pragma omp parallel for num_threads(8)
    for (auto m=0; m<M; m++) {
        for (auto n=0; n<N&&correct; n++) {
            DTYPE sum = 0;
            for (auto k=0; k<K; k++) {
                sum += A[m*K+k] * B[k*N+n];
            }
            if (IN_RANGE(sum, C[m*N+n]) == false) {
                #pragma omp critical
                {
                    std::cout << "[[[ERROR]]] Checking the result fails at [" << m << "," << n << "]\n";
                    std::cout << "--- Expected : " << sum << ", but result : " << C[m*N+n] << std::endl;
                    correct = false;
                }
            }
        }
    }

    if (correct)
        std::cout << "--- Checking the result succeed!!! \n";
    return;
}