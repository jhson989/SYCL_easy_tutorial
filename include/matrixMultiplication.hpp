#ifndef __JH_MATRIX_MULTIPLICATION__
#define __JH_MATRIX_MULTIPLICATION__

#include <CL/sycl.hpp>
namespace sycl = cl::sycl;



template <typename T>
void parallel_matrix_multiplication(sycl::queue queue, std::vector<T>& inA, std::vector<T>& inB, std::vector<T>& out, const size_t M, const size_t N, const size_t K) {

    // Define device buffers
    sycl::buffer<T, 1> bufA(inA.data(), inA.size());
    sycl::buffer<T, 1> bufB(inB.data(), inB.size());
    sycl::buffer<T, 1> bufOut(out.data(), out.size());

    queue.submit([&] (sycl::handler& cgh) {

        // Define accessors
        auto accA = bufA.template get_access<sycl::access::mode::read>(cgh);
        auto accB = bufB.template get_access<sycl::access::mode::read>(cgh);
        auto accOut = bufOut.template get_access<sycl::access::mode::write>(cgh);

        // Kernel submission
        cgh.parallel_for(sycl::range<1>(out.size()), [=] (sycl::id<1> idx) {
            size_t x=idx%N, y=idx/N;
            float sum = 0;
            for (auto k=0; k<K; k++) {
                sum += accA[y*K+k] * accB[k*N+x];
            }
            accOut[y*N+x] = sum;
        });
        

    });
    
}

#endif

