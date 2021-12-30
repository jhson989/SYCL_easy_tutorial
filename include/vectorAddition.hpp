#ifndef __JH_VECTOR_ADDITION__
#define __JH_VECTOR_ADDITION__

#include <CL/sycl.hpp>
namespace sycl = cl::sycl;


template <typename T>
void parallel_vector_addition(sycl::queue queue, std::vector<T>& inA, std::vector<T>& inB, std::vector<T>& out) {

    // Define device buffers
    sycl::buffer<T, 1> bufA(inA.data(), out.size());
    sycl::buffer<T, 1> bufB(inB.data(), out.size());
    sycl::buffer<T, 1> bufOut(out.data(), out.size());

    queue.submit([&] (sycl::handler & cgh) {

        // Define accessors
        auto accA = bufA.template get_access<sycl::access::mode::read>(cgh);
        auto accB = bufB.template get_access<sycl::access::mode::read>(cgh);
        auto accOut = bufOut.template get_access<sycl::access::mode::write>(cgh);

        // Kernel submission
        cgh.parallel_for(sycl::range<1>(out.size()), [=] (sycl::id<1> idx) {
            accOut[idx] = accA[idx] + accB[idx];
        });
    });

}

#endif