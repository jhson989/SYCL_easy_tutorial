#ifndef __JH_REDUCTION__
#define __JH_REDUCTION__

#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

template<typename T>
void parallel_reduction(sycl::queue queue, std::vector<T>& A, int per_workitem=256) {

    size_t num_total = A.size();
    {
        sycl::buffer<T, 1> bufA(A.data(), A.size());

        for (size_t num_workitem=(num_total+per_workitem-1)/per_workitem; num_workitem>256; num_workitem=(num_total+per_workitem-1)/per_workitem) {

            queue.submit([&] (sycl::handler& cgh) {
                auto accA = bufA.template get_access<sycl::access::mode::read_write>(cgh);

                cgh.parallel_for(sycl::range<1>(num_workitem), [=] (sycl::id<1> idx) {
                    T result = 0.0f;
                    for (size_t i=0; i<per_workitem; i++) {
                        if (idx+num_workitem*i < num_total){
                            result += accA[idx+num_workitem*i];
                        }
                            
                    }
                    accA[idx] = result;
                });

            });
            num_total = num_workitem;
        }
    }
    T result = 0.0f;
    for (size_t i=0; i<num_total; i++)
        result += A[i];
    A[0] = result;
}



#endif