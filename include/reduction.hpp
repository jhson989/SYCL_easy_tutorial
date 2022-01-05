#ifndef __JH_REDUCTION__
#define __JH_REDUCTION__

#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

template<typename T>
void parallel_reduction(sycl::queue queue, std::vector<T>& A, int per_workitem=256, int wgroup_size=256) {

    size_t num_total = A.size();
    {
        sycl::buffer<T, 1> bufA(A.data(), A.size());

        for (size_t num_workitem=(num_total+per_workitem-1)/per_workitem; num_workitem>256; num_workitem=(num_total+per_workitem-1)/per_workitem) {

            queue.submit([&] (sycl::handler& cgh) {
                auto accA = bufA.template get_access<sycl::access::mode::read_write>(cgh);
                sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local> local_mem(sycl::range<1>(wgroup_size), cgh);

                cgh.parallel_for(sycl::nd_range<1>(((num_workitem+wgroup_size-1)/wgroup_size)*wgroup_size, wgroup_size), [=] (sycl::nd_item<1> item) {
                    size_t global_id = item.get_global_id(0);
                    size_t local_id = item.get_local_id(0);
                    local_mem[local_id] = accA[global_id];
                    for (size_t i=1; i<per_workitem; i++) {
                        if (global_id+num_workitem*i < num_total){
                            local_mem[local_id] += accA[global_id+num_workitem*i];
                        }
                            
                    }
                    accA[global_id] = local_mem[local_id];
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