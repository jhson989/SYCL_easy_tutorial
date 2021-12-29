    #include <vector>
#include <cstdlib>

#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

const size_t num_data = 1<<25;

template <typename T>
void parallel_add(sycl::queue queue, std::vector<T>& inA, std::vector<T>& inB, std::vector<T>& out) {

    sycl::buffer<T, 1> bufA(inA.data(), out.size());
    sycl::buffer<T, 1> bufB(inB.data(), out.size());
    sycl::buffer<T, 1> bufOut(out.data(), out.size());

    queue.submit([&] (sycl::handler & cgh) {

        auto accA = bufA.template get_access<sycl::access::mode::read>(cgh);
        auto accB = bufB.template get_access<sycl::access::mode::read>(cgh);
        auto accOut = bufOut.template get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for(sycl::range<1>(out.size()), [=] (sycl::id<1> idx) {
            accOut[idx] = accA[idx] + accB[idx];
        });
    });
}


int main(void) {

    // Explicitly context selection
    sycl::platform platform(sycl::gpu_selector{});
    sycl::device device = platform.get_devices(sycl::info::device_type::gpu)[0];
    sycl::context context(device);
    sycl::queue queue(context, device);

    // Data initialization
    std::vector<float> inA;
    std::vector<float> inB;
    std::vector<float> out;
    for (auto i=0; i<num_data; i++) {
        inA.push_back(rand()%1000);
        inB.push_back(rand()%1000);
        out.push_back(0.0f);
    }


    // Run kernel
    printf("==============================================================\n");
    printf("Vector addition, num_data = %lu\n", num_data);
    printf("==============================================================\n");

    parallel_add(queue, inA, inB, out);

    printf("kernel run finished\n");

    // Result test
    for (auto i=0; i<num_data; i++) {
        if (out[i] != inA[i] + inB[i]) {
            printf("Result test failed!\n");
            exit(1);
        }
    }
    printf("Result test succeeded\n");
    
    return 0;
}