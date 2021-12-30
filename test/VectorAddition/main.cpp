#include <vector>
#include <cstdlib>
#include <sys/time.h>

#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#include <vectorAddition.hpp>
#include <devceiProperties.hpp>

const size_t num_data = 1<<25;


int main(void) {

    timeval st, ed;
    

    // Explicitly context selection
    sycl::platform platform(sycl::gpu_selector{});
    sycl::device device = platform.get_devices(sycl::info::device_type::gpu)[0];
    sycl::context context(device);
    sycl::queue queue(context, device);

    print_properties(queue);

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


    gettimeofday(&st, NULL);
    parallel_add(queue, inA, inB, out);
    gettimeofday(&ed, NULL);

    float time = (ed.tv_sec - st.tv_sec) + ((ed.tv_usec-st.tv_usec)*1e-6);
    printf("kernel run finished\n. Elapsed time: %.3f s\n", time);
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