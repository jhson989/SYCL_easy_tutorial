#include <vector>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <algorithm>

#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#include <reduction.hpp>
#include <deviceProperties.hpp>

#define TYPE long long


timeval st, ed;
const size_t num_data = (1.7)*(2<<28)+121;

int main(void) {

    // Run the kernel
    printf("==============================================================\n");
    printf("Parallel reduction\n");
    printf("Result = A[0] + A[1] + ... + A[%lu], total memory : %.4f GB\n", num_data-1, num_data*sizeof(TYPE)/1024.0/1024.0/1024.0);

    
    // Explicitly context selection
    sycl::platform platform(sycl::gpu_selector{});
    sycl::device device = platform.get_devices(sycl::info::device_type::gpu)[0];
    sycl::context context(device);
    sycl::queue queue(context, device);

    // Querying device info.
    print_properties(queue);

    // Data initialization
    std::vector<TYPE> A(num_data);
    std::generate(A.begin(), A.end(), [](){return (rand());});

    // Calculate the ground truth.
    printf("    -- CPU sequential run\n");
    TYPE answer = 0.0f;
    gettimeofday(&st, NULL);
    for (auto i=0; i<num_data; i++) {answer += A[i];}
    gettimeofday(&ed, NULL);
    printf("    -- Elapsed time: %.3f s\n", (ed.tv_sec - st.tv_sec) + ((ed.tv_usec-st.tv_usec)*1e-6));


    // Launch the kernel
    printf("    -- Kernel run\n");
    gettimeofday(&st, NULL);
    parallel_reduction(queue, A);
    gettimeofday(&ed, NULL);
    printf("    -- Elapsed time: %.3f s\n", (ed.tv_sec - st.tv_sec) + ((ed.tv_usec-st.tv_usec)*1e-6));


    // Check the result correctness
    if (A[0] != answer) {
        std::cout << "    [[ERR]] result check failed!, the answer is " << answer << " but result of reduction is " << A[0] << std::endl;
    }
    else {
        std::cout << "    -- Result test succeeded, the answer is "<< answer << " and result of reduction is " << A[0] << std::endl;
    }
    
    return 0;
}