#include <vector>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <algorithm>

#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#include <reduction.hpp>
#include <deviceProperties.hpp>

//#define DEBUG_MODE
#define DTYPE unsigned long long
#define ELAPSED_TIME(st, ed) ((ed.tv_sec - st.tv_sec) + ((ed.tv_usec-st.tv_usec)*1e-6))

timeval start, end;
const size_t num_data = (1<<29)*1.5;
const size_t num_iter = 10;

void execute_USM_device(sycl::queue&, std::vector<DTYPE>&, std::vector<DTYPE>&);
void execute_USM_host(sycl::queue&, std::vector<DTYPE>&, std::vector<DTYPE>&);
void execute_USM_shared(sycl::queue&, std::vector<DTYPE>&, std::vector<DTYPE>&);
void execute_buffer(sycl::queue&, std::vector<DTYPE>&, std::vector<DTYPE>&);
void check_result(std::vector<DTYPE>& out, size_t, int);

int main(void) {

    // Run the kernel
    printf("==============================================================\n");
    printf("Memory Management Performance Test with a GPU Device\n");
    printf("1. Explicit data transfer\n");
    printf("--- Unified Shared Memory - Device-sided memory\n");
    printf("2. Implicit data transfer\n");
    printf("--- Unified Shared Memory - Host-sided memory\n");
    printf("--- Unified Shared Memory - Shared memory\n");
    printf("--- Buffer\n");
    printf("Total transferred memory size (%lu iterations) : %.4f GB\n", num_iter, 2*(num_data*sizeof(DTYPE)/1024.0/1024.0/1024.0)*num_iter);
    printf("--- Memcpy Host to Device per iteration: %.4f GB\n", (num_data*sizeof(DTYPE)/1024.0/1024.0/1024.0));
    printf("--- Memcpy Device to Host per iteration: %.4f GB\n", (num_data*sizeof(DTYPE)/1024.0/1024.0/1024.0));

    sycl::gpu_selector selector;
    sycl::queue queue(selector);

    // Querying device info.
    print_properties(queue);

    // Data initialization
    std::vector<DTYPE> in(num_data, 0);
    std::vector<DTYPE> out(num_data);

    execute_USM_device(queue, in, out);
    execute_USM_host(queue, in, out);
    execute_USM_shared(queue, in, out);
    execute_buffer(queue, in, out);




    return 0;
}


void execute_USM_device(sycl::queue& queue, std::vector<DTYPE>& in, std::vector<DTYPE>& out) {

    DTYPE* device_mem = sycl::malloc_device<DTYPE>(num_data, queue);
    printf("[USM-Device] Performance measure start...\n");
    gettimeofday(&start, NULL);
    for (int i=0; i<num_iter; i++) {

        queue.submit([&] (sycl::handler& cgh) {
            cgh.memcpy(device_mem, &in[0], num_data*sizeof(DTYPE));
        });
        queue.wait();

        queue.submit([&] (sycl::handler& cgh) {
            cgh.parallel_for(num_data, [=](sycl::id<1> idx) {
                device_mem[idx] = idx * i;
            });
        });
        queue.wait();

        queue.submit([&] (sycl::handler& cgh) {
            cgh.memcpy(&out[0], device_mem, num_data*sizeof(DTYPE));
        });
        queue.wait();
        
        #ifdef DEBUG_MODE
        check_result(out, i, __LINE__);
        #endif

    }
    gettimeofday(&end, NULL);
    printf("--- Average elasped time: %.4f s (total: %.4f s) \n\n", ELAPSED_TIME(start, end)/num_iter, ELAPSED_TIME(start, end));

    free (device_mem ,queue);

}

void execute_USM_host(sycl::queue& queue, std::vector<DTYPE>& in, std::vector<DTYPE>& out) {

    DTYPE* host_mem = sycl::malloc_host<DTYPE>(num_data, queue);

    printf("[USM-Host] Performance measure start...\n");
    gettimeofday(&start, NULL);
    for (int i=0; i<num_iter; i++) {

        queue.submit([&] (sycl::handler& cgh) {
            cgh.memcpy(host_mem, &in[0], num_data*sizeof(DTYPE));
        });
        queue.wait();

        queue.submit([&] (sycl::handler& cgh) {
            cgh.parallel_for(num_data, [=](sycl::id<1> idx) {
                host_mem[idx] = idx * i;
            });
        });
        queue.wait();

        queue.submit([&] (sycl::handler& cgh) {
            cgh.memcpy(&out[0], host_mem, num_data*sizeof(DTYPE));
        });
        queue.wait();

        #ifdef DEBUG_MODE
        check_result(out, i, __LINE__);
        #endif

    }
    gettimeofday(&end, NULL);
    printf("--- Average elasped time: %.4f s (total: %.4f s) \n\n", ELAPSED_TIME(start, end)/num_iter, ELAPSED_TIME(start, end));

    free (host_mem, queue);

}


void execute_USM_shared(sycl::queue& queue, std::vector<DTYPE>& in, std::vector<DTYPE>& out) {

    DTYPE* shared_mem = sycl::malloc_shared<DTYPE>(num_data, queue);

    printf("[USM-Shared] Performance measure start...\n");
    gettimeofday(&start, NULL);
    for (int i=0; i<num_iter; i++) {

        queue.submit([&] (sycl::handler& cgh) {
            cgh.memcpy(shared_mem, &in[0], num_data*sizeof(DTYPE));
        });
        queue.wait();

        queue.submit([&] (sycl::handler& cgh) {
            cgh.parallel_for(num_data, [=](sycl::id<1> idx) {
                shared_mem[idx] = idx * i;
            });
        });
        queue.wait();

        queue.submit([&] (sycl::handler& cgh) {
            cgh.memcpy(&out[0], shared_mem, num_data*sizeof(DTYPE));
        });
        queue.wait();

        #ifdef DEBUG_MODE
        check_result(out, i, __LINE__);
        #endif

    }
    gettimeofday(&end, NULL);
    printf("--- Average elasped time: %.4f s (total: %.4f s) \n\n", ELAPSED_TIME(start, end)/num_iter, ELAPSED_TIME(start, end));

    free (shared_mem, queue);

}

void execute_buffer(sycl::queue& queue, std::vector<DTYPE>& in, std::vector<DTYPE>& out) {

    printf("[Buffer] Performance measure start...\n");
    gettimeofday(&start, NULL);
    for (int i=0; i<num_iter; i++) {

        {
            sycl::buffer<DTYPE, 1> buffer_in(in.data(), in.size());
            queue.submit([&] (sycl::handler& cgh) {
                auto acc_in = buffer_in.template get_access<sycl::access::mode::read_write>(cgh);
                cgh.parallel_for(num_data, [=](sycl::id<1> idx) {
                    acc_in[idx] = idx * i;
                });
            });
            queue.wait();
        }

        #ifdef DEBUG_MODE
        check_result(in, i, __LINE__);
        #endif

    }
    gettimeofday(&end, NULL);
    printf("--- Average elasped time: %.4f s (total: %.4f s) \n\n", ELAPSED_TIME(start, end)/num_iter, ELAPSED_TIME(start, end));
}

void check_result(std::vector<DTYPE>& out, size_t iter, int line) {

    #pragma omp parallel for num_threads(8)
    for (size_t i=0; i<out.size(); i++) {
        if (out[i] != ((DTYPE)i)*iter) {
            printf("[[[ERROR]]] the output is not correct in function at %d (%llu, %llu)\n", line, out[i], ((DTYPE)i)*iter);
            exit(1);
        }
    }

    printf("[[[Debug]]] %lu/%lu : the output is correct\n", iter, num_iter);
}