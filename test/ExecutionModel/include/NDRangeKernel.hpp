#ifndef __JH_ND_RANGE_KERNEL__
#define __JH_ND_RANGE_KERNEL__

namespace NDRange {

    template <typename T>
    void matrix_multiplication(sycl::queue queue, std::vector<T>& A, std::vector<T>& B, std::vector<T>& C, const size_t M, const size_t N, const size_t K, size_t wg_size=256) {

        T* device_A = sycl::malloc_device<T>(A.size(), queue);
        T* device_B = sycl::malloc_device<T>(B.size(), queue);
        T* device_C = sycl::malloc_device<T>(C.size(), queue);

        // Copy input A from Host to Device
        queue.submit([&] (sycl::handler& cgh) {
            cgh.memcpy(device_A, &A[0], A.size()*sizeof(T));
        });
        // Copy input B from Host to Device
        queue.submit([&] (sycl::handler& cgh) {
            cgh.memcpy(device_B, &B[0], B.size()*sizeof(T));
        });

        // Launch the matnul kernel
        queue.wait();
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::nd_range<1>(M*N, wg_size), [=](sycl::nd_item<1> idx) {
                int m = idx.get_global_id() / N;
                int n = idx.get_global_id() % N;
                T sum = 0;
                for (auto k=0; k<K; k++) {
                    sum += device_A[m*K+k] * device_B[k*N+n];
                }
                device_C[m*N+n] = sum;
            });
        });

        // Copy result C from Device to Host
        queue.wait();
        queue.submit([&](sycl::handler& cgh) {
            cgh.memcpy(&C[0], device_C, C.size()*sizeof(T));
        });
        queue.wait();

        sycl::free (device_A, queue);
        sycl::free (device_B, queue);
        sycl::free (device_C, queue);
    }

}


#endif