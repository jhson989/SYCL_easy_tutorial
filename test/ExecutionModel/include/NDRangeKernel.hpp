#ifndef __JH_ND_RANGE_KERNEL__
#define __JH_ND_RANGE_KERNEL__

namespace NDRange {

    template <typename T>
    void matrix_multiplication(sycl::queue queue, std::vector<T>& A, std::vector<T>& B, std::vector<T>& C, const size_t M, const size_t N, const size_t K, size_t wg_size=16) {

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

            cgh.parallel_for(sycl::nd_range<2>({(M+wg_size-1)/wg_size*wg_size, (N+wg_size-1)/wg_size*wg_size}, {wg_size, wg_size}), [=](sycl::nd_item<2> idx) {
                int m = idx.get_global_id(0);
                int n = idx.get_global_id(1);
                
                T sum = 0;
                for (auto k=0; k<K; k++) {
                    sum += device_A[m*K+k] * device_B[k*N+n];
                }
                if (m<M && n<N)
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

    template <typename T>
    void matrix_multiplication_gpu_optimized(sycl::queue queue, std::vector<T>& A, std::vector<T>& B, std::vector<T>& C, const size_t M, const size_t N, const size_t K, size_t wg_size=16) {

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

            sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local> local_A(sycl::range<1>(wg_size*wg_size), cgh);
            sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local> local_B(sycl::range<1>(wg_size*wg_size), cgh);

            cgh.parallel_for(sycl::nd_range<2>({(M+wg_size-1)/wg_size*wg_size, (N+wg_size-1)/wg_size*wg_size}, {wg_size, wg_size}), [=](sycl::nd_item<2> idx) {

                int m = idx.get_global_id(0);
                int n = idx.get_global_id(1);
                int lm = idx.get_local_id(0);
                int ln = idx.get_local_id(1);

                T sum = 0;

                int num_tiles = (K+wg_size-1)/wg_size;
                for (int tile=0; tile<num_tiles; tile++) {
                    if (ln+tile*wg_size < K)
                        local_A[lm*wg_size+ln] = device_A[(m)*K+(ln+tile*wg_size)];
                    else
                        local_A[lm*wg_size+ln] = 0;
                    if (lm+tile*wg_size < K)
                        local_B[lm*wg_size+ln] = device_B[(lm+tile*wg_size)*N+n];
                    else
                        local_B[lm*wg_size+ln] = 0;

                    idx.barrier(sycl::access::fence_space::local_space);
                    
                    for (auto lk=0; lk<wg_size; lk++) {
                        sum += local_A[lm*wg_size+lk] * local_B[lk*wg_size+ln];
                    }
                    idx.barrier(sycl::access::fence_space::local_space);
                }
                if (m<M && n<N)
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
