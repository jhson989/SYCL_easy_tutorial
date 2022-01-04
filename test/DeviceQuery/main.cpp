#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#include <deviceProperties.hpp>


int main(void) {

    
    // Print host device info
    print_properties_host();

    // Print CPU device info
    print_properties_cpu();

    // Print GPU device info
    print_properties_gpu();

    // Print explicitly selected device info
    sycl::platform platform(sycl::gpu_selector{});
    sycl::device device = platform.get_devices(sycl::info::device_type::gpu)[0];
    sycl::context context(device);
    sycl::queue queue(context, device);
    print_properties(queue);    


    return 0;
}