#include <iostream>
#include <cstdio>
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

void print_properties_host() {
    
    sycl::host_selector host;
    sycl::queue host_q(host);

    sycl::device dev = host_q.get_device();
    std::cout << "=============== Device Properties ==============" << std::endl;
    std::cout << "Name: " << dev.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Vendor: " << dev.get_info<sycl::info::device::vendor>() << std::endl;
    std::cout << "Memory size: " << dev.get_info<sycl::info::device::global_mem_size>()/1024.0f/1024.0f/1024.0f << " GB"  << std::endl;
    std::cout << "================================================" << std::endl << std::endl;
    
}

void print_properties_gpu() {

    sycl::gpu_selector gpu;
    sycl::queue gpu_q(gpu);
        
    sycl::device dev = gpu_q.get_device();
    std::cout << "=============== Device Properties ==============" << std::endl;
    std::cout << "Name: " << dev.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Vendor: " << dev.get_info<sycl::info::device::vendor>() << std::endl;
    std::cout << "Memory size: " << dev.get_info<sycl::info::device::global_mem_size>()/1024.0f/1024.0f/1024.0f << " GB" << std::endl;
    std::cout << "================================================" << std::endl << std::endl;

}

void print_properties(sycl::queue& queue) {

    sycl::device dev = queue.get_device();

    std::cout << "=============== Device Properties ==============" << std::endl;
    std::cout << "Name: " << dev.get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Vendor: " << dev.get_info<sycl::info::device::vendor>() << std::endl;
    std::cout << "Memory size: " << dev.get_info<sycl::info::device::global_mem_size>()/1024.0f/1024.0f/1024.0f << " GB"  << std::endl;
    std::cout << "================================================" << std::endl << std::endl;
}

