#ifndef __JH_DEVICE_PROPERTIES__
#define __JH_DEVICE_PROPERTIES__

#include <CL/sycl.hpp>

void print_properties_host();
void print_properties_cpu();
void print_properties_gpu();
void print_properties(sycl::queue& queue);

#endif