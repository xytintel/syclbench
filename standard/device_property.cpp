#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <assert.h>
using namespace std;

// ref: https://github.com/OpenSYCL/OpenSYCL/blob/b1f06a404741b8329d6e1ab5b634cbf960c5a78c/include/hipSYCL/sycl/info/device.hpp#L97

#define PRINT_PROP(PARAM) std::cout << #PARAM << ": " << device.get_info<sycl::info::device::PARAM>() << std::endl;
#define PRINT_PROP_VEC(PARAM)                                  \
    {                                                          \
        auto p = device.get_info<sycl::info::device::PARAM>(); \
        std::cout << #PARAM << ": ";                           \
        for (auto x : p) std::cout << x << ", ";               \
        std::cout << std::endl;                                \
    }

int main() {
    auto platforms = sycl::platform::get_platforms();
    int count = 0;
    for (const auto &p : platforms) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero)
            continue;
        auto device_list = p.get_devices();
        for (const auto &device : device_list) {
            if (device.is_gpu()) {
                std::cout << "[" << count << "] " << device.get_info<sycl::info::device::name>() << std::endl;
                PRINT_PROP(vendor)
                PRINT_PROP(vendor_id)
                PRINT_PROP(driver_version)
                PRINT_PROP(version)

                PRINT_PROP(max_compute_units)
                PRINT_PROP(max_work_item_dimensions)
                PRINT_PROP(max_work_group_size)
                PRINT_PROP(sub_group_independent_forward_progress)

                PRINT_PROP_VEC(sub_group_sizes)

                PRINT_PROP(preferred_vector_width_char)
                PRINT_PROP(native_vector_width_char)

                PRINT_PROP(max_clock_frequency)
                PRINT_PROP(address_bits)
                PRINT_PROP(max_mem_alloc_size)

                PRINT_PROP(global_mem_cache_line_size)
                PRINT_PROP(global_mem_cache_size)
                PRINT_PROP(global_mem_size)

                PRINT_PROP(local_mem_size)

                PRINT_PROP(error_correction_support)
                PRINT_PROP(profiling_timer_resolution)
                PRINT_PROP(is_available)

                std::cout << std::endl;
            }
        }
    }
}
