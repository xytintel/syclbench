#pragma once

#include <CL/sycl.hpp>
#include <iostream>

template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_array {
    scalar_t val[vec_size];
};

void flush_cache(sycl::queue &queue) {
    uint32_t size = 1024 * 1024 * 1024;
    auto temp_ptr_cpu = new int[size];
    auto temp_ptr_xpu = sycl::aligned_alloc_device<int>(256, size, queue);
    queue.memcpy(temp_ptr_xpu, temp_ptr_cpu, size * sizeof(int)).wait();
    queue.memcpy(temp_ptr_cpu, temp_ptr_xpu, size * sizeof(int)).wait();
    delete[] temp_ptr_cpu;
    sycl::free(temp_ptr_xpu, queue);
}
