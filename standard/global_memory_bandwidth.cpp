#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <assert.h>
#include "utils.h"
using namespace std;

template <typename T, int vec_size, typename item_t>
inline void threads_copy_kernel(item_t &item, const T *in, T *out, const size_t n) {
    const int group_work_size = item.get_local_range(0) * vec_size;
    auto index = item.get_group(0) * group_work_size + item.get_local_id(0) * vec_size;
    auto remaining = n - index;
    if (remaining < vec_size) {
        for (auto i = index; i < n; i++) {
            out[i] = in[i];
        }
    } else {
        using vec_t = aligned_array<T, vec_size>;
        auto in_vec = reinterpret_cast<vec_t *>(const_cast<T *>(&in[index]));
        auto out_vec = reinterpret_cast<vec_t *>(&out[index]);
        *out_vec = *in_vec;
    }
}

template <typename T, int vec_size>
float threads_copy(sycl::queue &queue, const T *in, T *out, size_t n) {
    const int group_size = 1024;
    const int group_work_size = group_size * vec_size;
    const int num_groups = (n + group_work_size - 1) / group_work_size;
    auto event = queue.submit([&](sycl::handler &h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_groups * group_size), sycl::range<1>(group_size)),
            [=](sycl::nd_item<1> item) {
                threads_copy_kernel<T, vec_size>(item, in, out, n);
            });
    });
    event.wait();
    return timeit(event);
}

template <int vec_size>
void test_threads_copy(size_t n) {
    sycl::queue queue(sycl::gpu_selector_v, cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()});

    auto in_cpu = new float[n];
    auto out_cpu = new float[n];
    for (int i = 0; i < n; i++)
        in_cpu[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    auto in_xpu = sycl::aligned_alloc_device<float>(512, n, queue);
    auto out_xpu = sycl::aligned_alloc_device<float>(512, n, queue);
    queue.memcpy(in_xpu, in_cpu, n * sizeof(float)).wait();

    float timems;
    for (int i = 0; i < 300; i++)
        timems = threads_copy<float, vec_size>(queue, in_xpu, out_xpu, n);

    float total_GBytes = (n + n) * sizeof(float) / 1000.0 / 1000.0;
    std::cout << total_GBytes / (timems) << " GBPS ... ";

    queue.memcpy(out_cpu, out_xpu, n * sizeof(float)).wait();

    for (int i = 0; i < n; i++) {
        auto diff = out_cpu[i] - in_cpu[i];
        diff = diff > 0 ? diff : -diff;
        if (diff > 0.01) {
            std::cout << "error\n";
            return;
        }
    }
    std::cout << "ok\n";

    sycl::free(in_xpu, queue);
    sycl::free(out_xpu, queue);
    delete[] in_cpu;
    delete[] out_cpu;
}

int main() {
    std::cout << "1GB threads copy test ...\n";
    std::cout << "float1: ";
    test_threads_copy<1>(1024 * 1024 * 256 + 2);
    std::cout << "float2: ";
    test_threads_copy<2>(1024 * 1024 * 256 + 2);
    std::cout << "float4: ";
    test_threads_copy<4>(1024 * 1024 * 256 + 2);
    std::cout << "float8: ";
    test_threads_copy<8>(1024 * 1024 * 256 + 2);
    std::cout << "float16: ";
    test_threads_copy<16>(1024 * 1024 * 256 + 2);
}
