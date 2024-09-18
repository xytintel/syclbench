#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <assert.h>
#include "utils.h"
using namespace std;
using half = sycl::half;

template <typename T, int vec_size, typename item_t>
inline void threads_unroll_copy_kernel(item_t &item, const T *in, T *out, const size_t n) {
    const int block_work_size = item.get_local_range(0) * vec_size;
    auto index = item.get_group(0) * block_work_size + item.get_local_id(0);
#pragma unroll
    for (int i = 0; i < vec_size; i++) {
        if (index < n) {
            out[index] = in[index];
        }
        index += item.get_local_range(0);
    }
}

template <typename T, int vec_size>
float threads_unroll_copy(sycl::queue &queue, const T *in, T *out, size_t n) {
    const int group_size = 1024;
    const int group_work_size = group_size * vec_size;
    const int num_groups = (n + group_work_size - 1) / group_work_size;
    auto event = queue.submit([&](sycl::handler &h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_groups * group_size), sycl::range<1>(group_size)),
            [=](sycl::nd_item<1> item) {
                threads_unroll_copy_kernel<T, vec_size>(item, in, out, n);
            });
    });
    event.wait();
    return timeit(event);
}

template <typename T, int vec_size>
void test_threads_unroll_copy(size_t n) {
    sycl::queue queue(sycl::gpu_selector_v, cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()});

    auto in_cpu = new T[n];
    auto out_cpu = new T[n];
    for (int i = 0; i < n; i++)
        in_cpu[i] = (T)(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));

    auto in_xpu = sycl::aligned_alloc_device<T>(512, n, queue);
    auto out_xpu = sycl::aligned_alloc_device<T>(512, n, queue);
    queue.memcpy(in_xpu, in_cpu, n * sizeof(T)).wait();

    float timems;
    for (int i = 0; i < 300; i++)
        timems = threads_unroll_copy<T, vec_size>(queue, in_xpu, out_xpu, n);

    float total_GBytes = (n + n) * sizeof(T) / 1000.0 / 1000.0;
    std::cout << total_GBytes / (timems) << " GBPS ... ";

    queue.memcpy(out_cpu, out_xpu, n * sizeof(T)).wait();

    for (int i = 0; i < n; i++) {
        auto diff = (float)(out_cpu[i] - in_cpu[i]);
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
    std::cout << "1GB threads unroll copy test ...\n";

    std::cout << "float1: ";
    test_threads_unroll_copy<float, 1>(1024 * 1024 * 256 + 2);
    std::cout << "float2: ";
    test_threads_unroll_copy<float, 2>(1024 * 1024 * 256 + 2);
    std::cout << "float4: ";
    test_threads_unroll_copy<float, 4>(1024 * 1024 * 256 + 2);
    std::cout << "float8: ";
    test_threads_unroll_copy<float, 8>(1024 * 1024 * 256 + 2);
    std::cout << "float16: ";
    test_threads_unroll_copy<float, 16>(1024 * 1024 * 256 + 2);

    std::cout << "half1: ";
    test_threads_unroll_copy<half, 1>(1024 * 1024 * 256 + 2);
    std::cout << "half2: ";
    test_threads_unroll_copy<half, 2>(1024 * 1024 * 256 + 2);
    std::cout << "half4: ";
    test_threads_unroll_copy<half, 4>(1024 * 1024 * 256 + 2);
    std::cout << "half8: ";
    test_threads_unroll_copy<half, 8>(1024 * 1024 * 256 + 2);
    std::cout << "half16: ";
    test_threads_unroll_copy<half, 16>(1024 * 1024 * 256 + 2);
}
