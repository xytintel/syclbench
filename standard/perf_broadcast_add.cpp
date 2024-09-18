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
inline void threads_unroll_add_kernel(item_t &item, const T *in_dense, const T *in_bc, T *out, const size_t n, const size_t stride0, const size_t stride1) {
    const int block_work_size = item.get_local_range(0) * vec_size;
    auto index = item.get_group(0) * block_work_size + item.get_local_id(0);
#pragma unroll
    for (int i = 0; i < vec_size; i++) {
        if (index < n) {
            out[index] = in_dense[index] + in_bc[index / stride0 * stride1 + index % stride1];
        }
        index += item.get_local_range(0);
    }
}

template <typename T, int vec_size>
float threads_unroll_add(sycl::queue &queue, const T *in_dense, const T *in_bc, T *out, const size_t n, const size_t stride0, const size_t stride1) {
    const int group_size = 1024;
    const int group_work_size = group_size * vec_size;
    const int num_groups = (n + group_work_size - 1) / group_work_size;
    auto event = queue.submit([&](sycl::handler &h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_groups * group_size), sycl::range<1>(group_size)),
            [=](sycl::nd_item<1> item) {
                threads_unroll_add_kernel<T, vec_size>(item, in_dense, in_bc, out, n, stride0, stride1);
            });
    });
    event.wait();
    return timeit(event);
}

template <typename T, int vec_size>
void test_threads_unroll_add(size_t left, size_t mid, size_t right) {
    sycl::queue queue(sycl::gpu_selector_v, cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()});
    auto problem_size = left * mid * right;
    auto bc_size = left * 1 * right;

    auto in_dense_cpu = new T[problem_size];
    auto in_bc_cpu = new T[bc_size];
    auto out_cpu = new T[problem_size];

    for (int i = 0; i < problem_size; i++)
        in_dense_cpu[i] = (T)(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
    for (int i = 0; i < bc_size; i++)
        in_bc_cpu[i] = (T)(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));

    auto in_dense_xpu = sycl::aligned_alloc_device<T>(512, problem_size, queue);
    auto in_bc_xpu = sycl::aligned_alloc_device<T>(512, bc_size, queue);
    auto out_xpu = sycl::aligned_alloc_device<T>(512, problem_size, queue);

    queue.memcpy(in_dense_xpu, in_dense_cpu, problem_size * sizeof(T)).wait();
    queue.memcpy(in_bc_xpu, in_bc_cpu, bc_size * sizeof(T)).wait();

    float timems;
    for (int i = 0; i < 300; i++)
        timems = threads_unroll_add<T, vec_size>(queue, in_dense_xpu, in_bc_xpu, out_xpu, problem_size, mid * right, right);

    float total_GBytes = (2 * problem_size + bc_size) * sizeof(T) / 1000.0 / 1000.0;
    std::cout << total_GBytes / (timems) << " GBPS ... ";

    queue.memcpy(out_cpu, out_xpu, problem_size * sizeof(T)).wait();

    for (int i = 0; i < problem_size; i++) {
        auto bc_idx = i / (mid * right) * right + i % right;
        auto diff = (float)(out_cpu[i] - in_dense_cpu[i] - in_bc_cpu[bc_idx]);
        diff = diff > 0 ? diff : -diff;
        if (diff > 0.01) {
            std::cout << "error\n";
            return;
        }
    }
    std::cout << "ok\n";

    sycl::free(in_dense_xpu, queue);
    sycl::free(in_bc_xpu, queue);
    sycl::free(out_xpu, queue);
    delete[] in_dense_cpu;
    delete[] in_bc_cpu;
    delete[] out_cpu;
}

int main() {
    std::cout << "threads unroll broadcast add test ...\n";

    std::cout << "float4: ";
    test_threads_unroll_add<float, 4>(2, 12, 2048 * 2048);

    std::cout << "half4: ";
    test_threads_unroll_add<half, 4>(2, 12, 2048 * 2048);
}
