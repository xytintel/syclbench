#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <assert.h>
using namespace std;

double timeit(cl::sycl::event &event) {
    auto submit_time = event.get_profiling_info<cl::sycl::info::event_profiling::command_submit>();
    auto start_time = event.get_profiling_info<cl::sycl::info::event_profiling::command_start>();
    auto end_time = event.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
    auto submission_time = (start_time - submit_time) / 1000000.0f;
    auto execution_time = (end_time - start_time) / 1000000.0f;
    return execution_time;
}

template <int LOOP, typename item_t>
inline void fmad_loop_kernel(item_t &item, float *x) {
    int index = item.get_local_id(0) + item.get_group(0) * item.get_local_range(0);
    float a = x[index], b = -1.0f, c = 1.5f;
    for (int i = 0; i < LOOP; i++) {
        for (int j = 0; j < LOOP; j++) {
            a = a * b + c;
        }
    }
    x[index] = a;
}

template <int LOOP, int group_size, int num_groups>
float fmad_test(sycl::queue &queue) {
    constexpr int n = group_size * num_groups;
    auto x = new float[n];

    auto dx = sycl::aligned_alloc_device<float>(512, n, queue);
    queue.memcpy(dx, x, n * sizeof(float)).wait();

    auto event = queue.submit([&](sycl::handler &h) {
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(num_groups * group_size), sycl::range<1>(group_size)),
            [=](sycl::nd_item<1> item) {
                fmad_loop_kernel<LOOP>(item, dx);
            });
    });
    event.wait();
    float ms = timeit(event);
    queue.memcpy(x, dx, n * sizeof(float)).wait();

    sycl::free(dx, queue);
    delete[] x;
    return ms;
}

int main() {
    sycl::queue queue(sycl::gpu_selector_v, cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()});
    constexpr int LOOP = 10000;
    constexpr int block_size = 256;
    constexpr int num_blocks = 2048;
    for (int i = 0; i < 3; i++) {
        auto timems = fmad_test<LOOP, block_size, num_blocks>(queue);
        auto tflops =
            2.0 * LOOP * LOOP * num_blocks * block_size / (timems / 1000) * 1e-12;
        std::cout << tflops << " TFLOPS" << std::endl;
    }
    return 0;
}
