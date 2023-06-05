#include <CL/sycl.hpp>
#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include "xetla.hpp"

double timeit(cl::sycl::event &event) {
    auto submit_time = event.get_profiling_info<cl::sycl::info::event_profiling::command_submit>();
    auto start_time = event.get_profiling_info<cl::sycl::info::event_profiling::command_start>();
    auto end_time = event.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
    auto submission_time = (start_time - submit_time) / 1000000.0f;
    auto execution_time = (end_time - start_time) / 1000000.0f;
    return execution_time;
}

using namespace cl::sycl;
using namespace gpu::xetla;
using namespace gpu;

template <typename scalar_t>
float gemm_xpu(
    sycl::queue &q,
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const scalar_t alpha,
    const scalar_t beta) {
    //Define the shape of workgroup and subgroup
    //It's tunable parameters based on different input shape and hardware for better performance
    constexpr uint32_t wg_tile_m = 256;
    constexpr uint32_t wg_tile_n = 256;
    constexpr uint32_t sg_tile_m = 32;
    constexpr uint32_t sg_tile_n = 64;

    //There are implicit requirement for sg_tile_k range
    constexpr uint32_t sg_tile_k = 32;

    // Org the compute shape for sub-matrix
    using tile_shape
            = xetla::group::tile_shape_t<wg_tile_n, // workgroup size in dim0
                    wg_tile_m, //	workgroup size in dim1
                    sg_tile_n, //	subgroup size in dim0
                    sg_tile_m>; //	subgroup size in dim1
    
    // Mirco-kernel configuration
    using brgemm_config = xetla::group::brgemm_selector_t<
            scalar_t, // input datatype for A
            scalar_t, // input datatype for B
            mem_layout::row_major, // memory layout for A
            mem_layout::row_major, // memory layout for B
            mem_space::global, // memory reading from global mem for A
            mem_space::global, // memory reading from global mem for B
            8, // buffer alignment for A, in unit of element
            8, // buffer alignment for B, in unit of element
            float, // accumulator data type for intermediate resutls
            tile_shape, // computation tile shape
            sg_tile_k, // elements in each iteration
            mma_engine::xmx, // compute engine
            gpu_arch::Xe> // GPU arch
            ::brgemm;

    using epilogue_t = xetla::group::epilogue_t<
            xetla::group::epilogue_policy_default<gpu_arch::Xe>, tile_shape,
            mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;

    using gemm_op_t = xetla::kernel::gemm_t<
            gpu::xetla::kernel::dispatch_policy_default<gpu_arch::Xe>,
            brgemm_config, epilogue_t>;

    sycl::nd_range<3> NDRange = gemm_op_t::get_nd_range(m, n);
    auto event = q.submit([&](handler &cgh) {
        cgh.parallel_for(NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
                xetla_exec_item<3> ei(item);
                gemm_op_t gemm_op;
                // allocate slm and nbarrier resource
                slm_barrier_init<gemm_op_t>();
                // set up gemm arguments
                typename gemm_op_t::arguments_t arg(m, k,
                        n, const_cast<scalar_t*>(a), k, const_cast<scalar_t*>(b), n, const_cast<scalar_t*>(out), n);
                gemm_op(ei, arg);
            });
        });
    event.wait();
    return timeit(event);
}

template <typename scalar_t, typename item_t>
inline void gemm_xpu_ref_kernel(
    item_t &item,
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const scalar_t alpha,
    const scalar_t beta) {
    auto mi = item.get_group(0) * 32 + item.get_local_id(0); // y
    auto ni = item.get_group(1) * 32 + item.get_local_id(1);
    if (mi < m && ni < n) {
        float acc = 0.f;
        for (int ki = 0; ki < k; ki++) {
            acc += (float)a[mi * k + ki] * (float)b[ki * n + ni];
        }
        auto r = (float)alpha * acc;
        out[mi * n + ni] = acc; //r + (float)beta * (float)out[mi * n + ni];
    }
}

template <typename scalar_t>
void gemm_xpu_ref(
    sycl::queue &q,
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m, const int n, const int k,
    const scalar_t alpha,
    const scalar_t beta) {
    auto m_groups = (m + 32 - 1) / 32; // y
    auto n_groups = (n + 32 - 1) / 32;
    auto event = q.submit([&](sycl::handler& h){
        h.parallel_for(
            sycl::nd_range<2>(
                sycl::range<2>(m_groups * 32, n_groups * 32), 
                sycl::range<2>(32, 32)),
            [=](sycl::nd_item<2> item) {
                gemm_xpu_ref_kernel<scalar_t>(item, out, a, b, m, n, k, alpha, beta);
        });
    });
    event.wait();
}

struct gemm_sizes {
    int m, n, k;
    float alpha, beta;
    gemm_sizes(int m_, int n_, int k_, float a, float b) :
        m(m_), n(n_), k(k_), alpha(a), beta(b) {
    }
};

int main() {
    std::cout << "hgemm_xetla\n";
    sycl::queue queue(sycl::gpu_selector_v, cl::sycl::property_list {cl::sycl::property::queue::enable_profiling()});
    using scalar_t = sycl::half;

    std::vector<gemm_sizes> sizes;
    // sizes.push_back(gemm_sizes(512, 512, 512, 0.5, 0.5));
    sizes.push_back(gemm_sizes(1024, 1024, 1024, 0.5, 0.5));
    // sizes.push_back(gemm_sizes(1028, 1028, 1028, 0.5, 0.5));
    sizes.push_back(gemm_sizes(2048, 2048, 2048, 0.5, 0.5));
    sizes.push_back(gemm_sizes(4096, 4096, 4096, 0.5, 0.5));
    sizes.push_back(gemm_sizes(8192, 8192, 8192, 0.5, 0.5));
    sizes.push_back(gemm_sizes(1<<14, 1<<14, 1<<14, 0.5, 0.5));

    for (auto size : sizes) {
        int m = size.m;
        int n = size.n;
        int k = size.k;
        auto alpha = size.alpha;
        auto beta = size.beta;

        std::cout << "m=" << m << ", n=" << n << ", k=" << k
                  << ", alpha=" << alpha << ", beta=" << beta << "\n";

        auto a_cpu = new scalar_t[m * k];
        auto b_cpu = new scalar_t[k * n];
        auto out_cpu = new scalar_t[m * n];
        for (int i = 0; i < m * k; i++)
            a_cpu[i] = static_cast<scalar_t>((float)rand() / (float)RAND_MAX);
        for (int i = 0; i < k * n; i++)
            b_cpu[i] = static_cast<scalar_t>((float)rand() / (float)RAND_MAX);
        for (int i = 0; i < m * n; i++)
            out_cpu[i] = static_cast<scalar_t>((float)rand() / (float)RAND_MAX);

        auto a_xpu = sycl::aligned_alloc_device<scalar_t>(256, m * k, queue);
        auto b_xpu = sycl::aligned_alloc_device<scalar_t>(256, k * n, queue);
        auto out_xpu = sycl::aligned_alloc_device<scalar_t>(256, m * n, queue);
        queue.memcpy(a_xpu, a_cpu,  m * k * sizeof(scalar_t)).wait();
        queue.memcpy(b_xpu, b_cpu,  k * n * sizeof(scalar_t)).wait();
        queue.memcpy(out_xpu, out_cpu,  m * n * sizeof(scalar_t)).wait();

        auto a_xpu_ref = sycl::aligned_alloc_device<scalar_t>(256, m * k, queue);
        auto b_xpu_ref = sycl::aligned_alloc_device<scalar_t>(256, k * n, queue);
        auto out_xpu_ref = sycl::aligned_alloc_device<scalar_t>(256, m * n, queue);
        queue.memcpy(a_xpu_ref, a_cpu,  m * k * sizeof(scalar_t)).wait();
        queue.memcpy(b_xpu_ref, b_cpu,  k * n * sizeof(scalar_t)).wait();
        queue.memcpy(out_xpu_ref, out_cpu,  m * n * sizeof(scalar_t)).wait();

        gemm_xpu_ref<scalar_t>(queue, out_xpu_ref, a_xpu_ref, b_xpu_ref, m, n, k, alpha, beta);
        auto timems = gemm_xpu<scalar_t>(queue, out_xpu, a_xpu, b_xpu, m, n, k, alpha, beta);

        double total_gbytes = ((double)m * k + k * n + m * n + m * n) * sizeof(scalar_t) / 1000.0 / 1000 / 1000;
        std::cout << timems << " ms, " << total_gbytes / (timems / 1000.0) << " gbps, ";

        double tflops = ((double)2 * m * n * k) / (timems / 1000) * 1e-12;
        std::cout << tflops << " tflops\n";

        auto out_xpu_ref_ = new scalar_t[m * n];
        auto out_xpu_ = new scalar_t[m * n];
        queue.memcpy(out_xpu_ref_, out_xpu_ref,  m * n * sizeof(scalar_t)).wait();
        queue.memcpy(out_xpu_, out_xpu,  m * n * sizeof(scalar_t)).wait();
        auto maxdiff = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < m * n; i++) {
            // if (i < 10)
            // std::cout << (float)out_xpu_[i] << " " << (float)out_xpu_ref_[i] << "\n";
            auto diff = std::abs((float)out_xpu_[i] - (float)out_xpu_ref_[i]);
            maxdiff = std::max(maxdiff, diff);
        }
        std::cout << "maxdiff: " << maxdiff << std::endl;

        sycl::free(a_xpu, queue);
        sycl::free(b_xpu, queue);
        sycl::free(out_xpu, queue);
        sycl::free(a_xpu_ref, queue);
        sycl::free(b_xpu_ref, queue);
        sycl::free(out_xpu_ref, queue);
        delete[] a_cpu;
        delete[] b_cpu;
        delete[] out_cpu;
        delete[] out_xpu_;
        delete[] out_xpu_ref_;
    }
    return 0;
}
