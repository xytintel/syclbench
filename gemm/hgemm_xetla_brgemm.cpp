#include "xetla.hpp"
#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <assert.h>

using namespace cl::sycl;
using namespace gpu;
using namespace gpu::xetla;
using namespace gpu::xetla::group;
using namespace gpu::xetla::kernel;
using namespace gpu::xetla::subgroup;

double timeit(cl::sycl::event &event) {
    auto submit_time = event.get_profiling_info<
        cl::sycl::info::event_profiling::command_submit>();
    auto start_time =
        event
            .get_profiling_info<cl::sycl::info::event_profiling::command_start>();
    auto end_time =
        event.get_profiling_info<cl::sycl::info::event_profiling::command_end>();
    auto submission_time = (start_time - submit_time) / 1000000.0f;
    auto execution_time = (end_time - start_time) / 1000000.0f;
    return execution_time;
}

template <
    typename scalar_t,
    int WG_M = 8,
    int WG_N = 32,
    int SG_M = 8,
    int SG_N = 16,
    int KN = 32,
    bool B_ROW_MAJOR = false>
float gemm_xpu(sycl::queue &q, scalar_t *out, const scalar_t *a,
               const scalar_t *b, const int m, const int n, const int k) {
    uint32_t matrix_m = m;
    uint32_t matrix_n = n;
    uint32_t matrix_k = k;
    constexpr uint32_t wg_tile_m = WG_M;
    constexpr uint32_t wg_tile_n = WG_N;
    constexpr uint32_t sg_tile_m = SG_M;
    constexpr uint32_t sg_tile_n = SG_N;
    uint32_t group_range_m = (matrix_m + wg_tile_m - 1) / wg_tile_m;
    uint32_t group_range_n = (matrix_n + wg_tile_n - 1) / wg_tile_n;
    uint32_t thread_range_m = wg_tile_m / sg_tile_m;
    uint32_t thread_range_n = wg_tile_n / sg_tile_n;
    uint32_t lda = matrix_k;
    uint32_t ldb = B_ROW_MAJOR ? matrix_n : matrix_k;
    uint32_t ldc = matrix_n;
    cl::sycl::range<3> GroupRange{1, group_range_m, group_range_n};
    cl::sycl::range<3> LocalRange{1, thread_range_m, thread_range_n};
    cl::sycl::nd_range<3> NDRange(GroupRange * LocalRange, LocalRange);

    auto gpu_event = q.submit([&](handler &cgh) {
        cgh.parallel_for(NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
            xetla_exec_item<3> ei(item);

            using data_type_b = scalar_t;
            using data_type_a = scalar_t;
            using data_type_c = scalar_t;
            using data_type_acc = float;
            using compute_attr =
                compute_attr_t<data_type_a, data_type_b, data_type_acc>;

            static constexpr uint32_t periodic_sync_interval = 1;
            static constexpr uint32_t prefetch_distance = 3;
            static constexpr uint32_t k_iter_num = KN;
            using perf_tuning_knob = perf_tuning_knob_t<k_iter_num, prefetch_distance,
                                                        periodic_sync_interval>;
            using compute_policy =
                compute_policy_default_xmx<compute_attr, perf_tuning_knob,
                                           gpu_arch::Xe>;

            using mem_desc_input_a =
                mem_desc_t<data_type_a, mem_layout::row_major, mem_space::global>;
            using mem_desc_input_b =
                std::conditional<B_ROW_MAJOR,
                                 mem_desc_t<data_type_b, mem_layout::row_major, mem_space::global>,
                                 mem_desc_t<data_type_b, mem_layout::col_major, mem_space::global>>::type;
            using mem_desc_output_c =
                mem_desc_t<data_type_c, mem_layout::row_major, mem_space::global>;

            using tile_shape =
                tile_shape_t<wg_tile_n, wg_tile_m, sg_tile_n, sg_tile_m>;
            using brgemm_t = brgemm_t<compute_policy, tile_shape, mem_desc_input_a,
                                      mem_desc_input_b>;
            brgemm_t brgemm;
            using epilogue_t = epilogue_t<epilogue_policy_tile_op<chained_tile_op_t<>,
                                                                  result_overwrite, gpu_arch::Xe>,
                                          tile_shape, mem_desc_output_c>;

            static constexpr uint32_t barrier_count = brgemm_t::barrier_count;
            static constexpr uint32_t slm_size = brgemm_t::slm_size;
            xetla_nbarrier_init<barrier_count>();
            xetla_local_init<slm_size>();
            int start_n = ei.get_group(2) * wg_tile_n;
            int start_m = ei.get_group(1) * wg_tile_m;
            int start_k = 0;
            uint32_t wg_tile_k = matrix_k;
            uint32_t inner_loop_count = wg_tile_k / k_iter_num;

            mem_desc_input_a md_a({const_cast<scalar_t *>(a)},
                                  {matrix_k, matrix_m, lda}, {start_k, start_m});
            mem_desc_input_b md_b({const_cast<scalar_t *>(b)},
                                  {matrix_n, matrix_k, ldb}, {start_n, start_k});
            mem_desc_output_c md_c({out}, {matrix_n, matrix_m, ldc},
                                   {start_n, start_m});

            typename brgemm_t::matAcc_t matAcc;
            matAcc.init(0);
            typename brgemm_t::arguments_t brgemm_args(md_a, md_b, inner_loop_count);
            typename brgemm_t::work_group_t g(ei.get_local_linear_id());
            brgemm(g, matAcc, brgemm_args);
            epilogue_t epilogue;
            epilogue(g, matAcc, md_c);
        });
    });
    gpu_event.wait();
    float time = timeit(gpu_event);
    return time;
}

template <typename scalar_t, typename item_t>
inline void gemm_xpu_ref_kernel(item_t &item, scalar_t *out, const scalar_t *a,
                                const scalar_t *b, const int m, const int n,
                                const int k, const scalar_t alpha,
                                const scalar_t beta) {
    auto mi = item.get_group(0) * 32 + item.get_local_id(0); // y
    auto ni = item.get_group(1) * 32 + item.get_local_id(1);
    if (mi < m && ni < n) {
        float acc = 0.f;
        for (int ki = 0; ki < k; ki++) {
            acc += (float)a[mi * k + ki] * (float)b[ni * k + ki];
        }
        auto r = (float)alpha * acc;
        out[mi * n + ni] = r + (float)beta * (float)out[mi * n + ni];
    }
}

template <typename scalar_t>
void gemm_xpu_ref(sycl::queue &q, scalar_t *out, const scalar_t *a,
                  const scalar_t *b, const int m, const int n, const int k,
                  const scalar_t alpha, const scalar_t beta) {
    auto m_groups = (m + 32 - 1) / 32; // y
    auto n_groups = (n + 32 - 1) / 32;
    auto event = q.submit([&](sycl::handler &h) {
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(m_groups * 32, n_groups * 32),
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
    std::cout << "hgemm_xetla_brgemm\n";
    sycl::queue queue(
        sycl::gpu_selector_v,
        cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()});
    using scalar_t = sycl::half;

    std::vector<gemm_sizes> sizes;

    sizes.push_back(gemm_sizes(1, 4096, 4096, 1.0, 0.0));
    sizes.push_back(gemm_sizes(1, 16384, 4096, 1.0, 0.0));
    sizes.push_back(gemm_sizes(1, 4096, 16384, 1.0, 0.0));
    sizes.push_back(gemm_sizes(1, 32000, 4096, 1.0, 0.0));

    for (auto size : sizes) {
        int m = size.m;
        int n = size.n;
        int k = size.k;
        auto alpha = size.alpha;
        auto beta = size.beta;

        std::cout << "m=" << m << ", n=" << n << ", k=" << k << ", alpha=" << alpha
                  << ", beta=" << beta << "\n";

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
        queue.memcpy(a_xpu, a_cpu, m * k * sizeof(scalar_t)).wait();
        queue.memcpy(b_xpu, b_cpu, k * n * sizeof(scalar_t)).wait();
        queue.memcpy(out_xpu, out_cpu, m * n * sizeof(scalar_t)).wait();

        auto a_xpu_ref = sycl::aligned_alloc_device<scalar_t>(256, m * k, queue);
        auto b_xpu_ref = sycl::aligned_alloc_device<scalar_t>(256, k * n, queue);
        auto out_xpu_ref = sycl::aligned_alloc_device<scalar_t>(256, m * n, queue);
        queue.memcpy(a_xpu_ref, a_cpu, m * k * sizeof(scalar_t)).wait();
        queue.memcpy(b_xpu_ref, b_cpu, k * n * sizeof(scalar_t)).wait();
        queue.memcpy(out_xpu_ref, out_cpu, m * n * sizeof(scalar_t)).wait();

        gemm_xpu_ref<scalar_t>(queue, out_xpu_ref, a_xpu_ref, b_xpu_ref, m, n, k,
                               alpha, beta);
        auto timems =
            gemm_xpu<scalar_t>(queue, out_xpu, a_xpu, b_xpu, m, n, k);

        double total_gbytes = ((double)m * k + k * n + m * n + m * n) * sizeof(scalar_t) / 1000.0 / 1000 / 1000;
        std::cout << timems << " ms, " << total_gbytes / (timems / 1000.0)
                  << " gbps, ";

        double tflops = ((double)2 * m * n * k) / (timems / 1000) * 1e-12;
        std::cout << tflops << " tflops\n";

        auto out_xpu_ref_ = new scalar_t[m * n];
        auto out_xpu_ = new scalar_t[m * n];
        queue.memcpy(out_xpu_ref_, out_xpu_ref, m * n * sizeof(scalar_t)).wait();
        queue.memcpy(out_xpu_, out_xpu, m * n * sizeof(scalar_t)).wait();
        auto maxdiff = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < m * n; i++) {
            // if (i < 10)
            // std::cout << (float)out_xpu_[i] << " " << (float)out_xpu_ref_[i] <<
            // "\n";
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
