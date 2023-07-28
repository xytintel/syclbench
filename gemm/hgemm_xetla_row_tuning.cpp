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
    int SG_K = 64,
    int SLM_KS = 8,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3,
    bool B_ROW_MAJOR = true>
inline double gemm_xpu_impl(
    sycl::queue &queue,
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m,
    const int n,
    const int k,
    const bool verbose) {
    constexpr mem_layout layout_a = mem_layout::row_major;
    constexpr mem_layout layout_b =
        B_ROW_MAJOR ? mem_layout::row_major : mem_layout::col_major;
    uint32_t group_range_m = (m + WG_M - 1) / WG_M;
    uint32_t group_range_n = (n + WG_N - 1) / WG_N;
    uint32_t thread_range_m = WG_M / SG_M;
    uint32_t thread_range_n = WG_N / SG_N;
    if (verbose) {
        std::cout << "m=" << m << ", n=" << n << ", k=" << k << ", n_ss=" << group_range_m * group_range_n << ", N_SG_PER_SS=" << SLM_KS * thread_range_m * thread_range_n
                  << ", WG_M=" << WG_M << ", WG_N=" << WG_N << ", SG_M=" << SG_M << ", SG_N=" << SG_N << ", SG_K=" << SG_K << ", SLM_KS=" << SLM_KS;
    }
    uint32_t lda = k;
    uint32_t ldb = B_ROW_MAJOR ? n : k;
    uint32_t ldc = n;
    cl::sycl::range<3> GroupRange{L3_KS, group_range_m, group_range_n};
    cl::sycl::range<3> LocalRange{SLM_KS, thread_range_m, thread_range_n};
    cl::sycl::nd_range<3> NDRange(GroupRange * LocalRange, LocalRange);

    auto gpu_event = queue.submit([&](handler &cgh) {
        cgh.parallel_for(NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
            xetla_exec_item<3> ei(item);

            using data_type_b = scalar_t;
            using data_type_a = scalar_t;
            using data_type_c = scalar_t;
            using data_type_acc = float;
            static constexpr uint32_t periodic_sync_interval = SYNC_FREQ;
            static constexpr uint32_t prefetch_distance = STAGES;
            using tile_shape = tile_shape_t<WG_N, WG_M, SG_N, SG_M>;

            using brgemm_t = typename brgemm_selector_t<
                data_type_a,
                data_type_b,
                layout_a,
                layout_b,
                mem_space::global,
                mem_space::global,
                8,
                8,
                data_type_acc,
                tile_shape,
                SG_K,
                mma_engine::xmx,
                gpu_arch::Xe,
                prefetch_distance,
                periodic_sync_interval>::brgemm;
            using update_method = typename std::
                conditional<(L3_KS > 1), result_reduce_sum, result_overwrite>::type;
            using epilogue_t = epilogue_t<
                epilogue_policy_tile_op<
                    chained_tile_op_t<>,
                    update_method,
                    gpu_arch::Xe>,
                tile_shape,
                mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
            using gemm_op_t = gemm_t<
                dispatch_policy_kslicing<L3_KS, SLM_KS, gpu_arch::Xe>,
                brgemm_t,
                epilogue_t>;
            typename gemm_op_t::arguments_t arg(
                m,
                k,
                n,
                const_cast<scalar_t *>(a),
                lda,
                const_cast<scalar_t *>(b),
                ldb,
                out,
                ldc);
            slm_barrier_init<gemm_op_t>();
            gemm_op_t gemm_op;
            gemm_op(ei, arg);
        });
    });
    gpu_event.wait();
    float time = timeit(gpu_event);
    return time;
}

#define DISPATCH_POLICY(WG_M, WG_N, SG_M, SG_N, SG_K)                                                                                                  \
    double gemm_xpu_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_(                                                                                     \
        sycl::queue &queue,                                                                                                                            \
        sycl::half *out,                                                                                                                               \
        const sycl::half *a,                                                                                                                           \
        const sycl::half *b,                                                                                                                           \
        const int m,                                                                                                                                   \
        const int n,                                                                                                                                   \
        const int k, const bool verbose) {                                                                                                             \
        return gemm_xpu_impl<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, 32 / (WG_M * WG_N / SG_M / SG_N), 1, 1, 3>(queue, out, a, b, m, n, k, verbose); \
    }

#define DESC_POLICY(WG_M, WG_N, SG_M, SG_N, SG_K) gemm_xpu_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_

#define NUM_POLICIES 13

DISPATCH_POLICY(32, 64, 8, 16, 16)
DISPATCH_POLICY(256, 256, 32, 64, 16)
DISPATCH_POLICY(8, 64, 8, 16, 64)
DISPATCH_POLICY(8, 128, 8, 16, 32)
DISPATCH_POLICY(8, 128, 8, 16, 16)
DISPATCH_POLICY(16, 256, 8, 16, 16)
DISPATCH_POLICY(8, 512, 8, 16, 16)
DISPATCH_POLICY(8, 256, 8, 16, 32)
DISPATCH_POLICY(8, 512, 8, 16, 32)
DISPATCH_POLICY(256, 256, 32, 64, 32)
DISPATCH_POLICY(32, 256, 8, 32, 16)
DISPATCH_POLICY(8, 512, 8, 32, 16)
DISPATCH_POLICY(32, 64, 8, 16, 32)

double (*policies[NUM_POLICIES])(sycl::queue &, sycl::half *, const sycl::half *, const sycl::half *, const int, const int, const int, const bool) = {
    DESC_POLICY(32, 64, 8, 16, 16),
    DESC_POLICY(256, 256, 32, 64, 16),
    DESC_POLICY(8, 64, 8, 16, 64),
    DESC_POLICY(8, 128, 8, 16, 32),
    DESC_POLICY(8, 128, 8, 16, 16),
    DESC_POLICY(16, 256, 8, 16, 16),
    DESC_POLICY(8, 512, 8, 16, 16),
    DESC_POLICY(8, 256, 8, 16, 32),
    DESC_POLICY(8, 512, 8, 16, 32),
    DESC_POLICY(256, 256, 32, 64, 32),
    DESC_POLICY(32, 256, 8, 32, 16),
    DESC_POLICY(8, 512, 8, 32, 16),
    DESC_POLICY(32, 64, 8, 16, 32),
};

template <typename scalar_t>
inline void add_noice(sycl::queue &queue, scalar_t *out, int size) {
    auto temp_ptr_cpu = new scalar_t[size];
    queue.memcpy(temp_ptr_cpu, out, size * sizeof(scalar_t)).wait();
    for (int i = 0; i < size; i++) temp_ptr_cpu[i] += 1e-6;
    queue.memcpy(out, temp_ptr_cpu, size * sizeof(scalar_t)).wait();
    delete[] temp_ptr_cpu;
}

template <typename scalar_t>
inline double gemm_xpu(
    sycl::queue &queue,
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const int m,
    const int n,
    const int k,
    const bool is_warmup) {
    double timems_min = 100000.0f;
    int index_min = 0;
    for (int i = 0; i < NUM_POLICIES; i++) {
        add_noice<scalar_t>(queue, const_cast<scalar_t *>(a), m * k);
        add_noice<scalar_t>(queue, const_cast<scalar_t *>(b), k * n);
        add_noice<scalar_t>(queue, out, m * n);
        auto t = policies[i](queue, out, a, b, m, n, k, false);
        if (!is_warmup)
            std::cout << "policy_" << i << ": " << t << "\n";
        if (t < timems_min) {
            timems_min = t;
            index_min = i;
        }
    }
    add_noice<scalar_t>(queue, const_cast<scalar_t *>(a), m * k);
    add_noice<scalar_t>(queue, const_cast<scalar_t *>(b), k * n);
    add_noice<scalar_t>(queue, out, m * n);
    auto timems = policies[index_min](queue, out, a, b, m, n, k, !is_warmup);
    if (!is_warmup)
        std::cout << ", policy_id=" << index_min << "\n";
    return timems;
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
            acc += (float)a[mi * k + ki] * (float)b[ki * n + ni];
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
    std::cout << "hgemm_xetla_row_tuning\n";
    sycl::queue queue(
        sycl::gpu_selector_v,
        cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()});
    using scalar_t = sycl::half;

    std::vector<gemm_sizes> sizes;
    // warmup
    for (int i = 0; i < 3; i++)
        sizes.push_back(gemm_sizes(2048, 2048, 2048, 1.0, 0.0));

    // int ms[10] = {1, 8, 16, 32, 48, 64, 80, 96, 112, 128};

    // for (int i = 0; i < 10; i++) {
    //     sizes.push_back(gemm_sizes(ms[i], 4096, 4096, 1.0, 0.0));
    //     sizes.push_back(gemm_sizes(ms[i], 5120, 5120, 1.0, 0.0));
    //     sizes.push_back(gemm_sizes(ms[i], 14336, 14336, 1.0, 0.0));

    //     sizes.push_back(gemm_sizes(ms[i], 1792, 14336, 1.0, 0.0));
    //     sizes.push_back(gemm_sizes(ms[i], 4096, 11008, 1.0, 0.0));
    //     sizes.push_back(gemm_sizes(ms[i], 4096, 16384, 1.0, 0.0));
    //     sizes.push_back(gemm_sizes(ms[i], 5120, 13824, 1.0, 0.0));
    //     sizes.push_back(gemm_sizes(ms[i], 7168, 14336, 1.0, 0.0));

    //     sizes.push_back(gemm_sizes(ms[i], 11008, 4096, 1.0, 0.0));
    //     sizes.push_back(gemm_sizes(ms[i], 13824, 5120, 1.0, 0.0));
    //     sizes.push_back(gemm_sizes(ms[i], 14336, 1792, 1.0, 0.0));
    //     sizes.push_back(gemm_sizes(ms[i], 14336, 7168, 1.0, 0.0));
    //     sizes.push_back(gemm_sizes(ms[i], 16384, 4096, 1.0, 0.0));

    //     sizes.push_back(gemm_sizes(ms[i], 50400, 4096, 1.0, 0.0));
    // }

    // sizes.push_back(gemm_sizes(1, 50400, 4096, 1.0, 0.0)); // 6
    // sizes.push_back(gemm_sizes(8, 50400, 4096, 1.0, 0.0)); // 6
    // sizes.push_back(gemm_sizes(16, 50400, 4096, 1.0, 0.0));

    // sizes.push_back(gemm_sizes(1, 65536, 4096, 1.0, 0.0)); // 6
    // sizes.push_back(gemm_sizes(1, 32768, 4096, 1.0, 0.0)); // 6
    // sizes.push_back(gemm_sizes(1, 16384, 4096, 1.0, 0.0)); // 6

    // sizes.push_back(gemm_sizes(8, 65536, 2048, 1.0, 0.0));
    // sizes.push_back(gemm_sizes(8, 65536, 4096, 1.0, 0.0));
    // sizes.push_back(gemm_sizes(8, 65536, 8192, 1.0, 0.0));
    // sizes.push_back(gemm_sizes(8, 65536, 16384, 1.0, 0.0));

    // sizes.push_back(gemm_sizes(1, 65536, 32768, 1.0, 0.0));
    // sizes.push_back(gemm_sizes(1, 65536, 65536, 1.0, 0.0));

    // sizes.push_back(gemm_sizes(1, 8192, 4096, 1.0, 0.0));

    // sizes.push_back(gemm_sizes(32, 50400, 4096, 1.0, 0.0));

    // sizes.push_back(gemm_sizes(1, 50400, 4096, 1.0, 0.0));
    // sizes.push_back(gemm_sizes(1, 40600, 4096, 1.0, 0.0));

    sizes.push_back(gemm_sizes(60, 4096, 4096, 1.0, 0.0));
    sizes.push_back(gemm_sizes(60, 16384, 4096, 1.0, 0.0));
    sizes.push_back(gemm_sizes(60, 16384, 4096, 1.0, 0.0));
    // sizes.push_back(gemm_sizes(114960, 50400, 4096, 1.0, 0.0));
    sizes.push_back(gemm_sizes(60, 50400, 4096, 1.0, 0.0));
    sizes.push_back(gemm_sizes(28740, 4096, 4096, 1.0, 0.0));

    int count = 0;
    for (auto size : sizes) {
        int m = size.m;
        int n = size.n;
        int k = size.k;
        auto alpha = size.alpha;
        auto beta = size.beta;

        bool is_warmup = false;
        if (count < 3) is_warmup = true;

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
            gemm_xpu<scalar_t>(queue, out_xpu, a_xpu, b_xpu, m, n, k, is_warmup);

        double total_gbytes = ((double)m * k + k * n + m * n) * sizeof(scalar_t) / 1000.0 / 1000 / 1000;
        if (count >= 3)
            std::cout << timems << " ms, " << total_gbytes / (timems / 1000.0)
                      << " gbps, ";

        double tflops = ((double)2 * m * n * k) / (timems / 1000) * 1e-12;
        if (count >= 3)
            std::cout << tflops << " tflops\n";

        auto out_xpu_ref_ = new scalar_t[m * n];
        auto out_xpu_ = new scalar_t[m * n];
        queue.memcpy(out_xpu_ref_, out_xpu_ref, m * n * sizeof(scalar_t)).wait();
        queue.memcpy(out_xpu_, out_xpu, m * n * sizeof(scalar_t)).wait();
        auto maxdiff = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < m * n; i++) {
            auto diff = std::abs((float)out_xpu_[i] - (float)out_xpu_ref_[i]);
            maxdiff = std::max(maxdiff, diff);
        }
        if (count >= 3)
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
        count++;
    }
    return 0;
}
