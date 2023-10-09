#include "xetla.hpp"
#include "utils.h"
#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <assert.h>
#include <map>
#include <algorithm>
#include <cstdlib>

using namespace cl::sycl;
using namespace gpu;
using namespace gpu::xetla;
using namespace gpu::xetla::group;
using namespace gpu::xetla::kernel;
using namespace gpu::xetla::subgroup;

template <
    typename scalar_t,
    int WG_M,
    int WG_N,
    int SG_M,
    int SG_N,
    int SG_K,
    int SLM_KS,
    int L3_KS = 1,
    int SYNC_FREQ = 1,
    int STAGES = 3,
    bool A_ROW_MAJOR = true,
    bool B_ROW_MAJOR = true>
inline sycl::event gemm_core(
    sycl::queue &queue,
    scalar_t *out0,
    scalar_t *out1,
    scalar_t *out2,
    const scalar_t *a,
    const scalar_t *b,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
    constexpr mem_layout layout_a =
        A_ROW_MAJOR ? mem_layout::row_major : mem_layout::col_major;
    constexpr mem_layout layout_b =
        B_ROW_MAJOR ? mem_layout::row_major : mem_layout::col_major;

    uint32_t group_range_m = (m + WG_M - 1) / WG_M;
    uint32_t group_range_n = (n + WG_N - 1) / WG_N;
    constexpr uint32_t thread_range_m = WG_M / SG_M;
    constexpr uint32_t thread_range_n = WG_N / SG_N;
    uint32_t lda = A_ROW_MAJOR ? k : m;
    uint32_t ldb = B_ROW_MAJOR ? n : k;
    uint32_t ldc = n;
    cl::sycl::range<3> GroupRange{3, group_range_m, group_range_n};
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
            using epilogue_t = epilogue_t<
                epilogue_policy_tile_op<
                    chained_tile_op_t<>,
                    gpu_arch::Xe>,
                tile_shape,
                mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
            using gemm_op_t = gemm_t<
                dispatch_policy_kslicing<L3_KS, SLM_KS, gpu_arch::Xe>,
                brgemm_t,
                epilogue_t>;

            uint32_t batch_id = ei.get_group(0);
            slm_barrier_init<gemm_op_t>();
            scalar_t *out = (batch_id == 0) ? out0 : ((batch_id == 1) ? out1 : out2);

            uint32_t size_b = k * n;

            typename gemm_op_t::arguments_t arg(
                m,
                k,
                n,
                const_cast<scalar_t *>(a),
                lda,
                const_cast<scalar_t *>(b) + size_b * batch_id,
                ldb,
                out,
                ldc);

            gemm_op_t gemm_op;
            gemm_op(ei, arg);
        });
    });
    gpu_event.wait();
    return gpu_event;
}

#define HGEMM_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS) \
    hgemm_core_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_##SLM_KS##_
#define HGEMM_IMPL(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS)                               \
    sycl::event HGEMM_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS)(                 \
        sycl::queue & queue,                                                           \
        sycl::half * out0,                                                             \
        sycl::half * out1,                                                             \
        sycl::half * out2,                                                             \
        const sycl::half *a,                                                           \
        const sycl::half *b,                                                           \
        const uint32_t m,                                                              \
        const uint32_t n,                                                              \
        const uint32_t k) {                                                            \
        return gemm_core<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,             \
                         1, 1, 3, true, true>(queue, out0, out1, out2, a, b, m, n, k); \
    }

// clang-format off
#define HGEMM_COMMA ,
#define HGEMM_NUM_POLICIES 26
#define HGEMM_ENUMERATE_POLICIES(_, T) \
  _(8, 64, 8, 16, 32, 8)T      \
  _(8, 128, 8, 16, 16, 2)T     \
  _(8, 128, 8, 16, 32, 4)T     \
  _(8, 256, 8, 16, 16, 2)T     \
  _(8, 512, 8, 16, 16, 1)T     \
  _(16, 64, 16, 16, 16, 8)T    \
  _(16, 256, 8, 16, 16, 1)T    \
  _(16, 256, 16, 16, 16, 2)T   \
  _(16, 512, 16, 16, 16, 1)T   \
  _(32, 64, 32, 16, 16, 8)T    \
  _(32, 64, 8, 16, 16, 2)T     \
  _(32, 128, 32, 16, 16, 4)T   \
  _(32, 256, 32, 16, 16, 2)T   \
  _(32, 512, 32, 16, 16, 1)T   \
  _(64, 128, 64, 16, 16, 4)T   \
  _(64, 256, 64, 16, 16, 2)T   \
  _(64, 512, 64, 16, 16, 1)T   \
  _(128, 128, 32, 32, 32, 2)T  \
  _(128, 256, 64, 16, 16, 1)T  \
  _(128, 512, 64, 32, 16, 1)T  \
  _(256, 256, 64, 32, 16, 1)T  \
  _(256, 256, 32, 64, 16, 1)T  \
  _(256, 256, 32, 64, 32, 1)T  \
  _(128, 64, 16, 16, 64, 1)T   \
  _(128, 128, 16, 32, 64, 1)T  \
  _(128, 256, 32, 32, 16, 1)T
// clang-format on

HGEMM_ENUMERATE_POLICIES(HGEMM_IMPL, )
sycl::event (*policies[HGEMM_NUM_POLICIES])(sycl::queue &, sycl::half *, sycl::half *, sycl::half *, const sycl::half *, const sycl::half *,
                                            const uint32_t, const uint32_t, const uint32_t) = {
    HGEMM_ENUMERATE_POLICIES(HGEMM_IMPL_NAME, HGEMM_COMMA)};

struct PolicyTraits {
    int wg_m, wg_n;
    int sg_m, sg_n, sg_k;
    int slm_ks;
    PolicyTraits(int wg_m_, int wg_n_, int sg_m_, int sg_n_, int sg_k_, int slm_ks_) :
        wg_m(wg_m_), wg_n(wg_n_), sg_m(sg_m_), sg_n(sg_n_), sg_k(sg_k_), slm_ks(slm_ks_) {
    }
};

#define HGEMM_POLICY_TRAITS(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS) PolicyTraits(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS)
PolicyTraits policy_traits[HGEMM_NUM_POLICIES] = {
    HGEMM_ENUMERATE_POLICIES(HGEMM_POLICY_TRAITS, HGEMM_COMMA)};

#define HGEMM_POLICY_NAME(WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS) "hgemm_core_" #WG_M "x" #WG_N "_" #SG_M "x" #SG_N "x" #SG_K "x" #SLM_KS "_"
const char *policy_names[HGEMM_NUM_POLICIES] = {
    HGEMM_ENUMERATE_POLICIES(HGEMM_POLICY_NAME, HGEMM_COMMA)};

template <typename scalar_t>
inline double gemm_xpu(
    sycl::queue &queue,
    scalar_t *out0,
    scalar_t *out1,
    scalar_t *out2,
    const scalar_t *a,
    const scalar_t *b,
    const int m,
    const int n,
    const int k,
    const int policy_id) {
    auto event = policies[policy_id](queue, out0, out1, out2, a, b, m, n, k);
    return timeit(event);
}

struct gemm_sizes {
    int m, n, k;
    float alpha, beta;
    gemm_sizes(int m_, int n_, int k_, float a = 1.2, float b = 0.0) :
        m(m_), n(n_), k(k_), alpha(a), beta(b) {
        assert(b == 0.0);
    }
};

int main(int argc, char *argv[]) {
    std::cout << "hgemm_qkv_xetla_row_tuning\n";
    sycl::queue queue(
        sycl::gpu_selector_v,
        cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()});
    using scalar_t = sycl::half;

    std::vector<gemm_sizes> sizes;

    int arg_m = std::atoi(argv[1]);
    int arg_n = std::atoi(argv[2]);
    int arg_k = std::atoi(argv[3]);
    sizes.push_back(gemm_sizes(arg_m, arg_n, arg_k));

    for (auto size : sizes) {
        int m = size.m;
        int n = size.n;
        int k = size.k;
        auto alpha = size.alpha;
        auto beta = size.beta;
        uint64_t memory_size = (m * k + 3 * k * n + 3 * m * n) * sizeof(scalar_t);
        uint64_t rounds = (memory_size + LLC_SIZE - 1) / memory_size + 1;

        auto a_cpu = new scalar_t[m * k];
        auto b_cpu = new scalar_t[3 * k * n];
        for (int i = 0; i < m * k; i++)
            a_cpu[i] = static_cast<scalar_t>((float)rand() / (float)RAND_MAX);
        for (int i = 0; i < k * n; i++)
            b_cpu[i] = static_cast<scalar_t>((float)rand() / (float)RAND_MAX);

        scalar_t **a_xpu_pool = new scalar_t *[rounds];
        scalar_t **b_xpu_pool = new scalar_t *[rounds];
        scalar_t **out0_xpu_pool = new scalar_t *[rounds];
        scalar_t **out1_xpu_pool = new scalar_t *[rounds];
        scalar_t **out2_xpu_pool = new scalar_t *[rounds];
        for (int i = 0; i < rounds; i++) {
            int index = i % rounds;
            auto &a_xpu = a_xpu_pool[index];
            auto &b_xpu = b_xpu_pool[index];
            auto &out0_xpu = out0_xpu_pool[index];
            auto &out1_xpu = out1_xpu_pool[index];
            auto &out2_xpu = out2_xpu_pool[index];
            a_xpu = sycl::aligned_alloc_device<scalar_t>(256, m * k, queue);
            b_xpu = sycl::aligned_alloc_device<scalar_t>(256, 3 * k * n, queue);
            out0_xpu = sycl::aligned_alloc_device<scalar_t>(256, m * n, queue);
            out1_xpu = sycl::aligned_alloc_device<scalar_t>(256, m * n, queue);
            out2_xpu = sycl::aligned_alloc_device<scalar_t>(256, m * n, queue);
            queue.memcpy(a_xpu, a_cpu, m * k * sizeof(scalar_t)).wait();
            queue.memcpy(b_xpu, b_cpu, 3 * k * n * sizeof(scalar_t)).wait();
        }

        constexpr int WARM_UP = 10;
        constexpr int RUNS = 100;

        int min_policy_id;
        double min_policy_timems = 99999999.99;
        for (int policy_id = 0; policy_id < HGEMM_NUM_POLICIES; policy_id++) {
            double total_timems = 0.0;
            for (int i = 0; i < WARM_UP + RUNS; i++) {
                int index = i % rounds;
                auto &a_xpu = a_xpu_pool[index];
                auto &b_xpu = b_xpu_pool[index];
                auto &out0_xpu = out0_xpu_pool[index];
                auto &out1_xpu = out1_xpu_pool[index];
                auto &out2_xpu = out2_xpu_pool[index];
                double timems = gemm_xpu<scalar_t>(queue, out0_xpu, out1_xpu, out2_xpu, a_xpu, b_xpu, m, n, k, policy_id);
                if (i >= WARM_UP)
                    total_timems += timems;
            }
            double policy_timems = total_timems / RUNS;

            auto traits = policy_traits[policy_id];
            uint32_t group_range_m = (m + traits.wg_m - 1) / traits.wg_m;
            uint32_t group_range_n = (n + traits.wg_n - 1) / traits.wg_n;
            uint32_t thread_range_m = traits.wg_m / traits.sg_m;
            uint32_t thread_range_n = traits.wg_n / traits.sg_n;
            auto slm_ks = traits.slm_ks;
            std::cout << "{ \"policy\":" << policy_id << ", \"m\":" << m << ", \"n\":" << n << ", \"k\":" << k << ", \"n_ss\":" << group_range_m * group_range_n;
            std::cout << ", \"N_SG_PER_SS\":" << slm_ks * thread_range_m * thread_range_n << ", \"WG_M\":" << traits.wg_m << ", \"WG_N\":" << traits.wg_n;
            std::cout << ", \"SG_M\":" << traits.sg_m << ", \"SG_N\":" << traits.sg_n << ", \"SG_K\":" << traits.sg_k << ", \"SLM_KS\":" << slm_ks;
            std::cout << ", \"timems\":" << policy_timems;

            double total_bytes = ((double)m * k + 3 * k * n + 3 * m * n) * sizeof(scalar_t);
            if (beta != 0.0f) total_bytes += m * n * sizeof(scalar_t);
            double total_gbytes = total_bytes / 1000.0 / 1000 / 1000;
            double total_flop = (double)2 * m * n * k * 3;
            double tflops = total_flop / (policy_timems / 1000) * 1e-12;
            std::cout << ", \"gbps\":" << total_gbytes / (policy_timems / 1000.0)
                      << ", \"tflops\":" << tflops << ", \"compute_pressure\":" << total_flop / total_bytes << " }\n";

            if (policy_timems < min_policy_timems) {
                min_policy_timems = policy_timems;
                min_policy_id = policy_id;
            }
        }

        std::cout << "min_policy_id=" << min_policy_id << ", min_policy_timems=" << min_policy_timems << std::endl;

        delete[] a_cpu;
        delete[] b_cpu;
        for (int i = 0; i < rounds; i++) {
            sycl::free(a_xpu_pool[i], queue);
            sycl::free(b_xpu_pool[i], queue);
            sycl::free(out0_xpu_pool[i], queue);
            sycl::free(out1_xpu_pool[i], queue);
            sycl::free(out2_xpu_pool[i], queue);
        }
        delete[] a_xpu_pool;
        delete[] b_xpu_pool;
        delete[] out0_xpu_pool;
        delete[] out1_xpu_pool;
        delete[] out2_xpu_pool;
    }

    return 0;
}
