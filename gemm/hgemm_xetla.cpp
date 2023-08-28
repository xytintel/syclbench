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

using namespace cl::sycl;
using namespace gpu;
using namespace gpu::xetla;
using namespace gpu::xetla::group;
using namespace gpu::xetla::kernel;
using namespace gpu::xetla::subgroup;

template <typename dtype_in_>
struct alpha_beta_t {
    using dtype_in = dtype_in_;
    using mem_desc_in_t = mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>;
    using shape_t = typename mem_desc_in_t::shape_t;
    using coord_t = typename mem_desc_in_t::coord_t;
    using base_t = typename mem_desc_in_t::base_t;

    struct arguments_t {
        shape_t shape;
        base_t base;
        dtype_in alpha, beta;
        inline arguments_t() = default;
        inline arguments_t(base_t base_, shape_t shape_, dtype_in alpha_, dtype_in beta_) :
            base(base_), shape(shape_), alpha(alpha_), beta(beta_) {
        }
    };
    template <typename matAcc_t>
    __XETLA_API KERNEL_FUNC void operator()(matAcc_t &matAcc,
                                            const coord_t &coord, const arguments_t &args,
                                            uint32_t slm_base = 0, uint32_t nbarrier_base = 0) {
        using dtype_acc = typename matAcc_t::dtype;
        static constexpr uint32_t tile_size_x = matAcc_t::tile_size_x;
        static constexpr uint32_t tile_size_y = matAcc_t::tile_size_y;
        static constexpr uint32_t block_size_x = matAcc_t::block_size_x;
        static constexpr uint32_t block_size_y = matAcc_t::block_size_y;
        static constexpr int32_t num_block_x = matAcc_t::num_block_x;
        static constexpr int32_t num_block_y = matAcc_t::num_block_y;
        static constexpr uint32_t tile_elems = matAcc_t::tile_elems;
        static constexpr uint32_t block_elems = matAcc_t::block_elems;

        using mat_in_tile_desc_t = tile_desc_t<tile_size_x, tile_size_y,
                                               block_size_x, block_size_y, reg_layout::tiled>;
        using mat_in_tile_t = tile_t<dtype_in, mat_in_tile_desc_t>;
        using mat_in_payload_t = mem_payload_t<dtype_in, mat_in_tile_desc_t,
                                               msg_type_v<mat_in_tile_desc_t, mem_desc_in_t::space>,
                                               mem_desc_in_t::layout, mem_desc_in_t::space, gpu_arch::Xe>;
        using mat_in_tile_acc_t = tile_t<dtype_acc, mat_in_tile_desc_t>;
        mem_desc_in_t mem_desc_in(args.base, args.shape, coord);
        mat_in_tile_t mat_in;
        mat_in_payload_t mat_in_payload(mem_desc_in);
        tile_load<cache_hint::cached, cache_hint::cached>(
            mat_in, mat_in_payload);
        mat_in_tile_acc_t mat_in_acc;
        elemwise_cvt(mat_in_acc, mat_in);

#pragma unroll
        for (int i = 0; i < tile_size_y / block_size_y; i++) {
#pragma unroll
            for (int j = 0; j < num_block_x; j++) {
                auto dst_reg = matAcc.reg.xetla_select<block_elems, 1>(
                    (i * num_block_x + j) * block_elems);
                auto src_reg = mat_in_acc.reg.xetla_select<block_elems, 1>(
                    (i * num_block_x + j) * block_elems);
                dst_reg = args.alpha * dst_reg + args.beta * src_reg;
            }
        }
        // process the tail
        if constexpr ((tile_size_y % block_size_y) != 0) {
            constexpr uint32_t tail_start_y = tile_size_y / block_size_y * block_size_y;
            constexpr int32_t tail_size_y = tile_size_y % block_size_y;
            constexpr int32_t tail_block_elems = tail_size_y * block_size_x;
#pragma unroll
            for (int j = 0; j < num_block_x; j++) {
                auto dst_reg = matAcc.reg.xetla_select<tail_block_elems, 1>(
                    tail_start_y * tile_size_x + j * tail_block_elems);
                auto src_reg = mat_in_acc.reg.xetla_select<tail_block_elems, 1>(
                    tail_start_y * tile_size_x + j * tail_block_elems);
                dst_reg = args.alpha * dst_reg + args.beta * src_reg;
            }
        }
    }
};

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
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k,
    const float alpha,
    const float beta) {
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
            using epilogue_t = epilogue_t<
                epilogue_policy_tile_op<
                    chained_tile_op_t<
                        alpha_beta_t<data_type_c>>,
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
                ldc,
                {{{const_cast<scalar_t *>(out), {n, m, n}, alpha, beta}}});
            slm_barrier_init<gemm_op_t>();
            gemm_op_t gemm_op;
            gemm_op(ei, arg);
        });
    });
    gpu_event.wait();
    return gpu_event;
}

#define HGEMM_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K) \
    hgemm_core_##WG_M##x##WG_N##_##SG_M##x##SG_N##x##SG_K##_
#define HGEMM_IMPL(WG_M, WG_N, SG_M, SG_N, SG_K)                                                     \
    sycl::event HGEMM_IMPL_NAME(WG_M, WG_N, SG_M, SG_N, SG_K)(                                       \
        sycl::queue & queue,                                                                         \
        sycl::half * out,                                                                            \
        const sycl::half *a,                                                                         \
        const sycl::half *b,                                                                         \
        const uint32_t m,                                                                            \
        const uint32_t n,                                                                            \
        const uint32_t k,                                                                            \
        const float alpha,                                                                           \
        const float beta) {                                                                          \
        return gemm_core<sycl::half, WG_M, WG_N, SG_M, SG_N, SG_K, 32 / (WG_M * WG_N / SG_M / SG_N), \
                         1, 1, 3, true, true>(queue, out, a, b, m, n, k, alpha, beta);               \
    }

// clang-format off
#define HGEMM_COMMA ,
#define HGEMM_NUM_POLICIES 23
#define HGEMM_ENUMERATE_POLICIES(_, T) \
  _(8, 64, 8, 16, 32)T      \
  _(8, 256, 8, 16, 16)T     \
  _(8, 512, 8, 16, 16)T     \
  _(16, 64, 16, 16, 16)T    \
  _(16, 256, 16, 16, 16)T   \
  _(16, 512, 16, 16, 16)T   \
  _(32, 64, 32, 16, 16)T    \
  _(32, 64, 8, 16, 16)T     \
  _(32, 128, 32, 16, 16)T   \
  _(32, 256, 32, 16, 16)T   \
  _(32, 512, 32, 16, 16)T   \
  _(64, 128, 64, 16, 16)T   \
  _(64, 256, 64, 16, 16)T   \
  _(64, 512, 64, 16, 16)T   \
  _(128, 128, 32, 32, 32)T  \
  _(128, 256, 64, 16, 16)T  \
  _(128, 512, 64, 32, 16)T  \
  _(256, 256, 64, 32, 16)T  \
  _(256, 256, 32, 64, 16)T  \
  _(256, 256, 32, 64, 32)T  \
  _(128, 64, 16, 16, 64)T   \
  _(128, 128, 16, 32, 64)T  \
  _(128, 256, 32, 32, 16)T
// clang-format on

HGEMM_ENUMERATE_POLICIES(HGEMM_IMPL, )
sycl::event (*policies[HGEMM_NUM_POLICIES])(sycl::queue &, sycl::half *, const sycl::half *, const sycl::half *,
                                            const uint32_t, const uint32_t, const uint32_t, const float, const float) = {
    HGEMM_ENUMERATE_POLICIES(HGEMM_IMPL_NAME, HGEMM_COMMA)};

struct PolicyTraits {
    int wg_m, wg_n;
    int sg_m, sg_n, sg_k;
    PolicyTraits(int wg_m_, int wg_n_, int sg_m_, int sg_n_, int sg_k_) :
        wg_m(wg_m_), wg_n(wg_n_), sg_m(sg_m_), sg_n(sg_n_), sg_k(sg_k_) {
    }
};

#define HGEMM_POLICY_TRAITS(WG_M, WG_N, SG_M, SG_N, SG_K) PolicyTraits(WG_M, WG_N, SG_M, SG_N, SG_K)
PolicyTraits policy_traits[HGEMM_NUM_POLICIES] = {
    HGEMM_ENUMERATE_POLICIES(HGEMM_POLICY_TRAITS, HGEMM_COMMA)};

#define HGEMM_POLICY_NAME(WG_M, WG_N, SG_M, SG_N, SG_K) "hgemm_core_" #WG_M "x" #WG_N "_" #SG_M "x" #SG_N "x" #SG_K
const char *policy_names[HGEMM_NUM_POLICIES] = {
    HGEMM_ENUMERATE_POLICIES(HGEMM_POLICY_NAME, HGEMM_COMMA)};

inline int select_gemm_config(
    const int m,
    const int n,
    const int k,
    const bool is_b_row_major,
    const int TOTAL_SS = 64) {
    struct gemm_cfg_meta {
        float wg_eff;
        float num_ss;
        float aspect_r;
        int idx;
    };

    std::vector<gemm_cfg_meta> metas;
    for (int i = 0; i < HGEMM_NUM_POLICIES; i++) {
        auto traits = policy_traits[i];
        gemm_cfg_meta meta;
        int wg_m = traits.wg_m;
        int wg_n = traits.wg_n;
        int ms = (m + wg_m - 1) / wg_m;
        int ns = (n + wg_n - 1) / wg_n;
        meta.num_ss = ms * ns;
        int vm = m > wg_m ? wg_m : m;
        int vn = n > wg_n ? wg_n : n;
        meta.wg_eff = (float)vm * vn / (float)wg_m / (float)wg_n;
        meta.idx = i;
        meta.aspect_r = std::max((float)wg_m / wg_n, (float)wg_n / wg_m);
        metas.push_back(meta);
    }
    std::sort(metas.begin(), metas.end(), [](const auto &lhs, const auto &rhs) {
        if (lhs.wg_eff != rhs.wg_eff)
            return lhs.wg_eff > rhs.wg_eff;
        else if (lhs.num_ss != rhs.num_ss)
            return lhs.num_ss < rhs.num_ss;
        else
            return lhs.aspect_r < rhs.aspect_r;
    });
    int idx = metas[0].idx;
    return is_b_row_major ? idx : idx + HGEMM_NUM_POLICIES;
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
    const float alpha,
    const float beta,
    const bool is_warmup) {
    auto auto_selected_policy_id = select_gemm_config(m, n, k, true, 64);
    if (!is_warmup) {
        for (int i = 0; i < HGEMM_NUM_POLICIES; i++) {
            auto out_ = sycl::aligned_alloc_device<scalar_t>(256, m * n, queue);
            queue.memcpy(out_, out, m * n * sizeof(scalar_t)).wait();
            flush_cache(queue);
            auto traits = policy_traits[i];
            uint32_t group_range_m = (m + traits.wg_m - 1) / traits.wg_m;
            uint32_t group_range_n = (n + traits.wg_n - 1) / traits.wg_n;
            uint32_t thread_range_m = traits.wg_m / traits.sg_m;
            uint32_t thread_range_n = traits.wg_n / traits.sg_n;
            auto slm_ks = 32 / (traits.wg_m * traits.wg_n / traits.sg_m / traits.sg_n);
            std::cout << "policy=" << i << ", m=" << m << ", n=" << n << ", k=" << k << ", n_ss=" << group_range_m * group_range_n << ", N_SG_PER_SS=" << slm_ks * thread_range_m * thread_range_n << ", WG_M=" << traits.wg_m << ", WG_N=" << traits.wg_n << ", SG_M=" << traits.sg_m << ", SG_N=" << traits.sg_n << ", SG_K=" << traits.sg_k << ", SLM_KS=" << slm_ks;
            auto event = policies[i](queue, out_, a, b, m, n, k, alpha, beta);
            auto timems = timeit(event);
            std::cout << ", timems=" << timems << std::endl;
            sycl::free(out_, queue);
        }
        std::cout << "auto_selected_policy_id=" << auto_selected_policy_id << "\n";
    }
    auto event = policies[auto_selected_policy_id](queue, out, a, b, m, n, k, alpha, beta);
    return timeit(event);
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
            });
    });
    event.wait();
}

struct gemm_sizes {
    int m, n, k;
    float alpha, beta;
    gemm_sizes(int m_, int n_, int k_, float a = 1.2, float b = 0.5) :
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

    sizes.push_back(gemm_sizes(100, 4096, 4096));
    sizes.push_back(gemm_sizes(8192, 8192, 8192));

    sizes.push_back(gemm_sizes(4, 16384, 2048));
    sizes.push_back(gemm_sizes(4, 16384, 4096));
    sizes.push_back(gemm_sizes(4, 16384, 8192));

    sizes.push_back(gemm_sizes(9, 16384, 2048));
    sizes.push_back(gemm_sizes(9, 16384, 4096));
    sizes.push_back(gemm_sizes(9, 16384, 8192));

    sizes.push_back(gemm_sizes(40, 8192, 2048));
    sizes.push_back(gemm_sizes(40, 16384, 4096));
    sizes.push_back(gemm_sizes(40, 8192, 8192));

    sizes.push_back(gemm_sizes(16384, 4, 2048));
    sizes.push_back(gemm_sizes(16384, 4, 4096));
    sizes.push_back(gemm_sizes(16384, 4, 8192));

    for (auto size : sizes) {
        int m = size.m;
        int n = size.n;
        int k = size.k;
        auto alpha = size.alpha;
        auto beta = size.beta;

        for (int count = 0; count < 3; count++) {
            bool is_warmup = count < 2;

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

            flush_cache(queue);

            gemm_xpu_ref<scalar_t>(queue, out_xpu_ref, a_xpu_ref, b_xpu_ref, m, n, k,
                                   alpha, beta);
            auto timems =
                gemm_xpu<scalar_t>(queue, out_xpu, a_xpu, b_xpu, m, n, k, alpha, beta, is_warmup);

            double total_bytes = ((double)m * k + k * n + m * n) * sizeof(scalar_t);
            if (beta != 0.0f) total_bytes += m * n * sizeof(scalar_t);
            double total_gbytes = total_bytes / 1000.0 / 1000 / 1000;
            double total_flop = (double)2 * m * n * k;
            double tflops = total_flop / (timems / 1000) * 1e-12;

            if (!is_warmup) {
                std::cout << "timems=" << timems << ", gbps=" << total_gbytes / (timems / 1000.0)
                          << ", tflops=" << tflops << ", compute_pressure=" << total_flop / total_bytes << "\n";
            }

            using MaxDiff = CompareMaxdiff<scalar_t>;
            auto diff = MaxDiff(queue, out_xpu_ref, MaxDiff::XPU, out_xpu, MaxDiff::XPU, m * n);
            auto maxdiff = diff();

            assert(maxdiff <= (k / 4096.0 * 1.01));
            if (!is_warmup)
                std::cout << "maxdiff=" << maxdiff << std::endl;

            sycl::free(a_xpu, queue);
            sycl::free(b_xpu, queue);
            sycl::free(out_xpu, queue);
            sycl::free(a_xpu_ref, queue);
            sycl::free(b_xpu_ref, queue);
            sycl::free(out_xpu_ref, queue);
            delete[] a_cpu;
            delete[] b_cpu;
            delete[] out_cpu;
        }
    }
    return 0;
}
