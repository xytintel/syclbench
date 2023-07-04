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

template <typename dtype_in_>
struct xres_op_t {
    using dtype_in = dtype_in_;
    using mem_desc_in_t = mem_desc_t<dtype_in, mem_layout::row_major, mem_space::global>;
    using shape_t = typename mem_desc_in_t::shape_t;
    using coord_t = typename mem_desc_in_t::coord_t;
    using base_t = typename mem_desc_in_t::base_t;

    struct arguments_t {
        shape_t shape;
        base_t base;
        dtype_in x;
        inline arguments_t() = default;
        inline arguments_t(base_t base_, shape_t shape_, dtype_in x_) :
            base(base_), shape(shape_), x(x_) {
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
                dst_reg = dst_reg + args.x * src_reg;
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
                dst_reg = dst_reg + args.x * src_reg;
            }
        }
    }
};

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
float gemm_xpu(
    sycl::queue &queue,
    scalar_t *out,
    const scalar_t *a,
    const scalar_t *b,
    const scalar_t *bias,
    const scalar_t *res,
    const int m,
    const int n,
    const int k,
    const scalar_t x) {
    uint32_t matrix_m = m;
    uint32_t matrix_n = n;
    uint32_t matrix_k = k;
    constexpr uint32_t slm_kslicing = SLM_KS;
    constexpr uint32_t l3_kslicing = L3_KS;
    constexpr uint32_t wg_tile_m = WG_M;
    constexpr uint32_t wg_tile_n = WG_N;
    constexpr uint32_t sg_tile_m = SG_M;
    constexpr uint32_t sg_tile_n = SG_N;
    static_assert(l3_kslicing == 1, "for fused op, l3_kslicing should be 1");
    constexpr mem_layout layout_a = mem_layout::row_major;
    constexpr mem_layout layout_b =
        B_ROW_MAJOR ? mem_layout::row_major : mem_layout::col_major;
    uint32_t group_range_m = (matrix_m + wg_tile_m - 1) / wg_tile_m;
    uint32_t group_range_n = (matrix_n + wg_tile_n - 1) / wg_tile_n;
    uint32_t thread_range_m = wg_tile_m / sg_tile_m;
    uint32_t thread_range_n = wg_tile_n / sg_tile_n;
    uint32_t lda = matrix_k;
    uint32_t ldb = B_ROW_MAJOR ? matrix_n : matrix_k;
    uint32_t ldc = matrix_n;
    cl::sycl::range<3> GroupRange{l3_kslicing, group_range_m, group_range_n};
    cl::sycl::range<3> LocalRange{slm_kslicing, thread_range_m, thread_range_n};
    cl::sycl::nd_range<3> NDRange(GroupRange * LocalRange, LocalRange);

    auto gpu_event = queue.submit([&](handler &cgh) {
        cgh.parallel_for(NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
            xetla_exec_item<3> ei(item);

            using data_type_b = scalar_t;
            using data_type_a = scalar_t;
            using data_type_c = scalar_t;
            using data_type_bias = scalar_t;
            using data_type_res = scalar_t;
            using data_type_acc = float;
            static constexpr uint32_t periodic_sync_interval = SYNC_FREQ;
            static constexpr uint32_t prefetch_distance = STAGES;
            using tile_shape =
                tile_shape_t<wg_tile_n, wg_tile_m, sg_tile_n, sg_tile_m>;

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
            using update_method = typename std::conditional<
                (l3_kslicing > 1),
                result_reduce_sum,
                result_overwrite>::type;
            using epilogue_t = epilogue_t<
                epilogue_policy_tile_op<
                    chained_tile_op_t<
                        bias_add_op_t<data_type_bias, gpu_arch::Xe>,
                        xres_op_t<data_type_res>>,
                    update_method,
                    gpu_arch::Xe>,
                tile_shape,
                mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
            using gemm_op_t = gemm_t<
                dispatch_policy_kslicing<l3_kslicing, slm_kslicing, gpu_arch::Xe>,
                brgemm_t,
                epilogue_t>;

            typename gemm_op_t::arguments_t arg(
                matrix_m,
                matrix_k,
                matrix_n,
                const_cast<scalar_t *>(a),
                lda,
                const_cast<scalar_t *>(b),
                ldb,
                out,
                ldc,
                {{{const_cast<scalar_t *>(bias), {matrix_n, 1, matrix_n}},
                  {const_cast<scalar_t *>(res), {matrix_n, matrix_m, matrix_n}, x}}});
            slm_barrier_init<gemm_op_t>();
            gemm_op_t gemm_op;
            gemm_op(ei, arg);
        });
    });
    gpu_event.wait();
    float time = timeit(gpu_event);
    return time;
}

template <typename scalar_t, typename item_t>
inline void gemm_xpu_ref_kernel(item_t &item, scalar_t *out, const scalar_t *a,
                                const scalar_t *b, const scalar_t *bias, const scalar_t *res,
                                const int m, const int n,
                                const int k, const scalar_t x) {
    auto mi = item.get_group(0) * 32 + item.get_local_id(0); // y
    auto ni = item.get_group(1) * 32 + item.get_local_id(1);
    if (mi < m && ni < n) {
        float acc = 0.f;
        for (int ki = 0; ki < k; ki++) {
            acc += (float)a[mi * k + ki] * (float)b[ki * n + ni];
        }
        out[mi * n + ni] = acc + (float)bias[ni] + (float)x * (float)res[mi * n + ni];
    }
}

template <typename scalar_t>
void gemm_xpu_ref(sycl::queue &q, scalar_t *out, const scalar_t *a,
                  const scalar_t *b, const scalar_t *bias, const scalar_t *res,
                  const int m, const int n, const int k, const scalar_t x) {
    auto m_groups = (m + 32 - 1) / 32; // y
    auto n_groups = (n + 32 - 1) / 32;
    auto event = q.submit([&](sycl::handler &h) {
        h.parallel_for(
            sycl::nd_range<2>(sycl::range<2>(m_groups * 32, n_groups * 32),
                              sycl::range<2>(32, 32)),
            [=](sycl::nd_item<2> item) {
                gemm_xpu_ref_kernel<scalar_t>(item, out, a, b, bias, res, m, n, k, x);
            });
    });
    event.wait();
}

struct gemm_sizes {
    int m, n, k;
    float x;
    gemm_sizes(int m_, int n_, int k_, float x_) :
        m(m_), n(n_), k(k_), x(x_) {
    }
};

int main() {
    std::cout << "hgemm_xetla_bias_xres\n";
    sycl::queue queue(
        sycl::gpu_selector_v,
        cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()});
    using scalar_t = sycl::half;

    std::vector<gemm_sizes> sizes;

    sizes.push_back(gemm_sizes(12, 4096, 4096, 0.2));
    // sizes.push_back(gemm_sizes(1, 4096, 4096, 0.5));
    // sizes.push_back(gemm_sizes(1, 16384, 4096, 0.5));
    // sizes.push_back(gemm_sizes(1, 4096, 16384, 0.5));
    // sizes.push_back(gemm_sizes(1, 32000, 4096, 0.5));

    for (auto size : sizes) {
        int m = size.m;
        int n = size.n;
        int k = size.k;
        auto x = size.x;

        std::cout << "m=" << m << ", n=" << n << ", k=" << k << ", x=" << x << "\n";

        auto a_cpu = new scalar_t[m * k];
        auto b_cpu = new scalar_t[k * n];
        auto bias_cpu = new scalar_t[n];
        auto res_cpu = new scalar_t[m * n];
        auto out_cpu = new scalar_t[m * n];

        for (int i = 0; i < m * k; i++)
            a_cpu[i] = static_cast<scalar_t>((float)rand() / (float)RAND_MAX);
        for (int i = 0; i < k * n; i++)
            b_cpu[i] = static_cast<scalar_t>((float)rand() / (float)RAND_MAX);
        for (int i = 0; i < n; i++)
            bias_cpu[i] = static_cast<scalar_t>((float)rand() / (float)RAND_MAX);
        for (int i = 0; i < m * n; i++)
            res_cpu[i] = static_cast<scalar_t>((float)rand() / (float)RAND_MAX);
        for (int i = 0; i < m * n; i++)
            out_cpu[i] = static_cast<scalar_t>((float)rand() / (float)RAND_MAX);

        auto a_xpu = sycl::aligned_alloc_device<scalar_t>(256, m * k, queue);
        auto b_xpu = sycl::aligned_alloc_device<scalar_t>(256, k * n, queue);
        auto bias_xpu = sycl::aligned_alloc_device<scalar_t>(256, n, queue);
        auto res_xpu = sycl::aligned_alloc_device<scalar_t>(256, m * n, queue);
        auto out_xpu = sycl::aligned_alloc_device<scalar_t>(256, m * n, queue);
        queue.memcpy(a_xpu, a_cpu, m * k * sizeof(scalar_t)).wait();
        queue.memcpy(b_xpu, b_cpu, k * n * sizeof(scalar_t)).wait();
        queue.memcpy(bias_xpu, bias_cpu, n * sizeof(scalar_t)).wait();
        queue.memcpy(res_xpu, res_cpu, m * n * sizeof(scalar_t)).wait();
        queue.memcpy(out_xpu, out_cpu, m * n * sizeof(scalar_t)).wait();

        auto a_xpu_ref = sycl::aligned_alloc_device<scalar_t>(256, m * k, queue);
        auto b_xpu_ref = sycl::aligned_alloc_device<scalar_t>(256, k * n, queue);
        auto bias_xpu_ref = sycl::aligned_alloc_device<scalar_t>(256, n, queue);
        auto res_xpu_ref = sycl::aligned_alloc_device<scalar_t>(256, m * n, queue);
        auto out_xpu_ref = sycl::aligned_alloc_device<scalar_t>(256, m * n, queue);
        queue.memcpy(a_xpu_ref, a_cpu, m * k * sizeof(scalar_t)).wait();
        queue.memcpy(b_xpu_ref, b_cpu, k * n * sizeof(scalar_t)).wait();
        queue.memcpy(bias_xpu_ref, bias_cpu, n * sizeof(scalar_t)).wait();
        queue.memcpy(res_xpu_ref, res_cpu, m * n * sizeof(scalar_t)).wait();
        queue.memcpy(out_xpu_ref, out_cpu, m * n * sizeof(scalar_t)).wait();

        gemm_xpu_ref<scalar_t>(queue, out_xpu_ref, a_xpu_ref, b_xpu_ref, bias_xpu_ref, res_xpu_ref, m, n, k, x);
        auto timems =
            gemm_xpu<scalar_t>(queue, out_xpu, a_xpu, b_xpu, bias_xpu, res_xpu, m, n, k, x);

        double total_gbytes = ((double)m * k + k * n + n + m * n + m * n) * sizeof(scalar_t) / 1000.0 / 1000 / 1000;
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
        sycl::free(bias_xpu, queue);
        sycl::free(res_xpu, queue);
        sycl::free(out_xpu, queue);
        sycl::free(a_xpu_ref, queue);
        sycl::free(b_xpu_ref, queue);
        sycl::free(bias_xpu_ref, queue);
        sycl::free(res_xpu_ref, queue);
        sycl::free(out_xpu_ref, queue);
        delete[] a_cpu;
        delete[] b_cpu;
        delete[] bias_cpu;
        delete[] res_cpu;
        delete[] out_cpu;
        delete[] out_xpu_;
        delete[] out_xpu_ref_;
    }
    return 0;
}
