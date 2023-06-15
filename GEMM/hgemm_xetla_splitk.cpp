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

template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) aligned_array {
    T val[vec_size];
};

template <typename in_t, typename out_t, int vec_size, typename item_t>
inline void dtype_cast_kernel(
    item_t &item, const in_t *in, out_t *out,
    const size_t n, const float alpha, const float beta) {
    const int group_work_size = item.get_local_range(0) * vec_size;
    auto index = item.get_group(0) * group_work_size + item.get_local_id(0) * vec_size;
    auto remaining = n - index;
    if (remaining < vec_size) {
        for (auto i = index; i < n; i++) {
            out[i] = (out_t)(alpha * (float)in[i] + beta * (float)out[i]);
        }
    } else {
        using in_vec_t = aligned_array<in_t, vec_size>;
        using out_vec_t = aligned_array<out_t, vec_size>;
        auto in_vec = *reinterpret_cast<in_vec_t *>(const_cast<in_t *>(&in[index]));
        auto out_vec = *reinterpret_cast<out_vec_t *>(&out[index]);
#pragma unroll
        for (int i = 0; i < vec_size; i++) {
            out_vec.val[i] = (out_t)(alpha * (float)in_vec.val[i] + beta * (float)out_vec.val[i]);
        }
        *reinterpret_cast<out_vec_t *>(&out[index]) = out_vec;
    }
}

template <typename in_t, typename out_t, int vec_size>
float dtype_cast(sycl::queue &q, const in_t *in, out_t *out, size_t n, const float alpha, const float beta) {
    const int group_size = 256;
    const int group_work_size = group_size * vec_size;
    auto event = q.submit([&](sycl::handler &h) {
        h.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>((n + group_work_size - 1) / group_work_size * group_size),
                sycl::range<1>(group_size)),
            [=](sycl::nd_item<1> item) {
                dtype_cast_kernel<in_t, out_t, vec_size>(item, in, out, n, alpha, beta);
            });
    });
    event.wait();
    return timeit(event);
}

template <
    typename scalar_t,
    int WG_M = 8,
    int WG_N = 32,
    int SG_M = 8,
    int SG_N = 16,
    int KS = 16,
    int KN = 32,
    bool B_ROW_MAJOR = false>
float gemm_xpu(
    sycl::queue &queue,
    scalar_t *output,
    float *acc,
    const scalar_t *a,
    const scalar_t *b,
    const int m,
    const int n,
    const int k,
    const float alpha,
    const float beta) {
    uint32_t matrix_m = m;
    uint32_t matrix_n = n;
    uint32_t matrix_k = k;
    constexpr uint32_t split_k_S = KS;
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
    cl::sycl::range<3> GroupRange{split_k_S, group_range_m, group_range_n};
    cl::sycl::range<3> LocalRange{1, thread_range_m, thread_range_n};
    cl::sycl::nd_range<3> NDRange(GroupRange * LocalRange, LocalRange);

    auto gpu_event = queue.submit([&](handler &cgh) {
        cgh.parallel_for(NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
            xetla_exec_item<3> ei(item);

            using data_type_b = scalar_t;
            using data_type_a = scalar_t;
            using data_type_c = float;
            using data_type_acc = float;
            using compute_attr =
                compute_attr_t<data_type_a, data_type_b, data_type_acc>;

            static constexpr uint32_t periodic_sync_interval = 8;
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
                                                                  result_reduce_sum, gpu_arch::Xe>,
                                          tile_shape, mem_desc_output_c>;

            static constexpr uint32_t barrier_count = brgemm_t::barrier_count;
            static constexpr uint32_t slm_size = brgemm_t::slm_size;
            xetla_nbarrier_init<barrier_count>();
            xetla_local_init<slm_size>();
            int start_n = ei.get_group(2) * wg_tile_n;
            int start_m = ei.get_group(1) * wg_tile_m;
            int split_k_R = matrix_k / split_k_S;
            int split_k_E = ei.get_group(0) * split_k_R;
            uint32_t split_k_D = split_k_E + split_k_R;
            int start_k = split_k_E;
            uint32_t wg_tile_k = split_k_R;
            uint32_t inner_loop_count = wg_tile_k / k_iter_num;

            mem_desc_input_a md_a({const_cast<scalar_t *>(a)},
                                  {split_k_D, matrix_m, lda}, {start_k, start_m});
            mem_desc_input_b md_b({const_cast<scalar_t *>(b)},
                                  {matrix_n, split_k_D, ldb}, {start_n, start_k});
            mem_desc_output_c md_c({acc}, {matrix_n, matrix_m, ldc},
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
    dtype_cast<float, scalar_t, 4>(queue, acc, output, m * n, alpha, beta);
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
    std::cout << "hgemm_xetla_spk\n";
    sycl::queue queue(
        sycl::gpu_selector_v,
        cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()});
    using scalar_t = sycl::half;

    std::vector<gemm_sizes> sizes;

    sizes.push_back(gemm_sizes(1, 4096, 4096, 0.5, 0.5));
    sizes.push_back(gemm_sizes(1, 16384, 4096, 0.5, 0.5));
    sizes.push_back(gemm_sizes(1, 4096, 16384, 0.5, 0.5));
    sizes.push_back(gemm_sizes(1, 32000, 4096, 0.5, 0.5));

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
        auto acc_xpu = sycl::aligned_alloc_device<float>(256, m * n, queue);
        auto out_xpu = sycl::aligned_alloc_device<scalar_t>(256, m * n, queue);
        queue.memcpy(a_xpu, a_cpu, m * k * sizeof(scalar_t)).wait();
        queue.memcpy(b_xpu, b_cpu, k * n * sizeof(scalar_t)).wait();
        queue.memcpy(out_xpu, out_cpu, m * n * sizeof(scalar_t)).wait();
        queue.fill<float>(acc_xpu, 0.0, m * n).wait();

        auto a_xpu_ref = sycl::aligned_alloc_device<scalar_t>(256, m * k, queue);
        auto b_xpu_ref = sycl::aligned_alloc_device<scalar_t>(256, k * n, queue);
        auto out_xpu_ref = sycl::aligned_alloc_device<scalar_t>(256, m * n, queue);
        queue.memcpy(a_xpu_ref, a_cpu, m * k * sizeof(scalar_t)).wait();
        queue.memcpy(b_xpu_ref, b_cpu, k * n * sizeof(scalar_t)).wait();
        queue.memcpy(out_xpu_ref, out_cpu, m * n * sizeof(scalar_t)).wait();

        gemm_xpu_ref<scalar_t>(queue, out_xpu_ref, a_xpu_ref, b_xpu_ref, m, n, k,
                               alpha, beta);
        auto timems =
            gemm_xpu<scalar_t>(queue, out_xpu, acc_xpu, a_xpu, b_xpu, m, n, k, alpha, beta);

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
        sycl::free(acc_xpu, queue);
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
