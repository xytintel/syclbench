#include "xetla.hpp"
#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

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

using namespace cl::sycl;
using namespace gpu::xetla;
using namespace gpu;

template <typename scalar_t>
float gemm_xpu(sycl::queue &q, scalar_t *out, const scalar_t *a,
               const scalar_t *b, const int m, const int n, const int k,
               const scalar_t alpha, const scalar_t beta) {
  float time;
  uint32_t matrix_m = m;
  uint32_t matrix_n = n;
  uint32_t matrix_k = k;
  // Define the shape of workgroup and subgroup
  // It's tunable parameters based on different input shape and hardware for
  // better performance
  constexpr uint32_t wg_tile_m = 256;
  constexpr uint32_t wg_tile_n = 256;
  constexpr uint32_t sg_tile_m = 32;
  constexpr uint32_t sg_tile_n = 64;

  // Workload mapping, linear mapping will be used in the code
  // Suppose it is divisible.
  uint32_t group_range_m = matrix_m / wg_tile_m;
  uint32_t group_range_n = matrix_n / wg_tile_n;

  // Each subgroup will be executed in one hardware thread
  // Calculate how many threads in a workgroup
  uint32_t thread_range_m = wg_tile_m / sg_tile_m;
  uint32_t thread_range_n = wg_tile_n / sg_tile_n;

  // leading dimension
  uint32_t lda = matrix_k;
  uint32_t ldb = matrix_n;
  uint32_t ldc = matrix_n;

  // Ndrange and workgroup shape
  cl::sycl::range<3> GroupRange{1, group_range_m, group_range_n};
  cl::sycl::range<3> LocalRange{1, thread_range_m, thread_range_n};

  cl::sycl::nd_range<3> NDRange(GroupRange * LocalRange, LocalRange);

  auto gpu_event = q.submit([&](handler &cgh) {
    // GPU kernel
    cgh.parallel_for(NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
      using namespace gpu::xetla;
      using namespace gpu::xetla::group;
      using namespace gpu::xetla::kernel;
      using namespace gpu::xetla::subgroup;
      using data_type_b = scalar_t;
      using data_type_a = scalar_t;
      using data_type_c = scalar_t;
      using data_type_acc = float;

      // wrap the NDrange to XeTLA range
      xetla_exec_item<3> ei(item);

      // Step 1: basic computation information
      // define A, B and accumulator datatype
      // Using float as accumuator for better accuracy
      using compute_attr =
          compute_attr_t<data_type_a, data_type_b, data_type_acc>;

      // Performance tuning setting based on different shapes
      static constexpr uint32_t periodic_sync_interval = 8;
      static constexpr uint32_t prefetch_distance = 3;
      // should larger than 8
      static constexpr uint32_t k_iter_num = 32;
      using perf_tuning_knob = perf_tuning_knob_t<k_iter_num, prefetch_distance,
                                                  periodic_sync_interval>;

      // specific the computation, performance tuning and computation core
      using compute_policy =
          compute_policy_default_xmx<compute_attr, perf_tuning_knob,
                                     gpu_arch::Xe>;

      // Step 2: define the memory layout & location of input/output
      // this setting could be used to optimize the data re-use in shared
      // local memory
      using mem_desc_input_a =
          mem_desc_t<data_type_a, mem_layout::row_major, mem_space::global>;
      using mem_desc_input_b =
          mem_desc_t<data_type_b, mem_layout::row_major, mem_space::global>;
      using mem_desc_output_c =
          mem_desc_t<data_type_c, mem_layout::row_major, mem_space::global>;

      // Step 3: define mirco-kernel's configuration
      using tile_shape =
          tile_shape_t<wg_tile_n, wg_tile_m, sg_tile_n, sg_tile_m>;
      using brgemm_t = brgemm_t<compute_policy, tile_shape, mem_desc_input_a,
                                mem_desc_input_b>;
      brgemm_t brgemm;

      // Step 4: epilogue function to overwrite the result
      using epilogue_t = epilogue_t<epilogue_policy_default<gpu_arch::Xe>,
                                    tile_shape, mem_desc_output_c>;

      // Step 5: define the shared local memory usages
      // developers have the responsibility to set
      // shared loacal memory through XeTLA API
      static constexpr uint32_t barrier_count = brgemm_t::barrier_count;
      static constexpr uint32_t slm_size = brgemm_t::slm_size;
      xetla_nbarrier_init<barrier_count>();
      xetla_local_init<slm_size>();

      // Step 6: ecah workgroup gets it individual index to start computation
      int start_n = ei.get_group(2) * wg_tile_n;
      int start_m = ei.get_group(1) * wg_tile_m;
      // no slicing in K direction so start from zero for all WG
      int start_k = 0;

      // Each workgroup will compute all data in K based on no k_sliciing
      // The developer can set how much data a subgroup compute by k_iter_num
      uint32_t wg_tile_k = matrix_k;
      uint32_t inner_loop_count = wg_tile_k / k_iter_num;

      // Step 7: define the workgroup start point for each workgroup
      mem_desc_input_a md_a({const_cast<scalar_t *>(a)},
                            {matrix_k, matrix_m, lda}, {start_k, start_m});
      mem_desc_input_b md_b({const_cast<scalar_t *>(b)},
                            {matrix_n, matrix_k, ldb}, {start_n, start_k});
      mem_desc_output_c md_c({out}, {matrix_n, matrix_m, ldc},
                             {start_n, start_m});

      // Step 8: real calculation with accumulator varibales which suppose
      // will be in register.
      typename brgemm_t::matAcc_t matAcc;
      matAcc.init(0);

      typename brgemm_t::arguments_t brgemm_args(md_a, md_b, inner_loop_count);

      // the results is in the matAcc rather than real output C
      typename brgemm_t::work_group_t g(ei.get_local_linear_id());
      brgemm(g, matAcc, brgemm_args);

      // Step 9: write the results from matACC to real output C
      epilogue_t epilogue;
      epilogue(g, matAcc, md_c);
    });
  });
  gpu_event.wait();
  time = timeit(gpu_event);

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
      acc += (float)a[mi * k + ki] * (float)b[ki * n + ni];
    }
    auto r = (float)alpha * acc;
    out[mi * n + ni] = acc; // r + (float)beta * (float)out[mi * n + ni];
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
  gemm_sizes(int m_, int n_, int k_, float a, float b)
      : m(m_), n(n_), k(k_), alpha(a), beta(b) {}
};

int main() {
  std::cout << "hgemm_xetla\n";
  sycl::queue queue(
      sycl::gpu_selector_v,
      cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()});
  using scalar_t = sycl::half;

  std::vector<gemm_sizes> sizes;
  // sizes.push_back(gemm_sizes(512, 512, 512, 0.5, 0.5));
  sizes.push_back(gemm_sizes(4096, 4096, 4096, 0.5, 0.5));
  sizes.push_back(gemm_sizes(4096, 4096, 4096, 0.5, 0.5));
  sizes.push_back(gemm_sizes(4096, 4096, 4096, 0.5, 0.5));
  // sizes.push_back(gemm_sizes(4, 4096, 4096, 0.5, 0.5));
  // sizes.push_back(gemm_sizes(1028, 1028, 1028, 0.5, 0.5));
  // sizes.push_back(gemm_sizes(2048, 2048, 2048, 0.5, 0.5));
  // sizes.push_back(gemm_sizes(4096, 4096, 4096, 0.5, 0.5));
  // sizes.push_back(gemm_sizes(8192, 8192, 8192, 0.5, 0.5));
  // sizes.push_back(gemm_sizes(1<<14, 1<<14, 1<<14, 0.5, 0.5));

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
        gemm_xpu<scalar_t>(queue, out_xpu, a_xpu, b_xpu, m, n, k, alpha, beta);

    double total_gbytes = ((double)m * k + k * n + m * n + m * n) *
                          sizeof(scalar_t) / 1000.0 / 1000 / 1000;
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
