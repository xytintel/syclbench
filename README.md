#### Usage

1. Update "xetla_base" within build.sh (To specify XETLA root path)
2. Build the GEMM benchmark
```bash
bash build.sh gemm/hgemm_xetla.cpp
# Or "bash build.sh gemm/hgemm_qkv_xetla.cpp" when you want to test qkv policy
```
3. Run the specific shape for gemm
```bash
./a.out 4 2048 4096
# ordered by m n k
```

Here is the example output

```txt
hgemm_xetla_row_tuning
{ "policy":0, "m":4, "n":2048, "k":4096, "n_ss":32, "N_SG_PER_SS":16, "WG_M":8, "WG_N":64, "SG_M":8, "SG_N":16, "SG_K":16, "SLM_KS":4, "timems":0.0389456, "gbps":432.048, "tflops":1.72314, "compute_pressure":3.98832 }
{ "policy":1, "m":4, "n":2048, "k":4096, "n_ss":32, "N_SG_PER_SS":32, "WG_M":8, "WG_N":64, "SG_M":8, "SG_N":16, "SG_K":16, "SLM_KS":8, "timems":0.0376928, "gbps":446.408, "tflops":1.78042, "compute_pressure":3.98832 }
{ "policy":2, "m":4, "n":2048, "k":4096, "n_ss":32, "N_SG_PER_SS":16, "WG_M":8, "WG_N":64, "SG_M":8, "SG_N":16, "SG_K":32, "SLM_KS":4, "timems":0.037544, "gbps":448.177, "tflops":1.78747, "compute_pressure":3.98832 }
{ "policy":3, "m":4, "n":2048, "k":4096, "n_ss":32, "N_SG_PER_SS":32, "WG_M":8, "WG_N":64, "SG_M":8, "SG_N":16, "SG_K":32, "SLM_KS":8, "timems":0.0370048, "gbps":454.708, "tflops":1.81352, "compute_pressure":3.98832 }
{ "policy":4, "m":4, "n":2048, "k":4096, "n_ss":16, "N_SG_PER_SS":16, "WG_M":8, "WG_N":128, "SG_M":8, "SG_N":16, "SG_K":16, "SLM_KS":2, "timems":0.0358256, "gbps":469.674, "tflops":1.87321, "compute_pressure":3.98832 }
{ "policy":5, "m":4, "n":2048, "k":4096, "n_ss":16, "N_SG_PER_SS":32, "WG_M":8, "WG_N":128, "SG_M":8, "SG_N":16, "SG_K":32, "SLM_KS":4, "timems":0.0352624, "gbps":477.176, "tflops":1.90313, "compute_pressure":3.98832 }
{ "policy":6, "m":4, "n":2048, "k":4096, "n_ss":8, "N_SG_PER_SS":32, "WG_M":8, "WG_N":256, "SG_M":8, "SG_N":16, "SG_K":16, "SLM_KS":2, "timems":0.0510304, "gbps":329.732, "tflops":1.31508, "compute_pressure":3.98832 }
{ "policy":7, "m":4, "n":2048, "k":4096, "n_ss":4, "N_SG_PER_SS":32, "WG_M":8, "WG_N":512, "SG_M":8, "SG_N":16, "SG_K":16, "SLM_KS":1, "timems":0.0949856, "gbps":177.147, "tflops":0.706516, "compute_pressure":3.98832 }
{ "policy":8, "m":4, "n":2048, "k":4096, "n_ss":32, "N_SG_PER_SS":32, "WG_M":16, "WG_N":64, "SG_M":8, "SG_N":16, "SG_K":16, "SLM_KS":4, "timems":0.0376848, "gbps":446.503, "tflops":1.78079, "compute_pressure":3.98832 }
{ "policy":9, "m":4, "n":2048, "k":4096, "n_ss":32, "N_SG_PER_SS":16, "WG_M":16, "WG_N":64, "SG_M":8, "SG_N":32, "SG_K":16, "SLM_KS":4, "timems":0.0379968, "gbps":442.836, "tflops":1.76617, "compute_pressure":3.98832 }
{ "policy":10, "m":4, "n":2048, "k":4096, "n_ss":32, "N_SG_PER_SS":16, "WG_M":16, "WG_N":64, "SG_M":16, "SG_N":16, "SG_K":16, "SLM_KS":4, "timems":0.0373536, "gbps":450.462, "tflops":1.79658, "compute_pressure":3.98832 }
{ "policy":11, "m":4, "n":2048, "k":4096, "n_ss":32, "N_SG_PER_SS":32, "WG_M":16, "WG_N":64, "SG_M":16, "SG_N":16, "SG_K":16, "SLM_KS":8, "timems":0.0367104, "gbps":458.354, "tflops":1.82806, "compute_pressure":3.98832 }
{ "policy":12, "m":4, "n":2048, "k":4096, "n_ss":8, "N_SG_PER_SS":32, "WG_M":16, "WG_N":256, "SG_M":8, "SG_N":16, "SG_K":16, "SLM_KS":1, "timems":0.0607104, "gbps":277.158, "tflops":1.10539, "compute_pressure":3.98832 }
{ "policy":13, "m":4, "n":2048, "k":4096, "n_ss":8, "N_SG_PER_SS":32, "WG_M":16, "WG_N":256, "SG_M":16, "SG_N":16, "SG_K":16, "SLM_KS":2, "timems":0.0516896, "gbps":325.527, "tflops":1.2983, "compute_pressure":3.98832 }
{ "policy":14, "m":4, "n":2048, "k":4096, "n_ss":4, "N_SG_PER_SS":32, "WG_M":16, "WG_N":512, "SG_M":16, "SG_N":16, "SG_K":16, "SLM_KS":1, "timems":0.097184, "gbps":173.139, "tflops":0.690534, "compute_pressure":3.98832 }
{ "policy":15, "m":4, "n":2048, "k":4096, "n_ss":32, "N_SG_PER_SS":32, "WG_M":32, "WG_N":64, "SG_M":8, "SG_N":16, "SG_K":16, "SLM_KS":2, "timems":0.03832, "gbps":439.101, "tflops":1.75128, "compute_pressure":3.98832 }
{ "policy":16, "m":4, "n":2048, "k":4096, "n_ss":32, "N_SG_PER_SS":32, "WG_M":32, "WG_N":64, "SG_M":32, "SG_N":16, "SG_K":16, "SLM_KS":8, "timems":0.0375152, "gbps":448.521, "tflops":1.78884, "compute_pressure":3.98832 }
{ "policy":17, "m":4, "n":2048, "k":4096, "n_ss":16, "N_SG_PER_SS":32, "WG_M":32, "WG_N":128, "SG_M":32, "SG_N":16, "SG_K":16, "SLM_KS":4, "timems":0.035312, "gbps":476.506, "tflops":1.90045, "compute_pressure":3.98832 }
{ "policy":18, "m":4, "n":2048, "k":4096, "n_ss":8, "N_SG_PER_SS":32, "WG_M":32, "WG_N":256, "SG_M":32, "SG_N":16, "SG_K":16, "SLM_KS":2, "timems":0.0548112, "gbps":306.988, "tflops":1.22436, "compute_pressure":3.98832 }
{ "policy":19, "m":4, "n":2048, "k":4096, "n_ss":4, "N_SG_PER_SS":32, "WG_M":32, "WG_N":512, "SG_M":32, "SG_N":16, "SG_K":16, "SLM_KS":1, "timems":0.123571, "gbps":136.167, "tflops":0.543079, "compute_pressure":3.98832 }
{ "policy":20, "m":4, "n":2048, "k":4096, "n_ss":16, "N_SG_PER_SS":32, "WG_M":64, "WG_N":128, "SG_M":64, "SG_N":16, "SG_K":16, "SLM_KS":4, "timems":0.0366288, "gbps":459.375, "tflops":1.83213, "compute_pressure":3.98832 }
{ "policy":21, "m":4, "n":2048, "k":4096, "n_ss":8, "N_SG_PER_SS":32, "WG_M":64, "WG_N":256, "SG_M":64, "SG_N":16, "SG_K":16, "SLM_KS":2, "timems":0.0806976, "gbps":208.511, "tflops":0.831609, "compute_pressure":3.98832 }
{ "policy":22, "m":4, "n":2048, "k":4096, "n_ss":4, "N_SG_PER_SS":32, "WG_M":64, "WG_N":512, "SG_M":64, "SG_N":16, "SG_K":16, "SLM_KS":1, "timems":0.133917, "gbps":125.648, "tflops":0.501124, "compute_pressure":3.98832 }
{ "policy":23, "m":4, "n":2048, "k":4096, "n_ss":32, "N_SG_PER_SS":32, "WG_M":128, "WG_N":64, "SG_M":16, "SG_N":16, "SG_K":64, "SLM_KS":1, "timems":0.0581984, "gbps":289.121, "tflops":1.1531, "compute_pressure":3.98832 }
{ "policy":24, "m":4, "n":2048, "k":4096, "n_ss":16, "N_SG_PER_SS":32, "WG_M":128, "WG_N":128, "SG_M":16, "SG_N":32, "SG_K":64, "SLM_KS":1, "timems":0.0502816, "gbps":334.643, "tflops":1.33466, "compute_pressure":3.98832 }
{ "policy":25, "m":4, "n":2048, "k":4096, "n_ss":16, "N_SG_PER_SS":32, "WG_M":128, "WG_N":128, "SG_M":32, "SG_N":32, "SG_K":32, "SLM_KS":2, "timems":0.0360272, "gbps":467.046, "tflops":1.86273, "compute_pressure":3.98832 }
{ "policy":26, "m":4, "n":2048, "k":4096, "n_ss":8, "N_SG_PER_SS":32, "WG_M":128, "WG_N":256, "SG_M":32, "SG_N":32, "SG_K":16, "SLM_KS":1, "timems":0.0851456, "gbps":197.619, "tflops":0.788166, "compute_pressure":3.98832 }
{ "policy":27, "m":4, "n":2048, "k":4096, "n_ss":8, "N_SG_PER_SS":32, "WG_M":128, "WG_N":256, "SG_M":64, "SG_N":16, "SG_K":16, "SLM_KS":1, "timems":0.0871248, "gbps":193.129, "tflops":0.770261, "compute_pressure":3.98832 }
{ "policy":28, "m":4, "n":2048, "k":4096, "n_ss":4, "N_SG_PER_SS":32, "WG_M":128, "WG_N":512, "SG_M":64, "SG_N":32, "SG_K":16, "SLM_KS":1, "timems":0.134491, "gbps":125.111, "tflops":0.498983, "compute_pressure":3.98832 }
{ "policy":29, "m":4, "n":2048, "k":4096, "n_ss":8, "N_SG_PER_SS":32, "WG_M":256, "WG_N":256, "SG_M":32, "SG_N":64, "SG_K":16, "SLM_KS":1, "timems":0.0929648, "gbps":180.997, "tflops":0.721874, "compute_pressure":3.98832 }
{ "policy":30, "m":4, "n":2048, "k":4096, "n_ss":8, "N_SG_PER_SS":32, "WG_M":256, "WG_N":256, "SG_M":32, "SG_N":64, "SG_K":32, "SLM_KS":1, "timems":0.146598, "gbps":114.779, "tflops":0.457774, "compute_pressure":3.98832 }
{ "policy":31, "m":4, "n":2048, "k":4096, "n_ss":8, "N_SG_PER_SS":32, "WG_M":256, "WG_N":256, "SG_M":64, "SG_N":32, "SG_K":16, "SLM_KS":1, "timems":0.120242, "gbps":139.938, "tflops":0.558117, "compute_pressure":3.98832 }
auto_policy_id=1, auto_policy_timems=0.0376928, min_policy_id=5, min_policy_timems=0.0352624
```

4. For generating policy selection file

```bash
python gemm/select_best.py
```
