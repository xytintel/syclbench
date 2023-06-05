xetla_base=/home/gta/xyt/xetla/
icpx -fsycl -std=c++20 \
  -DXETPP_NEW_XMAIN  \
  -isystem ${xetla_base} \
  -isystem ${xetla_base}/include \
  -Xs '-doubleGRF -Xfinalizer -printregusage  -Xfinalizer -DPASTokenReduction  -Xfinalizer -enableBCR' \
  $1 -o a.out
