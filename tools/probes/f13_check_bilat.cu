// f13: mcMPP bilateral on the probe26 image with step 1 and 2.
#include "npp.h"
#include "nppi_filter_tail_executor.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
int main() {
  const int W = 8, H = 8;
  std::vector<Npp8u> h((size_t)W * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) h[y * W + x] = (Npp8u)((x * 11 + y * 23) % 97);
  Npp8u *d, *o;
  cudaMalloc(&d, W * H);
  cudaMalloc(&o, W * H);
  cudaMemcpy(d, h.data(), W * H, cudaMemcpyHostToDevice);
  for (int step : {1, 2}) {
    cudaMemset(o, 0, W * H);
    NppStatus st = nppi::filter::bilateralGaussBorder<Npp8u, Npp8u, 1>(
        d, W, {W, H}, {0, 0}, o, W, {W, H}, 1, step, 10.f, 2.f, NPP_BORDER_REPLICATE, 0);
    std::vector<Npp8u> v(W * H);
    cudaMemcpy(v.data(), o, W * H, cudaMemcpyDeviceToHost);
    printf("step%d st=%d:", step, (int)st);
    for (int i = 0; i < W * H; ++i) printf(" %d", (int)v[i]);
    printf("\n");
  }
  return 0;
}
