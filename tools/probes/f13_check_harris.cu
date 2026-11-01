// f13: run mcMPP's HarrisCornersBorder on the probe28 input and print it.
#include "npp.h"
#include "nppi_filter_tail_executor.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
int main() {
  const int W = 6, H = 6;
  std::vector<Npp8u> h((size_t)W * H);
  for (int i = 0; i < W * H; ++i) h[i] = (Npp8u)((i * 37 + 11) % 97);
  Npp8u *d, *buf;
  Npp32f *o;
  cudaMalloc(&d, W * H);
  cudaMalloc(&o, W * H * 4);
  cudaMalloc(&buf, 1 << 16);
  cudaMemcpy(d, h.data(), W * H, cudaMemcpyHostToDevice);
  NppStatus st = nppi::filter::harrisCornersBorder(d, W, {W, H}, {0, 0}, o, W * 4, {W, H},
                                                   NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3,
                                                   NPP_MASK_SIZE_3_X_3, 0.04f, 1.f,
                                                   NPP_BORDER_REPLICATE, buf, 0);
  std::vector<Npp32f> v(W * H);
  cudaMemcpy(v.data(), o, W * H * 4, cudaMemcpyDeviceToHost);
  printf("mine st=%d:", (int)st);
  for (int i = 0; i < W * H; ++i) printf(" %.9g", v[i]);
  printf("\n");
  return 0;
}
