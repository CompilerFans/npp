// f13: does the existing 8u watershed match NPP?
#include "npp.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
extern "C" NppStatus nppiSegmentWatershed_8u_C1IR_Ctx_impl(Npp8u *, Npp32s, Npp32u *, Npp32s,
                                                            NppiNorm, NppiSize, Npp8u *,
                                                            NppStreamContext);
int main(int argc, char **argv) {
  int norm = argc > 1 ? atoi(argv[1]) : 0;
  const int W = 10, H = 8;
  std::vector<Npp8u> h((size_t)W * H);
  std::vector<Npp32u> m((size_t)W * H, 0);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) h[y * W + x] = (Npp8u)((x * 20 + y * 7) % 200);
  // markers: 1 in the left half cell, 2 in the right half cell
  for (int y = 2; y < 5; ++y) {
    m[y * W + 2] = 1;
    m[y * W + 7] = 2;
  }
  Npp8u *d, *buf;
  Npp32u *ma, *mb;
  cudaMalloc(&d, W * H);
  cudaMalloc(&ma, W * H * 4);
  cudaMalloc(&mb, W * H * 4);
  cudaMalloc(&buf, 1 << 20);
  std::vector<Npp8u> ha = h, hb = h;
  NppStreamContext c{};
  c.hStream = 0;
  
  cudaMemcpy(d, h.data(), W * H, cudaMemcpyHostToDevice);
  cudaMemcpy(ma, m.data(), W * H * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(mb, m.data(), W * H * 4, cudaMemcpyHostToDevice);
  NppiNorm eNorm = norm ? nppiNormInf : nppiNormL1;
  NppStatus rs = nppiSegmentWatershed_8u_C1IR_Ctx(d, W, ma, W * 4, eNorm, (NppiWatershedSegmentBoundaryType)0, {W, H}, buf, c);
  NppStatus ms = nppiSegmentWatershed_8u_C1IR_Ctx_impl(d, W, mb, W * 4, eNorm, {W, H}, buf, c);
  std::vector<Npp32u> va(W * H), vb(W * H);
  cudaMemcpy(va.data(), ma, W * H * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(vb.data(), mb, W * H * 4, cudaMemcpyDeviceToHost);
  int diff = 0;
  for (int i = 0; i < W * H; ++i) diff += va[i] != vb[i];
  printf("ref st=%d mine st=%d diffs=%d bufsize=%zu\n", (int)rs, (int)ms, diff, (size_t)0);
  printf("npp :");
  for (int i = 0; i < W * H; ++i) printf(" %u", va[i]);
  printf("\nmcpp:");
  for (int i = 0; i < W * H; ++i) printf(" %u", vb[i]);
  printf("\n");
  return 0;
}
