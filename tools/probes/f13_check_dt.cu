// f13: does the existing PBA distance-transform match NPP?
#include "npp.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
extern "C" {
NppStatus nppiDistanceTransformPBA_8u32f_C1R_Ctx_impl(Npp8u *, int, Npp8u, Npp8u, Npp16s *, int,
                                                       Npp16s *, int, Npp16s *, int, Npp32f *, int,
                                                       NppiSize, Npp8u *, NppStreamContext);
NppStatus nppiDistanceTransformAbsPBA_8u16u_C1R_Ctx_impl(
    Npp8u *, int, Npp8u, Npp8u, Npp16s *, int, Npp16s *, int, Npp16u *, int, Npp16u *, int,
    NppiSize, Npp8u *, NppStreamContext);
}
int main() {
  const int W = 10, H = 8;
  std::vector<Npp8u> h((size_t)W * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) h[y * W + x] = (Npp8u)(((x % 4) == 0 && (y % 3) == 0) ? 255 : 0);
  Npp8u *d, *buf;
  Npp16s *va, *vb, *ia, *ib, *ra, *rb;
  Npp32f *ta, *tb;
  cudaMalloc(&d, W * H);
  cudaMalloc(&buf, 1 << 20);
  cudaMalloc(&va, W * H * 2); cudaMalloc(&vb, W * H * 2);
  cudaMalloc(&ia, W * H * 2); cudaMalloc(&ib, W * H * 2);
  cudaMalloc(&ra, W * H * 2); cudaMalloc(&rb, W * H * 2);
  cudaMalloc(&ta, W * H * 4); cudaMalloc(&tb, W * H * 4);
  cudaMemcpy(d, h.data(), W * H, cudaMemcpyHostToDevice);
  NppStreamContext c{};
  c.hStream = 0;
  NppStatus rs = nppiDistanceTransformPBA_8u32f_C1R_Ctx(d, W, 200, 255, va, W * 4, ia, W * 4, ra,
                                                        W * 4, ta, W * 4, {W, H}, buf, c);
  NppStatus ms = nppiDistanceTransformPBA_8u32f_C1R_Ctx_impl(d, W, 200, 255, vb, W * 4, ib, W * 4,
                                                             rb, W * 4, tb, W * 4, {W, H}, buf, c);
  std::vector<Npp32f> A(W * H), B(W * H);
  std::vector<Npp16s> VA(W * H), VB(W * H), IA(W * H), IB(W * H);
  cudaMemcpy(A.data(), ta, W * H * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(B.data(), tb, W * H * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(VA.data(), va, W * H * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(VB.data(), vb, W * H * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(IA.data(), ia, W * H * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(IB.data(), ib, W * H * 2, cudaMemcpyDeviceToHost);
  int dt = 0, dv = 0, di = 0;
  for (int i = 0; i < W * H; ++i) {
    dt += A[i] != B[i];
    dv += VA[i] != VB[i];
    di += IA[i] != IB[i];
  }
  printf("8u32f ref st=%d mine st=%d: transform diffs=%d voronoi diffs=%d indices diffs=%d\n",
         (int)rs, (int)ms, dt, dv, di);
  printf("npp  t:");
  for (int i = 0; i < 20; ++i) printf(" %.4g", A[i]);
  printf("\nmopp t:");
  for (int i = 0; i < 20; ++i) printf(" %.4g", B[i]);
  printf("\n");
  return 0;
}
