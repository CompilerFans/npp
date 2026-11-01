// f13: does the existing mcMPP label-markers Union-Find match NPP 13.4?
#include "npp.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" NppStatus nppiLabelMarkersUF_8u32u_C1R_Ctx_impl(const Npp8u *, int, Npp32u *, int,
                                                           NppiSize, NppiNorm, Npp8u *,
                                                           NppStreamContext);
extern "C" NppStatus nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx_impl(
    const NppiImageDescriptor *, NppiImageDescriptor *, int, NppiSize, NppiNorm, NppStreamContext);

int main(int argc, char **argv) {
  int norm = argc > 1 ? atoi(argv[1]) : 0;  // 0 = L1, 1 = Inf
  const int W = 12, H = 9;
  std::vector<Npp8u> h((size_t)W * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) h[y * W + x] = (Npp8u)(((x / 3 + y / 2) % 3) * 40);
  printf("src:");
  for (int i = 0; i < W * H; ++i) printf(" %d", (int)h[i]);
  printf("\n");
  Npp8u *d, *buf;
  Npp32u *a, *b;
  cudaMalloc(&d, W * H);
  cudaMalloc(&a, W * H * 4);
  cudaMalloc(&b, W * H * 4);
  cudaMalloc(&buf, 1 << 20);
  cudaMemcpy(d, h.data(), W * H, cudaMemcpyHostToDevice);
  NppiNorm eNorm = norm ? nppiNormInf : nppiNormL1;
  NppStreamContext c{};
  c.hStream = 0;
  int bsz = 0;
  nppiLabelMarkersUFGetBufferSize_32u_C1R(NppiSize{W, H}, &bsz);
  NppStatus rs = nppiLabelMarkersUF_8u32u_C1R_Ctx(d, W, a, W * 4, NppiSize{W, H}, eNorm, buf, c);
  NppStatus ms = nppiLabelMarkersUF_8u32u_C1R_Ctx_impl(d, W, b, W * 4, NppiSize{W, H}, eNorm, buf, c);
  printf("ref st=%d mine st=%d (bufsize %d)\n", (int)rs, (int)ms, bsz);
  std::vector<Npp32u> va(W * H), vb(W * H);
  cudaMemcpy(va.data(), a, W * H * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(vb.data(), b, W * H * 4, cudaMemcpyDeviceToHost);
  printf("npp :");
  for (int i = 0; i < W * H; ++i) printf(" %u", va[i]);
  printf("\nmcpp:");
  for (int i = 0; i < W * H; ++i) printf(" %u", vb[i]);
  printf("\n");
  // batch comparison
  {
    NppiImageDescriptor srcImg{d, W, {W, H}};
    Npp32u *ba, *bb;
    cudaMalloc(&ba, W * H * 4);
    cudaMalloc(&bb, W * H * 4);
    NppiImageDescriptor dstA{ba, W * 4, {W, H}};
    NppiImageDescriptor dstB{bb, W * 4, {W, H}};
    NppStatus brs = nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx(&srcImg, &dstA, 1, {W, H},
                                                                   eNorm, c);
    NppStatus bms = nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx_impl(&srcImg, &dstB, 1, {W, H},
                                                                        eNorm, c);
    std::vector<Npp32u> b1(W * H), b2(W * H);
    cudaMemcpy(b1.data(), ba, W * H * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(b2.data(), bb, W * H * 4, cudaMemcpyDeviceToHost);
    int diff = 0;
    for (int i = 0; i < W * H; ++i) diff += (b1[i] != b2[i]);
    printf("batch ref st=%d mine st=%d diffs=%d\n", (int)brs, (int)bms, diff);
    printf("batch npp :");
    for (int i = 0; i < W * H; ++i) printf(" %u", b1[i]);
    printf("\n");
    printf("batch mcpp:");
    for (int i = 0; i < W * H; ++i) printf(" %u", b2[i]);
    printf("\n");
  }
  return 0;
}
