// f13: replay the label-markers / watershed variants against NPP on the 5090.
#include "npp.h"
#include "npp_internal.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
NppStatus nppiLabelMarkersUF_16u32u_C1R_Ctx_impl(const Npp16u *, int, Npp32u *, int, NppiSize,
                                                  NppiNorm, Npp8u *, NppStreamContext);
NppStatus nppiLabelMarkersUF_32u_C1R_Ctx_impl(const Npp32u *, int, Npp32u *, int, NppiSize,
                                               NppiNorm, Npp8u *, NppStreamContext);
NppStatus nppiLabelMarkersUFBatch_16u32u_C1R_Advanced_Ctx_impl(const NppiImageDescriptor *,
                                                                NppiImageDescriptor *, int,
                                                                NppiSize, NppiNorm,
                                                                NppStreamContext);
NppStatus nppiLabelMarkersUFBatch_32u_C1R_Advanced_Ctx_impl(const NppiImageDescriptor *,
                                                             NppiImageDescriptor *, int, NppiSize,
                                                             NppiNorm, NppStreamContext);
}

static int g_fail = 0, g_cases = 0;

static NppStreamContext ctx() {
  NppStreamContext c{};
  c.hStream = 0;
  return c;
}

template <typename Src>
static void single(const char *name, int W, int H, NppiNorm norm, bool is32) {
  std::vector<Src> h((size_t)W * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) h[y * W + x] = (Src)(((x / 3 + y / 2 + (x * y) % 2) % 4) * 17);
  Src *d;
  Npp32u *a, *b;
  Npp8u *buf;
  cudaMalloc(&d, (size_t)W * H * sizeof(Src));
  cudaMalloc(&a, (size_t)W * H * 4);
  cudaMalloc(&b, (size_t)W * H * 4);
  cudaMalloc(&buf, 1 << 20);
  cudaMemcpy(d, h.data(), (size_t)W * H * sizeof(Src), cudaMemcpyHostToDevice);
  NppStatus rs, ms;
  if (is32) {
    rs = nppiLabelMarkersUF_32u_C1R_Ctx((Npp32u *)d, W * 4, a, W * 4, {W, H}, norm, buf, ctx());
    ms = nppiLabelMarkersUF_32u_C1R_Ctx_impl((const Npp32u *)d, W * 4, b, W * 4, {W, H}, norm, buf, ctx());
  } else {
    rs = nppiLabelMarkersUF_16u32u_C1R_Ctx((Npp16u *)d, W * 2, a, W * 4, {W, H}, norm, buf, ctx());
    ms = nppiLabelMarkersUF_16u32u_C1R_Ctx_impl((const Npp16u *)d, W * 2, b, W * 4, {W, H}, norm, buf, ctx());
  }
  std::vector<Npp32u> va((size_t)W * H), vb((size_t)W * H);
  cudaMemcpy(va.data(), a, (size_t)W * H * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(vb.data(), b, (size_t)W * H * 4, cudaMemcpyDeviceToHost);
  ++g_cases;
  int diff = 0;
  for (int i = 0; i < W * H; ++i) diff += va[i] != vb[i];
  if (rs != ms || diff) {
    ++g_fail;
    printf("FAIL %-28s st ref=%d mine=%d diffs=%d\n", name, (int)rs, (int)ms, diff);
    for (int i = 0; i < 8; ++i) printf("   [%d] %u vs %u\n", i, va[i], vb[i]);
  }
  cudaFree(d);
  cudaFree(a);
  cudaFree(b);
  cudaFree(buf);
}

template <typename Src>
static void batch(const char *name, int W, int H, NppiNorm norm, bool is32) {
  const int nImg = 2;
  std::vector<Src> h((size_t)W * H * nImg);
  for (auto &v : h) v = (Src)(rand() % 4 * 11);
  Src *d;
  Npp32u *a, *b;
  cudaMalloc(&d, (size_t)W * H * nImg * sizeof(Src));
  cudaMalloc(&a, (size_t)W * H * nImg * 4);
  cudaMalloc(&b, (size_t)W * H * nImg * 4);
  cudaMemcpy(d, h.data(), (size_t)W * H * nImg * sizeof(Src), cudaMemcpyHostToDevice);
  NppiImageDescriptor sImg[2] = {{d, W * (int)sizeof(Src), {W, H}},
                                 {(char *)d + (size_t)W * H * sizeof(Src), W * (int)sizeof(Src),
                                  {W, H}}};
  NppiImageDescriptor dA[2] = {{a, W * 4, {W, H}},
                               {(char *)a + (size_t)W * H * 4, W * 4, {W, H}}};
  NppiImageDescriptor dB[2] = {{b, W * 4, {W, H}},
                               {(char *)b + (size_t)W * H * 4, W * 4, {W, H}}};
  NppStatus rs, ms;
  if (is32) {
    rs = nppiLabelMarkersUFBatch_32u_C1R_Advanced_Ctx(sImg, dA, nImg, {W, H}, norm, ctx());
    ms = nppiLabelMarkersUFBatch_32u_C1R_Advanced_Ctx_impl(sImg, dB, nImg, {W, H}, norm, ctx());
  } else {
    rs = nppiLabelMarkersUFBatch_16u32u_C1R_Advanced_Ctx(sImg, dA, nImg, {W, H}, norm, ctx());
    ms = nppiLabelMarkersUFBatch_16u32u_C1R_Advanced_Ctx_impl(sImg, dB, nImg, {W, H}, norm, ctx());
  }
  std::vector<Npp32u> va((size_t)W * H * nImg), vb((size_t)W * H * nImg);
  cudaMemcpy(va.data(), a, va.size() * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(vb.data(), b, vb.size() * 4, cudaMemcpyDeviceToHost);
  ++g_cases;
  int diff = 0;
  for (size_t i = 0; i < va.size(); ++i) diff += va[i] != vb[i];
  if (rs != ms || diff) {
    ++g_fail;
    printf("FAIL %-28s st ref=%d mine=%d diffs=%d\n", name, (int)rs, (int)ms, diff);
  }
  cudaFree(d);
  cudaFree(a);
  cudaFree(b);
}

int main() {
  single<Npp16u>("16u32u L1", 14, 10, nppiNormL1, false);
  single<Npp16u>("16u32u Inf", 14, 10, nppiNormInf, false);
  single<Npp32u>("32u L1", 14, 10, nppiNormL1, true);
  single<Npp32u>("32u Inf", 14, 10, nppiNormInf, true);
  batch<Npp16u>("batch 16u32u L1", 12, 9, nppiNormL1, false);
  batch<Npp16u>("batch 16u32u Inf", 12, 9, nppiNormInf, false);
  batch<Npp32u>("batch 32u L1", 12, 9, nppiNormL1, true);
  batch<Npp32u>("batch 32u Inf", 12, 9, nppiNormInf, true);
  printf("%d cases, %d failed\n", g_cases, g_fail);
  return g_fail ? 1 : 0;
}
