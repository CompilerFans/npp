// f13: replay the bilateral / threshold-adaptive-box / Harris cases against
// mcMPP's executors on the RTX 5090 (NVIDIA NPP is the reference).
#include "npp.h"
#include "nppi_filter_tail_executor.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0, g_cases = 0;
static unsigned st = 0x9e3779b9u;
static unsigned rnd() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; }
static NppStreamContext ctx() { NppStreamContext c{}; c.hStream = 0; return c; }

template <typename T> static double tol();
template <> double tol<Npp8u>() { return 0; }
template <> double tol<Npp16u>() { return 0; }
template <> double tol<Npp32f>() { return 0; }  // relative check below

template <typename T>
static void check(const char *name, NppStatus refSt, NppStatus mySt, T *ref, T *mine, int n) {
  ++g_cases;
  cudaError_t e = cudaDeviceSynchronize();
  if (refSt != mySt || e != cudaSuccess) {
    ++g_fail;
    printf("FAIL %-40s status ref=%d mine=%d cuda=%s\n", name, (int)refSt, (int)mySt,
           cudaGetErrorString(e));
    return;
  }
  if (refSt != NPP_NO_ERROR) return;
  std::vector<T> a(n), b(n);
  cudaMemcpy(a.data(), ref, n * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(b.data(), mine, n * sizeof(T), cudaMemcpyDeviceToHost);
  int mism = 0;
  double maxrel = 0;
  for (int i = 0; i < n; ++i) {
    double x = (double)a[i], y = (double)b[i];
    bool bad;
    if (sizeof(T) == 1 || sizeof(T) == 2) {
      bad = (x != y && std::fabs(x - y) > 1.0);
    } else {
      double d = std::fabs(x - y);
      double r = d / (1.0 + std::fabs(x));
      maxrel = r > maxrel ? r : maxrel;
      bad = r > 1e-4;
    }
    if (bad) ++mism;
  }
  if (mism) {
    ++g_fail;
    printf("FAIL %-40s mism=%d maxrel=%.3g\n", name, mism, maxrel);
    for (int i = 0; i < n && i < 6; ++i) printf("      [%d] ref=%g mine=%g\n", i, (double)a[i], (double)b[i]);
  }
}

// concrete wrappers (templates cannot dispatch on the NPP symbol by type, so
// spell the three dtypes out)
static void bil8(const char *n, int W, int H, int r, int stp, float vs, float ps, NppiBorderType bt) {
  Npp8u *s, *a, *b;
  cudaMalloc(&s, W * H); cudaMalloc(&a, W * H); cudaMalloc(&b, W * H);
  std::vector<Npp8u> h(W * H);
  for (auto &v : h) v = rnd() % 256;
  cudaMemcpy(s, h.data(), W * H, cudaMemcpyHostToDevice);
  cudaMemset(b, 0xEE, W * H);
  NppStatus rs = nppiFilterBilateralGaussBorder_8u_C1R_Ctx(s, W, {W, H}, {0, 0}, a, W, {W, H}, r, stp, vs, ps, bt, ctx());
  NppStatus ms = nppi::filter::bilateralGaussBorder<Npp8u, Npp8u, 1>(s, W, {W, H}, {0, 0}, b, W, {W, H}, r, stp, vs, ps, bt, 0);
  check(n, rs, ms, a, b, W * H);
  cudaFree(s); cudaFree(a); cudaFree(b);
}
static void bil16(const char *n, int W, int H, int r, int stp, float vs, float ps, NppiBorderType bt) {
  Npp16u *s, *a, *b;
  cudaMalloc(&s, W * H * 2); cudaMalloc(&a, W * H * 2); cudaMalloc(&b, W * H * 2);
  std::vector<Npp16u> h(W * H);
  for (auto &v : h) v = rnd() % 65536;
  cudaMemcpy(s, h.data(), W * H * 2, cudaMemcpyHostToDevice);
  cudaMemset(b, 0xEE, W * H * 2);
  NppStatus rs = nppiFilterBilateralGaussBorder_16u_C1R_Ctx(s, W * 2, {W, H}, {0, 0}, a, W * 2, {W, H}, r, stp, vs, ps, bt, ctx());
  NppStatus ms = nppi::filter::bilateralGaussBorder<Npp16u, Npp16u, 1>(s, W * 2, {W, H}, {0, 0}, b, W * 2, {W, H}, r, stp, vs, ps, bt, 0);
  check(n, rs, ms, a, b, W * H);
  cudaFree(s); cudaFree(a); cudaFree(b);
}
static void bil32(const char *n, int W, int H, int r, int stp, float vs, float ps, NppiBorderType bt) {
  Npp32f *s, *a, *b;
  cudaMalloc(&s, W * H * 4); cudaMalloc(&a, W * H * 4); cudaMalloc(&b, W * H * 4);
  std::vector<Npp32f> h(W * H);
  for (auto &v : h) v = (float)(rnd() % 512) * 0.5f;
  cudaMemcpy(s, h.data(), W * H * 4, cudaMemcpyHostToDevice);
  cudaMemset(b, 0xEE, W * H * 4);
  NppStatus rs = nppiFilterBilateralGaussBorder_32f_C1R_Ctx(s, W * 4, {W, H}, {0, 0}, a, W * 4, {W, H}, r, stp, vs, ps, bt, ctx());
  NppStatus ms = nppi::filter::bilateralGaussBorder<Npp32f, Npp32f, 1>(s, W * 4, {W, H}, {0, 0}, b, W * 4, {W, H}, r, stp, vs, ps, bt, 0);
  check(n, rs, ms, a, b, W * H);
  cudaFree(s); cudaFree(a); cudaFree(b);
}
static void bil8c3(const char *n, int W, int H, int r, int stp, float vs, float ps, NppiBorderType bt) {
  Npp8u *s, *a, *b;
  cudaMalloc(&s, W * H * 3); cudaMalloc(&a, W * H * 3); cudaMalloc(&b, W * H * 3);
  std::vector<Npp8u> h(W * H * 3);
  for (auto &v : h) v = rnd() % 256;
  cudaMemcpy(s, h.data(), W * H * 3, cudaMemcpyHostToDevice);
  cudaMemset(b, 0xEE, W * H * 3);
  NppStatus rs = nppiFilterBilateralGaussBorder_8u_C3R_Ctx(s, W * 3, {W, H}, {0, 0}, a, W * 3, {W, H}, r, stp, vs, ps, bt, ctx());
  NppStatus ms = nppi::filter::bilateralGaussBorder<Npp8u, Npp8u, 3>(s, W * 3, {W, H}, {0, 0}, b, W * 3, {W, H}, r, stp, vs, ps, bt, 0);
  check(n, rs, ms, a, b, W * H * 3);
  cudaFree(s); cudaFree(a); cudaFree(b);
}
static void at8(const char *n, int W, int H, NppiSize m, float d, Npp8u gt, Npp8u le, NppiBorderType bt) {
  Npp8u *s, *a, *b;
  cudaMalloc(&s, W * H); cudaMalloc(&a, W * H); cudaMalloc(&b, W * H);
  std::vector<Npp8u> h(W * H);
  for (auto &v : h) v = rnd() % 256;
  cudaMemcpy(s, h.data(), W * H, cudaMemcpyHostToDevice);
  cudaMemset(b, 0xEE, W * H);
  NppStatus rs = nppiFilterThresholdAdaptiveBoxBorder_8u_C1R_Ctx(s, W, {W, H}, {0, 0}, a, W, {W, H}, m, d, gt, le, bt, ctx());
  NppStatus ms = nppi::filter::thresholdAdaptiveBoxBorder(s, W, {W, H}, {0, 0}, b, W, {W, H}, m, d, gt, le, bt, 0);
  check(n, rs, ms, a, b, W * H);
  cudaFree(s); cudaFree(a); cudaFree(b);
}
static void hc(const char *n, int W, int H, NppiDifferentialKernel f, NppiMaskSize m,
               NppiMaskSize aw, float k, float sc, NppiBorderType bt) {
  Npp8u *s;
  Npp32f *a, *b;
  Npp8u *buf;
  cudaMalloc(&s, W * H); cudaMalloc(&a, W * H * 4); cudaMalloc(&b, W * H * 4);
  cudaMalloc(&buf, 1 << 16);
  std::vector<Npp8u> h(W * H);
  for (auto &v : h) v = rnd() % 256;
  cudaMemcpy(s, h.data(), W * H, cudaMemcpyHostToDevice);
  cudaMemset(b, 0xEE, W * H * 4);
  NppStatus rs = nppiFilterHarrisCornersBorder_8u32f_C1R_Ctx(s, W, {W, H}, {0, 0}, a, W * 4, {W, H}, f, m, aw, k, sc, bt, buf, ctx());
  NppStatus ms = nppi::filter::harrisCornersBorder(s, W, {W, H}, {0, 0}, b, W * 4, {W, H}, f, m, aw, k, sc, bt, buf, 0);
  check(n, rs, ms, a, b, W * H);
  cudaFree(s); cudaFree(a); cudaFree(b); cudaFree(buf);
}

int main() {
  bil8("bil8 r1 s1 v10 p2", 24, 18, 1, 1, 10.f, 2.f, NPP_BORDER_REPLICATE);
  bil8("bil8 r2 s1 v10 p2", 24, 18, 2, 1, 10.f, 2.f, NPP_BORDER_REPLICATE);
  bil8("bil8 r3 s1 v50 p5 mirror", 24, 18, 3, 1, 50.f, 5.f, NPP_BORDER_MIRROR);
  bil8("bil8 r2 s1 v100 p8", 24, 18, 2, 1, 100.f, 8.f, NPP_BORDER_REPLICATE);
  bil16("bil16 r2 s1 v20 p3", 24, 18, 2, 1, 20.f, 3.f, NPP_BORDER_REPLICATE);
  bil16("bil16 r1 s1 v5 p1", 24, 18, 1, 1, 5.f, 1.f, NPP_BORDER_MIRROR);
  bil32("bil32 r2 s1 v10 p2", 24, 18, 2, 1, 10.f, 2.f, NPP_BORDER_REPLICATE);
  bil32("bil32 r1 s1 v1 p1", 24, 18, 1, 1, 1.f, 1.f, NPP_BORDER_MIRROR);
  bil8c3("bil8c3 r2 s1 v10 p2", 24, 18, 2, 1, 10.f, 2.f, NPP_BORDER_REPLICATE);
  bil8("bil8 r1 s2 v10 p2", 24, 18, 1, 2, 10.f, 2.f, NPP_BORDER_REPLICATE);
  bil8("bil8 r2 s3 v10 p2", 24, 18, 2, 3, 10.f, 2.f, NPP_BORDER_REPLICATE);
  bil8("bil8 r1 s4 v50 p5 mirror", 24, 18, 1, 4, 50.f, 5.f, NPP_BORDER_MIRROR);
  bil16("bil16 r2 s2 v20 p3", 24, 18, 2, 2, 20.f, 3.f, NPP_BORDER_REPLICATE);
  bil32("bil32 r1 s2 v10 p2", 24, 18, 1, 2, 10.f, 2.f, NPP_BORDER_REPLICATE);

  at8("at8 m3 d0", 24, 18, {3, 3}, 0.f, 1, 0, NPP_BORDER_REPLICATE);
  at8("at8 m5 d10", 24, 18, {5, 5}, 10.f, 200, 7, NPP_BORDER_REPLICATE);
  at8("at8 m9 d3.5", 24, 18, {9, 9}, 3.5f, 255, 1, NPP_BORDER_REPLICATE);
  at8("at8 m1 d0", 24, 18, {1, 1}, 0.f, 1, 0, NPP_BORDER_REPLICATE);

  hc("hc sobel3 avg3", 24, 18, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, 0.04f, 1.f, NPP_BORDER_REPLICATE);
  hc("hc sobel3 avg5", 24, 18, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_5_X_5, 0.04f, 1.f, NPP_BORDER_REPLICATE);
  hc("hc sobel5 avg3", 24, 18, NPP_FILTER_SOBEL, NPP_MASK_SIZE_5_X_5, NPP_MASK_SIZE_3_X_3, 0.1f, 0.5f, NPP_BORDER_REPLICATE);
  hc("hc scharr avg3", 24, 18, NPP_FILTER_SCHARR, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, 0.04f, 2.f, NPP_BORDER_REPLICATE);
  hc("hc scharr avg5", 24, 18, NPP_FILTER_SCHARR, NPP_MASK_SIZE_5_X_5, NPP_MASK_SIZE_5_X_5, 0.04f, 1.f, NPP_BORDER_REPLICATE);

  printf("%d cases, %d failed\n", g_cases, g_fail);
  return g_fail ? 1 : 0;
}
