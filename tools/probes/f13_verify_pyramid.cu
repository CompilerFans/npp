// f13: replay the pyramid-down differential cases against mcMPP's executor on
// the RTX 5090 (NVIDIA NPP is the reference).  This links the mcMPP executor
// templates directly, so it validates the implementation without an MPP build.
#include "npp.h"
#include "nppi_gauss_pyramid_executor.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>

static int g_fail = 0, g_cases = 0;

static unsigned rng_state = 0x1234567u;
static unsigned next_u32() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return rng_state;
}
template <typename T>
static std::vector<T> uniform_ints(std::size_t n, int lo, int hi) {
  std::vector<T> v(n);
  for (auto &x : v) x = (T)(lo + (int)(next_u32() % (unsigned)(hi - lo + 1)));
  return v;
}

static NppStreamContext ctx() {
  NppStreamContext c{};
  c.hStream = 0;
  return c;
}

template <typename S, typename D>
static void compare(const std::string &name, S *src, int sStep, NppiSize srcSize, NppiPoint off,
                    D *nppDst, int dStep, NppiSize roi, float rate, int taps, float *kern,
                    NppiBorderType bt, bool nppOk, int ch, int elems) {
  // mcMPP
  D *mine = nullptr;
  cudaMalloc(&mine, (size_t)(roi.height > 0 ? roi.height : 1) * dStep + 64);
  cudaMemset(mine, 0xEE, (size_t)(roi.height > 0 ? roi.height : 1) * dStep + 64);
  NppStatus st = NPP_NO_ERROR;
  switch (ch) {
    case 1:
      st = nppi::filter::gaussPyramidBorder<S, D, 1, nppi::filter::kPyramidDown>(
          src, sStep, srcSize, off, (D *)((char *)mine + 0), dStep, roi, rate, taps, kern, bt,
          ctx().hStream);
      break;
    default:
      st = nppi::filter::gaussPyramidBorder<S, D, 3, nppi::filter::kPyramidDown>(
          src, sStep, srcSize, off, (D *)((char *)mine + 0), dStep, roi, rate, taps, kern, bt,
          ctx().hStream);
      break;
  }
  cudaError_t e = cudaDeviceSynchronize();
  ++g_cases;
  if (e != cudaSuccess) {
    ++g_fail;
    printf("FAIL %-42s cuda %s\n", name.c_str(), cudaGetErrorString(e));
    cudaFree(mine);
    return;
  }
  std::vector<D> ref(elems), got(elems);
  if (nppOk) {
    cudaMemcpy(ref.data(), nppDst, elems * sizeof(D), cudaMemcpyDeviceToHost);
  }
  cudaMemcpy(got.data(), mine, elems * sizeof(D), cudaMemcpyDeviceToHost);
  double maxdiff = 0.0;
  int mism = 0;
  for (int i = 0; i < elems; ++i) {
    double a = (double)ref[i], b = (double)got[i];
    double d = fabs(a - b);
    if (d > maxdiff) maxdiff = d;
    if (d > (sizeof(D) == 4 ? 1e-4 : 1.0)) ++mism;
  }
  if (mism) {
    ++g_fail;
    printf("FAIL %-42s st=%d mism=%d maxdiff=%.6g\n", name.c_str(), (int)st, mism, maxdiff);
    for (int i = 0; i < elems && i < 8; ++i) printf("      [%d] ref=%g got=%g\n", i, (double)ref[i], (double)got[i]);
  }
  cudaFree(mine);
}

template <typename S, typename D, typename Fn>
static void run_case(const std::string &name, int ch, int srcW, int srcH, int offX, int offY,
                     int dw, int dh, float rate, int taps, std::vector<float> kern,
                     NppiBorderType bt, int lo, int hi, Fn nppCall) {
  const int spx = ch * (int)sizeof(S);
  const int sStep = srcW * spx;
  const int dStep = dw * ch * (int)sizeof(D);
  std::vector<S> hs = uniform_ints<S>((size_t)srcH * sStep / sizeof(S), lo, hi);
  S *src = nullptr;
  D *dst = nullptr;
  float *kk = nullptr;
  cudaMalloc(&src, (size_t)srcH * sStep);
  cudaMalloc(&dst, (size_t)(dh > 0 ? dh : 1) * dStep);
  cudaMalloc(&kk, kern.size() * 4);
  cudaMemcpy(src, hs.data(), (size_t)srcH * sStep, cudaMemcpyHostToDevice);
  cudaMemcpy(kk, kern.data(), kern.size() * 4, cudaMemcpyHostToDevice);
  const S *ps = (const S *)((const char *)src + (size_t)offY * sStep + (size_t)offX * spx);
  NppStatus ref_st = nppCall(ps, sStep, NppiSize{srcW, srcH}, NppiPoint{offX, offY}, dst, dStep,
                             NppiSize{dw, dh}, rate, taps, kk, bt, ctx());
  compare<S, D>(name, const_cast<S *>(ps), sStep, NppiSize{srcW, srcH}, NppiPoint{offX, offY}, dst,
                dStep, NppiSize{dw, dh}, rate, taps, kk, bt, ref_st == NPP_NO_ERROR, ch,
                dw * dh * ch);
  cudaFree(src);
  cudaFree(dst);
  cudaFree(kk);
}


// ------------------------------------------------------------------ statuses
template <typename S, typename D>
static void status_case(const char *name, S *src, int sStep, NppiSize ss, NppiPoint off, D *dst,
                        int dStep, NppiSize roi, float rate, int taps, float *kern,
                        NppiBorderType bt, int ch, NppStatus (*nppFn)(const S *, int, NppiSize,
                        NppiPoint, D *, int, NppiSize, Npp32f, int, const Npp32f *, NppiBorderType,
                        NppStreamContext)) {
  NppStatus ref = nppFn(src, sStep, ss, off, dst, dStep, roi, rate, taps, kern, bt, ctx());
  NppStatus mine;
  if (ch == 1)
    mine = nppi::filter::gaussPyramidBorder<S, D, 1, nppi::filter::kPyramidDown>(
        src, sStep, ss, off, dst, dStep, roi, rate, taps, kern, bt, ctx().hStream);
  else
    mine = nppi::filter::gaussPyramidBorder<S, D, 3, nppi::filter::kPyramidDown>(
        src, sStep, ss, off, dst, dStep, roi, rate, taps, kern, bt, ctx().hStream);
  cudaError_t e = cudaDeviceSynchronize();
  ++g_cases;
  if (ref != mine || e != cudaSuccess) {
    ++g_fail;
    printf("FAIL status %-34s ref=%d mine=%d cuda=%s\n", name, (int)ref, (int)mine,
           cudaGetErrorString(e));
  }
}

static void run_statuses() {
  std::vector<float> k3(100, 0.01f);
  k3[0] = k3[1] = k3[2] = 0.25f;
  Npp32f *src32;
  Npp8u *src8;
  Npp16u *src16;
  Npp32f *dst32;
  Npp8u *dst8;
  Npp16u *dst16;
  float *kk;
  cudaMalloc(&src32, 24 * 18 * 4);
  cudaMalloc(&src8, 24 * 18);
  cudaMalloc(&src16, 24 * 18 * 2);
  cudaMalloc(&dst32, 1024 * 4);
  cudaMalloc(&dst8, 1024);
  cudaMalloc(&dst16, 1024 * 2);
  cudaMalloc(&kk, 100 * 4);
  cudaMemset(src32, 0, 24 * 18 * 4);
  cudaMemset(src8, 1, 24 * 18);
  cudaMemset(src16, 1, 24 * 18 * 2);
  cudaMemcpy(kk, k3.data(), 100 * 4, cudaMemcpyHostToDevice);
  using F32 = NppStatus (*)(const Npp32f *, int, NppiSize, NppiPoint, Npp32f *, int, NppiSize,
                            Npp32f, int, const Npp32f *, NppiBorderType, NppStreamContext);
  using F8 = NppStatus (*)(const Npp8u *, int, NppiSize, NppiPoint, Npp8u *, int, NppiSize, Npp32f,
                           int, const Npp32f *, NppiBorderType, NppStreamContext);
  using F16 = NppStatus (*)(const Npp16u *, int, NppiSize, NppiPoint, Npp16u *, int, NppiSize,
                            Npp32f, int, const Npp32f *, NppiBorderType, NppStreamContext);
  F32 f32 = nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx;
  F8 f8 = nppiFilterGaussPyramidLayerDownBorder_8u_C1R_Ctx;
  F16 f16 = nppiFilterGaussPyramidLayerDownBorder_16u_C1R_Ctx;
  const NppiSize ss{24, 18};
  const NppiSize roi{8, 6};
  const int dRow = 8 * 4;
  struct C { const char *n; int sstep, dstep; NppiSize ssz, r; NppiPoint off; float rate;
             int taps; float *k; NppiBorderType bt; };
  C cs[] = {
      {"ok", 96, dRow, ss, roi, {0, 0}, 2.0f, 3, kk, NPP_BORDER_REPLICATE},
      {"srcsize0", 96, dRow, {0, 18}, roi, {0, 0}, 2.0f, 3, kk, NPP_BORDER_REPLICATE},
      {"roineg", 96, dRow, ss, {-1, 6}, {0, 0}, 2.0f, 3, kk, NPP_BORDER_REPLICATE},
      {"roizero", 96, 1, ss, {0, 6}, {0, 0}, 2.0f, 3, kk, NPP_BORDER_REPLICATE},
      {"roizero+badrate", 96, 1, ss, {0, 6}, {0, 0}, 1.0f, 3, kk, NPP_BORDER_REPLICATE},
      {"roizero+badtaps", 96, 1, ss, {0, 6}, {0, 0}, 2.0f, 1, kk, NPP_BORDER_REPLICATE},
      {"roizero+badborder", 96, 1, ss, {0, 6}, {0, 0}, 2.0f, 3, kk, NPP_BORDER_CONSTANT},
      {"offneg", 96, dRow, ss, roi, {-1, 0}, 2.0f, 3, kk, NPP_BORDER_REPLICATE},
      {"offeq", 96, dRow, ss, roi, {24, 0}, 2.0f, 3, kk, NPP_BORDER_REPLICATE},
      {"borderconst", 96, dRow, ss, roi, {0, 0}, 2.0f, 3, kk, NPP_BORDER_CONSTANT},
      {"taps1", 96, dRow, ss, roi, {0, 0}, 2.0f, 1, kk, NPP_BORDER_REPLICATE},
      {"taps2", 96, dRow, ss, roi, {0, 0}, 2.0f, 2, kk, NPP_BORDER_REPLICATE},
      {"taps99", 96, dRow, ss, roi, {0, 0}, 2.0f, 99, kk, NPP_BORDER_REPLICATE},
      {"taps100", 96, dRow, ss, roi, {0, 0}, 2.0f, 100, kk, NPP_BORDER_REPLICATE},
      {"rate1", 96, dRow, ss, roi, {0, 0}, 1.0f, 3, kk, NPP_BORDER_REPLICATE},
      {"rate10", 96, dRow, ss, roi, {0, 0}, 10.0f, 3, kk, NPP_BORDER_REPLICATE},
      {"rate10.1", 96, dRow, ss, roi, {0, 0}, 10.1f, 3, kk, NPP_BORDER_REPLICATE},
      {"stepsmall", 96, 31, ss, roi, {0, 0}, 2.0f, 3, kk, NPP_BORDER_REPLICATE},
      {"stepunaligned", 96, 34, ss, roi, {0, 0}, 2.0f, 3, kk, NPP_BORDER_REPLICATE},
      {"srcstep0", 0, dRow, ss, roi, {0, 0}, 2.0f, 3, kk, NPP_BORDER_REPLICATE},
      {"badrate+badtaps", 96, dRow, ss, roi, {0, 0}, 1.0f, 1, kk, NPP_BORDER_REPLICATE},
      {"badtaps+badborder", 96, dRow, ss, roi, {0, 0}, 2.0f, 1, kk, NPP_BORDER_CONSTANT},
      {"badborder+badstep", 96, 1, ss, roi, {0, 0}, 2.0f, 3, kk, NPP_BORDER_CONSTANT},
      {"offneg+badrate", 96, dRow, ss, roi, {-1, 0}, 1.0f, 3, kk, NPP_BORDER_REPLICATE},
  };
  for (auto &c : cs) {
    status_case<Npp32f, Npp32f>(c.n, src32, c.sstep, c.ssz, c.off, dst32, c.dstep, c.r, c.rate,
                                c.taps, c.k, c.bt, 1, f32);
  }
  // 8u / 16u matrices (channel 1) -- same table with the matching element type
  for (auto &c : cs) {
    status_case<Npp8u, Npp8u>(c.n, src8, c.sstep, c.ssz, c.off, dst8, c.dstep, c.r, c.rate, c.taps,
                              c.k, c.bt, 1, f8);
    status_case<Npp16u, Npp16u>(c.n, src16, c.sstep, c.ssz, c.off, dst16, c.dstep, c.r, c.rate,
                                c.taps, c.k, c.bt, 1, f16);
  }
  cudaFree(src32);
  cudaFree(src8);
  cudaFree(src16);
  cudaFree(dst32);
  cudaFree(dst8);
  cudaFree(dst16);
  cudaFree(kk);
}

int main() {
  const std::vector<float> k3 = {0.25f, 0.5f, 0.25f};
  const std::vector<float> k3u = {1.f / 3.f, 1.f / 3.f, 1.f / 3.f};
  const std::vector<float> k5 = {1.f / 16, 4.f / 16, 6.f / 16, 4.f / 16, 1.f / 16};
  const std::vector<float> k4 = {0.1f, 0.2f, 0.3f, 0.4f};
  struct RC { float rate; int dw, dh; };
  const RC rcs[] = {{1.5f, 16, 12}, {2.0f, 12, 9}, {2.5f, 9, 7}, {3.0f, 8, 6}, {3.7f, 6, 4}, {1.25f, 19, 14}};

#define DOWN8(fn, ch)                                                                        \
  nppiFilterGaussPyramidLayerDownBorder_##fn##_Ctx
  for (const auto &rc : rcs) {
    for (int bt = 0; bt < 2; ++bt) {
      const NppiBorderType b = bt ? NPP_BORDER_MIRROR : NPP_BORDER_REPLICATE;
      const char *bn = bt ? "mirror" : "repl";
      char nm[128];
      snprintf(nm, 128, "32f C1 r%.2f %s k3", rc.rate, bn);
      run_case<Npp32f, Npp32f>(nm, 1, 24, 18, 0, 0, rc.dw, rc.dh, rc.rate, 3, k3u, b, -100, 400,
                               nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx);
      snprintf(nm, 128, "8u C1 r%.2f %s k3", rc.rate, bn);
      run_case<Npp8u, Npp8u>(nm, 1, 24, 18, 0, 0, rc.dw, rc.dh, rc.rate, 3, k3, b, 0, 255,
                             nppiFilterGaussPyramidLayerDownBorder_8u_C1R_Ctx);
      snprintf(nm, 128, "16u C1 r%.2f %s k3", rc.rate, bn);
      run_case<Npp16u, Npp16u>(nm, 1, 24, 18, 0, 0, rc.dw, rc.dh, rc.rate, 3, k3, b, 0, 65535,
                               nppiFilterGaussPyramidLayerDownBorder_16u_C1R_Ctx);
      snprintf(nm, 128, "32f C3 r%.2f %s k3", rc.rate, bn);
      run_case<Npp32f, Npp32f>(nm, 3, 24, 18, 0, 0, rc.dw, rc.dh, rc.rate, 3, k3, b, -100, 400,
                               nppiFilterGaussPyramidLayerDownBorder_32f_C3R_Ctx);
      snprintf(nm, 128, "8u C3 r%.2f %s k3", rc.rate, bn);
      run_case<Npp8u, Npp8u>(nm, 3, 24, 18, 0, 0, rc.dw, rc.dh, rc.rate, 3, k3, b, 0, 255,
                             nppiFilterGaussPyramidLayerDownBorder_8u_C3R_Ctx);
      snprintf(nm, 128, "16u C3 r%.2f %s k3", rc.rate, bn);
      run_case<Npp16u, Npp16u>(nm, 3, 24, 18, 0, 0, rc.dw, rc.dh, rc.rate, 3, k3, b, 0, 65535,
                               nppiFilterGaussPyramidLayerDownBorder_16u_C3R_Ctx);
    }
  }
  // k5 / even taps / offsets / overhang / rate10
  run_case<Npp32f, Npp32f>("32f C1 r2 k5", 1, 24, 18, 0, 0, 12, 9, 2.0f, 5, k5, NPP_BORDER_REPLICATE, -100, 400, nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx);
  run_case<Npp16u, Npp16u>("16u C3 r2 k5 mirror", 3, 24, 18, 0, 0, 12, 9, 2.0f, 5, k5, NPP_BORDER_MIRROR, 0, 65535, nppiFilterGaussPyramidLayerDownBorder_16u_C3R_Ctx);
  run_case<Npp32f, Npp32f>("32f C1 r2 even4", 1, 24, 18, 0, 0, 10, 8, 2.0f, 4, k4, NPP_BORDER_REPLICATE, -100, 400, nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx);
  run_case<Npp8u, Npp8u>("8u C1 r2 even4", 1, 24, 18, 0, 0, 10, 8, 2.0f, 4, k4, NPP_BORDER_REPLICATE, 0, 255, nppiFilterGaussPyramidLayerDownBorder_8u_C1R_Ctx);
  run_case<Npp32f, Npp32f>("32f C1 off(2,3) r2", 1, 24, 18, 2, 3, 8, 6, 2.0f, 3, k3, NPP_BORDER_REPLICATE, -100, 400, nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx);
  run_case<Npp32f, Npp32f>("32f C1 off(3,2) r2.5 k5 mirror", 1, 24, 18, 3, 2, 8, 6, 2.5f, 5, k5, NPP_BORDER_MIRROR, -100, 400, nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx);
  run_case<Npp8u, Npp8u>("8u C3 off(2,2) r1.5", 3, 24, 18, 2, 2, 12, 8, 1.5f, 3, k3, NPP_BORDER_REPLICATE, 0, 255, nppiFilterGaussPyramidLayerDownBorder_8u_C3R_Ctx);
  run_case<Npp32f, Npp32f>("32f C1 overhang r1.5", 1, 24, 18, 0, 0, 16, 12, 1.5f, 3, k3, NPP_BORDER_REPLICATE, -100, 400, nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx);
  run_case<Npp16u, Npp16u>("16u C1 rate10", 1, 24, 18, 0, 0, 3, 2, 10.0f, 3, k3u, NPP_BORDER_REPLICATE, 0, 65535, nppiFilterGaussPyramidLayerDownBorder_16u_C1R_Ctx);
  run_case<Npp32f, Npp32f>("32f C1 large taps 7", 1, 24, 18, 0, 0, 8, 6, 2.0f, 7, {1.f/64,6.f/64,15.f/64,20.f/64,15.f/64,6.f/64,1.f/64}, NPP_BORDER_MIRROR, -100, 400, nppiFilterGaussPyramidLayerDownBorder_32f_C1R_Ctx);

  run_statuses();
  printf("%d cases, %d failed\n", g_cases, g_fail);
  return g_fail ? 1 : 0;
}
