// Probe Q: end-to-end check of the mcMPP unsharp executor against the recorded
// NPP 13.4 goldens (f11).  Reproduces a few differential-test cases with the
// harness RNG (make_uniform) and compares the executor output with
// ~/f11u_records/nppiFilterUnsharp/<case>.bin.
#include "npp.h"
#include "nppi_filter_unsharp_executor.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#define CK(x) do { cudaError_t e_=(x); if(e_){printf("CUDAERR %s\n", cudaGetErrorString(e_)); exit(1);} } while(0)

static NppStreamContext mkctx() {
  NppStreamContext c{};
  c.hStream = 0;
  int dev = 0; CK(cudaGetDevice(&dev)); c.nCudaDeviceId = dev;
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  c.nMultiProcessorCount = prop.multiProcessorCount;
  c.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  c.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
  c.nSharedMemPerBlock = prop.sharedMemPerBlock;
  return c;
}

struct Rng {
  uint64_t s_;
  explicit Rng(uint64_t seed) : s_(seed ? seed : 0x9e3779b97f4a7c15ULL) {}
  uint64_t next() {
    uint64_t z = (s_ += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  float unit_float() { return (float)(next() >> 40) / 16777216.0f; }
  double unit_double() { return (double)(next() >> 11) / 9007199254740992.0; }
};

template <typename T> std::vector<T> uniform(size_t n, uint64_t seed) {
  std::vector<T> v(n);
  Rng r(seed);
  for (auto &x : v) {
    if constexpr (std::is_floating_point_v<T>) x = (T)(r.unit_double() * 4.0 - 2.0);
    else if constexpr (std::is_same_v<T, Npp8u>) x = (Npp8u)(r.next() % 256);
    else x = (T)(r.next() % 65536);
  }
  return v;
}

static int align_step(int step, int es) { return ((step + es - 1) / es) * es; }

template <typename S, typename D, int Ch, bool AC4>
static bool run_case(const char *label, const char *golden, int offX, int offY, int W, int H, int pad,
                     uint64_t seed, float radius, float sigma, float weight, float threshold) {
  const int es = (int)sizeof(S);
  const int spx = Ch * es;
  const int srcW = W + offX, srcH = H + offY;
  const int sStep = align_step(srcW * spx, spx);
  const int dpx = Ch * (int)sizeof(D);
  const int dAlign = Ch == 4 ? dpx : (int)sizeof(D);
  const int dStep = align_step(W * dpx + pad, dAlign);
  const size_t se = (size_t)srcH * sStep / sizeof(S);
  const size_t de = (size_t)H * dStep / sizeof(D);
  std::vector<S> hs = uniform<S>(se, seed + 11);
  std::vector<D> hd = uniform<D>(de, seed + 29);
  S *ds; D *dd;
  CK(cudaMalloc(&ds, se * sizeof(S)));
  CK(cudaMalloc(&dd, de * sizeof(D)));
  CK(cudaMemcpy(ds, hs.data(), se * sizeof(S), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dd, hd.data(), de * sizeof(D), cudaMemcpyHostToDevice));
  Npp8u *buf;
  CK(cudaMalloc(&buf, 4096));
  NppStreamContext ctx = mkctx();
  const S *pSrc = ds + (size_t)offY * (sStep / sizeof(S)) + (size_t)offX * Ch;
  NppStatus st = nppi::filter::filterUnsharpBorder<S, D, Ch, AC4>(pSrc, sStep, NppiPoint{offX, offY}, dd,
                                                                 dStep, NppiSize{W, H}, radius, sigma,
                                                                 weight, threshold,
                                                                 NPP_BORDER_REPLICATE, buf, ctx.hStream);
  std::vector<D> out(de);
  CK(cudaMemcpy(out.data(), dd, de * sizeof(D), cudaMemcpyDeviceToHost));
  std::vector<D> gold(de);
  FILE *f = fopen(golden, "rb");
  size_t got = f ? fread(gold.data(), sizeof(D), de, f) : 0;
  if (f) fclose(f);
  int mism = 0, maxdiff = 0;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      for (int c = 0; c < Ch; ++c) {
        if (AC4 && c == 3) continue;  // alpha untouched
        const size_t i = (size_t)y * (dStep / sizeof(D)) + (size_t)x * Ch + c;
        const double a = (double)out[i], b = (double)gold[i];
        const double d = std::fabs(a - b);
        if (d > 1e-3 + 1e-3 * std::fabs(b)) ++mism;
        if (d > maxdiff) maxdiff = (int)d;
      }
  printf("%-28s status=%d golden_bytes=%zu mism=%d maxdiff=%d\n", label, (int)st, got * sizeof(D), mism,
         maxdiff);
  cudaFree(ds); cudaFree(dd); cudaFree(buf);
  return mism == 0 && st == NPP_NO_ERROR;
}

static void gbs(const char *label, int got, int want) {
  printf("%-28s size=%d expected=%d %s\n", label, got, want, got == want ? "ok" : "MISMATCH");
}

int main() {
  cudaFree(0);
  const char *R = "/home/metax/f11u_records/nppiFilterUnsharp/";
  char path[512];
  int fails = 0;
  auto P = [&](const char *case_name) {
    snprintf(path, sizeof(path), "%s%s.bin", R, case_name);
    return (const char *)path;
  };
  const int ch1 = 0x1000 + 1 * 7 + 13;  // channels*7 + sizeof(S)*13
  fails += !run_case<Npp8u, Npp8u, 1, false>("8u/C1 r2s1", P("8u/C1/r2s1"), 0, 0, 13, 10, 2,
                                             ch1 + 3, 2.f, 1.f, 1.2f, 0.f);
  fails += !run_case<Npp8u, Npp8u, 1, false>("8u/C1 r1s1/off21", P("8u/C1/r1s1/off21"), 2, 1, 12, 9, 3,
                                             ch1 + 2, 1.f, 1.f, 1.2f, 0.f);
  fails += !run_case<Npp8u, Npp8u, 4, true>("8u/AC4 r2s1/wneg", P("8u/AC4/r2s1/wneg"), 2, 2, 13, 10, 2,
                                            0x1000 + 4 * 7 + 13 + 5, 2.f, 1.f, -0.7f, 0.f);
  fails += !run_case<Npp16u, Npp16u, 3, false>("16u/C3 auto_s1", P("16u/C3/auto_s1"), 1, 1, 17, 17, 2,
                                               0x1000 + 3 * 7 + 2 * 13 + 6, 0.f, 1.f, 1.2f, 0.f);
  fails += !run_case<Npp16s, Npp16s, 4, false>("16s/C4 r2s1", P("16s/C4/r2s1"), 0, 0, 13, 10, 2,
                                               0x1000 + 4 * 7 + 2 * 13 + 3, 2.f, 1.f, 1.2f, 0.f);
  fails += !run_case<Npp32f, Npp32f, 1, false>("32f/C1 auto_s1", P("32f/C1/auto_s1"), 1, 1, 17, 17, 2,
                                               0x1000 + 1 * 7 + 4 * 13 + 6, 0.f, 1.f, 1.2f, 0.f);
  fails += !run_case<Npp32f, Npp32f, 4, true>("32f/AC4 r1s1", P("32f/AC4/r1s1"), 0, 0, 12, 9, 2,
                                              0x1000 + 4 * 7 + 4 * 13 + 1, 1.f, 1.f, 1.2f, 0.f);
  fails += !run_case<Npp8u, Npp8u, 1, false>("8u/C1 thr1_delta", P("8u/C1/thr1_delta"), 0, 0, 17, 11, 2,
                                             0x3001, 2.f, 1.f, 1.0f, 1.f);
  // buffer sizes
  struct { float r, s; int want8u, want32f; } cs[] = {
      {1.f, 1.f, 36, 36}, {2.f, 1.f, 100, 100}, {0.f, 1.f, 196, 900}, {0.f, 0.5f, 36, 196},
      {2.f, 0.f, 100, 100}, {0.f, 2.f, 484, 3364}, {0.f, 0.25f, 36, 36}};
  for (auto &c : cs) {
    int a = -1, b = -1;
    nppi::filter::unsharpGetBufferSizeT<Npp8u>(c.r, c.s, &a);
    nppi::filter::unsharpGetBufferSizeT<Npp32f>(c.r, c.s, &b);
    gbs("gbs 8u", a, c.want8u);
    gbs("gbs 32f", b, c.want32f);
    if (a != c.want8u || b != c.want32f) ++fails;
  }
  printf("FAILS=%d\n", fails);
  return fails ? 1 : 0;
}
