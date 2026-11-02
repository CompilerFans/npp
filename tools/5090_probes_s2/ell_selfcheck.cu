// Compares the mcMPP elliptical radial profile executor against NVIDIA NPP
// 13.4 on the RTX 5090 (package s2).
//
//   nvcc -O2 -std=c++17 -o /tmp/ell_selfcheck ell_selfcheck.cu \
//       -lnppist -lnppial -lnppc -lcudart && /tmp/ell_selfcheck
//
// Two checks:
//   * stats: per-ring nPixels must be identical and mean/std within 2e-5 on
//     pattern/constant/zero images, padded steps and centres in/out of the ROI;
//   * map: for every pixel impulse, the ring assigned by NPP and by the
//     executor must match (this is what pins the per-ring scan crop).
// The current executor reports 10/10 stats and 6/6 maps with zero mismatch.
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "../../src/nppi/nppi_statistics_functions/nppi_radial_profile_executor.cu"

extern "C" {
NppStatus nppGetStreamContext(NppStreamContext *c);
}
static NppStreamContext ctx() {
  NppStreamContext c{};
  nppGetStreamContext(&c);
  return c;
}
static int gFail = 0;

static void pattern(std::vector<float> &img, int stride, int H) {
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < stride; ++x)
      img[(size_t)y * stride + x] = (float)(((x * 11 + y * 7) % 23) - 11) * 0.5f + (float)(x & 3) * 0.25f;
}

static void stats_case(const char *tag, int W, int H, float cx, float cy, float ratio, float orient, int S,
                       int kind) {
  const int stride = W + 3;
  std::vector<float> img((size_t)stride * H);
  pattern(img, stride, H);
  if (kind == 1) for (auto &v : img) v = 3.5f;
  if (kind == 2) for (auto &v : img) v = 0.0f;
  float *d = nullptr;
  cudaMalloc(&d, img.size() * sizeof(float));
  cudaMemcpy(d, img.data(), img.size() * sizeof(float), cudaMemcpyHostToDevice);
  NppiProfileData *ref = nullptr, *mine = nullptr;
  cudaMalloc(&ref, (size_t)S * sizeof(NppiProfileData));
  cudaMalloc(&mine, (size_t)S * sizeof(NppiProfileData));
  std::vector<NppiProfileData> zero((size_t)S, NppiProfileData{-777, -7.f, -7.f});
  cudaMemcpy(ref, zero.data(), zero.size() * sizeof(NppiProfileData), cudaMemcpyHostToDevice);
  cudaMemcpy(mine, zero.data(), zero.size() * sizeof(NppiProfileData), cudaMemcpyHostToDevice);
  NppStatus s1 = nppiEllipticalRadialProfile_32f_C1R_Ctx(d, stride * 4, NppiSize{W, H}, NppiPoint32f{cx, cy},
                                                         ratio, orient, ref, (Npp16u)S, ctx());
  NppStatus s2 = nppi::statistics::runEllipticalRadialProfile(d, stride * 4, NppiSize{W, H}, NppiPoint32f{cx, cy},
                                                              ratio, orient, mine, (Npp16u)S, 0);
  std::vector<NppiProfileData> r((size_t)S), m((size_t)S);
  cudaMemcpy(r.data(), ref, r.size() * sizeof(NppiProfileData), cudaMemcpyDeviceToHost);
  cudaMemcpy(m.data(), mine, m.size() * sizeof(NppiProfileData), cudaMemcpyDeviceToHost);
  int bad = 0, countBad = 0;
  for (int i = 0; i < S; ++i) {
    if (r[i].nPixels != m[i].nPixels) ++countBad;
    if (r[i].nPixels != m[i].nPixels || fabsf(r[i].nMeanIntensity - m[i].nMeanIntensity) > 2e-5f ||
        fabsf(r[i].nStdDevIntensity - m[i].nStdDevIntensity) > 2e-5f) {
      if (bad < 3)
        printf("   [%s] ring %d npp(n=%d, m=%.7g, sd=%.7g) mcpp(n=%d, m=%.7g, sd=%.7g)\n", tag, i, r[i].nPixels,
               r[i].nMeanIntensity, r[i].nStdDevIntensity, m[i].nPixels, m[i].nMeanIntensity, m[i].nStdDevIntensity);
      ++bad;
    }
  }
  printf("%-42s st=%d/%d rings=%d count_mismatch=%d stat_mismatch=%d\n", tag, (int)s1, (int)s2, S, countBad, bad);
  if (bad || s1 != s2) ++gFail;
  cudaFree(d);
  cudaFree(ref);
  cudaFree(mine);
}

// Impulse map: which ring each pixel lands in (for both implementations).
static void map_case(const char *tag, int W, int H, float cx, float cy, float ratio, float orient, int S) {
  float *src = nullptr;
  NppiProfileData *ref = nullptr, *mine = nullptr;
  cudaMalloc(&src, (size_t)W * H * sizeof(float));
  cudaMalloc(&ref, (size_t)S * sizeof(NppiProfileData));
  cudaMalloc(&mine, (size_t)S * sizeof(NppiProfileData));
  std::vector<float> img((size_t)W * H, 0.0f);
  std::vector<NppiProfileData> zero((size_t)S, NppiProfileData{0, -1e30f, -1e30f});
  int mism = 0, nppHits = 0;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      img[(size_t)y * W + x] = 1.0f;
      cudaMemcpy(src, img.data(), img.size() * sizeof(float), cudaMemcpyHostToDevice);
      img[(size_t)y * W + x] = 0.0f;
      cudaMemcpy(ref, zero.data(), zero.size() * sizeof(NppiProfileData), cudaMemcpyHostToDevice);
      cudaMemcpy(mine, zero.data(), zero.size() * sizeof(NppiProfileData), cudaMemcpyHostToDevice);
      NppStatus s1 = nppiEllipticalRadialProfile_32f_C1R_Ctx(src, W * 4, NppiSize{W, H}, NppiPoint32f{cx, cy}, ratio,
                                                             orient, ref, (Npp16u)S, ctx());
      NppStatus s2 = nppi::statistics::runEllipticalRadialProfile(src, W * 4, NppiSize{W, H},
                                                                  NppiPoint32f{cx, cy}, ratio, orient, mine,
                                                                  (Npp16u)S, 0);
      if (s1 != s2) { ++mism; continue; }
      std::vector<NppiProfileData> r((size_t)S), m((size_t)S);
      cudaMemcpy(r.data(), ref, r.size() * sizeof(NppiProfileData), cudaMemcpyDeviceToHost);
      cudaMemcpy(m.data(), mine, m.size() * sizeof(NppiProfileData), cudaMemcpyDeviceToHost);
      int rr = -1, mr = -1;
      for (int i = 0; i < S; ++i) {
        if (r[i].nPixels > 0 && r[i].nMeanIntensity != 0.0f) rr = i;
        if (m[i].nPixels > 0 && m[i].nMeanIntensity != 0.0f) mr = i;
      }
      if (rr >= 0) ++nppHits;
      if (rr != mr) {
        if (mism < 5) printf("   [%s] pixel (%d,%d): npp ring %d, mcpp ring %d\n", tag, x, y, rr, mr);
        ++mism;
      }
    }
  }
  printf("%-42s map_mismatch=%d npp_hits=%d\n", tag, mism, nppHits);
  if (mism) ++gFail;
  cudaFree(src);
  cudaFree(ref);
  cudaFree(mine);
}

int main() {
  stats_case("stats 33 c16.5 S24 r2 o0", 33, 33, 16.5f, 16.5f, 2.f, 0.f, 24, 0);
  stats_case("stats 33 c16.5 S24 r2 o30deg", 33, 33, 16.5f, 16.5f, 2.f, 0.52359878f, 24, 0);
  stats_case("stats 33 c16.5 S24 r1.5 o0.7", 33, 33, 16.5f, 16.5f, 1.5f, 0.7f, 24, 0);
  stats_case("stats 33 c16 S24 r1 o0.3", 33, 33, 16.0f, 16.0f, 1.0f, 0.3f, 24, 0);
  stats_case("stats 40x25 c20.5 S24 r2 o0.9", 40, 25, 20.5f, 12.5f, 2.f, 0.9f, 24, 0);
  stats_case("stats 33 c-3.5,-2.5 S8 r2 o0", 33, 33, -3.5f, -2.5f, 2.f, 0.f, 8, 0);
  stats_case("stats 33 c31.75,0.25 S8 r1.1 o1.2", 33, 33, 31.75f, 0.25f, 1.1f, 1.2f, 8, 0);
  stats_case("stats 21x15 const S12 r1.5 o0.4", 21, 15, 10.5f, 7.5f, 1.5f, 0.4f, 12, 1);
  stats_case("stats 17x9 zeros S12 r2 o0", 17, 9, 8.5f, 4.5f, 2.f, 0.f, 12, 2);
  stats_case("stats 33 c16.5 S48 r1.25 o2.0", 33, 33, 16.5f, 16.5f, 1.25f, 2.0f, 48, 0);
  map_case("map 33 r2 o0", 33, 33, 16.5f, 16.5f, 2.f, 0.f, 24);
  map_case("map 33 r2 o30deg", 33, 33, 16.5f, 16.5f, 2.f, 0.52359878f, 24);
  map_case("map 33 r2 o45deg", 33, 33, 16.5f, 16.5f, 2.f, 0.78539816f, 24);
  map_case("map 33 r1.5 o0.7", 33, 33, 16.5f, 16.5f, 1.5f, 0.7f, 24);
  map_case("map 33 r1 o0.3", 33, 33, 16.5f, 16.5f, 1.f, 0.3f, 24);
  map_case("map 33 r2 o60deg", 33, 33, 16.5f, 16.5f, 2.f, 1.04719755f, 24);
  printf("FAILURES=%d\n", gFail);
  return gFail ? 1 : 0;
}
