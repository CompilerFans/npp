// NPP 13.4 elliptical radial profile probe (package s2).
//
//   usage: ell_probe map <ratio> <orientation_radians>
//          ell_probe case <name>
//
// `map` injects a unit impulse at every pixel and prints "x y hits ring" where
// `ring` is the profile entry with nPixels > 0 (and a non-zero mean), or -1.
// The membership map pins the per-ring scan crop: for 33x33, ratio 2,
// orientation 0 and 24 samples only 1065 of 1089 pixels are assigned.
//
// `case` checks one host-side status contract per process (the reference can
// leave a sticky CUDA error or spin on an infinite ratio, so each case exits
// with _exit before teardown).  Measured order: null pointer -> negative ROI
// -> ratio < 1 -> empty ROI/samples no-op.  NaN and +inf pass `ratio < 1`;
// -inf is rejected.
//
// Build and run on the RTX 5090:
//   nvcc -O2 -std=c++17 -o /tmp/ell_probe ell_probe.cu \
//       -lnppist -lnppial -lnppc -lcudart
//   /tmp/ell_probe map 2 0.52359878
//   for c in r0.5 r0.999 rnan rninf r05_zeroW r05_negROI; do \
//       timeout 20 /tmp/ell_probe case $c; done
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <unistd.h>

extern "C" {
NppStatus nppGetStreamContext(NppStreamContext *pNppStreamContext);
}

static NppStreamContext ctx() {
  NppStreamContext c{};
  nppGetStreamContext(&c);
  return c;
}

static const char *sname(NppStatus s) {
  switch (s) {
    case NPP_SUCCESS: return "OK";
    case NPP_NULL_POINTER_ERROR: return "NULL";
    case NPP_SIZE_ERROR: return "SIZE";
    case NPP_STEP_ERROR: return "STEP";
    case NPP_RANGE_ERROR: return "RANGE";
    default: return "OTHER";
  }
}

static int mapSection(float ratio, float orient) {
  const int W = 33, H = 33, S = 24;
  NppStreamContext c = ctx();
  float *src = nullptr;
  NppiProfileData *prof = nullptr;
  cudaMalloc(&src, (size_t)W * H * sizeof(float));
  cudaMalloc(&prof, (size_t)(S + 4) * sizeof(NppiProfileData));
  std::vector<float> img((size_t)W * H, 0.0f);
  std::vector<NppiProfileData> zero((size_t)S + 4, NppiProfileData{0, -1e30f, -1e30f});
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      img[(size_t)y * W + x] = 1.0f;
      cudaMemcpy(src, img.data(), img.size() * sizeof(float), cudaMemcpyHostToDevice);
      img[(size_t)y * W + x] = 0.0f;
      cudaMemcpy(prof, zero.data(), zero.size() * sizeof(NppiProfileData), cudaMemcpyHostToDevice);
      NppStatus st = nppiEllipticalRadialProfile_32f_C1R_Ctx(
          src, W * 4, NppiSize{W, H}, NppiPoint32f{16.5f, 16.5f}, ratio, orient, prof, (Npp16u)S, c);
      if (st != NPP_SUCCESS) {
        printf("status %d (%s)\n", (int)st, sname(st));
        return 1;
      }
      std::vector<NppiProfileData> v((size_t)S + 4);
      cudaMemcpy(v.data(), prof, v.size() * sizeof(NppiProfileData), cudaMemcpyDeviceToHost);
      int ring = -1, hits = 0;
      for (int i = 0; i < S; ++i) {
        if (v[i].nPixels > 0 && v[i].nMeanIntensity != 0.0f) {
          ring = i;
          ++hits;
        }
      }
      printf("%d %d %d %d\n", x, y, hits == 1 ? ring : -1, hits);
    }
  }
  cudaFree(src);
  cudaFree(prof);
  return 0;
}

static int caseSection(const char *m) {
  const int W = 33, H = 33, S = 8;
  float *src = nullptr;
  NppiProfileData *prof = nullptr;
  cudaMalloc(&src, (size_t)W * H * sizeof(float));
  cudaMalloc(&prof, (size_t)S * sizeof(NppiProfileData));
  cudaMemset(src, 0, (size_t)W * H * sizeof(float));
  NppStreamContext c = ctx();
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  float ratio = 2.0f, orient = 0.0f;
  NppiSize roi{W, H};
  const float *sptr = src;
  NppiProfileData *pptr = prof;
  Npp16u samples = S;
  if (!strcmp(m, "r0.5")) ratio = 0.5f;
  else if (!strcmp(m, "r0.75")) ratio = 0.75f;
  else if (!strcmp(m, "r0.999")) ratio = 0.999f;
  else if (!strcmp(m, "r1.0")) ratio = 1.0f;
  else if (!strcmp(m, "r1.0001")) ratio = 1.0001f;
  else if (!strcmp(m, "r1e5")) ratio = 1e5f;
  else if (!strcmp(m, "rinf")) ratio = inf;
  else if (!strcmp(m, "rninf")) ratio = -inf;
  else if (!strcmp(m, "rnan")) ratio = nan;
  else if (!strcmp(m, "oinf")) orient = inf;
  else if (!strcmp(m, "oninf")) orient = -inf;
  else if (!strcmp(m, "onan")) orient = nan;
  else if (!strcmp(m, "r05_nullS")) { ratio = 0.5f; sptr = nullptr; }
  else if (!strcmp(m, "r05_nullP")) { ratio = 0.5f; pptr = nullptr; }
  else if (!strcmp(m, "r05_zeroW")) { ratio = 0.5f; roi = {0, H}; }
  else if (!strcmp(m, "r05_S0")) { ratio = 0.5f; samples = 0; }
  else if (!strcmp(m, "r05_negROI")) { ratio = 0.5f; roi = {-1, H}; }
  else if (!strcmp(m, "nanOri_negROI")) { orient = nan; roi = {-1, H}; }
  else if (!strcmp(m, "nanOri_S0")) { orient = nan; samples = 0; }
  else if (!strcmp(m, "nanOri_zeroW")) { orient = nan; roi = {0, H}; }
  else if (!strcmp(m, "nanOri_nullS")) { orient = nan; sptr = nullptr; }
  else return 2;
  NppStatus s = nppiEllipticalRadialProfile_32f_C1R_Ctx(sptr, W * 4, roi, NppiPoint32f{16.5f, 16.5f}, ratio, orient,
                                                        pptr, samples, c);
  printf("%-14s %s (%d)\n", m, sname(s), (int)s);
  fflush(stdout);
  _exit(0);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s map <ratio> <orientation> | case <name>\n", argv[0]);
    return 2;
  }
  if (!strcmp(argv[1], "map")) {
    if (argc < 4) return 2;
    return mapSection(strtof(argv[2], nullptr), strtof(argv[3], nullptr));
  }
  if (!strcmp(argv[1], "case")) {
    if (argc < 3) return 2;
    return caseSection(argv[2]);
  }
  fprintf(stderr, "unknown mode %s\n", argv[1]);
  return 2;
}
