// Probe the NPP 13.4 unsigned PBA distance-transform outputs:
// layout, rounding, no-site behaviour and tie-breaking among equidistant
// sites.  All images are 64x64 (the family's minimum ROI).
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <nppi.h>
#include <nppi_filtering_functions.h>

extern "C" {
NppStatus nppGetStreamContext(NppStreamContext *c);
}

#define CK(x)                                                                                                      \
  do {                                                                                                             \
    cudaError_t e_ = (x);                                                                                          \
    if (e_ != cudaSuccess) {                                                                                       \
      printf("cuda err %s %d\n", cudaGetErrorString(e_), __LINE__);                                                \
      return;                                                                                                      \
    }                                                                                                              \
  } while (0)

namespace {

constexpr int W = 64, H = 64;
constexpr int STEP16 = W * 2 * (int)sizeof(Npp16s);
constexpr int STEPU16 = W * (int)sizeof(Npp16u);

struct Pixel {
  int x, y;
};

void runCase(const char *name, const std::vector<Pixel> &sites) {
  std::vector<unsigned char> src(W * H, 255);
  for (const Pixel &p : sites) src[p.y * W + p.x] = 0;

  size_t bufferSize = 0;
  NppStatus st = nppiDistanceTransformPBAGetBufferSize({W, H}, &bufferSize);
  printf("=== %s (%zu sites, buffer=%zu, getbuf status=%d)\n", name, sites.size(), bufferSize, (int)st);
  if (st != NPP_NO_ERROR) return;

  unsigned char *dSrc = nullptr;
  Npp8u *dBuffer = nullptr;
  Npp16s *dVoronoi = nullptr, *dIndices = nullptr, *dRel = nullptr;
  Npp16u *dTransform = nullptr;
  CK(cudaMalloc(&dSrc, src.size()));
  CK(cudaMalloc(&dBuffer, bufferSize));
  CK(cudaMalloc(&dVoronoi, (size_t)STEP16 * H));
  CK(cudaMalloc(&dIndices, (size_t)STEP16 * H));
  CK(cudaMalloc(&dRel, (size_t)STEP16 * H));
  CK(cudaMalloc(&dTransform, (size_t)STEPU16 * H));
  CK(cudaMemcpy(dSrc, src.data(), src.size(), cudaMemcpyHostToDevice));
  CK(cudaMemset(dVoronoi, 0xEE, (size_t)STEP16 * H));
  CK(cudaMemset(dIndices, 0xCC, (size_t)STEP16 * H));
  CK(cudaMemset(dRel, 0xAA, (size_t)STEP16 * H));
  CK(cudaMemset(dTransform, 0xEE, (size_t)STEPU16 * H));

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  st = nppiDistanceTransformPBA_8u16u_C1R_Ctx(dSrc, W, 0, 0, dVoronoi, STEP16, dIndices, STEP16, dRel, STEP16,
                                              dTransform, STEPU16, {W, H}, dBuffer, ctx);
  CK(cudaDeviceSynchronize());
  printf("main status=%d\n", (int)st);

  std::vector<Npp16s> vor(STEP16 / 2 * H), ind(STEP16 / 2 * H), rel(STEP16 / 2 * H);
  std::vector<Npp16u> tr(STEPU16 / 2 * H);
  CK(cudaMemcpy(vor.data(), dVoronoi, vor.size() * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(ind.data(), dIndices, ind.size() * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(rel.data(), dRel, rel.size() * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(tr.data(), dTransform, tr.size() * 2, cudaMemcpyDeviceToHost));

  const Pixel probes[] = {{0, 0},   {1, 0},   {0, 1},   {1, 1},   {2, 3},   {3, 4},   {5, 12},
                          {31, 32}, {32, 31}, {32, 32}, {33, 32}, {32, 33}, {20, 32}, {44, 32},
                          {32, 20}, {32, 44}, {63, 63}, {63, 0},  {0, 63},  {10, 10}, {50, 50}};
  for (const Pixel &p : probes) {
    const int i = p.y * (STEP16 / 2) + p.x * 2;
    const int t = p.y * (STEPU16 / 2) + p.x;
    printf("  (%2d,%2d) vor=(%d,%d) ind=(%d,%d) rel=(%d,%d) tr16=%u\n", p.x, p.y, vor[i], vor[i + 1],
           ind[i], ind[i + 1], rel[i], rel[i + 1], tr[t]);
  }

  cudaFree(dSrc);
  cudaFree(dBuffer);
  cudaFree(dVoronoi);
  cudaFree(dIndices);
  cudaFree(dRel);
  cudaFree(dTransform);
}

void runFloatCase(const char *name, const std::vector<Pixel> &sites, bool is64f) {
  std::vector<unsigned char> src(W * H, 255);
  for (const Pixel &p : sites) src[p.y * W + p.x] = 0;

  size_t bufferSize = 0;
  NppStatus st = nppiDistanceTransformPBAGetBufferSize({W, H}, &bufferSize);
  printf("=== %s (64f=%d, buffer=%zu, getbuf=%d)\n", name, (int)is64f, bufferSize, (int)st);
  if (st != NPP_NO_ERROR) return;

  unsigned char *dSrc = nullptr;
  Npp8u *dBuffer = nullptr;
  Npp16s *dVoronoi = nullptr, *dIndices = nullptr, *dRel = nullptr;
  Npp32f *dT32 = nullptr;
  Npp64f *dT64 = nullptr;
  CK(cudaMalloc(&dSrc, src.size()));
  CK(cudaMalloc(&dBuffer, bufferSize));
  CK(cudaMalloc(&dVoronoi, (size_t)STEP16 * H));
  CK(cudaMalloc(&dIndices, (size_t)STEP16 * H));
  CK(cudaMalloc(&dRel, (size_t)STEP16 * H));
  CK(cudaMalloc(&dT32, (size_t)W * sizeof(Npp32f) * H));
  CK(cudaMalloc(&dT64, (size_t)W * sizeof(Npp64f) * H));
  CK(cudaMemcpy(dSrc, src.data(), src.size(), cudaMemcpyHostToDevice));
  CK(cudaMemset(dT32, 0xEE, (size_t)W * sizeof(Npp32f) * H));
  CK(cudaMemset(dT64, 0xEE, (size_t)W * sizeof(Npp64f) * H));

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  if (is64f) {
    st = nppiDistanceTransformPBA_8u64f_C1R_Ctx(dSrc, W, 0, 0, dVoronoi, STEP16, dIndices, STEP16, dRel, STEP16,
                                                dT64, W * (int)sizeof(Npp64f), {W, H}, dBuffer, nullptr, ctx);
  } else {
    st = nppiDistanceTransformPBA_8u32f_C1R_Ctx(dSrc, W, 0, 0, dVoronoi, STEP16, dIndices, STEP16, dRel, STEP16,
                                                dT32, W * (int)sizeof(Npp32f), {W, H}, dBuffer, ctx);
  }
  CK(cudaDeviceSynchronize());
  printf("main status=%d\n", (int)st);

  const Pixel probes[] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}, {2, 3}, {3, 4}, {5, 12}, {32, 32}, {63, 63}};
  std::vector<Npp32f> t32((size_t)W * H);
  std::vector<Npp64f> t64((size_t)W * H);
  if (is64f) {
    CK(cudaMemcpy(t64.data(), dT64, t64.size() * sizeof(Npp64f), cudaMemcpyDeviceToHost));
  } else {
    CK(cudaMemcpy(t32.data(), dT32, t32.size() * sizeof(Npp32f), cudaMemcpyDeviceToHost));
  }
  for (const Pixel &p : probes) {
    const size_t i = (size_t)p.y * W + p.x;
    if (is64f) {
      printf("  (%2d,%2d) tr64=%.17g\n", p.x, p.y, t64[i]);
    } else {
      printf("  (%2d,%2d) tr32=%.9g\n", p.x, p.y, t32[i]);
    }
  }
  cudaFree(dSrc);
  cudaFree(dBuffer);
  cudaFree(dVoronoi);
  cudaFree(dIndices);
  cudaFree(dRel);
  cudaFree(dT32);
  cudaFree(dT64);
}

void runAbsCase(const char *name, const std::vector<Pixel> &sites) {
  std::vector<unsigned char> src(W * H, 255);
  for (const Pixel &p : sites) src[p.y * W + p.x] = 0;
  size_t bufferSize = 0;
  NppStatus st = nppiDistanceTransformPBAGetBufferSize({W, H}, &bufferSize);
  printf("=== abs %s (buffer=%zu, getbuf=%d)\n", name, bufferSize, (int)st);
  if (st != NPP_NO_ERROR) return;
  unsigned char *dSrc = nullptr;
  Npp8u *dBuffer = nullptr;
  Npp16s *dVoronoi = nullptr, *dIndices = nullptr;
  Npp16u *dAbs = nullptr, *dTransform = nullptr;
  CK(cudaMalloc(&dSrc, src.size()));
  CK(cudaMalloc(&dBuffer, bufferSize));
  CK(cudaMalloc(&dVoronoi, (size_t)STEP16 * H));
  CK(cudaMalloc(&dIndices, (size_t)STEP16 * H));
  CK(cudaMalloc(&dAbs, (size_t)STEP16 * H));
  CK(cudaMalloc(&dTransform, (size_t)STEPU16 * H));
  CK(cudaMemcpy(dSrc, src.data(), src.size(), cudaMemcpyHostToDevice));
  CK(cudaMemset(dAbs, 0xEE, (size_t)STEP16 * H));
  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  st = nppiDistanceTransformAbsPBA_8u16u_C1R_Ctx(dSrc, W, 0, 0, dVoronoi, STEP16, dIndices, STEP16, dAbs, STEP16,
                                                 dTransform, STEPU16, {W, H}, dBuffer, ctx);
  CK(cudaDeviceSynchronize());
  printf("main status=%d\n", (int)st);
  std::vector<Npp16u> abs((size_t)STEP16 / 2 * H);
  CK(cudaMemcpy(abs.data(), dAbs, abs.size() * 2, cudaMemcpyDeviceToHost));
  const Pixel probes[] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}, {2, 3}, {32, 32}, {32, 20}, {20, 32}, {63, 63}};
  for (const Pixel &p : probes) {
    const int i = p.y * (STEP16 / 2) + p.x * 2;
    printf("  (%2d,%2d) abs=(%u,%u)\n", p.x, p.y, abs[i], abs[i + 1]);
  }
  cudaFree(dSrc);
  cudaFree(dBuffer);
  cudaFree(dVoronoi);
  cudaFree(dIndices);
  cudaFree(dAbs);
  cudaFree(dTransform);
}

void runRangeCase() {
  std::vector<unsigned char> src(W * H, 5);
  src[10 * W + 10] = 15;
  src[20 * W + 20] = 25;
  src[30 * W + 30] = 10;
  size_t bufferSize = 0;
  NppStatus st = nppiDistanceTransformPBAGetBufferSize({W, H}, &bufferSize);
  printf("=== range [10,20] buffer=%zu getbuf=%d\n", bufferSize, (int)st);
  if (st != NPP_NO_ERROR) return;
  unsigned char *dSrc = nullptr;
  Npp8u *dBuffer = nullptr;
  Npp16s *dVoronoi = nullptr;
  CK(cudaMalloc(&dSrc, src.size()));
  CK(cudaMalloc(&dBuffer, bufferSize));
  CK(cudaMalloc(&dVoronoi, (size_t)STEP16 * H));
  CK(cudaMemcpy(dSrc, src.data(), src.size(), cudaMemcpyHostToDevice));
  CK(cudaMemset(dVoronoi, 0xEE, (size_t)STEP16 * H));
  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  st = nppiDistanceTransformPBA_8u16u_C1R_Ctx(dSrc, W, 10, 20, dVoronoi, STEP16, nullptr, 0, nullptr, 0, nullptr, 0,
                                              {W, H}, dBuffer, ctx);
  CK(cudaDeviceSynchronize());
  printf("main status=%d\n", (int)st);
  std::vector<Npp16s> vor((size_t)STEP16 / 2 * H);
  CK(cudaMemcpy(vor.data(), dVoronoi, vor.size() * 2, cudaMemcpyDeviceToHost));
  const Pixel probes[] = {{0, 0}, {10, 10}, {20, 20}, {30, 30}, {15, 15}, {32, 32}};
  for (const Pixel &p : probes) {
    const int i = p.y * (STEP16 / 2) + p.x * 2;
    printf("  (%2d,%2d) vor=(%d,%d)\n", p.x, p.y, vor[i], vor[i + 1]);
  }
  cudaFree(dSrc);
  cudaFree(dBuffer);
  cudaFree(dVoronoi);
}

void runValidation() {
  size_t bufferSize = 0;
  NppStatus st = nppiDistanceTransformPBAGetBufferSize({63, 64}, &bufferSize);
  printf("=== validate getbuf 63x64 status=%d size=%zu\n", (int)st, bufferSize);
  st = nppiDistanceTransformPBAGetBufferSize({64, 63}, &bufferSize);
  printf("=== validate getbuf 64x63 status=%d size=%zu\n", (int)st, bufferSize);
  st = nppiDistanceTransformPBAGetBufferSize({32, 32}, &bufferSize);
  printf("=== validate getbuf 32x32 status=%d size=%zu\n", (int)st, bufferSize);

  std::vector<unsigned char> src(W * H, 255);
  src[32 * W + 32] = 0;
  size_t goodBuffer = 0;
  nppiDistanceTransformPBAGetBufferSize({W, H}, &goodBuffer);
  unsigned char *dSrc = nullptr;
  Npp8u *dBuffer = nullptr;
  Npp16s *dOut = nullptr;
  Npp16u *dT = nullptr;
  CK(cudaMalloc(&dSrc, src.size()));
  CK(cudaMalloc(&dBuffer, goodBuffer));
  CK(cudaMalloc(&dOut, (size_t)STEP16 * H));
  CK(cudaMalloc(&dT, (size_t)STEPU16 * H));
  CK(cudaMemcpy(dSrc, src.data(), src.size(), cudaMemcpyHostToDevice));
  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);

  st = nppiDistanceTransformPBA_8u16u_C1R_Ctx(dSrc, W, 0, 0, dOut, STEP16, nullptr, 0, nullptr, 0, dT, STEPU16,
                                              {63, 64}, dBuffer, ctx);
  printf("=== main 63x64 status=%d\n", (int)st);
  st = nppiDistanceTransformPBA_8u16u_C1R_Ctx(dSrc, W, 0, 0, dOut, 32, nullptr, 0, nullptr, 0, dT, STEPU16,
                                              {64, 64}, dBuffer, ctx);
  printf("=== main voronoi step 32 status=%d\n", (int)st);
  st = nppiDistanceTransformPBA_8u16u_C1R_Ctx(dSrc, 0, 0, 0, dOut, STEP16, nullptr, 0, nullptr, 0, dT, STEPU16,
                                              {64, 64}, dBuffer, ctx);
  printf("=== main src step 0 status=%d\n", (int)st);
  st = nppiDistanceTransformPBA_8u16u_C1R_Ctx(dSrc, W, 0, 0, nullptr, 0, nullptr, 0, nullptr, 0, nullptr,
                                              0, {64, 64}, dBuffer, ctx);
  printf("=== main all outputs null status=%d\n", (int)st);
  cudaFree(dSrc);
  cudaFree(dBuffer);
  cudaFree(dOut);
  cudaFree(dT);
}

}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  runCase("empty", {});
  runCase("single_center", {{32, 32}});
  runCase("single_corner", {{0, 0}});
  runCase("two_horizontal", {{20, 32}, {44, 32}});
  runCase("two_vertical", {{32, 20}, {32, 44}});
  runCase("four_square", {{20, 20}, {44, 20}, {20, 44}, {44, 44}});
  runCase("three_triangle", {{10, 10}, {50, 10}, {30, 50}});
  runCase("two_diagonal_main", {{20, 20}, {44, 44}});
  runCase("two_diagonal_anti", {{44, 20}, {20, 44}});
  runCase("cross", {{12, 32}, {52, 32}, {32, 12}, {32, 52}});
  runFloatCase("single_corner_32f", {{0, 0}}, false);
  runFloatCase("single_corner_64f", {{0, 0}}, true);
  runAbsCase("two_horizontal", {{20, 32}, {44, 32}});
  runRangeCase();
  runCase("far_corner", {{63, 63}});
  runValidation();
  return 0;
}
