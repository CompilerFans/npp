// Determine whether the HOG gradient border applies at the source image or at
// the source ROI, and verify the C3 channel-selection rule.
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>
#include <nppi.h>
#include <nppi_filtering_functions.h>

#define CK(x)                                                                                                      \
  do {                                                                                                             \
    cudaError_t e_ = (x);                                                                                          \
    if (e_ != cudaSuccess) {                                                                                       \
      printf("cuda err %s %d\n", cudaGetErrorString(e_), __LINE__);                                                \
      return;                                                                                                      \
    }                                                                                                              \
  } while (0)

namespace {
constexpr int W = 32, H = 32, cell = 4, block = 8, bins = 4, winW = 8, winH = 8;
constexpr int step = 128;

void runC1(const char *name, int locx, int locy, int roiW, int roiH) {
  std::vector<Npp8u> img((size_t)step * H, 0);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) img[y * step + x] = (Npp8u)(x * 8);
  Npp8u *dSrc = nullptr;
  Npp32f *dOut = nullptr;
  Npp8u *dScratch = nullptr;
  NppiHOGConfig cfg{cell, block, bins, {winW, winH}};
  const int bxN = (winW / block) * 2 - 1, byN = (winH / block) * 2 - 1;
  const int descBytes = bxN * byN * (block / cell) * (block / cell) * bins * 4;
  NppiPoint locs[1] = {{locx, locy}};
  int bufSize = 0;
  NppStatus sb = nppiHistogramOfGradientsBorderGetBufferSize(cfg, locs, 1, {roiW, roiH}, &bufSize);
  printf("=== %s loc=(%d,%d) roi=%dx%d getbuf=%d buf=%d\n", name, locx, locy, roiW, roiH, (int)sb, bufSize);
  if (sb != NPP_NO_ERROR) return;
  CK(cudaMalloc(&dSrc, (size_t)step * H));
  CK(cudaMalloc(&dOut, descBytes));
  CK(cudaMalloc(&dScratch, bufSize > 0 ? bufSize : 1));
  CK(cudaMemcpy(dSrc, img.data(), (size_t)step * H, cudaMemcpyHostToDevice));
  CK(cudaMemset(dOut, 0, descBytes));
  NppStreamContext ctx{};
  ctx.hStream = 0;
  NppStatus st = nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(
      dSrc, step, {W, H}, {0, 0}, locs, 1, dOut, {roiW, roiH}, cfg, dScratch, NPP_BORDER_REPLICATE, ctx);
  CK(cudaDeviceSynchronize());
  std::vector<Npp32f> desc(descBytes / 4);
  CK(cudaMemcpy(desc.data(), dOut, descBytes, cudaMemcpyDeviceToHost));
  printf("st=%d desc", (int)st);
  for (auto v : desc) printf(" %.9g", v);
  printf("\n");
  cudaFree(dSrc);
  cudaFree(dOut);
  cudaFree(dScratch);
}

void runC3(const char *name, int sx, int sy) {
  std::vector<Npp8u> img((size_t)step * H, 0);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      // channel 0: horizontal ramp; channel 1: vertical ramp; channel 2: flat
      img[y * step + x * 3 + 0] = (Npp8u)(x * sx);
      img[y * step + x * 3 + 1] = (Npp8u)(y * sy);
      img[y * step + x * 3 + 2] = 0;
    }
  }
  const int step3 = step;  // bytes
  Npp8u *dSrc = nullptr;
  Npp32f *dOut = nullptr;
  Npp8u *dScratch = nullptr;
  NppiHOGConfig cfg{cell, block, bins, {winW, winH}};
  const int bxN = (winW / block) * 2 - 1, byN = (winH / block) * 2 - 1;
  const int descBytes = bxN * byN * (block / cell) * (block / cell) * bins * 4;
  NppiPoint locs[1] = {{0, 0}};
  int bufSize = 0;
  NppStatus sb = nppiHistogramOfGradientsBorderGetBufferSize(cfg, locs, 1, {W, H}, &bufSize);
  printf("=== %s getbuf=%d buf=%d\n", name, (int)sb, bufSize);
  if (sb != NPP_NO_ERROR) return;
  CK(cudaMalloc(&dSrc, (size_t)step3 * H));
  CK(cudaMalloc(&dOut, descBytes));
  CK(cudaMalloc(&dScratch, bufSize > 0 ? bufSize : 1));
  CK(cudaMemcpy(dSrc, img.data(), (size_t)step3 * H, cudaMemcpyHostToDevice));
  CK(cudaMemset(dOut, 0, descBytes));
  NppStreamContext ctx{};
  ctx.hStream = 0;
  NppStatus st = nppiHistogramOfGradientsBorder_8u32f_C3R_Ctx(
      dSrc, step3, {W, H}, {0, 0}, locs, 1, dOut, {W, H}, cfg, dScratch, NPP_BORDER_REPLICATE, ctx);
  CK(cudaDeviceSynchronize());
  std::vector<Npp32f> desc(descBytes / 4);
  CK(cudaMemcpy(desc.data(), dOut, descBytes, cudaMemcpyDeviceToHost));
  printf("st=%d desc", (int)st);
  for (auto v : desc) printf(" %.9g", v);
  printf("\n");
  cudaFree(dSrc);
  cudaFree(dOut);
  cudaFree(dScratch);
}
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  runC1("roi_left", 0, 0, 16, 16);
  runC1("roi_right_edge", 8, 0, 16, 16);
  runC1("image_right_edge", 24, 0, 32, 32);
  runC3("c3_x8_y2", 8, 2);
  runC3("c3_x2_y8", 2, 8);
  runC3("c3_x8_y8", 8, 8);
  runC3("c3_x5_y0", 5, 0);
  runC3("c3_x0_y0", 0, 0);
  return 0;
}
