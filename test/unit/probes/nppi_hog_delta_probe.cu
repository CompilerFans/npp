// Map HOG descriptor cell indices to image cells by placing a single delta
// at known positions and dumping the nonzero descriptor entries.
#include <cstdio>
#include <cstdlib>
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

constexpr int W = 32, H = 32, cell = 8, block = 16, bins = 4, winW = 32, winH = 32;
constexpr int step = 128;

void run(int px, int py) {
  std::vector<Npp8u> img((size_t)step * H, 0);
  img[py * step + px] = 255;
  Npp8u *dSrc = nullptr;
  Npp32f *dOut = nullptr;
  Npp8u *dScratch = nullptr;
  NppiHOGConfig cfg{cell, block, bins, {winW, winH}};
  const int blocksX = (winW / block) * 2 - 1;
  const int blocksY = (winH / block) * 2 - 1;
  const int descBytes = blocksX * blocksY * (block / cell) * (block / cell) * bins * 4;
  int bufSize = 0;
  NppiPoint locs[1] = {{0, 0}};
  NppStatus sb = nppiHistogramOfGradientsBorderGetBufferSize(cfg, locs, 1, {W, H}, &bufSize);
  if (sb != NPP_NO_ERROR) {
    printf("DELTA %d %d getbuf=%d\n", px, py, (int)sb);
    return;
  }
  CK(cudaMalloc(&dSrc, (size_t)step * H));
  CK(cudaMalloc(&dOut, descBytes));
  CK(cudaMalloc(&dScratch, bufSize > 0 ? bufSize : 1));
  CK(cudaMemcpy(dSrc, img.data(), (size_t)step * H, cudaMemcpyHostToDevice));
  CK(cudaMemset(dOut, 0, descBytes));
  NppStreamContext ctx{};
  ctx.hStream = 0;
  NppStatus st = nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(
      dSrc, step, {W, H}, {0, 0}, locs, 1, dOut, {W, H}, cfg, dScratch, NPP_BORDER_REPLICATE, ctx);
  CK(cudaDeviceSynchronize());
  std::vector<Npp32f> desc(descBytes / 4);
  CK(cudaMemcpy(desc.data(), dOut, descBytes, cudaMemcpyDeviceToHost));
  printf("DELTA %d %d st=%d nz", px, py, (int)st);
  for (int i = 0; i < (int)desc.size(); ++i) {
    if (desc[i] != 0.0f) printf(" %d:%.9g", i, desc[i]);
  }
  printf("\n");
  cudaFree(dSrc);
  cudaFree(dOut);
  cudaFree(dScratch);
}

}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const int positions[] = {8, 12, 16, 20, 24};
  for (int py : positions) {
    for (int px : positions) {
      run(px, py);
    }
  }
  return 0;
}
