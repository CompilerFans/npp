// Dump NPP HOG descriptors for controlled 8u C1 images so the gradient,
// binning, normalization and ordering model can be fitted.
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

int cell = 8, block = 16, bins = 4, winW = 32, winH = 32, W = 32, H = 32;

void run(const char *name, const std::vector<Npp8u> &img) {
  const int step = ((W + 127) / 128) * 128;
  Npp8u *dSrc = nullptr;
  Npp32f *dOut = nullptr;
  Npp8u *dScratch = nullptr;
  NppiHOGConfig cfg{cell, block, bins, {winW, winH}};
  const int blocksX = (winW / block) * 2 - 1;
  const int blocksY = (winH / block) * 2 - 1;
  const int cellsPerBlock = (block / cell) * (block / cell);
  const int descBytes = blocksX * blocksY * cellsPerBlock * bins * 4;
  int bufSize = 0;
  NppiPoint locs[1] = {{0, 0}};
  NppStatus sb = nppiHistogramOfGradientsBorderGetBufferSize(cfg, locs, 1, {W, H}, &bufSize);
  printf("=== %s cfg=%d/%d/%d win=%dx%d img=%dx%d descBytes=%d getbuf=%d buf=%d\n", name, cell,
         block, bins, winW, winH, W, H, descBytes, (int)sb, bufSize);
  if (sb != NPP_NO_ERROR) return;
  CK(cudaMalloc(&dSrc, (size_t)step * H));
  CK(cudaMemset(dSrc, 0, (size_t)step * H));
  CK(cudaMalloc(&dOut, descBytes));
  CK(cudaMalloc(&dScratch, bufSize > 0 ? bufSize : 1));
  CK(cudaMemcpy2D(dSrc, step, img.data(), W, W, H, cudaMemcpyHostToDevice));
  CK(cudaMemset(dOut, 0, descBytes));
  NppStreamContext ctx{};
  ctx.hStream = 0;
  NppStatus st = nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(
      dSrc, step, {W, H}, {0, 0}, locs, 1, dOut, {W, H}, cfg, dScratch, NPP_BORDER_REPLICATE, ctx);
  CK(cudaDeviceSynchronize());
  std::vector<Npp32f> desc(descBytes / 4);
  CK(cudaMemcpy(desc.data(), dOut, descBytes, cudaMemcpyDeviceToHost));
  printf("st=%d\nimg", (int)st);
  for (auto v : img) printf(" %u", (unsigned)v);
  printf("\ndesc");
  for (auto v : desc) printf(" %.9g", v);
  printf("\n");
  cudaFree(dSrc);
  cudaFree(dOut);
  cudaFree(dScratch);
}

}  // namespace

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc > 7) {
    cell = atoi(argv[1]);
    block = atoi(argv[2]);
    bins = atoi(argv[3]);
    winW = atoi(argv[4]);
    winH = atoi(argv[5]);
    W = atoi(argv[6]);
    H = atoi(argv[7]);
  }

  std::vector<Npp8u> rampX(W * H), rampY(W * H), rampDiag(W * H), delta(W * H, 0), flat(W * H, 128),
      pattern(W * H);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      rampX[y * W + x] = (Npp8u)(x * 8);
      rampY[y * W + x] = (Npp8u)(y * 8);
      rampDiag[y * W + x] = (Npp8u)((x + y) * 4);
      pattern[y * W + x] = (Npp8u)(((x * 7 + y * 13) * 3) & 0xFF);
    }
  }
  delta[(H / 2) * W + W / 2] = 255;

  run("rampX", rampX);
  run("rampY", rampY);
  run("rampDiag", rampDiag);
  run("delta", delta);
  run("flat", flat);
  run("pattern", pattern);
  return 0;
}
