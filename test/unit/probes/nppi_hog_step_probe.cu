// Probe the HOG source-step validation threshold.
#include <cstdio>

#include <cuda_runtime.h>
#include <nppi.h>
#include <nppi_filtering_functions.h>

extern "C" {
NppStatus nppGetStreamContext(NppStreamContext *c);
}

int main() {
  const int W = 16, H = 16;
  NppiHOGConfig cfg{8, 16, 9, {16, 16}};
  NppiPoint locs[3] = {{0, 0}, {0, 0}, {0, 0}};
  const int steps[] = {32, 64, 96, 127, 128, 129, 255, 256};
  for (int n = 1; n <= 3; ++n) {
    for (int step : steps) {
      unsigned char *ds = nullptr;
      unsigned char *sc = nullptr;
      float *dd = nullptr;
      cudaMalloc(&ds, step * H);
      cudaMemset(ds, 0, step * H);
      int bs = 0;
      NppStatus gs = nppiHistogramOfGradientsBorderGetBufferSize(cfg, locs, n, {W, H}, &bs);
      cudaMalloc(&sc, bs > 0 ? bs : 1);
      cudaMalloc(&dd, 324 * 4 * n);
      NppStreamContext ctx{};
      nppGetStreamContext(&ctx);
      NppStatus st = nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(
          ds, step, {W, H}, {0, 0}, locs, n, dd, {W, H}, cfg, sc, NPP_BORDER_REPLICATE, ctx);
      printf("nLoc=%d step=%d getbuf=%d status=%d\n", n, step, (int)gs, (int)st);
      cudaFree(ds);
      cudaFree(sc);
      cudaFree(dd);
    }
  }
  return 0;
}
