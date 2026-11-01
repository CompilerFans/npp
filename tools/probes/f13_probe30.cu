// f13 probe 30: minimal accepted nSrcStep for HistogramOfGradientsBorder.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
int main() {
  NppStreamContext c{};
  c.hStream = 0;
  Npp8u *d, *scratch;
  Npp32f *desc;
  cudaMalloc(&d, 1 << 20);
  cudaMalloc(&scratch, 1 << 20);
  cudaMalloc(&desc, 1 << 20);
  for (int W : {4, 8, 12, 16, 24, 32, 40}) {
    for (int H : {8, 16, 32}) {
      NppiHOGConfig cfg{2, 4, 4, {8, 8}};
      std::vector<NppiPoint> locs = {{0, 0}};
      int first = -1;
      for (int step = 4; step <= 256; step += 4) {
        NppStatus st = nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(
            d, step, {(int)W, H}, {0, 0}, locs.data(), 1, desc, {(int)W, H}, cfg, scratch,
            NPP_BORDER_REPLICATE, c);
        if (st == 0) { first = step; break; }
      }
      printf("W%-3d H%-3d first ok step=%d\n", W, H, first);
    }
  }
  return 0;
}
