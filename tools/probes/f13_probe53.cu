// f13: HOG config validity matrix via GetDescriptorsSize.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

int main() {
  struct C { int cell, block, bins, w, h; const char *n; };
  C cs[] = {
      {8, 16, 4, 16, 16, "8/16/4 16x16"},
      {8, 16, 4, 32, 32, "8/16/4 32x32"},
      {8, 16, 4, 24, 32, "8/16/4 24x32"},
      {2, 8, 4, 8, 8, "2/8/4 8x8"},
      {2, 8, 4, 16, 16, "2/8/4 16x16"},
      {2, 4, 4, 8, 8, "2/4/4 8x8"},
      {4, 8, 4, 8, 8, "4/8/4 8x8"},
      {4, 8, 4, 16, 16, "4/8/4 16x16"},
      {4, 16, 4, 16, 16, "4/16/4 16x16"},
      {8, 32, 4, 32, 32, "8/32/4 32x32"},
      {8, 32, 9, 64, 64, "8/32/9 64x64"},
      {8, 16, 9, 64, 128, "8/16/9 64x128"},
      {8, 16, 9, 64, 128, "x"},
      {16, 64, 16, 128, 128, "16/64/16 128x128"},
  };
  for (auto &c : cs) {
    NppiHOGConfig cfg{c.cell, c.block, c.bins, {c.w, c.h}};
    int sz = -1;
    NppStatus st = nppiHistogramOfGradientsBorderGetDescriptorsSize(cfg, 1, &sz);
    printf("%-22s st=%d size=%d\n", c.n, (int)st, sz);
  }
  return 0;
}
