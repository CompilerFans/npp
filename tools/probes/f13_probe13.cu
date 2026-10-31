// f13 probe 13: nppiHistogramOfGradientsBorder semantics, part 1.
// - GetDescriptorsSize / GetBufferSize values across configs
// - validation statuses of the size helpers
// - one happy-path 8u32f_C1R descriptor dump (delta-pixel source)
#include <npp.h>
#include <nppcore.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const char *es(NppStatus s) { static char b[32]; snprintf(b, 32, "%d", (int)s); return b; }

static void try_desc(const char *tag, int cell, int block, int bins, int ww, int wh, int nLoc) {
  NppiHOGConfig cfg{};
  cfg.cellSize = cell; cfg.histogramBlockSize = block; cfg.nHistogramBins = bins;
  cfg.detectionWindowSize.width = ww; cfg.detectionWindowSize.height = wh;
  int sz = -1;
  NppStatus st = nppiHistogramOfGradientsBorderGetDescriptorsSize(cfg, nLoc, &sz);
  printf("%-34s st=%-34s size=%d floats=%d\n", tag, es(st), sz, sz / 4);
}

static void try_buf(const char *tag, int cell, int block, int bins, int ww, int wh,
                    std::vector<NppiPoint> locs, NppiSize roi) {
  NppiHOGConfig cfg{};
  cfg.cellSize = cell; cfg.histogramBlockSize = block; cfg.nHistogramBins = bins;
  cfg.detectionWindowSize.width = ww; cfg.detectionWindowSize.height = wh;
  int sz = -1;
  NppStatus st = nppiHistogramOfGradientsBorderGetBufferSize(cfg, locs.data(), (int)locs.size(), roi, &sz);
  printf("%-34s st=%-34s size=%d\n", tag, es(st), sz);
}

int main(int argc, char **argv) {
  int part = argc > 1 ? atoi(argv[1]) : 0;

  if (part == 1) {
    // descriptor size across configs (nLocations = 1)
    try_desc("c2 b4 bins4 win8x8", 2, 4, 4, 8, 8, 1);
    try_desc("c2 b4 bins4 win8x8 x2", 2, 4, 4, 8, 8, 2);
    try_desc("c2 b4 bins9 win8x8", 2, 4, 9, 8, 8, 1);
    try_desc("c4 b8 bins9 win16x16", 4, 8, 9, 16, 16, 1);
    try_desc("c1 b1 bins4 win8x8", 1, 1, 4, 8, 8, 1);
    try_desc("c2 b2 bins4 win8x8", 2, 2, 4, 8, 8, 1);
    try_desc("c2 b8 bins4 win8x8", 2, 8, 4, 8, 8, 1);
    try_desc("c2 b4 bins4 win12x8", 2, 4, 4, 12, 8, 1);
    try_desc("c2 b4 bins4 win8x12", 2, 4, 4, 8, 12, 1);
    try_desc("c3 b4 bins4 win8x8(badcell?)", 3, 4, 4, 8, 8, 1);
    try_desc("c2 b3 bins4 win8x8(badblock?)", 2, 3, 4, 8, 8, 1);
    try_desc("c2 b4 bins4 win7x8(badwin?)", 2, 4, 4, 7, 8, 1);
    try_desc("c2 b4 bins18 win8x8(toomanybins)", 2, 4, 18, 8, 8, 1);
    try_desc("c2 b4 bins1 win8x8", 2, 4, 1, 8, 8, 1);
    try_desc("c2 b4 bins0 win8x8", 2, 4, 0, 8, 8, 1);
    try_desc("c0 b4 bins4 win8x8", 0, 4, 4, 8, 8, 1);
    try_desc("c2 b0 bins4 win8x8", 2, 0, 4, 8, 8, 1);
    try_desc("c2 b4 bins4 win0x8", 2, 4, 4, 0, 8, 1);
    try_desc("c17 b4 bins4 win8x8(bigcell)", 17, 4, 4, 8, 8, 1);
    try_desc("c8 b65 bins4 win8x8(bigblock)", 8, 65, 4, 8, 8, 1);
    try_desc("c2 b4 bins4 win8x8 nloc0", 2, 4, 4, 8, 8, 0);
    try_desc("c2 b4 bins4 win128x128 manycell", 2, 4, 4, 128, 128, 1);
    // null hpBufferSize
    {
      NppiHOGConfig cfg{2, 4, 4, {8, 8}};
      NppStatus st = nppiHistogramOfGradientsBorderGetDescriptorsSize(cfg, 1, nullptr);
      printf("%-34s st=%s\n", "desc null hpDescriptorsSize", es(st));
    }
  }

  if (part == 2) {
    std::vector<NppiPoint> l1 = {{0, 0}};
    try_buf("c2 b4 bins4 win8x8 roi8x8 loc1", 2, 4, 4, 8, 8, l1, NppiSize{8, 8});
    std::vector<NppiPoint> l2 = {{0, 0}, {4, 4}};
    try_buf("c2 b4 bins4 win8x8 roi16x16 loc2", 2, 4, 4, 8, 8, l2, NppiSize{16, 16});
    try_buf("c2 b4 bins4 win8x8 roi4x4(small)", 2, 4, 4, 8, 8, l1, NppiSize{4, 4});
    try_buf("c2 b4 bins4 win8x8 roi0x0", 2, 4, 4, 8, 8, l1, NppiSize{0, 0});
    try_buf("c2 b4 bins4 win8x8 locneg", 2, 4, 4, 8, 8, std::vector<NppiPoint>{{-1, 0}}, NppiSize{8, 8});
    try_buf("c2 b4 bins4 win8x8 locoob", 2, 4, 4, 8, 8, std::vector<NppiPoint>{{1, 0}}, NppiSize{8, 8});
    try_buf("c2 b4 bins4 win8x8 locoob2", 2, 4, 4, 8, 8, std::vector<NppiPoint>{{8, 8}}, NppiSize{8, 8});
    try_buf("c2 b4 bins4 win8x8 nullloc nloc1", 2, 4, 4, 8, 8, std::vector<NppiPoint>{}, NppiSize{8, 8});
    {
      NppiHOGConfig cfg{2, 4, 4, {8, 8}};
      int sz = -1;
      NppStatus st = nppiHistogramOfGradientsBorderGetBufferSize(cfg, l1.data(), 1, NppiSize{8, 8}, nullptr);
      printf("%-34s st=%s\n", "buf null hpBufferSize", es(st));
    }
  }

  if (part == 3) {
    // happy path: 16x16 image, one delta pixel, window 8x8 at (0,0),
    // cell 2 block 4 bins 4 -> dump descriptor floats.
    const int W = 16, H = 16, STEP = 64;
    std::vector<Npp8u> h((size_t)STEP * H, 0);
    // delta at the center of the window: (3,3)
    h[3 * STEP + 3] = 100;
    Npp8u *d; cudaMalloc(&d, STEP * H); cudaMemcpy(d, h.data(), STEP * H, cudaMemcpyHostToDevice);
    NppiHOGConfig cfg{2, 4, 4, {8, 8}};
    int bsz = -1, dsz = -1;
    std::vector<NppiPoint> locs = {{0, 0}};
    nppiHistogramOfGradientsBorderGetBufferSize(cfg, locs.data(), 1, NppiSize{16, 16}, &bsz);
    nppiHistogramOfGradientsBorderGetDescriptorsSize(cfg, 1, &dsz);
    printf("bufsize=%d descsize=%d\n", bsz, dsz);
    Npp8u *scratch; cudaMalloc(&scratch, bsz);
    Npp32f *desc; cudaMalloc(&desc, dsz);
    cudaMemset(desc, 0xCC, dsz);
    NppStreamContext c{}; c.hStream = 0;
    NppStatus st = nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(
        d, 64, NppiSize{W, H}, NppiPoint{0, 0}, locs.data(), 1, desc, NppiSize{16, 16}, cfg,
        scratch, NPP_BORDER_REPLICATE, c);
    printf("st=%s\n", es(st));
    cudaError_t e = cudaDeviceSynchronize();
    printf("sync=%s\n", cudaGetErrorString(e));
    std::vector<Npp32f> out(dsz / 4);
    cudaMemcpy(out.data(), desc, dsz, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < out.size(); ++i)
      if (out[i] != 0.0f || i < 4) printf("desc[%zu] = %.9g\n", i, out[i]);
    // block/cell/bin grid view (3x3 blocks of 2x2 cells x 4 bins)
    int bi = 0;
    for (int by = 0; by < 3; ++by)
      for (int bx = 0; bx < 3; ++bx, ++bi) {
        printf("block(%d,%d):\n", bx, by);
        for (int cy = 0; cy < 2; ++cy) {
          printf("  cellrow %d:", cy);
          for (int cx = 0; cx < 2; ++cx) {
            size_t base = (size_t)bi * 16 + (cy * 2 + cx) * 4;
            printf("  [");
            for (int b = 0; b < 4; ++b) printf(" %.6g", out[base + b]);
            printf(" ]");
          }
          printf("\n");
        }
      }
  }
  return 0;
}
