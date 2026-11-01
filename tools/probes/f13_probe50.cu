// f13: HOG main-function status matrix.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv) {
  int W = argc > 1 ? atoi(argv[1]) : 64;
  int H = argc > 2 ? atoi(argv[2]) : 64;
  int step = argc > 3 ? atoi(argv[3]) : W;
  int roiW = argc > 4 ? atoi(argv[4]) : W;
  int roiH = argc > 5 ? atoi(argv[5]) : H;
  int nLoc = argc > 6 ? atoi(argv[6]) : 1;
  int cell = argc > 7 ? atoi(argv[7]) : 8;
  int block = argc > 8 ? atoi(argv[8]) : 16;
  int bins = argc > 9 ? atoi(argv[9]) : 9;
  int winW = argc > 10 ? atoi(argv[10]) : 32;
  int winH = argc > 11 ? atoi(argv[11]) : 32;
  int border = argc > 12 ? atoi(argv[12]) : (int)NPP_BORDER_REPLICATE;

  std::vector<Npp8u> img((size_t)W * H);
  unsigned s = 12345;
  for (auto &v : img) {
    s = s * 1103515245u + 12345u;
    v = (Npp8u)((s >> 16) & 0xFF);
  }
  Npp8u *ds = nullptr;
  cudaMalloc(&ds, (size_t)step * H);
  cudaMemcpy2D(ds, step, img.data(), W, W, H, cudaMemcpyHostToDevice);
  NppiHOGConfig cfg{cell, block, bins, {winW, winH}};
  std::vector<NppiPoint> locs((size_t)(nLoc > 0 ? nLoc : 1), {0, 0});
  int dsz = 4096;
  if (nLoc > 0) nppiHistogramOfGradientsBorderGetDescriptorsSize(cfg, nLoc, &dsz);
  Npp32f *out = nullptr;
  cudaMalloc(&out, dsz > 0 ? dsz : 4096);
  cudaMemset(out, 0, dsz > 0 ? dsz : 4096);
  int bsz = 1 << 21;
  nppiHistogramOfGradientsBorderGetBufferSize(cfg, locs.data(), nLoc, {roiW, roiH}, &bsz);
  Npp8u *scratch = nullptr;
  cudaMalloc(&scratch, (size_t)(bsz > 0 ? bsz : 4096));
  NppStreamContext ctx{};
  ctx.hStream = 0;
  int variant = argc > 13 ? atoi(argv[13]) : 0;
  Npp8u *ps = ds;
  Npp32f *po = out;
  Npp8u *pb = scratch;
  NppiPoint *pl = locs.data();
  NppiPoint off = {0, 0};
  switch (variant) {
    case 1: ps = nullptr; break;
    case 2: po = nullptr; break;
    case 3: pb = nullptr; break;
    case 4: pl = nullptr; break;
    case 5: locs[0] = {roiW - 8, 0}; break;
    case 6: locs[0] = {-1, 0}; break;
    case 7: off = {2, 3}; break;
    case 8: cfg.cellSize = 3; break;
    case 9: cfg.detectionWindowSize = {48, 32}; break;
    case 10: cfg.nHistogramBins = 1; break;
    default: break;
  }
  NppStatus st = nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(
      ps, step, {W, H}, off, pl, nLoc, po, {roiW, roiH}, cfg, pb,
      (NppiBorderType)border, ctx);
  cudaError_t e = cudaDeviceSynchronize();
  printf("W=%d H=%d step=%d roi=%dx%d nLoc=%d v=%d st=%d cuda=%s dsz=%d bsz=%d\n", W, H, step, roiW,
         roiH, nLoc, variant, (int)st, cudaGetErrorName(e), dsz, bsz);
  return 0;
}
