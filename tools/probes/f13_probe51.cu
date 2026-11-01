// f13: HOG GetBufferSize / -16 threshold investigation.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv) {
  int W = argc > 1 ? atoi(argv[1]) : 64;
  int step = argc > 2 ? atoi(argv[2]) : ((W + 127) / 128) * 128;
  int nLoc = argc > 3 ? atoi(argv[3]) : 2;
  NppiHOGConfig cfg{8, 16, 9, {32, 32}};
  std::vector<NppiPoint> locs;
  for (int i = 0; i < nLoc; ++i) locs.push_back({i * 32, i * 32});
  int bsz = -1;
  NppStatus sb = nppiHistogramOfGradientsBorderGetBufferSize(cfg, locs.data(), nLoc, {W, W}, &bsz);
  Npp8u *ds = nullptr;
  cudaMalloc(&ds, (size_t)step * W);
  cudaMemset(ds, 7, (size_t)step * W);
  int cpb = 2, bx = 3, by = 3;
  int dsz = nLoc * bx * by * cpb * cpb * 9 * 4;
  Npp32f *out = nullptr;
  cudaMalloc(&out, dsz);
  Npp8u *scratch = nullptr;
  cudaMalloc(&scratch, bsz > 0 ? bsz : 4096);
  NppStreamContext ctx{};
  ctx.hStream = 0;
  NppStatus st = nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(
      ds, step, {W, W}, {0, 0}, locs.data(), nLoc, out, {W, W}, cfg, scratch, NPP_BORDER_REPLICATE,
      ctx);
  cudaError_t e = cudaDeviceSynchronize();
  printf("W=%d step=%d nLoc=%d bufst=%d bsz=%d st=%d cuda=%s\n", W, step, nLoc, (int)sb, bsz,
         (int)st, cudaGetErrorName(e));
  return 0;
}
