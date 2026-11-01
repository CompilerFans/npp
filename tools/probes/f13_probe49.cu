// f13: HOG differential replay (build once against NPP, once against mcMPP).
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int cell = 8, block = 16, bins = 9, winW = 32, winH = 32;

int main(int argc, char **argv) {
  int W = 64, H = 64;
  if (argc > 8) {
    W = atoi(argv[8]);
    H = W;
  }
  if (argc > 1) cell = atoi(argv[1]);
  if (argc > 2) block = atoi(argv[2]);
  if (argc > 3) bins = atoi(argv[3]);
  if (argc > 4) winW = atoi(argv[4]);
  if (argc > 5) winH = atoi(argv[5]);
  int nLoc = argc > 6 ? atoi(argv[6]) : 3;

  std::vector<Npp8u> img((size_t)W * H);
  unsigned s = 12345;
  for (auto &v : img) {
    s = s * 1103515245u + 12345u;
    v = (Npp8u)((s >> 16) & 0xFF);
  }
  const int step = ((W + 127) / 128) * 128 * (nLoc > 0 ? nLoc : 1);  // NPP needs 128*nLoc
  Npp8u *ds = nullptr;
  cudaMalloc(&ds, (size_t)step * H);
  cudaMemcpy2D(ds, step, img.data(), W, W, H, cudaMemcpyHostToDevice);
  NppiHOGConfig cfg{cell, block, bins, {winW, winH}};
  std::vector<NppiPoint> locs;
  for (int i = 0; i < nLoc; ++i) locs.push_back({i * 32, i * 32});
#ifdef USE_HOG_HELPERS
  int dsz = 0;
  nppiHistogramOfGradientsBorderGetDescriptorsSize(cfg, (int)locs.size(), &dsz);
#else
  // mcMPP does not implement the size helpers yet: compute them locally.
  int cpb = block / cell;
  int bx = (winW / block) * 2 - 1, by = (winH / block) * 2 - 1;
  int dsz = (int)locs.size() * bx * by * cpb * cpb * bins * 4;
#endif
  Npp32f *out = nullptr;
  cudaMalloc(&out, dsz > 0 ? dsz : 1024);
  cudaMemset(out, 0, dsz > 0 ? dsz : 1024);
  Npp8u *scratch = nullptr;
  int bsz = 1 << 20;
#ifdef USE_HOG_HELPERS
  nppiHistogramOfGradientsBorderGetBufferSize(cfg, locs.data(), (int)locs.size(), {W, H}, &bsz);
#endif
  if (argc > 7) bsz *= atoi(argv[7]);
  cudaMalloc(&scratch, bsz > 0 ? bsz : 1024);
  printf("bufsize=%d\n", bsz);
  NppStreamContext ctx{};
  ctx.hStream = 0;
  NppStatus st = nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(
      ds, step, {W, H}, {0, 0}, locs.data(), (int)locs.size(), out, {W, H}, cfg, scratch,
      NPP_BORDER_REPLICATE, ctx);
  cudaError_t e = cudaDeviceSynchronize();
  std::vector<Npp32f> v(dsz > 0 ? dsz / 4 : 0);
  if (!v.empty()) cudaMemcpy(v.data(), out, v.size() * 4, cudaMemcpyDeviceToHost);
  printf("st=%d cuda=%s dsz=%d n=%zu\n", (int)st, cudaGetErrorName(e), dsz, v.size());
  for (size_t i = 0; i < v.size(); ++i) printf("%zu %.6g\n", i, v[i]);
  return 0;
}
