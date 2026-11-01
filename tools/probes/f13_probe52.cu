// f13: small HOG descriptor dump (image + descriptor) for model fitting.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv) {
  int cell = argc > 1 ? atoi(argv[1]) : 8;
  int block = argc > 2 ? atoi(argv[2]) : 16;
  int bins = argc > 3 ? atoi(argv[3]) : 4;
  int winW = argc > 4 ? atoi(argv[4]) : 16;
  int winH = argc > 5 ? atoi(argv[5]) : 16;
  int W = argc > 6 ? atoi(argv[6]) : winW;
  int H = argc > 7 ? atoi(argv[7]) : winH;
  int locx = argc > 8 ? atoi(argv[8]) : 0;
  int locy = argc > 9 ? atoi(argv[9]) : 0;
  const int step = 128;

  std::vector<Npp8u> img((size_t)W * H);
  unsigned s = 999;
  for (auto &v : img) {
    s = s * 1103515245u + 12345u;
    v = (Npp8u)((s >> 16) & 0xFF);
  }
  Npp8u *ds = nullptr;
  cudaMalloc(&ds, (size_t)step * H);
  cudaMemcpy2D(ds, step, img.data(), W, W, H, cudaMemcpyHostToDevice);
  NppiHOGConfig cfg{cell, block, bins, {winW, winH}};
  int cpb = block / cell, bx = (winW / block) * 2 - 1, by = (winH / block) * 2 - 1;
  int dsz = bx * by * cpb * cpb * bins * 4;
  Npp32f *out = nullptr;
  cudaMalloc(&out, dsz);
  cudaMemset(out, 0, dsz);
  Npp8u *scratch = nullptr;
  cudaMalloc(&scratch, 1 << 20);
  NppiPoint locs[1] = {{locx, locy}};
  NppStreamContext ctx{};
  ctx.hStream = 0;
  NppStatus st = nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(
      ds, step, {W, H}, {0, 0}, locs, 1, out, {W, H}, cfg, scratch, NPP_BORDER_REPLICATE, ctx);
  cudaError_t e = cudaDeviceSynchronize();
  std::vector<Npp32f> v(dsz / 4);
  cudaMemcpy(v.data(), out, dsz, cudaMemcpyDeviceToHost);
  printf("st=%d cuda=%s dsz=%d\n", (int)st, cudaGetErrorName(e), dsz);
  printf("img");
  for (auto x : img) printf(" %u", (unsigned)x);
  printf("\n");
  printf("desc");
  for (auto x : v) printf(" %.9g", x);
  printf("\n");
  return 0;
}
