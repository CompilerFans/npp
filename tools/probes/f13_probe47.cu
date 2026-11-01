// f13: decode the HOG descriptor layout on NPP 13.4.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

static void run(const char *name, std::vector<Npp32f> img, int W, int H, int cell, int block,
                int bins, NppiSize win, NppiPoint loc) {
  Npp32f *ds = nullptr;
  cudaMalloc(&ds, (size_t)W * H * 4);
  cudaMemcpy(ds, img.data(), img.size() * 4, cudaMemcpyHostToDevice);
  const int nLoc = 1;
  NppiHOGConfig cfg{cell, block, bins, win};
  int descSize = 0;
  NppStatus s0 = nppiHistogramOfGradientsBorderGetDescriptorsSize(cfg, nLoc, &descSize);
  int bufSize = 0;
  NppiPoint locs[1] = {loc};
  NppStatus s1 = nppiHistogramOfGradientsBorderGetBufferSize(cfg, locs, nLoc, {W, H}, &bufSize);
  Npp32f *out = nullptr;
  cudaMalloc(&out, descSize > 0 ? descSize : 4096);
  cudaMemset(out, 0, descSize > 0 ? descSize : 4096);
  Npp8u *scratch = nullptr;
  cudaMalloc(&scratch, bufSize > 0 ? bufSize : 4096);
  NppStreamContext ctx{};
  ctx.hStream = 0;
  NppStatus st = nppiHistogramOfGradientsBorder_32f_C1R_Ctx(
      ds, W * 4, {W, H}, {0, 0}, locs, nLoc, out, {W, H}, cfg, scratch, NPP_BORDER_REPLICATE, ctx);
  cudaError_t e = cudaDeviceSynchronize();
  std::vector<Npp32f> v(descSize > 0 ? descSize / 4 : 0);
  if (!v.empty()) cudaMemcpy(v.data(), out, v.size() * 4, cudaMemcpyDeviceToHost);
  printf("%-12s st=%d cuda=%s descSize=%d bufSize=%d (s0=%d s1=%d) n=%zu:", name, (int)st,
         cudaGetErrorName(e), descSize, bufSize, (int)s0, (int)s1, v.size());
  for (size_t i = 0; i < v.size(); ++i) printf(" %.6g", v[i]);
  printf("\n");
  cudaFree(ds);
  cudaFree(out);
  cudaFree(scratch);
}

int main() {
  const int W = 16, H = 16;
  // 1) horizontal ramp: gx = 1 everywhere, gy = 0 -> angle 0
  {
    std::vector<Npp32f> img((size_t)W * H);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) img[y * W + x] = (float)x;
    run("gx=1", img, W, H, 8, 16, 4, {16, 16}, {0, 0});
  }
  // 2) vertical ramp: gy = 1 -> angle 90
  {
    std::vector<Npp32f> img((size_t)W * H);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) img[y * W + x] = (float)y;
    run("gy=1", img, W, H, 8, 16, 4, {16, 16}, {0, 0});
  }
  // 3) diagonal ramp x+y: gx=1, gy=1 -> angle 45
  {
    std::vector<Npp32f> img((size_t)W * H);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) img[y * W + x] = (float)(x + y);
    run("gx=gy=1", img, W, H, 8, 16, 4, {16, 16}, {0, 0});
  }
  // 4) cell-sized structure: only the top-left 8x8 quadrant has a horizontal ramp
  {
    std::vector<Npp32f> img((size_t)W * H, 0.0f);
    for (int y = 0; y < 8; ++y) for (int x = 0; x < 8; ++x) img[y * W + x] = (float)x;
    run("tl-ramp", img, W, H, 8, 16, 4, {16, 16}, {0, 0});
  }
  // 5) each 8x8 quadrant ramps in x independently -> identify cell order
  const char *qname[4] = {"q-tl", "q-tr", "q-bl", "q-br"};
  for (int q = 0; q < 4; ++q) {
    std::vector<Npp32f> img((size_t)W * H, 0.0f);
    int x0 = (q & 1) ? 8 : 0, y0 = (q & 2) ? 8 : 0;
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) img[(y0 + y) * W + x0 + x] = (float)x;
    run(qname[q], img, W, H, 8, 16, 4, {16, 16}, {0, 0});
  }
  // 6) angle accuracy: ramps with known angles
  {
    struct A { const char *n; float a, b; } cases[] = {
        {"ang0", 1, 0}, {"ang45", 1, 1}, {"ang26.5", 2, 1}, {"ang63.4", 1, 2},
        {"ang90", 0, 1}, {"ang135", -1, 1}, {"ang112.5", -1, 2.414f}};
    for (auto &c : cases) {
      std::vector<Npp32f> img((size_t)W * H);
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) img[y * W + x] = c.a * x + c.b * y;
      run(c.n, img, W, H, 8, 16, 4, {16, 16}, {0, 0});
    }
  }
  // 7) angle sweep: src = cos(a)*x + sin(a)*y (a in degrees)
  for (int deg = 0; deg < 180; deg += 10) {
    float a = (float)deg * 3.14159265358979f / 180.0f;
    std::vector<Npp32f> img((size_t)W * H);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) img[y * W + x] = cosf(a) * x + sinf(a) * y;
    char name[32];
    snprintf(name, sizeof name, "deg%d", deg);
    run(name, img, W, H, 8, 16, 4, {16, 16}, {0, 0});
  }
  // 8) interior window on a big ramp: no replicate-border pixels in the window
  {
    const int BW = 48, BH = 48;
    for (int deg = 0; deg < 90; deg += 15) {
      float a = (float)deg * 3.14159265358979f / 180.0f;
      std::vector<Npp32f> img((size_t)BW * BH);
      for (int y = 0; y < BH; ++y)
        for (int x = 0; x < BW; ++x) img[y * BW + x] = cosf(a) * x + sinf(a) * y;
      char name[32];
      snprintf(name, sizeof name, "big%d", deg);
      run(name, img, BW, BH, 8, 16, 4, {16, 16}, {16, 16});
    }
  }
  // 9) per-cell magnitude ratio: src = x^2/2 -> gx = 2x
  {
    const int BW = 48, BH = 48;
    std::vector<Npp32f> img((size_t)BW * BH);
    for (int y = 0; y < BH; ++y)
      for (int x = 0; x < BW; ++x) img[y * BW + x] = 0.5f * x * x;
    run("quad", img, BW, BH, 8, 16, 4, {16, 16}, {16, 16});
  }
  // 10) 9 bins: angle mapping
  {
    const int BW = 48, BH = 48;
    for (int deg = 0; deg < 180; deg += 20) {
      float a = (float)deg * 3.14159265358979f / 180.0f;
      std::vector<Npp32f> img((size_t)BW * BH);
      for (int y = 0; y < BH; ++y)
        for (int x = 0; x < BW; ++x) img[y * BW + x] = cosf(a) * x + sinf(a) * y;
      char name[32];
      snprintf(name, sizeof name, "b9_%d", deg);
      run(name, img, BW, BH, 8, 16, 9, {16, 16}, {16, 16});
    }
  }
  return 0;
}
