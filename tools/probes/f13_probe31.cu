// f13 probe 31: HOG descriptor layout with small windows.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const char *es(NppStatus s) { static char b[32]; snprintf(b, 32, "%d", (int)s); return b; }

static void run(const char *tag, int W, int H, const std::vector<Npp8u> &h, NppiHOGConfig cfg,
                NppiPoint loc) {
  Npp8u *d, *scratch;
  Npp32f *desc;
  cudaMalloc(&d, 1 << 20);
  cudaMalloc(&scratch, 1 << 20);
  cudaMalloc(&desc, 1 << 20);
  cudaMemset(d, 0, 1 << 20);
  cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice);
  cudaMemset(desc, 0, 1 << 20);
  std::vector<NppiPoint> locs = {loc};
  int bsz = -1, dsz = -1;
  nppiHistogramOfGradientsBorderGetBufferSize(cfg, locs.data(), 1, NppiSize{W, H}, &bsz);
  nppiHistogramOfGradientsBorderGetDescriptorsSize(cfg, 1, &dsz);
  NppStreamContext c{};
  c.hStream = 0;
  NppStatus st = nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(
      d, 32, NppiSize{W, H}, NppiPoint{0, 0}, locs.data(), 1, desc, NppiSize{W, H}, cfg, scratch,
      NPP_BORDER_REPLICATE, c);
  printf("%-22s st=%s descsz=%d:", tag, es(st), dsz);
  if (st == 0) {
    std::vector<Npp32f> v(dsz / 4);
    cudaMemcpy(v.data(), desc, dsz, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < v.size(); ++i) printf(" %.5g", v[i]);
  }
  printf("\n");
  cudaFree(d);
  cudaFree(scratch);
  cudaFree(desc);
}

int main() {
  const int W = 16, H = 16;
  std::vector<Npp8u> ramp((size_t)W * H), oneCell((size_t)W * H, 0), random((size_t)W * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) ramp[y * W + x] = (Npp8u)(x * 8);
  // a step only inside rows 4..5 / columns 6..9
  for (int y = 4; y <= 5; ++y)
    for (int x = 8; x < 12; ++x) oneCell[y * W + x] = 100;
  unsigned s = 7;
  for (auto &v : random) {
    s = s * 1103515245u + 12345u;
    v = (Npp8u)((s >> 16) % 256);
  }
  NppiHOGConfig cfg4{2, 4, 4, {4, 4}};
  NppiHOGConfig cfg84{2, 4, 4, {8, 4}};
  NppiHOGConfig cfg48{2, 4, 4, {4, 8}};
  run("ramp win4", W, H, ramp, cfg4, {0, 0});
  run("ramp win4 at(4,4)", W, H, ramp, cfg4, {4, 4});
  run("ramp win4 at(6,10)", W, H, ramp, cfg4, {6, 10});
  run("onecell win4 at(4,4)", W, H, oneCell, cfg4, {4, 4});
  run("onecell win4 at(6,4)", W, H, oneCell, cfg4, {6, 4});
  run("random win4", W, H, random, cfg4, {0, 0});
  run("ramp win8x4", W, H, ramp, cfg84, {0, 0});
  run("ramp win4x8", W, H, ramp, cfg48, {0, 0});
  return 0;
}
