// f13 probe 29: HistogramOfGradientsBorder descriptor layout and normalisation.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const char *es(NppStatus s) { static char b[32]; snprintf(b, 32, "%d", (int)s); return b; }

int main(int argc, char **argv) {
  int mode = argc > 1 ? atoi(argv[1]) : 0;
  const int W = 16, H = 16, STEP = argc > 2 ? atoi(argv[2]) : 32;
  std::vector<Npp8u> h((size_t)STEP * H, 0);
  if (mode == 0) {
    // ramp in x (gradient constant, angle 0)
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) h[y * STEP + x] = (Npp8u)(x * 8);
  } else if (mode == 1) {
    // column impulse: gradients at x=5 and x=7
    for (int y = 0; y < H; ++y) h[y * STEP + 6] = 100;
  } else {
    // random
    unsigned s = 12345;
    for (int i = 0; i < STEP * H; ++i) {
      s = s * 1103515245u + 12345u;
      h[i] = (Npp8u)((s >> 16) % 256);
    }
  }
  printf("src");
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) printf(" %d", (int)h[y * STEP + x]);
  printf("\n");
  Npp8u *d, *scratch;
  Npp32f *desc;
  cudaMalloc(&d, STEP * H);
  cudaMalloc(&scratch, 1 << 20);
  cudaMalloc(&desc, 1 << 16);
  cudaMemcpy(d, h.data(), (size_t)STEP * H, cudaMemcpyHostToDevice);
  cudaMemset(scratch, 0, 1 << 20);
  cudaMemset(desc, 0, 1 << 16);
  NppiHOGConfig cfg{2, 4, 4, {8, 8}};
  std::vector<NppiPoint> locs = {{0, 0}};
  int bsz = -1, dsz = -1;
  nppiHistogramOfGradientsBorderGetBufferSize(cfg, locs.data(), 1, NppiSize{W, H}, &bsz);
  nppiHistogramOfGradientsBorderGetDescriptorsSize(cfg, 1, &dsz);
  NppStreamContext c{};
  c.hStream = 0;
  NppStatus st = nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(
      d, STEP, NppiSize{W, H}, NppiPoint{0, 0}, locs.data(), 1, desc, NppiSize{W, H}, cfg, scratch,
      NPP_BORDER_REPLICATE, c);
  printf("st=%s bufsize=%d descsize=%d\n", es(st), bsz, dsz);
  std::vector<Npp32f> out(dsz / 4);
  cudaMemcpy(out.data(), desc, dsz, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < out.size(); ++i) printf("%zu %.9g\n", i, out[i]);
  // null scratch
  NppStatus st2 = nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(
      d, STEP, NppiSize{W, H}, NppiPoint{0, 0}, locs.data(), 1, desc, NppiSize{W, H}, cfg, nullptr,
      NPP_BORDER_REPLICATE, c);
  printf("nullscratch st=%s\n", es(st2));
  // 2 locations
  std::vector<NppiPoint> l2 = {{0, 0}, {4, 4}};
  cudaMemset(desc, 0, 1 << 16);
  NppStatus st3 = nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(
      d, STEP, NppiSize{W, H}, NppiPoint{0, 0}, l2.data(), 2, desc, NppiSize{W, H}, cfg, scratch,
      NPP_BORDER_REPLICATE, c);
  printf("2loc st=%s\n", es(st3));
  return 0;
}
