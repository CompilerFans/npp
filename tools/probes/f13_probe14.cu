// f13 probe 14: why does the HOG happy path return NPP_STEP_ERROR?
// Vary nSrcStep / oSrcSize / oSizeROI / oSrcOffset independently.
#include <npp.h>
#include <nppcore.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const char *es(NppStatus s) { static char b[32]; snprintf(b, 32, "%d", (int)s); return b; }

int main(int argc, char **argv) {
  int part = argc > 1 ? atoi(argv[1]) : 0;
  const int W = 16, H = 16;
  std::vector<Npp8u> h((size_t)W * H, 0);
  h[3 * W + 3] = 100;
  Npp8u *d; cudaMalloc(&d, 4096); cudaMemcpy(d, h.data(), W * H, cudaMemcpyHostToDevice);
  NppiHOGConfig cfg{2, 4, 4, {8, 8}};
  std::vector<NppiPoint> locs = {{0, 0}};
  int bsz = -1, dsz = -1;
  nppiHistogramOfGradientsBorderGetBufferSize(cfg, locs.data(), 1, NppiSize{16, 16}, &bsz);
  nppiHistogramOfGradientsBorderGetDescriptorsSize(cfg, 1, &dsz);
  printf("bufsize=%d descsize=%d\n", bsz, dsz);
  Npp8u *scratch; cudaMalloc(&scratch, 8192);
  Npp32f *desc; cudaMalloc(&desc, 4096);
  NppStreamContext c{}; c.hStream = 0;

  if (part == 2) {
    // fine step boundary scan for 8u C1R, srcW = 16
    for (int step = 16; step <= 34; ++step) {
      cudaMemset(desc, 0, 4096);
      NppStatus st = nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(
          d, step, NppiSize{16, 16}, NppiPoint{0, 0}, locs.data(), 1, desc, NppiSize{16, 16}, cfg,
          scratch, NPP_BORDER_REPLICATE, c);
      printf("W16 step %2d: st=%s\n", step, es(st));
    }
  }
  if (part == 3) {
    // step boundary vs smaller source: 8x8 image, window 8x8
    Npp8u *d2; cudaMalloc(&d2, 4096); cudaMemset(d2, 0, 4096);
    std::vector<NppiPoint> l0 = {{0, 0}};
    int b2 = -1;
    nppiHistogramOfGradientsBorderGetBufferSize(cfg, l0.data(), 1, NppiSize{8, 8}, &b2);
    for (int step = 8; step <= 26; step += 2) {
      cudaMemset(desc, 0, 4096);
      NppStatus st = nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(
          d2, step, NppiSize{8, 8}, NppiPoint{0, 0}, l0.data(), 1, desc, NppiSize{8, 8}, cfg,
          scratch, NPP_BORDER_REPLICATE, c);
      printf("W8 step %2d: st=%s\n", step, es(st));
    }
    // C3R: 8x8 3-channel (24 bytes per row)
    for (int step = 24; step <= 56; step += 4) {
      cudaMemset(desc, 0, 4096);
      NppStatus st = nppiHistogramOfGradientsBorder_8u32f_C3R_Ctx(
          d2, step, NppiSize{8, 8}, NppiPoint{0, 0}, l0.data(), 1, desc, NppiSize{8, 8}, cfg,
          scratch, NPP_BORDER_REPLICATE, c);
      printf("W8c3 step %2d: st=%s\n", step, es(st));
    }
  }
  if (part == 4) {
    // W8 c1: higher steps; also W4 and W24.
    Npp8u *d2; cudaMalloc(&d2, 8192); cudaMemset(d2, 0, 8192);
    std::vector<NppiPoint> l0 = {{0, 0}};
    for (int step = 27; step <= 48; ++step) {
      cudaMemset(desc, 0, 4096);
      NppStatus st = nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(
          d2, step, NppiSize{8, 8}, NppiPoint{0, 0}, l0.data(), 1, desc, NppiSize{8, 8}, cfg,
          scratch, NPP_BORDER_REPLICATE, c);
      if (st != 0) printf("W8 step %2d: st=%s\n", step, es(st));
    }
    printf("W8 scan done (only failures listed)\n");
    for (int step = 32; step <= 96; step += 4) {
      cudaMemset(desc, 0, 4096);
      NppStatus st = nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(
          d2, step, NppiSize{24, 8}, NppiPoint{0, 0}, l0.data(), 1, desc, NppiSize{24, 8}, cfg,
          scratch, NPP_BORDER_REPLICATE, c);
      if (st != 0) printf("W24 step %2d: st=%s\n", step, es(st));
    }
    printf("W24 scan done\n");
  }
  if (part == 5) {
    // does the 32-byte floor move with window size / channels / dtype?
    Npp8u *d2; cudaMalloc(&d2, 8192); cudaMemset(d2, 0, 8192);
    std::vector<NppiPoint> l0 = {{0, 0}};
    NppiHOGConfig cfg16{4, 8, 9, {16, 16}};
    for (int step = 16; step <= 68; step += 4) {
      cudaMemset(desc, 0, 4096);
      NppStatus st = nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(
          d2, step, NppiSize{16, 16}, NppiPoint{0, 0}, l0.data(), 1, desc, NppiSize{16, 16}, cfg16,
          scratch, NPP_BORDER_REPLICATE, c);
      if (st != 0) printf("win16 W16 step %2d: st=%s\n", step, es(st));
    }
    printf("win16 scan done\n");
    for (int step = 24; step <= 48; step += 4) {
      cudaMemset(desc, 0, 4096);
      NppStatus st = nppiHistogramOfGradientsBorder_8u32f_C3R_Ctx(
          d2, step, NppiSize{8, 8}, NppiPoint{0, 0}, l0.data(), 1, desc, NppiSize{8, 8}, cfg,
          scratch, NPP_BORDER_REPLICATE, c);
      if (st != 0) printf("W8c3 step %2d: st=%s\n", step, es(st));
    }
    printf("c3 scan done\n");
    // 32f source: element size 4
    Npp32f *f2; cudaMalloc(&f2, 8192); cudaMemset(f2, 0, 8192);
    for (int step = 32; step <= 68; step += 4) {
      cudaMemset(desc, 0, 4096);
      NppStatus st = nppiHistogramOfGradientsBorder_32f_C1R_Ctx(
          f2, step, NppiSize{8, 8}, NppiPoint{0, 0}, l0.data(), 1, desc, NppiSize{8, 8}, cfg,
          scratch, NPP_BORDER_REPLICATE, c);
      if (st != 0) printf("W8f32 step %2d: st=%s\n", step, es(st));
    }
    printf("f32 scan done\n");
  }
  if (part == 6) {
    Npp32f *f2; cudaMalloc(&f2, 8192); cudaMemset(f2, 0, 8192);
    std::vector<NppiPoint> l0 = {{0, 0}};
    // 32f low steps (win 8 -> floor 4*8 = 32 bytes?)
    for (int step = 8; step <= 40; step += 4) {
      NppStatus st = nppiHistogramOfGradientsBorder_32f_C1R_Ctx(
          f2, step, NppiSize{8, 8}, NppiPoint{0, 0}, l0.data(), 1, desc, NppiSize{8, 8}, cfg,
          scratch, NPP_BORDER_REPLICATE, c);
      if (st != 0) printf("f32 W8 step %2d: st=%s\n", step, es(st));
    }
    printf("f32 low scan done\n");
    // window 32 wide (cell 8, block 16): floor should be 4*32 = 128
    NppiHOGConfig cfg32{8, 16, 9, {32, 32}};
    for (int step = 96; step <= 136; step += 8) {
      NppStatus st = nppiHistogramOfGradientsBorder_32f_C1R_Ctx(
          f2, step, NppiSize{32, 32}, NppiPoint{0, 0}, l0.data(), 1, desc, NppiSize{32, 32}, cfg32,
          scratch, NPP_BORDER_REPLICATE, c);
      if (st != 0) printf("f32 W32 win32 step %3d: st=%s\n", step, es(st));
    }
    printf("win32 scan done\n");
    // 16s source, low steps
    Npp16s *s2; cudaMalloc(&s2, 8192); cudaMemset(s2, 0, 8192);
    for (int step = 16; step <= 40; step += 4) {
      NppStatus st = nppiHistogramOfGradientsBorder_16s32f_C1R_Ctx(
          s2, step, NppiSize{8, 8}, NppiPoint{0, 0}, l0.data(), 1, desc, NppiSize{8, 8}, cfg,
          scratch, NPP_BORDER_REPLICATE, c);
      if (st != 0) printf("16s W8 step %2d: st=%s\n", step, es(st));
    }
    printf("16s scan done\n");
  }
  if (part == 1) {
    struct { int step; NppiSize sz; NppiSize roi; NppiPoint off; const char *tag; } cases[] = {
      {16, {16, 16}, {16, 16}, {0, 0}, "step16 roi16"},
      {32, {16, 16}, {16, 16}, {0, 0}, "step32 roi16"},
      {64, {16, 16}, {16, 16}, {0, 0}, "step64 roi16"},
      {20, {16, 16}, {16, 16}, {0, 0}, "step20 roi16"},
      {17, {16, 16}, {16, 16}, {0, 0}, "step17 roi16"},
      {16, {16, 16}, {8, 8}, {0, 0}, "step16 roi8"},
      {32, {16, 16}, {8, 8}, {0, 0}, "step32 roi8"},
      {16, {16, 16}, {8, 8}, {1, 1}, "step16 roi8 off11"},
      {32, {16, 16}, {8, 8}, {1, 1}, "step32 roi8 off11"},
      {32, {16, 16}, {16, 16}, {1, 1}, "step32 roi16 off11"},
      {16, {16, 16}, {4, 4}, {0, 0}, "step16 roi4"},
    };
    for (auto &t : cases) {
      cudaMemset(desc, 0, 4096);
      NppStatus st = nppiHistogramOfGradientsBorder_8u32f_C1R_Ctx(
          d, t.step, t.sz, t.off, locs.data(), 1, desc, t.roi, cfg, scratch, NPP_BORDER_REPLICATE, c);
      printf("%-24s st=%s\n", t.tag, es(st));
    }
  }
  return 0;
}
