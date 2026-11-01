// f13 probe 1: measure NVIDIA NPP 13.4 semantics for
//   * nppiFilterScharrHoriz/Vert (+Border)          -> kernel taps / sign
//   * nppiGradientVectorSobel/Prewitt/ScharrBorder  -> X/Y kernels, mag/angle
// Run on the RTX 5090: nvcc -o probe1 f13_probe1.cu -lnppif -I/usr/local/cuda/include
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

#define P(fmt, ...) printf(fmt, __VA_ARGS__)
#define ST(x) do { NppStatus s_ = (x); if (s_ != 0) printf("    [status %d]\n", (int)s_); } while (0)

template <typename T>
static void dump(const char *tag, const T *h, int w, int hh, int step, const char *fmt) {
  printf("%s (%dx%d)\n", tag, w, hh);
  for (int y = 0; y < hh; ++y) {
    printf("   ");
    for (int x = 0; x < w; ++x) printf(fmt, h[(size_t)(y * step) / sizeof(T) + x]), printf(" ");
    printf("\n");
  }
}

static NppStreamContext ctx() {
  NppStreamContext c{};
  c.hStream = 0; // default stream
  return c;
}

// ---------------------------------------------------------------- Scharr
static void scharr_impulse() {
  printf("=== Scharr impulse (7x7 zeros, (3,3)=100, ROI 3x3 @ (2,2)) ===\n");
  const int W = 7, H = 7;
  Npp8u s8[W * H] = {0};
  s8[3 * W + 3] = 100;
  Npp32f s32[W * H] = {0};
  s32[3 * W + 3] = 100.0f;
  Npp8u *d8; Npp32f *d32; Npp16s *o16; Npp32f *o32;
  cudaMalloc(&d8, W * H); cudaMalloc(&d32, W * H * 4);
  cudaMalloc(&o16, 9 * 2); cudaMalloc(&o32, 9 * 4);
  cudaMemcpy(d8, s8, W * H, cudaMemcpyHostToDevice);
  cudaMemcpy(d32, s32, W * H * 4, cudaMemcpyHostToDevice);
  Npp16s h16[9]; Npp32f h32[9];
  NppiSize roi{3, 3};
  ST(nppiFilterScharrHoriz_8u16s_C1R_Ctx(d8, W, o16, 6, roi, ctx()));
  cudaMemcpy(h16, o16, 18, cudaMemcpyDeviceToHost);
  dump("ScharrHoriz 8u16s", h16, 3, 3, 6, "%5d");
  ST(nppiFilterScharrVert_8u16s_C1R_Ctx(d8, W, o16, 6, roi, ctx()));
  cudaMemcpy(h16, o16, 18, cudaMemcpyDeviceToHost);
  dump("ScharrVert 8u16s", h16, 3, 3, 6, "%5d");
  ST(nppiFilterScharrHoriz_32f_C1R_Ctx(d32, W * 4, o32, 12, roi, ctx()));
  cudaMemcpy(h32, o32, 36, cudaMemcpyDeviceToHost);
  dump("ScharrHoriz 32f", h32, 3, 3, 12, "%7.1f");
  ST(nppiFilterScharrVert_32f_C1R_Ctx(d32, W * 4, o32, 12, roi, ctx()));
  cudaMemcpy(h32, o32, 36, cudaMemcpyDeviceToHost);
  dump("ScharrVert 32f", h32, 3, 3, 12, "%7.1f");
  // Border variants on the same impulse: reads clamp, result must match plain
  NppiPoint off{2, 2}; NppiSize sz{7, 7};
  ST(nppiFilterScharrHorizBorder_8u16s_C1R_Ctx(d8, W, sz, off, o16, 6, roi, NPP_BORDER_REPLICATE, ctx()));
  cudaMemcpy(h16, o16, 18, cudaMemcpyDeviceToHost);
  dump("ScharrHorizBorder 8u16s replicate", h16, 3, 3, 6, "%5d");
  // non-replicate border types
  const NppiBorderType bt[4] = {NPP_BORDER_CONSTANT, NPP_BORDER_MIRROR, NPP_BORDER_WRAP, (NppiBorderType)77};
  const char *bn[4] = {"CONST", "MIRROR", "WRAP", "77"};
  for (int i = 0; i < 4; ++i) {
    printf("  ScharrHorizBorder border=%s -> ", bn[i]);
    ST(nppiFilterScharrHorizBorder_8u16s_C1R_Ctx(d8, W, sz, off, o16, 6, roi, bt[i], ctx()));
    printf("\n");
  }
  cudaFree(d8); cudaFree(d32); cudaFree(o16); cudaFree(o32);
}

static void scharr_errors() {
  printf("=== Scharr error matrix (W=H=4) ===\n");
  Npp8u *d8; Npp16s *o16; Npp32f *d32, *o32;
  cudaMalloc(&d8, 64); cudaMalloc(&o16, 64); cudaMalloc(&d32, 256); cudaMalloc(&o32, 256);
  NppiSize roi{4, 4}, sz{7, 7}; NppiPoint off{2, 2};
  NppStreamContext c = ctx();
  struct { const char *n; NppStatus s; } r[24];
  int n = 0;
  r[n++] = {"nullsrc", nppiFilterScharrHorizBorder_8u16s_C1R_Ctx(nullptr, 16, sz, off, o16, 16, roi, NPP_BORDER_REPLICATE, c)};
  r[n++] = {"nulldst", nppiFilterScharrHorizBorder_8u16s_C1R_Ctx(d8, 16, sz, off, nullptr, 16, roi, NPP_BORDER_REPLICATE, c)};
  r[n++] = {"negroi", nppiFilterScharrHorizBorder_8u16s_C1R_Ctx(d8, 16, sz, off, o16, 16, NppiSize{4, -1}, NPP_BORDER_REPLICATE, c)};
  r[n++] = {"zeroroi", nppiFilterScharrHorizBorder_8u16s_C1R_Ctx(d8, 16, sz, off, o16, 3, NppiSize{0, 0}, NPP_BORDER_REPLICATE, c)};
  r[n++] = {"zroi_badborder", nppiFilterScharrHorizBorder_8u16s_C1R_Ctx(d8, 16, sz, off, o16, 3, NppiSize{0, 0}, NPP_BORDER_CONSTANT, c)};
  r[n++] = {"zroi_offneg", nppiFilterScharrHorizBorder_8u16s_C1R_Ctx(d8, 16, sz, NppiPoint{0, -1}, o16, 3, NppiSize{0, 0}, NPP_BORDER_REPLICATE, c)};
  r[n++] = {"srcsize0", nppiFilterScharrHorizBorder_8u16s_C1R_Ctx(d8, 16, NppiSize{0, 5}, off, o16, 16, roi, NPP_BORDER_REPLICATE, c)};
  r[n++] = {"offneg", nppiFilterScharrHorizBorder_8u16s_C1R_Ctx(d8, 16, sz, NppiPoint{-1, 0}, o16, 16, roi, NPP_BORDER_REPLICATE, c)};
  r[n++] = {"badborder", nppiFilterScharrHorizBorder_8u16s_C1R_Ctx(d8, 16, sz, off, o16, 16, roi, NPP_BORDER_CONSTANT, c)};
  r[n++] = {"stepsmall", nppiFilterScharrHorizBorder_8u16s_C1R_Ctx(d8, 16, sz, off, o16, 7, roi, NPP_BORDER_REPLICATE, c)};
  r[n++] = {"stepodd", nppiFilterScharrHorizBorder_8u16s_C1R_Ctx(d8, 16, sz, off, o16, 9, roi, NPP_BORDER_REPLICATE, c)};
  r[n++] = {"negroi_badborder", nppiFilterScharrHorizBorder_8u16s_C1R_Ctx(d8, 16, sz, off, o16, 16, NppiSize{4, -1}, NPP_BORDER_CONSTANT, c)};
  r[n++] = {"plain_nullsrc", nppiFilterScharrHoriz_8u16s_C1R_Ctx(nullptr, 16, o16, 16, roi, c)};
  r[n++] = {"plain_negroi", nppiFilterScharrHoriz_8u16s_C1R_Ctx(d8, 16, o16, 16, NppiSize{4, -1}, c)};
  r[n++] = {"plain_zeroroi", nppiFilterScharrHoriz_8u16s_C1R_Ctx(d8, 16, o16, 3, NppiSize{0, 0}, c)};
  r[n++] = {"plain_stepsmall", nppiFilterScharrHoriz_8u16s_C1R_Ctx(d8, 16, o16, 7, roi, c)};
  r[n++] = {"plain_stepodd", nppiFilterScharrHoriz_8u16s_C1R_Ctx(d8, 16, o16, 9, roi, c)};
  r[n++] = {"32f_stepodd", nppiFilterScharrHoriz_32f_C1R_Ctx(d32, 64, o32, 17, roi, c)};
  r[n++] = {"32f_stepsmall", nppiFilterScharrHoriz_32f_C1R_Ctx(d32, 64, o32, 15, roi, c)};
  r[n++] = {"horiz_vertnull", nppiFilterScharrVert_8u16s_C1R_Ctx(nullptr, 16, o16, 16, roi, c)};
  for (int i = 0; i < n; ++i) printf("  %-18s -> %d\n", r[i].n, (int)r[i].s);
  cudaFree(d8); cudaFree(o16); cudaFree(d32); cudaFree(o32);
}

// ------------------------------------------------- GradientVector probes
// 8u16s C1: X/Y/Mag are Npp16s, Angle Npp32f
static void gv_impulse_8u16s(const char *fam, int mask, NppiNorm norm) {
  printf("=== %s 8u16s impulse (7x7, (3,3)=100, ROI 3x3 @ (2,2), mask=%s, norm=%d) ===\n",
         fam, mask == 0 ? "3x3" : "5x5", (int)norm);
  const int W = 7, H = 7;
  Npp8u s[W * H] = {0};
  s[3 * W + 3] = 100;
  Npp8u *d; Npp16s *ox, *oy, *om; Npp32f *oa;
  cudaMalloc(&d, W * H); cudaMalloc(&ox, 9 * 2); cudaMalloc(&oy, 9 * 2);
  cudaMalloc(&om, 9 * 2); cudaMalloc(&oa, 9 * 4);
  cudaMemcpy(d, s, W * H, cudaMemcpyHostToDevice);
  NppiSize roi{3, 3}, sz{7, 7}; NppiPoint off{2, 2};
  NppiMaskSize ms = mask ? NPP_MASK_SIZE_5_X_5 : NPP_MASK_SIZE_3_X_3;
  NppStatus st;
  if (!strcmp(fam, "Sobel"))
    st = nppiGradientVectorSobelBorder_8u16s_C1R_Ctx(d, W, sz, off, ox, 6, oy, 6, om, 6, oa, 12, roi, ms, norm, NPP_BORDER_REPLICATE, ctx());
  else if (!strcmp(fam, "Prewitt"))
    st = nppiGradientVectorPrewittBorder_8u16s_C1R_Ctx(d, W, sz, off, ox, 6, oy, 6, om, 6, oa, 12, roi, ms, norm, NPP_BORDER_REPLICATE, ctx());
  else
    st = nppiGradientVectorScharrBorder_8u16s_C1R_Ctx(d, W, sz, off, ox, 6, oy, 6, om, 6, oa, 12, roi, ms, norm, NPP_BORDER_REPLICATE, ctx());
  printf("  status=%d\n", (int)st);
  Npp16s hx[9], hy[9], hm[9]; Npp32f ha[9];
  cudaMemcpy(hx, ox, 18, cudaMemcpyDeviceToHost); cudaMemcpy(hy, oy, 18, cudaMemcpyDeviceToHost);
  cudaMemcpy(hm, om, 18, cudaMemcpyDeviceToHost); cudaMemcpy(ha, oa, 36, cudaMemcpyDeviceToHost);
  dump("X", hx, 3, 3, 6, "%6d"); dump("Y", hy, 3, 3, 6, "%6d");
  dump("Mag", hm, 3, 3, 6, "%6d"); dump("Angle", ha, 3, 3, 12, "%9.4f");
  cudaFree(d); cudaFree(ox); cudaFree(oy); cudaFree(om); cudaFree(oa);
}

// random 8u image, compare Mag/Angle against CPU candidates
static void gv_random_8u16s(const char *fam, NppiNorm norm) {
  printf("=== %s 8u16s random 9x9 ROI, mask 3x3, norm=%d, mag/angle convention ===\n", fam, (int)norm);
  const int W = 9, H = 9, RW = 7, RH = 7;
  Npp8u s[W * H];
  for (int i = 0; i < W * H; ++i) s[i] = (Npp8u)((i * 37 + 11) & 0xff);
  Npp8u *d; Npp16s *ox, *oy, *om; Npp32f *oa;
  cudaMalloc(&d, W * H); cudaMalloc(&ox, RW * RH * 2); cudaMalloc(&oy, RW * RH * 2);
  cudaMalloc(&om, RW * RH * 2); cudaMalloc(&oa, RW * RH * 4);
  cudaMemcpy(d, s, W * H, cudaMemcpyHostToDevice);
  NppiSize roi{RW, RH}, sz{W, H}; NppiPoint off{1, 1};
  NppStatus st;
  if (!strcmp(fam, "Sobel"))
    st = nppiGradientVectorSobelBorder_8u16s_C1R_Ctx(d, W, sz, off, ox, RW * 2, oy, RW * 2, om, RW * 2, oa, RW * 4, roi, NPP_MASK_SIZE_3_X_3, norm, NPP_BORDER_REPLICATE, ctx());
  else
    st = nppiGradientVectorScharrBorder_8u16s_C1R_Ctx(d, W, sz, off, ox, RW * 2, oy, RW * 2, om, RW * 2, oa, RW * 4, roi, NPP_MASK_SIZE_3_X_3, norm, NPP_BORDER_REPLICATE, ctx());
  printf("  status=%d\n", (int)st);
  std::vector<Npp16s> hx(RW * RH), hy(RW * RH), hm(RW * RH);
  std::vector<Npp32f> ha(RW * RH);
  cudaMemcpy(hx.data(), ox, RW * RH * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(hy.data(), oy, RW * RH * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(hm.data(), om, RW * RH * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(ha.data(), oa, RW * RH * 4, cudaMemcpyDeviceToHost);
  int trunc_ok = 1, round_ok = 1, l1_ok = 1;
  int atan_ok = 1;
  for (int y = 0; y < RH; ++y)
    for (int x = 0; x < RW; ++x) {
      size_t i = y * RW + x;
      double X = hx[i], Y = hy[i];
      double l2 = std::sqrt(X * X + Y * Y);
      if (norm == (NppiNorm)2) {
        if ((double)hm[i] != std::floor(l2)) trunc_ok = 0;
        if ((double)hm[i] != std::floor(l2 + 0.5)) round_ok = 0;
      } else {
        if ((double)hm[i] != std::fabs(X) + std::fabs(Y)) l1_ok = 0;
      }
      if (std::fabs((double)ha[i] - std::atan2(Y, X)) > 1e-4) atan_ok = 0;
    }
  printf("  mag: trunc_ok=%d round_ok=%d l1_ok=%d  angle atan2(Y,X)_ok=%d\n", trunc_ok, round_ok, l1_ok, atan_ok);
  if (!atan_ok) {
    int neg_ok = 1, deg_ok = 1, negdeg_ok = 1;
    for (size_t i = 0; i < hx.size(); ++i) {
      double X = hx[i], Y = hy[i];
      if (std::fabs((double)ha[i] - std::atan2(-Y, X)) > 1e-4) neg_ok = 0;
      if (std::fabs((double)ha[i] - std::atan2(Y, X) * 180.0 / M_PI) > 1e-3) deg_ok = 0;
      if (std::fabs((double)ha[i] - std::atan2(-Y, X) * 180.0 / M_PI) > 1e-3) negdeg_ok = 0;
    }
    printf("  angle: atan2(-Y,X)_ok=%d deg_ok=%d negdeg_ok=%d\n", neg_ok, deg_ok, negdeg_ok);
  }
  // a few raw samples for manual inspection
  for (int i = 0; i < 4; ++i)
    printf("   sample%d: X=%d Y=%d Mag=%d Angle=%.5f\n", i, hx[i], hy[i], hm[i], ha[i]);
  cudaFree(d); cudaFree(ox); cudaFree(oy); cudaFree(om); cudaFree(oa);
}

// C3C1R channel selection: 3-channel src with distinct per-channel gradients
static void gv_c3c1() {
  printf("=== GradientVectorSobelBorder 8u16s C3C1R selection (9x9 ramps) ===\n");
  const int W = 9, H = 9, RW = 5, RH = 5;
  Npp8u s[W * H * 3];
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      // ch0: strong vertical ramp; ch1: weak; ch2: medium horizontal
      s[(y * W + x) * 3 + 0] = (Npp8u)(x * 5);       // dI/dx strong
      s[(y * W + x) * 3 + 1] = (Npp8u)(x * 1 + 40);  // weak
      s[(y * W + x) * 3 + 2] = (Npp8u)(y * 3 + 30);  // dI/dy medium
    }
  Npp8u *d; Npp16s *ox, *oy, *om; Npp32f *oa;
  cudaMalloc(&d, W * H * 3); cudaMalloc(&ox, RW * RH * 2); cudaMalloc(&oy, RW * RH * 2);
  cudaMalloc(&om, RW * RH * 2); cudaMalloc(&oa, RW * RH * 4);
  cudaMemcpy(d, s, W * H * 3, cudaMemcpyHostToDevice);
  NppiSize roi{RW, RH}, sz{W, H}; NppiPoint off{1, 1};
  for (int normi = 1; normi <= 2; ++normi) {
    NppStatus st = nppiGradientVectorSobelBorder_8u16s_C3C1R_Ctx(d, W * 3, sz, off, ox, RW * 2, oy, RW * 2, om, RW * 2,
                                                                  oa, RW * 4, roi, NPP_MASK_SIZE_3_X_3, (NppiNorm)normi,
                                                                  NPP_BORDER_REPLICATE, ctx());
    printf("  norm=%d status=%d\n", normi, (int)st);
    Npp16s hx[4], hy[4], hm[4]; Npp32f ha[4];
    cudaMemcpy(hx, ox, 8, cudaMemcpyDeviceToHost); cudaMemcpy(hy, oy, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(hm, om, 8, cudaMemcpyDeviceToHost); cudaMemcpy(ha, oa, 16, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 4; ++i)
      printf("   (%d,%d): X=%d Y=%d Mag=%d Angle=%.5f\n", i % RW, i / RW, hx[i], hy[i], hm[i], ha[i]);
  }
  cudaFree(d); cudaFree(ox); cudaFree(oy); cudaFree(om); cudaFree(oa);
}

// error matrix + optional outputs
static void gv_errors() {
  printf("=== GradientVectorSobelBorder error matrix (W=H=8, ROI 4x4) ===\n");
  Npp8u *d; Npp16s *ox, *oy, *om; Npp32f *oa;
  cudaMalloc(&d, 256); cudaMalloc(&ox, 64); cudaMalloc(&oy, 64); cudaMalloc(&om, 64); cudaMalloc(&oa, 128);
  NppiSize roi{4, 4}, sz{8, 8}; NppiPoint off{1, 1};
  NppStreamContext c = ctx();
  struct { const char *n; NppStatus s; } r[32];
  int n = 0;
  auto call = [&](Npp8u *s_, Npp16s *x_, Npp16s *y_, Npp16s *m_, Npp32f *a_, NppiSize roi_, NppiMaskSize ms, NppiNorm nm, NppiBorderType bt, NppiSize sz_, NppiPoint off_) {
    return nppiGradientVectorSobelBorder_8u16s_C1R_Ctx(s_, 8, sz_, off_, x_, 8, y_, 8, m_, 8, a_, 16, roi_, ms, nm, bt, c);
  };
  r[n++] = {"allok", call(d, ox, oy, om, oa, roi, NPP_MASK_SIZE_3_X_3, (NppiNorm)2, NPP_BORDER_REPLICATE, sz, off)};
  r[n++] = {"nullsrc", call(nullptr, ox, oy, om, oa, roi, NPP_MASK_SIZE_3_X_3, (NppiNorm)2, NPP_BORDER_REPLICATE, sz, off)};
  r[n++] = {"nullx", call(d, nullptr, oy, om, oa, roi, NPP_MASK_SIZE_3_X_3, (NppiNorm)2, NPP_BORDER_REPLICATE, sz, off)};
  r[n++] = {"nully", call(d, ox, nullptr, om, oa, roi, NPP_MASK_SIZE_3_X_3, (NppiNorm)2, NPP_BORDER_REPLICATE, sz, off)};
  r[n++] = {"nullmag", call(d, ox, oy, nullptr, oa, roi, NPP_MASK_SIZE_3_X_3, (NppiNorm)2, NPP_BORDER_REPLICATE, sz, off)};
  r[n++] = {"nullangle", call(d, ox, oy, om, nullptr, roi, NPP_MASK_SIZE_3_X_3, (NppiNorm)2, NPP_BORDER_REPLICATE, sz, off)};
  r[n++] = {"allnullout", call(d, nullptr, nullptr, nullptr, nullptr, roi, NPP_MASK_SIZE_3_X_3, (NppiNorm)2, NPP_BORDER_REPLICATE, sz, off)};
  r[n++] = {"negroi", call(d, ox, oy, om, oa, NppiSize{4, -1}, NPP_MASK_SIZE_3_X_3, (NppiNorm)2, NPP_BORDER_REPLICATE, sz, off)};
  r[n++] = {"zeroroi", call(d, ox, oy, om, oa, NppiSize{0, 0}, NPP_MASK_SIZE_3_X_3, (NppiNorm)2, NPP_BORDER_REPLICATE, sz, off)};
  r[n++] = {"zroi_badborder", call(d, ox, oy, om, oa, NppiSize{0, 0}, NPP_MASK_SIZE_3_X_3, (NppiNorm)2, NPP_BORDER_CONSTANT, sz, off)};
  r[n++] = {"zroi_badmask", call(d, ox, oy, om, oa, NppiSize{0, 0}, (NppiMaskSize)0, (NppiNorm)2, NPP_BORDER_REPLICATE, sz, off)};
  r[n++] = {"srcsize0", call(d, ox, oy, om, oa, roi, NPP_MASK_SIZE_3_X_3, (NppiNorm)2, NPP_BORDER_REPLICATE, NppiSize{0, 8}, off)};
  r[n++] = {"offneg", call(d, ox, oy, om, oa, roi, NPP_MASK_SIZE_3_X_3, (NppiNorm)2, NPP_BORDER_REPLICATE, sz, NppiPoint{-1, 0})};
  r[n++] = {"badborder", call(d, ox, oy, om, oa, roi, NPP_MASK_SIZE_3_X_3, (NppiNorm)2, NPP_BORDER_CONSTANT, sz, off)};
  r[n++] = {"badmask", call(d, ox, oy, om, oa, roi, (NppiMaskSize)0, (NppiNorm)2, NPP_BORDER_REPLICATE, sz, off)};
  r[n++] = {"badmask_vbadborder", call(d, ox, oy, om, oa, roi, (NppiMaskSize)0, (NppiNorm)2, NPP_BORDER_CONSTANT, sz, off)};
  r[n++] = {"norminf", call(d, ox, oy, om, oa, roi, NPP_MASK_SIZE_3_X_3, (NppiNorm)0, NPP_BORDER_REPLICATE, sz, off)};
  r[n++] = {"badnorm7", call(d, ox, oy, om, oa, roi, NPP_MASK_SIZE_3_X_3, (NppiNorm)7, NPP_BORDER_REPLICATE, sz, off)};
  for (int i = 0; i < n; ++i) printf("  %-18s -> %d\n", r[i].n, (int)r[i].s);
  // mask/border supported combos on 32f variant (dsts are 32f)
  printf("=== GradientVectorSobelBorder 32f mask/border combos ===\n");
  Npp32f *df, *oxf, *oyf, *omf, *oaf;
  cudaMalloc(&df, 256); cudaMalloc(&oxf, 64); cudaMalloc(&oyf, 64); cudaMalloc(&omf, 64); cudaMalloc(&oaf, 64);
  const NppiBorderType bt[3] = {NPP_BORDER_MIRROR, NPP_BORDER_WRAP, NPP_BORDER_CONSTANT};
  const char *bn[3] = {"MIRROR", "WRAP", "CONST"};
  for (int i = 0; i < 3; ++i) {
    printf("  border=%s -> ", bn[i]);
    ST(nppiGradientVectorSobelBorder_32f_C1R_Ctx(df, 32, sz, off, oxf, 16, oyf, 16, omf, 16, oaf, 16, roi, NPP_MASK_SIZE_3_X_3, (NppiNorm)2, bt[i], ctx()));
    printf("\n");
  }
  printf("  mask=5x5 border=MIRROR -> ");
  ST(nppiGradientVectorSobelBorder_32f_C1R_Ctx(df, 32, sz, off, oxf, 16, oyf, 16, omf, 16, oaf, 16, roi, NPP_MASK_SIZE_5_X_5, (NppiNorm)2, NPP_BORDER_MIRROR, ctx()));
  printf("\n");
  cudaFree(df); cudaFree(oxf); cudaFree(oyf); cudaFree(omf); cudaFree(oaf);
  cudaFree(d); cudaFree(ox); cudaFree(oy); cudaFree(om); cudaFree(oa);
}

// 32f output scaling check: does the 32f variant scale the kernels?
static void gv_32f_scaling() {
  printf("=== GradientVectorSobelBorder 16s32f / 16u32f / 32f impulse (7x7 (3,3)=100) ===\n");
  const int W = 7, H = 7;
  Npp16s s16[W * H] = {0}; s16[3 * W + 3] = 100;
  Npp16u s16u[W * H] = {0}; s16u[3 * W + 3] = 100;
  Npp32f s32[W * H] = {0}; s32[3 * W + 3] = 100.0f;
  Npp16s *d16; Npp16u *d16u; Npp32f *d32, *ox, *oy, *om, *oa;
  cudaMalloc(&d16, W * H * 2); cudaMalloc(&d16u, W * H * 2); cudaMalloc(&d32, W * H * 4);
  cudaMalloc(&ox, 36); cudaMalloc(&oy, 36); cudaMalloc(&om, 36); cudaMalloc(&oa, 36);
  cudaMemcpy(d16, s16, W * H * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(d16u, s16u, W * H * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(d32, s32, W * H * 4, cudaMemcpyHostToDevice);
  NppiSize roi{3, 3}, sz{7, 7}; NppiPoint off{2, 2};
  Npp32f hx[9], hy[9], hm[9], ha[9];
  struct { const char *n; } names[3] = {"16s32f", "16u32f", "32f"};
  for (int k = 0; k < 3; ++k) {
    NppStatus st = NPP_NO_ERROR;
    if (k == 0) st = nppiGradientVectorSobelBorder_16s32f_C1R_Ctx(d16, 14, sz, off, ox, 12, oy, 12, om, 12, oa, 12, roi, NPP_MASK_SIZE_3_X_3, (NppiNorm)2, NPP_BORDER_REPLICATE, ctx());
    if (k == 1) st = nppiGradientVectorSobelBorder_16u32f_C1R_Ctx(d16u, 14, sz, off, ox, 12, oy, 12, om, 12, oa, 12, roi, NPP_MASK_SIZE_3_X_3, (NppiNorm)2, NPP_BORDER_REPLICATE, ctx());
    if (k == 2) st = nppiGradientVectorSobelBorder_32f_C1R_Ctx(d32, 28, sz, off, ox, 12, oy, 12, om, 12, oa, 12, roi, NPP_MASK_SIZE_3_X_3, (NppiNorm)2, NPP_BORDER_REPLICATE, ctx());
    cudaMemcpy(hx, ox, 36, cudaMemcpyDeviceToHost); cudaMemcpy(hy, oy, 36, cudaMemcpyDeviceToHost);
    cudaMemcpy(hm, om, 36, cudaMemcpyDeviceToHost); cudaMemcpy(ha, oa, 36, cudaMemcpyDeviceToHost);
    printf("  %s status=%d  X(center)=%.1f Y(center)=%.1f Mag(center)=%.3f Angle(1,1)=%.5f\n",
           names[k].n, (int)st, hx[4], hy[4], hm[4], ha[4]);
    dump("    X", hx, 3, 3, 12, "%6.0f");
  }
  cudaFree(d16); cudaFree(d16u); cudaFree(d32); cudaFree(ox); cudaFree(oy); cudaFree(om); cudaFree(oa);
}

int main() {
  scharr_impulse();
  scharr_errors();
  gv_impulse_8u16s("Sobel", 0, (NppiNorm)2);
  gv_impulse_8u16s("Sobel", 1, (NppiNorm)2);
  gv_impulse_8u16s("Prewitt", 0, (NppiNorm)1);
  gv_impulse_8u16s("Scharr", 0, (NppiNorm)2);
  gv_random_8u16s("Sobel", (NppiNorm)2);
  gv_random_8u16s("Sobel", (NppiNorm)1);
  gv_random_8u16s("Scharr", (NppiNorm)2);
  gv_c3c1();
  gv_errors();
  gv_32f_scaling();
  return 0;
}
