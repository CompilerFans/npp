// f13 probe 2: nppiGradientVector*Border semantics (NPP 13.4, RTX 5090).
// Correct pSrc placement (pSrc = image + offset) unlike probe 1.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

#define CK(x) do { if ((x) != cudaSuccess) { printf("CUDA ERR %s @%d\n", cudaGetErrorString(cudaGetLastError()), __LINE__); return; } } while (0)

static NppStreamContext ctx0() { NppStreamContext c{}; c.hStream = 0; return c; }

typedef NppStatus (*GvFn)(const Npp8u*, int, NppiSize, NppiPoint, Npp16s*, int, Npp16s*, int, Npp16s*, int, Npp32f*, int, NppiSize, NppiMaskSize, NppiNorm, NppiBorderType, NppStreamContext);
typedef NppStatus (*GvFnF)(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, Npp32f*, int, Npp32f*, int, Npp32f*, int, NppiSize, NppiMaskSize, NppiNorm, NppiBorderType, NppStreamContext);

static const char *es(NppStatus s) { switch (s) {
  case NPP_NO_ERROR: return "0"; case NPP_NULL_POINTER_ERROR: return "-8 null";
  case NPP_SIZE_ERROR: return "-6 size"; case NPP_OUT_OFF_RANGE_ERROR: return "-11 offrange";
  case NPP_NOT_SUPPORTED_MODE_ERROR: return "-14 mode"; case NPP_STEP_ERROR: return "-16 step";
  case NPP_NOT_EVEN_STEP_ERROR: return "-108 evenstep"; case NPP_MASK_SIZE_ERROR: return "-33 mask";
  case NPP_BAD_ARGUMENT_ERROR: return "-5 badarg"; default: { static char b[32]; snprintf(b,32,"%d",(int)s); return b; } } }

// ---------- full-kernel impulse (3x3 and 5x5), C1R 8u16s ----------
// Image 11x11, pSrc inset by (4,4); impulse 100 at abs (6,6) = pSrc-rel (2,2).
// With ROI 5x5 every kernel tap becomes visible:
//   3x3 mask: dst(i,j) = 100*K[3-j][3-i] for i,j in {1,2,3}
//   5x5 mask: dst(i,j) = 100*K[4-j][4-i] for i,j in {0..4}
static void kern_impulse(const char *fam, GvFn fn, NppiMaskSize ms) {
  const int W = 11, H = 11, R = 5;
  const int step = W * 2, astep = W * 4;
  std::vector<Npp8u> h((size_t)W * H, 0);
  h[6 * W + 6] = 100;
  Npp8u *d; cudaMalloc(&d, W * H); cudaMemcpy(d, h.data(), W * H, cudaMemcpyHostToDevice);
  Npp16s *dx, *dy; Npp32f *da;
  cudaMalloc(&dx, step * R); cudaMalloc(&dy, step * R); cudaMalloc(&da, astep * R);
  cudaMemset(dx, 0xAA, step * R); cudaMemset(dy, 0xAA, step * R); cudaMemset(da, 0xAA, astep * R);
  Npp16s hx[64] = {0}, hy[64] = {0}; Npp32f ha[64] = {0};
  const Npp8u *pSrc = d + 4 * W + 4;
  NppStatus st = fn(pSrc, W, NppiSize{W, H}, NppiPoint{4, 4}, dx, step, dy, step, nullptr, 0, da, astep,
                    NppiSize{R, R}, ms, nppiNormL2, NPP_BORDER_REPLICATE, ctx0());
  cudaMemcpy(hx, dx, step * R, cudaMemcpyDeviceToHost);
  cudaMemcpy(hy, dy, step * R, cudaMemcpyDeviceToHost);
  cudaMemcpy(ha, da, astep * R, cudaMemcpyDeviceToHost);
  printf("== %s impulse %s (%s)\n", fam, ms == NPP_MASK_SIZE_5_X_5 ? "5x5" : "3x3", es(st));
  for (int j = 0; j < R; ++j) {
    printf("  X:");
    for (int i = 0; i < R; ++i) printf(" %5d", hx[j * (step / 2) + i]);
    printf("   Y:");
    for (int i = 0; i < R; ++i) printf(" %5d", hy[j * (step / 2) + i]);
    printf("   A:");
    for (int i = 0; i < R; ++i) printf(" %6.3f", ha[j * (astep / 4) + i]);
    printf("\n");
  }
  cudaFree(d); cudaFree(dx); cudaFree(dy); cudaFree(da);
}

// ---------- C3C1R selection (correct placement) ----------
// CPU reference kernels (hypothesis): X = -Vert, Y = +Horiz for sobel/prewitt;
// X = +Vert, Y = +Horiz for scharr.  Returns channel with max L2.
struct K { int kx[3][3]; int ky[3][3]; };
static K gv_kernels(const char *fam) {
  K k; int sobV[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}}, sobH[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
  int preV[3][3] = {{1,0,-1},{1,0,-1},{1,0,-1}}, preH[3][3] = {{-1,-1,-1},{0,0,0},{1,1,1}};
  int schV[3][3] = {{-3,0,3},{-10,0,10},{-3,0,3}}, schH[3][3] = {{-3,-10,-3},{0,0,0},{3,10,3}};
  const int (*V)[3] = !strcmp(fam, "sobel") ? sobV : !strcmp(fam, "prewitt") ? preV : schV;
  const int (*Hh)[3] = !strcmp(fam, "sobel") ? sobH : !strcmp(fam, "prewitt") ? preH : schH;
  int sgn = !strcmp(fam, "scharr") ? +1 : -1;  // X = sgn * V
  for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) { k.kx[a][b] = sgn * V[a][b]; k.ky[a][b] = Hh[a][b]; }
  return k;
}

static void gv_c3c1(const char *fam, int mode) {
  const int W = 9, H = 9, C = 3;
  std::vector<Npp8u> h((size_t)W * H * C, 0);
  for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
    Npp8u *p = &h[((size_t)y * W + x) * C];
    if (mode == 0) { p[0] = (Npp8u)(x * 5); p[1] = (Npp8u)(x + 40); p[2] = (Npp8u)(y * 3 + 30); }
    else if (mode == 1) { p[0] = (Npp8u)(x * 5); p[1] = (Npp8u)(y * 5); p[2] = 7; }        // tie ch0/ch1
    else { p[0] = (Npp8u)((x * 37 + 11) & 0xff); p[1] = (Npp8u)((y * 53 + 5) & 0xff); p[2] = (Npp8u)(((x + y) * 7) & 0xff); }
  }
  const int step = W * C, R = 5, dstep = R * 2, astep = R * 4;
  Npp8u *d; cudaMalloc(&d, step * H); cudaMemcpy(d, h.data(), step * H, cudaMemcpyHostToDevice);
  Npp16s *dx, *dy, *dm; Npp32f *da;
  cudaMalloc(&dx, dstep * R); cudaMalloc(&dy, dstep * R); cudaMalloc(&dm, dstep * R); cudaMalloc(&da, astep * R);
  GvFn fn = !strcmp(fam, "sobel") ? nppiGradientVectorSobelBorder_8u16s_C3C1R_Ctx
          : !strcmp(fam, "prewitt") ? nppiGradientVectorPrewittBorder_8u16s_C3C1R_Ctx
          : nppiGradientVectorScharrBorder_8u16s_C3C1R_Ctx;
  const Npp8u *pSrc = d + 1 * step + 1 * C;
  NppStatus st = fn(pSrc, step, NppiSize{W, H}, NppiPoint{1, 1}, dx, dstep, dy, dstep, dm, dstep, da, astep,
                    NppiSize{R, R}, NPP_MASK_SIZE_3_X_3, nppiNormL1, NPP_BORDER_REPLICATE, ctx0());
  std::vector<Npp16s> hx((size_t)dstep / 2 * R), hy = hx, hm = hx;
  std::vector<Npp32f> ha((size_t)astep / 4 * R);
  cudaMemcpy(hx.data(), dx, dstep * R, cudaMemcpyDeviceToHost);
  cudaMemcpy(hy.data(), dy, dstep * R, cudaMemcpyDeviceToHost);
  cudaMemcpy(hm.data(), dm, dstep * R, cudaMemcpyDeviceToHost);
  cudaMemcpy(ha.data(), da, astep * R, cudaMemcpyDeviceToHost);
  printf("== c3c1 %s mode%d (%s)\n", fam, mode, es(st));
  K k = gv_kernels(fam);
  for (int j = 0; j < R; ++j) {
    printf("  y=%d:", j);
    for (int i = 0; i < R; ++i) {
      // per-channel gradients at ROI pixel (i,j) = abs (1+i, 1+j)
      int gx[3] = {0, 0, 0}, gy[3] = {0, 0, 0};
      for (int c = 0; c < 3; ++c) {
        for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) {
          int xx = 1 + i + b - 1, yy = 1 + j + a - 1;
          if (xx < 0) xx = 0; if (yy < 0) yy = 0; if (xx > W - 1) xx = W - 1; if (yy > H - 1) yy = H - 1;
          int v = h[((size_t)yy * W + xx) * C + c];
          gx[c] += k.kx[a][b] * v; gy[c] += k.ky[a][b] * v;
        }
      }
      int best = 0; double bl2 = -1;
      for (int c = 0; c < 3; ++c) { double l2 = (double)gx[c] * gx[c] + (double)gy[c] * gy[c]; if (l2 > bl2) { bl2 = l2; best = c; } }
      int ex = gx[best], ey = gy[best], em = abs(ex) + abs(ey);
      double ea = atan2((double)ey, (double)ex);
      int ox = hx[j * (dstep / 2) + i], oy = hy[j * (dstep / 2) + i], om = hm[j * (dstep / 2) + i];
      float oa = ha[j * (astep / 4) + i];
      printf(" [%d]%d/%d,%d,%d,%d,%d,%d,%.2f", best, ox == ex && oy == ey && om == em && fabsf(oa - (float)ea) < 1e-3, ox, oy, om, ex, ey, em, oa);
    }
    printf("\n");
  }
  cudaFree(d); cudaFree(dx); cudaFree(dy); cudaFree(dm); cudaFree(da);
}

// ---------- error matrix additions (8u16s C1R sobel) ----------
static void gv_errors() {
  const int W = 9, H = 9, R = 4;
  const int step = W * 2, dstep = R * 2, astep = R * 4;
  std::vector<Npp8u> h((size_t)W * H, 0x55);
  Npp8u *d; cudaMalloc(&d, W * H); cudaMemcpy(d, h.data(), W * H, cudaMemcpyHostToDevice);
  Npp16s *dx, *dy, *dm; Npp32f *da;
  cudaMalloc(&dx, step * H); cudaMalloc(&dy, step * H); cudaMalloc(&dm, step * H); cudaMalloc(&da, astep * H);
  GvFn fn = nppiGradientVectorSobelBorder_8u16s_C1R_Ctx;
  NppStreamContext c = ctx0();
  NppiSize sz{9, 9}; NppiPoint off{1, 1}; NppiSize roi{4, 4};
  auto call = [&](Npp16s *x, int xs, Npp16s *y, int ys, Npp16s *m, int ms, Npp32f *a, int as,
                  NppiSize rs = NppiSize{4, 4}, NppiMaskSize mk = NPP_MASK_SIZE_3_X_3, NppiNorm nm = nppiNormL2,
                  NppiBorderType bt = NPP_BORDER_REPLICATE, NppiSize sz_ = NppiSize{9, 9}, NppiPoint off_ = NppiPoint{1, 1}) {
    return fn(d, step, sz_, off_, x, xs, y, ys, m, ms, a, as, rs, mk, nm, bt, c);
  };
  printf("== gv errors (sobel 8u16s C1R)\n");
  printf("  xstep_small       %s\n", es(call(dx, 3, dy, dstep, dm, dstep, da, astep)));
  printf("  astep_odd         %s\n", es(call(dx, dstep, dy, dstep, dm, dstep, da, astep + 1)));
  printf("  ystep_small       %s\n", es(call(dx, dstep, dy, 3, dm, dstep, da, astep)));
  printf("  mstep_small       %s\n", es(call(dx, dstep, dy, dstep, dm, 3, da, astep)));
  printf("  nullx_badstep     %s\n", es(call(nullptr, 3, dy, dstep, dm, dstep, da, astep)));
  printf("  nullangle_badstep %s\n", es(call(dx, dstep, dy, dstep, dm, dstep, nullptr, 3)));
  printf("  allnull_badborder %s\n", es(call(nullptr, dstep, nullptr, dstep, nullptr, dstep, nullptr, astep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_CONSTANT)));
  printf("  allnull_badmask   %s\n", es(call(nullptr, dstep, nullptr, dstep, nullptr, dstep, nullptr, astep, roi, (NppiMaskSize)0)));
  printf("  allnull_badnorm   %s\n", es(call(nullptr, dstep, nullptr, dstep, nullptr, dstep, nullptr, astep, roi, NPP_MASK_SIZE_3_X_3, (NppiNorm)7)));
  printf("  allnull_zeroroi   %s\n", es(call(nullptr, dstep, nullptr, dstep, nullptr, dstep, nullptr, astep, NppiSize{0, 0})));
  printf("  allnull_badstep   %s\n", es(call(nullptr, 3, nullptr, dstep, nullptr, dstep, nullptr, astep)));
  printf("  zroi_srcsize0     %s\n", es(call(dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{0, 0}, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, NppiSize{0, 8}, off)));
  printf("  srcsize0_badmask  %s\n", es(call(dx, dstep, dy, dstep, dm, dstep, da, astep, roi, (NppiMaskSize)0, nppiNormL2, NPP_BORDER_REPLICATE, NppiSize{0, 8}, off)));
  printf("  offneg_badmask    %s\n", es(call(dx, dstep, dy, dstep, dm, dstep, da, astep, roi, (NppiMaskSize)0, nppiNormL2, NPP_BORDER_REPLICATE, sz, NppiPoint{-1, 1})));
  printf("  badnorm_badmask   %s\n", es(call(dx, dstep, dy, dstep, dm, dstep, da, astep, roi, (NppiMaskSize)0, (NppiNorm)7)));
  printf("  badnorm_badstep   %s\n", es(call(dx, 3, dy, dstep, dm, dstep, da, astep, roi, NPP_MASK_SIZE_3_X_3, (NppiNorm)7)));
  printf("  badmask_badstep   %s\n", es(call(dx, 3, dy, dstep, dm, dstep, da, astep, roi, (NppiMaskSize)0)));
  printf("  badborder_badstep %s\n", es(call(dx, 3, dy, dstep, dm, dstep, da, astep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_CONSTANT)));
  printf("  xonly_ok          %s\n", es(call(dx, dstep, nullptr, 0, nullptr, 0, nullptr, 0)));
  printf("  magonly_ok        %s\n", es(call(nullptr, 0, nullptr, 0, dm, dstep, nullptr, 0)));
  printf("  norminf_zeroroi   %s\n", es(call(dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{0, 0}, NPP_MASK_SIZE_3_X_3, nppiNormInf)));
  cudaFree(d); cudaFree(dx); cudaFree(dy); cudaFree(dm); cudaFree(da);
}

// ---------- 5x5 support per family + 32f C1 numeric ----------
static void misc() {
  const int W = 11, H = 11, step = W * 4, R = 5, dstep = R * 4;
  std::vector<Npp32f> h((size_t)W * H, 0.f);
  for (size_t i = 0; i < h.size(); ++i) h[i] = (float)((i * 29 + 3) & 0xff);
  Npp32f *d; cudaMalloc(&d, step * H); cudaMemcpy(d, h.data(), step * H, cudaMemcpyHostToDevice);
  Npp32f *dx, *dy, *dm; Npp32f *da;
  cudaMalloc(&dx, dstep * R); cudaMalloc(&dy, dstep * R); cudaMalloc(&dm, dstep * R); cudaMalloc(&da, dstep * R);
  const Npp32f *pSrc = d + 3 * (step / 4) + 3;
  GvFnF fs = nppiGradientVectorSobelBorder_32f_C1R_Ctx;
  GvFnF fp = nppiGradientVectorPrewittBorder_32f_C1R_Ctx;
  GvFnF fc = nppiGradientVectorScharrBorder_32f_C1R_Ctx;
  printf("== 32f C1R 5x5 mask support\n");
  printf("  sobel   %s\n", es(fs(pSrc, step, NppiSize{W, H}, NppiPoint{3, 3}, dx, dstep, dy, dstep, dm, dstep, da, dstep, NppiSize{R, R}, NPP_MASK_SIZE_5_X_5, nppiNormL2, NPP_BORDER_REPLICATE, ctx0())));
  printf("  prewitt %s\n", es(fp(pSrc, step, NppiSize{W, H}, NppiPoint{3, 3}, dx, dstep, dy, dstep, dm, dstep, da, dstep, NppiSize{R, R}, NPP_MASK_SIZE_5_X_5, nppiNormL2, NPP_BORDER_REPLICATE, ctx0())));
  printf("  scharr  %s\n", es(fc(pSrc, step, NppiSize{W, H}, NppiPoint{3, 3}, dx, dstep, dy, dstep, dm, dstep, da, dstep, NppiSize{R, R}, NPP_MASK_SIZE_5_X_5, nppiNormL2, NPP_BORDER_REPLICATE, ctx0())));
  printf("== 32f C1R 3x3 numeric (sobel L2, spot)\n");
  NppStatus st = fs(pSrc, step, NppiSize{W, H}, NppiPoint{3, 3}, dx, dstep, dy, dstep, dm, dstep, da, dstep, NppiSize{3, 3}, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, ctx0());
  std::vector<Npp32f> hx((size_t)dstep / 4 * 3), hy = hx, hm = hx, ha = hx;
  cudaMemcpy(hx.data(), dx, dstep * 3, cudaMemcpyDeviceToHost);
  cudaMemcpy(hy.data(), dy, dstep * 3, cudaMemcpyDeviceToHost);
  cudaMemcpy(hm.data(), dm, dstep * 3, cudaMemcpyDeviceToHost);
  cudaMemcpy(ha.data(), da, dstep * 3, cudaMemcpyDeviceToHost);
  printf("  (%s) X0=%.1f Y0=%.1f M0=%.2f A0=%.4f\n", es(st), hx[0], hy[0], hm[0], ha[0]);
  // CPU check pixel (0,0) => abs (3,3)
  int V[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}}, Hh[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
  float gx = 0, gy = 0;
  for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) {
    int xx = 3 + b - 1, yy = 3 + a - 1;
    float v = h[(size_t)yy * W + xx];
    gx += -V[a][b] * v; gy += Hh[a][b] * v;
  }
  printf("  cpu X0=%.1f Y0=%.1f M0=%.2f A0=%.4f\n", gx, gy, sqrtf(gx * gx + gy * gy), atan2(gy, gx));
  cudaFree(d); cudaFree(dx); cudaFree(dy); cudaFree(dm); cudaFree(da);
}

int main() {
  kern_impulse("sobel", nppiGradientVectorSobelBorder_8u16s_C1R_Ctx, NPP_MASK_SIZE_3_X_3);
  kern_impulse("prewitt", nppiGradientVectorPrewittBorder_8u16s_C1R_Ctx, NPP_MASK_SIZE_3_X_3);
  kern_impulse("scharr", nppiGradientVectorScharrBorder_8u16s_C1R_Ctx, NPP_MASK_SIZE_3_X_3);
  kern_impulse("sobel", nppiGradientVectorSobelBorder_8u16s_C1R_Ctx, NPP_MASK_SIZE_5_X_5);
  gv_c3c1("sobel", 0);
  gv_c3c1("sobel", 1);
  gv_c3c1("sobel", 2);
  gv_c3c1("scharr", 2);
  gv_errors();
  misc();
  printf("done\n");
  return 0;
}
