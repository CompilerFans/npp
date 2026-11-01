// f13 probe 7: clean-process re-verification of GV kernel signs, C3C1R
// semantics, magnitude/angle bit-exactness, and C3C1R launch statuses.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

static const char *es(NppStatus s) { switch (s) {
  case NPP_NO_ERROR: return "0"; case NPP_NULL_POINTER_ERROR: return "-8 null";
  case NPP_SIZE_ERROR: return "-6 size"; case NPP_OUT_OFF_RANGE_ERROR: return "-11 offrange";
  case NPP_NOT_SUPPORTED_MODE_ERROR: return "-14 mode"; case NPP_STEP_ERROR: return "-16 step";
  case NPP_NOT_EVEN_STEP_ERROR: return "-108 evenstep"; case NPP_MASK_SIZE_ERROR: return "-33 mask";
  case NPP_BAD_ARGUMENT_ERROR: return "-5 badarg"; default: { static char b[32]; snprintf(b,32,"%d",(int)s); return b; } } }

struct K { int n; int v[25]; int h[25]; };  // v/h = Vert/Horiz taps, row-major, top row first

static K kernels(const char *op) {
  K k{};
  if (!strcmp(op, "sobel3")) {
    k.n = 3;
    int v[9] = {-1,0,1,-2,0,2,-1,0,1}, h[9] = {-1,-2,-1,0,0,0,1,2,1};
    memcpy(k.v, v, sizeof v); memcpy(k.h, h, sizeof h);
  } else if (!strcmp(op, "prewitt3")) {
    k.n = 3;
    int v[9] = {-1,0,1,-1,0,1,-1,0,1}, h[9] = {-1,-1,-1,0,0,0,1,1,1};
    memcpy(k.v, v, sizeof v); memcpy(k.h, h, sizeof h);
  } else if (!strcmp(op, "scharr3")) {
    k.n = 3;
    int v[9] = {-3,0,3,-10,0,10,-3,0,3}, h[9] = {-3,-10,-3,0,0,0,3,10,3};
    memcpy(k.v, v, sizeof v); memcpy(k.h, h, sizeof h);
  } else {  // sobel5
    k.n = 5;
    int v[25] = {-1,-2,0,2,1,-4,-8,0,8,4,-6,-12,0,12,6,-4,-8,0,8,4,-1,-2,0,2,1};
    int h[25] = {-1,-4,-6,-4,-1,-2,-8,-12,-8,-2,0,0,0,0,0,2,8,12,8,2,1,4,6,4,1};
    memcpy(k.v, v, sizeof v); memcpy(k.h, h, sizeof h);
  }
  return k;
}

// CPU reference. xsign: X = xsign * Vert-response; Y = +Horiz-response.
// Border replicate clamp in pSrc-relative coords.
static void ref(const std::vector<Npp8u> &img, int W, int H, int step, int offX, int offY,
                int roiW, int roiH, const K &k, int xsign, int nch,
                std::vector<float> &gx, std::vector<float> &gy) {
  gx.assign((size_t)roiW * roiH * (nch == 3 ? 3 : 1), 0.f);
  gy.assign((size_t)roiW * roiH * (nch == 3 ? 3 : 1), 0.f);
  const int r = k.n >> 1;
  for (int y = 0; y < roiH; ++y)
    for (int x = 0; x < roiW; ++x)
      for (int c = 0; c < (nch == 3 ? 3 : 1); ++c) {
        int sv = 0, sh = 0;
        for (int a = 0; a < k.n; ++a) {
          int ry = y + a - r + offY;  // src-relative coords
          ry = ry < 0 ? 0 : (ry > H - 1 ? H - 1 : ry);
          for (int b = 0; b < k.n; ++b) {
            int rx = x + b - r + offX;
            rx = rx < 0 ? 0 : (rx > W - 1 ? W - 1 : rx);
            // img is the FULL image; pSrc = img + offY*step + offX*nch
            // so absolute pixel = pSrc-rel (rx-offX, ry-offY) => img row ry, col rx
            unsigned char p = img[(size_t)ry * step + (size_t)rx * nch + c];
            sv += p * k.v[a * k.n + b];
            sh += p * k.h[a * k.n + b];
          }
        }
        size_t idx = ((size_t)y * roiW + x) * (nch == 3 ? 3 : 1) + c;
        gx[idx] = (float)(xsign * sv);
        gy[idx] = (float)sh;
      }
}

int main() {
  NppStreamContext c{}; c.hStream = 0;
  const int W = 9, H = 9, OFF = 2, R = 5;   // pSrc inset (2,2), ROI 5x5
  const int step = W, dstep = R * 2, astep = R * 4;
  std::vector<Npp8u> img((size_t)W * H);
  for (size_t i = 0; i < img.size(); ++i) img[i] = (Npp8u)((i * 37 + 5) & 0xff);
  Npp8u *d; cudaMalloc(&d, step * H); cudaMemcpy(d, img.data(), step * H, cudaMemcpyHostToDevice);
  Npp16s *dx, *dy, *dm; Npp32f *da;
  cudaMalloc(&dx, dstep * H); cudaMalloc(&dy, dstep * H); cudaMalloc(&dm, dstep * H); cudaMalloc(&da, astep * H);
  const Npp8u *pSrc = d + OFF * step + OFF;
  typedef NppStatus (*F)(const Npp8u*, int, NppiSize, NppiPoint, Npp16s*, int, Npp16s*, int, Npp16s*, int, Npp32f*, int, NppiSize, NppiMaskSize, NppiNorm, NppiBorderType, NppStreamContext);
  F sob = nppiGradientVectorSobelBorder_8u16s_C1R_Ctx;
  F pre = nppiGradientVectorPrewittBorder_8u16s_C1R_Ctx;
  F sch = nppiGradientVectorScharrBorder_8u16s_C1R_Ctx;
  const NppiSize sz{W, H}; const NppiPoint off{OFF, OFF}; const NppiSize roi{R, R};
  printf("== kernel sign check (C1R, 8u16s)\n");
  const char *ops[4] = {"sobel3", "prewitt3", "scharr3", "sobel5"};
  for (int oi = 0; oi < 4; ++oi) {
    F f = oi == 0 ? sob : (oi == 1 ? pre : sch);
    NppiMaskSize mk = oi == 3 ? NPP_MASK_SIZE_5_X_5 : NPP_MASK_SIZE_3_X_3;
    NppStatus st = f(pSrc, step, sz, off, dx, dstep, dy, dstep, dm, dstep, da, astep, roi, mk, nppiNormL2, NPP_BORDER_REPLICATE, c);
    if (st != 0) { printf("  %-8s status %s\n", ops[oi], es(st)); continue; }
    std::vector<Npp16s> hx((size_t)dstep / 2 * R), hy((size_t)dstep / 2 * R), hm((size_t)dstep / 2 * R);
    std::vector<Npp32f> ha((size_t)astep / 4 * R);
    cudaMemcpy(hx.data(), dx, dstep * R, cudaMemcpyDeviceToHost);
    cudaMemcpy(hy.data(), dy, dstep * R, cudaMemcpyDeviceToHost);
    cudaMemcpy(hm.data(), dm, dstep * R, cudaMemcpyDeviceToHost);
    cudaMemcpy(ha.data(), da, astep * R, cudaMemcpyDeviceToHost);
    K k = kernels(ops[oi]);
    int xsign = oi == 2 ? +1 : -1;   // measured hypothesis: scharr X = +Vert
    std::vector<float> gx, gy;
    ref(img, W, H, step, OFF, OFF, R, R, k, xsign, 1, gx, gy);
    int mx = 0, my = 0, mm = 0, ma = 0, mmt = 0, mr = 0;
    int ulp = 0;
    for (int y = 0; y < R; ++y)
      for (int x = 0; x < R; ++x) {
        size_t i = (size_t)y * (dstep / 2) + x, j = (size_t)y * R + x;
        if (hx[i] == (Npp16s)gx[j]) ++mx;
        if (hy[i] == (Npp16s)gy[j]) ++my;
        float l2 = sqrtf(gx[j] * gx[j] + gy[j] * gy[j]);
        if (hm[i] == (Npp16s)l2) ++mm;
        if (hm[i] == (Npp16s)floorf(l2)) ++mmt;
        if (hm[i] == (Npp16s)(l2 >= 0 ? floorf(l2 + 0.5f) : -floorf(-l2 + 0.5f))) ++mr;
        float a = atan2f(gy[j], gx[j]);
        if (memcmp(&ha[(size_t)y * (astep / 4) + x], &a, 4) == 0) ++ma; else {
          int d = (int)(ha[(size_t)y * (astep / 4) + x] - a);
          if (d > ulp) ulp = d;
        }
      }
    printf("  %-8s X %d/%d Y %d/%d magL2 %d/%d (floor %d round %d) angle-bits %d/%d ulp<= %d\n",
           ops[oi], mx, R * R, my, R * R, mm, R * R, mmt, mr, ma, R * R, ulp);
  }
  printf("== C3C1R (8u16s)\n");
  {
    const int step3 = W * 3;
    std::vector<Npp8u> img3((size_t)step3 * H);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        for (int ch = 0; ch < 3; ++ch)
          img3[(size_t)y * step3 + (size_t)x * 3 + ch] = (Npp8u)(((size_t)(y * W + x) * (37 + ch * 25) + 5 + ch * 61) & 0xff);
    Npp8u *d3; cudaMalloc(&d3, step3 * H); cudaMemcpy(d3, img3.data(), step3 * H, cudaMemcpyHostToDevice);
    typedef NppStatus (*G)(const Npp8u*, int, NppiSize, NppiPoint, Npp16s*, int, Npp16s*, int, Npp16s*, int, Npp32f*, int, NppiSize, NppiMaskSize, NppiNorm, NppiBorderType, NppStreamContext);
    G g = nppiGradientVectorSobelBorder_8u16s_C3C1R_Ctx;
    const Npp8u *ps3 = d3 + OFF * step3 + OFF * 3;
    printf("  c3c1 3x3 L2 status %s\n", es(g(ps3, step3, sz, off, dx, dstep, dy, dstep, dm, dstep, da, astep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c)));
    std::vector<Npp16s> hx((size_t)dstep / 2 * R), hy((size_t)dstep / 2 * R), hm((size_t)dstep / 2 * R);
    std::vector<Npp32f> ha((size_t)astep / 4 * R);
    cudaMemcpy(hx.data(), dx, dstep * R, cudaMemcpyDeviceToHost);
    cudaMemcpy(hy.data(), dy, dstep * R, cudaMemcpyDeviceToHost);
    cudaMemcpy(hm.data(), dm, dstep * R, cudaMemcpyDeviceToHost);
    cudaMemcpy(ha.data(), da, astep * R, cudaMemcpyDeviceToHost);
    K k = kernels("sobel3");
    std::vector<float> gx, gy;
    ref(img3, W, H, step3, OFF, OFF, R, R, k, -1, 3, gx, gy);
    int okx = 0, oky = 0, okm = 0, oka = 0;
    for (int y = 0; y < R; ++y)
      for (int x = 0; x < R; ++x) {
        size_t j = (size_t)y * R + x;
        int sel = 0; float best = -1.f;
        for (int ch = 0; ch < 3; ++ch) {
          float l2 = gx[j * 3 + ch] * gx[j * 3 + ch] + gy[j * 3 + ch] * gy[j * 3 + ch];
          if (l2 > best) { best = l2; sel = ch; }
        }
        float X = gx[j * 3 + sel], Y = gy[j * 3 + sel];
        size_t i = (size_t)y * (dstep / 2) + x;
        if (hx[i] == (Npp16s)X) ++okx;
        if (hy[i] == (Npp16s)Y) ++oky;
        if (hm[i] == (Npp16s)sqrtf(X * X + Y * Y)) ++okm;
        float a = atan2f(Y, X);
        if (memcmp(&ha[(size_t)y * (astep / 4) + x], &a, 4) == 0) ++oka;
      }
    printf("  c3c1 L2 sel: X %d/%d Y %d/%d mag %d/%d angle %d/%d\n", okx, R * R, oky, R * R, okm, R * R, oka, R * R);
    // L1 with the same selection rule
    printf("  c3c1 3x3 L1 status %s\n", es(g(ps3, step3, sz, off, dx, dstep, dy, dstep, dm, dstep, da, astep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL1, NPP_BORDER_REPLICATE, c)));
    cudaMemcpy(hm.data(), dm, dstep * R, cudaMemcpyDeviceToHost);
    int okm1 = 0;
    for (int y = 0; y < R; ++y)
      for (int x = 0; x < R; ++x) {
        size_t j = (size_t)y * R + x;
        int sel = 0; float best = -1.f;
        for (int ch = 0; ch < 3; ++ch) {
          float l2 = gx[j * 3 + ch] * gx[j * 3 + ch] + gy[j * 3 + ch] * gy[j * 3 + ch];
          if (l2 > best) { best = l2; sel = ch; }
        }
        float X = gx[j * 3 + sel], Y = gy[j * 3 + sel];
        size_t i = (size_t)y * (dstep / 2) + x;
        if (hm[i] == (Npp16s)(fabsf(X) + fabsf(Y))) ++okm1;
      }
    printf("  c3c1 L1 sel: mag %d/%d\n", okm1, R * R);
    printf("  c3c1 5x5 status %s\n", es(g(ps3, step3, sz, off, dx, dstep, dy, dstep, dm, dstep, da, astep, roi, NPP_MASK_SIZE_5_X_5, nppiNormL2, NPP_BORDER_REPLICATE, c)));
    printf("  c3c1 srcstep9 status %s\n", es(g(ps3, 9, sz, off, dx, dstep, dy, dstep, dm, dstep, da, astep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c)));
    printf("  c3c1 prewitt status %s\n", es(nppiGradientVectorPrewittBorder_8u16s_C3C1R_Ctx(ps3, step3, sz, off, dx, dstep, dy, dstep, dm, dstep, da, astep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c)));
    printf("  c3c1 scharr status %s\n", es(nppiGradientVectorScharrBorder_8u16s_C3C1R_Ctx(ps3, step3, sz, off, dx, dstep, dy, dstep, dm, dstep, da, astep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c)));
    // 16u32f / 16s32f / 32f C3C1R
    printf("  c3c1 16u32f status %s\n", es(nppiGradientVectorSobelBorder_16u32f_C3C1R_Ctx((const Npp16u*)ps3, step3, sz, off, (Npp32f*)dx, dstep * 2, (Npp32f*)dy, dstep * 2, (Npp32f*)dm, dstep * 2, da, astep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c)));
    printf("  c3c1 16s32f status %s\n", es(nppiGradientVectorSobelBorder_16s32f_C3C1R_Ctx((const Npp16s*)ps3, step3, sz, off, (Npp32f*)dx, dstep * 2, (Npp32f*)dy, dstep * 2, (Npp32f*)dm, dstep * 2, da, astep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c)));
    printf("  c3c1 32f status %s\n", es(nppiGradientVectorSobelBorder_32f_C3C1R_Ctx((const Npp32f*)ps3, step3, sz, off, (Npp32f*)dx, dstep * 2, (Npp32f*)dy, dstep * 2, (Npp32f*)dm, dstep * 2, da, astep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c)));
    printf("  alive %s\n", es(sob(pSrc, step, sz, off, dx, dstep, dy, dstep, dm, dstep, da, astep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c)));
  }
  printf("done\n");
  return 0;
}
