// f13 probe 8: angle / magnitude numeric forensics for GradientVector.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

static const char *es(NppStatus s) { return s == 0 ? "0" : "ERR"; }
static unsigned bits(float f) { unsigned u; memcpy(&u, &f, 4); return u; }

int main() {
  NppStreamContext c{}; c.hStream = 0;
  const int W = 9, H = 9, OFF = 2, R = 5;
  const int step = W, dstep = R * 2, astep = R * 4;
  std::vector<Npp8u> img((size_t)W * H);
  for (size_t i = 0; i < img.size(); ++i) img[i] = (Npp8u)((i * 37 + 5) & 0xff);
  Npp8u *d; cudaMalloc(&d, step * H); cudaMemcpy(d, img.data(), step * H, cudaMemcpyHostToDevice);
  Npp16s *dx, *dy, *dm; Npp32f *da;
  cudaMalloc(&dx, dstep * H); cudaMalloc(&dy, dstep * H); cudaMalloc(&dm, dstep * H); cudaMalloc(&da, astep * H);
  const Npp8u *pSrc = d + OFF * step + OFF;
  const NppiSize sz{W, H}; const NppiPoint off{OFF, OFF}; const NppiSize roi{R, R};
  int hv[9] = {-1,-2,-1,0,0,0,1,2,1}, vv[9] = {-1,0,1,-2,0,2,-1,0,1};  // horiz / vert sobel3
  auto run16 = [&](const char *name, int *kh, int *kv, int xsign) {
    NppStatus st;
    if (!strcmp(name, "scharr")) {
      int sh[9] = {-3,-10,-3,0,0,0,3,10,3}, sv[9] = {-3,0,3,-10,0,10,-3,0,3};
      memcpy(kh, sh, 36); memcpy(kv, sv, 36);
      st = nppiGradientVectorScharrBorder_8u16s_C1R_Ctx(pSrc, step, sz, off, dx, dstep, dy, dstep, dm, dstep, da, astep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c);
    } else {
      st = nppiGradientVectorSobelBorder_8u16s_C1R_Ctx(pSrc, step, sz, off, dx, dstep, dy, dstep, dm, dstep, da, astep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c);
    }
    printf("== %s status %s\n", name, es(st));
    std::vector<Npp16s> hx((size_t)dstep / 2 * R), hy((size_t)dstep / 2 * R), hm((size_t)dstep / 2 * R);
    std::vector<Npp32f> ha((size_t)astep / 4 * R);
    cudaMemcpy(hx.data(), dx, dstep * R, cudaMemcpyDeviceToHost);
    cudaMemcpy(hy.data(), dy, dstep * R, cudaMemcpyDeviceToHost);
    cudaMemcpy(hm.data(), dm, dstep * R, cudaMemcpyDeviceToHost);
    cudaMemcpy(ha.data(), da, astep * R, cudaMemcpyDeviceToHost);
    int m_f = 0, m_d = 0, mg_f = 0, mg_d = 0;
    for (int y = 0; y < R; ++y)
      for (int x = 0; x < R; ++x) {
        size_t i = (size_t)y * (dstep / 2) + x, ia = (size_t)y * (astep / 4) + x;
        // CPU reference gradients
        int gx = 0, gy = 0;
        for (int a = 0; a < 3; ++a) {
          int ry = y + a - 1 + OFF; ry = ry < 0 ? 0 : (ry > H - 1 ? H - 1 : ry);
          for (int b = 0; b < 3; ++b) {
            int rx = x + b - 1 + OFF; rx = rx < 0 ? 0 : (rx > W - 1 ? W - 1 : rx);
            unsigned char p = img[(size_t)ry * step + rx];
            gx += xsign * p * kv[a * 3 + b];
            gy += p * kh[a * 3 + b];
          }
        }
        if (hx[i] != gx || hy[i] != gy) printf("  GRAD MISMATCH (%d,%d) npp %d,%d ref %d,%d\n", x, y, hx[i], hy[i], gx, gy);
        float af = atan2f((float)gy, (float)gx);
        float ad = (float)atan2((double)gy, (double)gx);
        float mf = sqrtf((float)gx * gx + (float)gy * gy);
        float md = (float)sqrt((double)gx * gx + (double)gy * gy);
        if (bits(ha[ia]) == bits(af)) ++m_f;
        if (bits(ha[ia]) == bits(ad)) ++m_d;
        if (hm[i] == (Npp16s)mf) ++mg_f;
        if (hm[i] == (Npp16s)md) ++mg_d;
        if (bits(ha[ia]) != bits(af))
          printf("  (%d,%d) gx=%d gy=%d npp=%.9g(%08x) atan2f=%.9g(%08x) atand=%.9g(%08x) |d|=%.3g\n",
                 x, y, gx, gy, ha[ia], bits(ha[ia]), af, bits(af), ad, bits(ad), ha[ia] - af);
      }
    printf("  angle: atan2f %d/%d double %d/%d ; mag: floor(sqrtf) %d/%d floor(dsqrt) %d/%d\n",
           m_f, R * R, m_d, R * R, mg_f, R * R, mg_d, R * R);
  };
  run16("sobel", hv, vv, -1);
  run16("scharr", hv, vv, +1);
  printf("== 32f C1R sobel forensics\n");
  {
    const int sstep = W * 4, dstep = R * 4;
    std::vector<Npp32f> imf((size_t)W * H);
    for (size_t i = 0; i < imf.size(); ++i) imf[i] = (float)((int)(i * 53 % 211) - 100) * 0.375f;
    Npp32f *s; cudaMalloc(&s, sstep * H); cudaMemcpy(s, imf.data(), sstep * H, cudaMemcpyHostToDevice);
    Npp32f *fx, *fy, *fm, *fa;
    cudaMalloc(&fx, dstep * H); cudaMalloc(&fy, dstep * H); cudaMalloc(&fm, dstep * H); cudaMalloc(&fa, dstep * H);
    const Npp32f *ps = s + OFF * (sstep / 4) + OFF;
    NppStatus st = nppiGradientVectorSobelBorder_32f_C1R_Ctx(ps, sstep, sz, off, fx, dstep, fy, dstep, fm, dstep, fa, dstep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c);
    printf("  status %s\n", es(st));
    std::vector<Npp32f> hx((size_t)dstep / 4 * R), hy((size_t)dstep / 4 * R), hm((size_t)dstep / 4 * R), ha((size_t)dstep / 4 * R);
    cudaMemcpy(hx.data(), fx, dstep * R, cudaMemcpyDeviceToHost);
    cudaMemcpy(hy.data(), fy, dstep * R, cudaMemcpyDeviceToHost);
    cudaMemcpy(hm.data(), fm, dstep * R, cudaMemcpyDeviceToHost);
    cudaMemcpy(ha.data(), fa, dstep * R, cudaMemcpyDeviceToHost);
    int m_x = 0, m_mf = 0, m_md = 0, m_af = 0, m_ad = 0;
    for (int y = 0; y < R; ++y)
      for (int x = 0; x < R; ++x) {
        size_t i = (size_t)y * (dstep / 4) + x;
        // CPU reference in float, row-major accumulation
        float gx = 0.f, gy = 0.f;
        for (int a = 0; a < 3; ++a) {
          int ry = y + a - 1 + OFF; ry = ry < 0 ? 0 : (ry > H - 1 ? H - 1 : ry);
          for (int b = 0; b < 3; ++b) {
            int rx = x + b - 1 + OFF; rx = rx < 0 ? 0 : (rx > W - 1 ? W - 1 : rx);
            float p = imf[(size_t)ry * W + rx];
            gx += -(float)(vv[a * 3 + b]) * p;
            gy += (float)(hv[a * 3 + b]) * p;
          }
        }
        if (bits(hx[i]) == bits(gx)) ++m_x;
        else if (m_x + 1 <= 3) printf("  (%d,%d) X npp=%.9g ref=%.9g\n", x, y, hx[i], gx);
        float mf = sqrtf(hx[i] * hx[i] + hy[i] * hy[i]);
        float md = (float)sqrt((double)hx[i] * hx[i] + (double)hy[i] * hy[i]);
        if (bits(hm[i]) == bits(mf)) ++m_mf;
        if (bits(hm[i]) == bits(md)) ++m_md;
        float af = atan2f(hy[i], hx[i]);
        float ad = (float)atan2((double)hy[i], (double)hx[i]);
        if (bits(ha[i]) == bits(af)) ++m_af;
        if (bits(ha[i]) == bits(ad)) ++m_ad;
      }
    printf("  X %d/%d mag: sqrtf %d/%d dsqrt %d/%d angle: atan2f %d/%d double %d/%d\n",
           m_x, R * R, m_mf, R * R, m_md, R * R, m_af, R * R, m_ad, R * R);
  }
  printf("done\n");
  return 0;
}
