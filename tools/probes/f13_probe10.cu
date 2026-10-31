// f13 probe 10: 8u16s 5x5 L2 magnitude float-vs-double pipeline; 32f angle/mag.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

static unsigned lcg(unsigned &s) { s = s * 1664525u + 1013904223u; return s >> 8; }

int main() {
  NppStreamContext c{}; c.hStream = 0;
  const int W = 64, H = 64, OFF = 2, R = 60;   // ROI 60x60, 5x5 sobel fully interior
  const int step = W, dstep = R * 2, astep = R * 4;
  std::vector<Npp8u> img((size_t)W * H);
  unsigned seed = 12345;
  for (auto &v : img) v = (Npp8u)(lcg(seed) & 0xff);
  Npp8u *d; cudaMalloc(&d, step * H); cudaMemcpy(d, img.data(), step * H, cudaMemcpyHostToDevice);
  Npp16s *dx, *dy, *dm; Npp32f *da;
  cudaMalloc(&dx, dstep * H); cudaMalloc(&dy, dstep * H); cudaMalloc(&dm, dstep * H); cudaMalloc(&da, astep * H);
  const Npp8u *pSrc = d + OFF * step + OFF;
  const NppiSize sz{W, H}; const NppiPoint off{OFF, OFF}; const NppiSize roi{R, R};
  NppStatus st = nppiGradientVectorSobelBorder_8u16s_C1R_Ctx(pSrc, step, sz, off, dx, dstep, dy, dstep, dm, dstep, da, astep, roi, NPP_MASK_SIZE_5_X_5, nppiNormL2, NPP_BORDER_REPLICATE, c);
  printf("status %d\n", (int)st);
  std::vector<Npp16s> hx((size_t)dstep / 2 * R), hy((size_t)dstep / 2 * R), hm((size_t)dstep / 2 * R);
  std::vector<Npp32f> ha((size_t)astep / 4 * R);
  cudaMemcpy(hx.data(), dx, dstep * R, cudaMemcpyDeviceToHost);
  cudaMemcpy(hy.data(), dy, dstep * R, cudaMemcpyDeviceToHost);
  cudaMemcpy(hm.data(), dm, dstep * R, cudaMemcpyDeviceToHost);
  cudaMemcpy(ha.data(), da, astep * R, cudaMemcpyDeviceToHost);
  // 5x5 sobel taps
  int kv[25] = {-1,-2,0,2,1,-4,-8,0,8,4,-6,-12,0,12,6,-4,-8,0,8,4,-1,-2,0,2,1};
  int kh[25] = {-1,-4,-6,-4,-1,-2,-8,-12,-8,-2,0,0,0,0,0,2,8,12,8,2,1,4,6,4,1};
  int n_match_x = 0, n_match_y = 0, n_mag_f32 = 0, n_mag_dbl = 0, n_disc = 0, n_ang_f = 0, n_ang_d = 0;
  for (int y = 0; y < R; ++y)
    for (int x = 0; x < R; ++x) {
      int gx = 0, gy = 0;
      for (int a = 0; a < 5; ++a) {
        int ry = y + a - 2 + OFF; ry = ry < 0 ? 0 : (ry > H - 1 ? H - 1 : ry);
        for (int b = 0; b < 5; ++b) {
          int rx = x + b - 2 + OFF; rx = rx < 0 ? 0 : (rx > W - 1 ? W - 1 : rx);
          unsigned char p = img[(size_t)ry * step + rx];
          gx += -(kv[a * 5 + b]) * p;
          gy += kh[a * 5 + b] * p;
        }
      }
      size_t i = (size_t)y * (dstep / 2) + x, ia = (size_t)y * (astep / 4) + x;
      if (hx[i] == (Npp16s)gx) ++n_match_x;
      if (hy[i] == (Npp16s)gy) ++n_match_y;
      float fx = (float)gx, fy = (float)gy;
      float m32 = sqrtf(fx * fx + fy * fy);
      double mdd = sqrt((double)gx * gx + (double)gy * gy);
      int f32mag = (int)m32, dblmag = (int)mdd;
      if (hm[i] == f32mag) ++n_mag_f32;
      if (hm[i] == dblmag) ++n_mag_dbl;
      if (f32mag != dblmag) {
        ++n_disc;
        if (n_disc <= 6) printf("  disc (%d,%d) gx=%d gy=%d npp=%d f32=%d dbl=%d\n", x, y, gx, gy, hm[i], f32mag, dblmag);
      }
      float af = atan2f((float)gy, (float)gx);
      float ad = (float)atan2((double)gy, (double)gx);
      if (memcmp(&ha[ia], &af, 4) == 0) ++n_ang_f;
      if (memcmp(&ha[ia], &ad, 4) == 0) ++n_ang_d;
    }
  printf("X %d/%d Y %d/%d\n", n_match_x, R * R, n_match_y, R * R);
  printf("mag16s: f32-pipeline %d/%d double-pipeline %d/%d (discriminating pixels %d)\n",
         n_mag_f32, R * R, n_mag_dbl, R * R, n_disc);
  printf("angle: atan2f %d/%d double %d/%d\n", n_ang_f, R * R, n_ang_d, R * R);
  printf("done\n");
  return 0;
}
