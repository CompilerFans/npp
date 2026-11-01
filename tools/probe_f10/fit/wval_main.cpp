// Compares the real mcMPP f10 executors against NVIDIA NPP on the same inputs.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdlib>
#include "wval_common.h"

static unsigned long long rng_state = 0x123456789abcdefULL;
static unsigned long long nextr() {
  unsigned long long z = (rng_state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static int elem_size(int dtype) {
  if (dtype == 0) return 1;              // 8u
  if (dtype == 1 || dtype == 2) return 2; // 16u / 16s
  return 4;                               // 32f
}
static void fill(void *p, size_t count, int dtype) {
  for (size_t i = 0; i < count; ++i) {
    if (dtype == 0) ((unsigned char *)p)[i] = (unsigned char)(nextr() & 0xFF);
    else if (dtype == 1) ((unsigned short *)p)[i] = (unsigned short)(nextr() & 0xFFFF);
    else if (dtype == 2) ((short *)p)[i] = (short)(int)(nextr() & 0xFFFF);
    else ((float *)p)[i] = (float)((int)(nextr() % 2001) - 1000) / 250.0f;
  }
}

struct Res { int so, sn; long bad; double maxd; };

static Res compare(const void *a, const void *b, size_t count, int dtype) {
  Res r{0, 0, 0, 0};
  for (size_t i = 0; i < count; ++i) {
    double va = 0, vb = 0;
    if (dtype == 0) { va = ((const unsigned char *)a)[i]; vb = ((const unsigned char *)b)[i]; }
    else if (dtype == 1) { va = ((const unsigned short *)a)[i]; vb = ((const unsigned short *)b)[i]; }
    else if (dtype == 2) { va = ((const short *)a)[i]; vb = ((const short *)b)[i]; }
    else { va = ((const float *)a)[i]; vb = ((const float *)b)[i]; }
    double d = fabs(va - vb);
    if (d > r.maxd) r.maxd = d;
    bool ok = (dtype == 3) ? (d <= 1e-3 * (1.0 + fabs(vb))) : (d == 0.0);
    if (!ok) ++r.bad;
  }
  return r;
}

static void run(const char *tag, bool plain, bool advanced, int dtype, int channels, int ac4, int mask, int taps,
                int mw, int mh, int ax, int ay, float noise) {
  const int es = elem_size(dtype);
  const int px = es * channels;
  const int fixedTaps = mask == 200 ? 3 : (mask == 201 ? 5 : mask / 100 + 2);
  const int m = plain ? (advanced ? taps / 2 + 1 : fixedTaps / 2 + 1) : 0;
  const int W = plain ? 21 : 13, H = plain ? 17 : 11;
  const int bw = plain ? W + 2 * m : 17, bh = plain ? H + 2 * m : 15;
  const int offX = plain ? m : 2, offY = plain ? m : 3;
  const int sStep = bw * px + (plain ? 0 : px);  // padded source step for border
  const int dStep = W * px;
  const size_t se = (size_t)bh * sStep, de = (size_t)H * dStep;
  std::vector<unsigned char> hs(se), ho(de), hn(de), hk(4 * std::max(taps, 1));
  fill(hs.data(), se / es, dtype);
  fill(ho.data(), de / es, dtype);
  hn = ho;
  if (advanced) {
    const int r = taps / 2;
    double sig = 0.6 + 0.4 * r, sum = 0;
    for (int i = 0; i < taps; ++i) {
      double d = i - r;
      ((float *)hk.data())[i] = (float)exp(-(d * d) / (2 * sig * sig));
      sum += ((float *)hk.data())[i];
    }
    if (taps == 7) {  // asymmetric case
      for (int i = 0; i < taps; ++i) ((float *)hk.data())[i] = 0.1f * (i + 1);
    } else {
      for (int i = 0; i < taps; ++i) ((float *)hk.data())[i] = (float)(((float *)hk.data())[i] / sum);
    }
  }
  float nz[4] = {noise, noise * 0.5f, noise * 1.0f, noise * 0.25f};
  unsigned char *ds, *da, *db;
  float *dk;
  cudaMalloc(&ds, se); cudaMalloc(&da, de); cudaMalloc(&db, de); cudaMalloc(&dk, 4 * std::max(taps, 1));
  cudaMemcpy(ds, hs.data(), se, cudaMemcpyHostToDevice);
  cudaMemcpy(da, ho.data(), de, cudaMemcpyHostToDevice);
  cudaMemcpy(db, hn.data(), de, cudaMemcpyHostToDevice);
  if (advanced) cudaMemcpy(dk, hk.data(), 4 * taps, cudaMemcpyHostToDevice);
  const unsigned char *ps = ds + (size_t)offY * sStep + (size_t)offX * px;
  int so = 0, sn = 0;
  cudaError_t eo = cudaSuccess, en = cudaSuccess;
  if (advanced) {
    if (plain) {
      so = ours_adv(ps, sStep, da, dStep, W, H, taps, dk, dtype, channels, ac4);
      eo = cudaDeviceSynchronize();
      if (eo != cudaSuccess) cudaGetLastError();
      sn = npp_adv(ps, sStep, db, dStep, W, H, taps, dk, dtype, channels, ac4);
      en = cudaDeviceSynchronize();
    } else {
      so = ours_adv_border(ps, sStep, bw, bh, offX, offY, da, dStep, W, H, taps, dk, 2, dtype, channels, ac4);
      eo = cudaDeviceSynchronize();
      if (eo != cudaSuccess) cudaGetLastError();
      sn = npp_adv_border(ps, sStep, bw, bh, offX, offY, db, dStep, W, H, taps, dk, 2, dtype, channels, ac4);
      en = cudaDeviceSynchronize();
    }
  } else if (mask == 0) {
    so = ours_wiener(ps, sStep, bw, bh, offX, offY, da, dStep, W, H, mw, mh, ax, ay, nz, 2, dtype, channels, ac4);
    eo = cudaDeviceSynchronize();
    if (eo != cudaSuccess) cudaGetLastError();
    sn = npp_wiener(ps, sStep, bw, bh, offX, offY, db, dStep, W, H, mw, mh, ax, ay, nz, 2, dtype, channels, ac4);
    en = cudaDeviceSynchronize();
  } else if (plain) {
    so = ours_gauss(ps, sStep, da, dStep, W, H, mask, dtype, channels, ac4);
    eo = cudaDeviceSynchronize();
    if (eo != cudaSuccess) cudaGetLastError();
    sn = npp_gauss(ps, sStep, db, dStep, W, H, mask, dtype, channels, ac4);
    en = cudaDeviceSynchronize();
  } else {
    so = ours_gauss_border(ps, sStep, bw, bh, offX, offY, da, dStep, W, H, mask, 2, dtype, channels, ac4);
    eo = cudaDeviceSynchronize();
    if (eo != cudaSuccess) cudaGetLastError();
    sn = npp_gauss_border(ps, sStep, bw, bh, offX, offY, db, dStep, W, H, mask, 2, dtype, channels, ac4);
    en = cudaDeviceSynchronize();
  }
  if (en != cudaSuccess) cudaGetLastError();
  std::vector<unsigned char> oa(de), ob(de);
  cudaMemcpy(oa.data(), da, de, cudaMemcpyDeviceToHost);
  cudaMemcpy(ob.data(), db, de, cudaMemcpyDeviceToHost);
  Res r = compare(oa.data(), ob.data(), de / es, dtype);
  printf("%-42s st=%d/%d bad=%ld maxd=%.5g eo=%d en=%d\n", tag, ours_canon(so), npp_canon(sn), r.bad,
         r.maxd, (int)eo, (int)en);
  if (r.bad && getenv("WVAL_DUMP")) {
    int shown = 0;
    for (size_t i = 0; i < de / es && shown < 8; ++i) {
      double va = 0, vb = 0;
      if (dtype == 0) { va = oa[i]; vb = ob[i]; }
      else if (dtype == 1) { va = ((unsigned short *)oa.data())[i]; vb = ((unsigned short *)ob.data())[i]; }
      else if (dtype == 2) { va = ((short *)oa.data())[i]; vb = ((short *)ob.data())[i]; }
      else { va = ((float *)oa.data())[i]; vb = ((float *)ob.data())[i]; }
      bool ok = (dtype == 3) ? (fabs(va - vb) <= 1e-3 * (1.0 + fabs(vb))) : (va == vb);
      if (!ok) {
        printf("   i=%zu (px=%zu ch=%d) ours=%.9g npp=%.9g\n", i, i / channels, (int)(i % channels), va, vb);
        ++shown;
      }
    }
  }
  cudaFree(ds); cudaFree(da); cudaFree(db); cudaFree(dk);
}

static int errdiff = 0;
static void chk(const char *tag, int so, int sn) {
  int a = ours_canon(so), b = npp_canon(sn);
  if (a != b) {
    ++errdiff;
    printf("ERRDIFF %-40s ours=%d npp=%d (raw %d/%d)\n", tag, a, b, so, sn);
  }
}

static void error_matrix() {
  const int es = 2, px = es * 4;
  const int sStep = 16 * px, dStep = 8 * px;
  unsigned char *ds, *dd;
  float *nz, *dk;
  cudaMalloc(&ds, 16 * sStep);
  cudaMalloc(&dd, 8 * dStep);
  cudaMalloc(&nz, 16);
  cudaMalloc(&dk, 64 * 4);
  float hn[4] = {0.5f, 0.5f, 0.5f, 0.5f};
  cudaMemcpy(nz, hn, 16, cudaMemcpyHostToDevice);
  float hk[65];
  for (int i = 0; i < 65; ++i) hk[i] = 1.f / 65.f;
  cudaMemcpy(dk, hk, sizeof hk, cudaMemcpyHostToDevice);
  const void *s = ds;
  void *d = dd;
  // Wiener (8u C1): noise validation precedes everything
  float hn2[4] = {1.5f, 0.5f, 0.5f, 0.5f};
  float hn3[4] = {0.5f, 0.5f, 0.5f, 1.5f};
#define WIO(S, SSW, SW, SH, OX, OY, D, DSW, RW, RH, MW, MH, AX, AY, NZ, B, CH, A4)   chk("wiener", ours_wiener(S, SSW, SW, SH, OX, OY, D, DSW, RW, RH, MW, MH, AX, AY, NZ, B, 0, CH, A4),       npp_wiener(S, SSW, SW, SH, OX, OY, D, DSW, RW, RH, MW, MH, AX, AY, NZ, B, 0, CH, A4))
  WIO(s, sStep, 16, 16, 2, 2, d, dStep, 8, 8, 3, 3, 1, 1, hn, 2, 1, 0);
  WIO(nullptr, sStep, 16, 16, 2, 2, d, dStep, 8, 8, 3, 3, 1, 1, hn, 2, 1, 0);
  WIO(s, sStep, 16, 16, 2, 2, nullptr, dStep, 8, 8, 3, 3, 1, 1, hn, 2, 1, 0);
  WIO(s, sStep, 16, 16, 2, 2, d, dStep, 8, 8, 3, 3, 1, 1, hn2, 2, 1, 0);
  WIO(s, sStep, 16, 16, 2, 2, d, dStep, 8, 8, 3, 3, 1, 1, hn3, 2, 4, 1);
  WIO(s, sStep, 16, 16, 2, 2, d, dStep, 8, -1, 3, 3, 1, 1, hn, 2, 1, 0);
  WIO(s, sStep, 16, 16, 2, 2, d, dStep, 8, 8, 3, 3, 1, 1, hn2, 2, 1, 0);
  WIO(s, sStep, 0, 16, 2, 2, d, dStep, 8, 8, 3, 3, 1, 1, hn, 2, 1, 0);
  WIO(s, sStep, 16, 16, -1, 2, d, dStep, 8, 8, 3, 3, 1, 1, hn, 2, 1, 0);
  WIO(s, sStep, 16, 16, 16, 2, d, dStep, 8, 8, 3, 3, 1, 1, hn, 2, 1, 0);
  WIO(s, sStep, 16, 16, 2, 2, d, dStep, 8, 8, 3, 3, 1, 1, hn, 0, 1, 0);
  WIO(s, sStep, 16, 16, 2, 2, d, dStep, 8, 8, 0, 3, 0, 0, hn, 2, 1, 0);
  WIO(s, sStep, 16, 16, 2, 2, d, dStep, 8, 8, -2, 3, 0, 0, hn, 2, 1, 0);
  WIO(s, sStep, 16, 16, 2, 2, d, dStep, 8, 8, 3, 3, 3, 1, hn, 2, 1, 0);
  WIO(s, sStep, 16, 16, 2, 2, d, dStep, 8, 8, 3, 3, 1, 3, hn, 2, 1, 0);
  WIO(s, sStep, 16, 16, 2, 2, d, dStep, 8, 8, 3, 3, 2, 2, hn, 2, 1, 0);
  WIO(s, sStep, 16, 16, 2, 2, d, dStep, 0, 0, 3, 3, 1, 1, hn, 2, 1, 0);
  WIO(s, sStep, 16, 16, 2, 2, d, dStep, 0, 0, 3, 3, 9, 9, hn, 2, 1, 0);
  WIO(s, sStep, 16, 16, 2, 2, d, dStep, 0, 0, 3, 3, 1, 1, hn, 0, 1, 0);
  WIO(s, sStep, 16, 16, 2, 2, d, 3, 8, 8, 3, 3, 1, 1, hn, 2, 1, 0);
  WIO(s, sStep, 16, 16, 2, 2, d, dStep + 1, 8, 8, 3, 3, 1, 1, hn, 2, 1, 0);
#undef WIO
  // Gauss plain (8u C1 / 16u C1): status-only paths (no kernel launch)
#define GC(DT, S, SSW, D, DSW, RW, RH, M) \
  chk("gauss", ours_gauss(S, SSW, D, DSW, RW, RH, M, DT, 1, 0), npp_gauss(S, SSW, D, DSW, RW, RH, M, DT, 1, 0))
  GC(0, nullptr, sStep, d, dStep, 8, 8, 200);
  GC(0, s, sStep, nullptr, dStep, 8, 8, 200);
  GC(0, s, sStep, d, dStep, 8, -1, 200);
  GC(0, s, sStep, d, dStep, 8, 8, 0);
  GC(0, s, sStep, d, dStep, 8, 8, 900);
  GC(0, s, sStep, d, dStep, 8, 8, 100);
  GC(0, s, sStep, d, dStep, 0, 0, 200);
  GC(0, s, sStep, d, dStep, 0, 8, 200);
  GC(0, s, sStep, d, dStep, 0, 0, 0);
  GC(0, s, sStep, d, dStep, 0, 0, 700);
  GC(0, s, sStep, d, 3, 0, 0, 200);
  GC(0, s, sStep, d, 7, 8, 8, 200);
  GC(1, s, sStep, d, 17, 8, 8, 200);
  GC(1, s, sStep, d, 15, 8, 8, 200);
#undef GC
  // Advanced plain / border: status-only paths
  chk("adv nullsrc", ours_adv(nullptr, sStep, d, dStep, 8, 8, 3, dk, 0, 1, 0),
      npp_adv(nullptr, sStep, d, dStep, 8, 8, 3, dk, 0, 1, 0));
  chk("adv nulldst", ours_adv(s, sStep, nullptr, dStep, 8, 8, 3, dk, 0, 1, 0),
      npp_adv(s, sStep, nullptr, dStep, 8, 8, 3, dk, 0, 1, 0));
  chk("adv nullkernel", ours_adv(s, sStep, d, dStep, 8, 8, 3, nullptr, 0, 1, 0),
      npp_adv(s, sStep, d, dStep, 8, 8, 3, nullptr, 0, 1, 0));
  chk("adv negroi", ours_adv(s, sStep, d, dStep, 8, -1, 3, dk, 0, 1, 0),
      npp_adv(s, sStep, d, dStep, 8, -1, 3, dk, 0, 1, 0));
  for (int t : {-1, 0, 1, 2, 4, 6}) {
    char tg[32];
    snprintf(tg, sizeof tg, "adv taps%d", t);
    chk(tg, ours_adv(s, sStep, d, dStep, 8, 8, t, dk, 0, 1, 0),
        npp_adv(s, sStep, d, dStep, 8, 8, t, dk, 0, 1, 0));
    snprintf(tg, sizeof tg, "adv zroi taps%d", t);
    chk(tg, ours_adv(s, sStep, d, dStep, 0, 0, t, dk, 0, 1, 0),
        npp_adv(s, sStep, d, dStep, 0, 0, t, dk, 0, 1, 0));
  }
  for (int t : {3, 4, 5}) {
    char tg[32];
    snprintf(tg, sizeof tg, "adv zroi taps%d ok", t);
    chk(tg, ours_adv(s, sStep, d, dStep, 0, 0, t, dk, 0, 1, 0),
        npp_adv(s, sStep, d, dStep, 0, 0, t, dk, 0, 1, 0));
  }
  chk("adv zroi nullk", ours_adv(s, sStep, d, dStep, 0, 0, 3, nullptr, 0, 1, 0),
      npp_adv(s, sStep, d, dStep, 0, 0, 3, nullptr, 0, 1, 0));
  chk("adv zroi badstep", ours_adv(s, sStep, d, 3, 0, 0, 3, dk, 0, 1, 0),
      npp_adv(s, sStep, d, 3, 0, 0, 3, dk, 0, 1, 0));
  chk("advb nullk", ours_adv_border(s, sStep, 16, 16, 2, 2, d, dStep, 8, 8, 3, nullptr, 2, 0, 1, 0),
      npp_adv_border(s, sStep, 16, 16, 2, 2, d, dStep, 8, 8, 3, nullptr, 2, 0, 1, 0));
  chk("advb srcsize0", ours_adv_border(s, sStep, 0, 16, 2, 2, d, dStep, 8, 8, 3, dk, 2, 0, 1, 0),
      npp_adv_border(s, sStep, 0, 16, 2, 2, d, dStep, 8, 8, 3, dk, 2, 0, 1, 0));
  chk("advb offneg", ours_adv_border(s, sStep, 16, 16, -1, 2, d, dStep, 8, 8, 3, dk, 2, 0, 1, 0),
      npp_adv_border(s, sStep, 16, 16, -1, 2, d, dStep, 8, 8, 3, dk, 2, 0, 1, 0));
  chk("advb badborder", ours_adv_border(s, sStep, 16, 16, 2, 2, d, dStep, 8, 8, 3, dk, 0, 0, 1, 0),
      npp_adv_border(s, sStep, 16, 16, 2, 2, d, dStep, 8, 8, 3, dk, 0, 0, 1, 0));
  for (int t : {-1, 0, 1, 2, 4, 6}) {
    char tg[32];
    snprintf(tg, sizeof tg, "advb taps%d", t);
    chk(tg, ours_adv_border(s, sStep, 16, 16, 2, 2, d, dStep, 8, 8, t, dk, 2, 0, 1, 0),
        npp_adv_border(s, sStep, 16, 16, 2, 2, d, dStep, 8, 8, t, dk, 2, 0, 1, 0));
    snprintf(tg, sizeof tg, "advb zroi taps%d", t);
    chk(tg, ours_adv_border(s, sStep, 16, 16, 2, 2, d, dStep, 0, 0, t, dk, 2, 0, 1, 0),
        npp_adv_border(s, sStep, 16, 16, 2, 2, d, dStep, 0, 0, t, dk, 2, 0, 1, 0));
  }
  chk("advb offeq", ours_adv_border(s, sStep, 16, 16, 16, 2, d, dStep, 8, 8, 3, dk, 2, 0, 1, 0),
      npp_adv_border(s, sStep, 16, 16, 16, 2, d, dStep, 8, 8, 3, dk, 2, 0, 1, 0));
  chk("advb badborder_zroi", ours_adv_border(s, sStep, 16, 16, 2, 2, d, dStep, 0, 0, 3, dk, 0, 0, 1, 0),
      npp_adv_border(s, sStep, 16, 16, 2, 2, d, dStep, 0, 0, 3, dk, 0, 0, 1, 0));
  // Gauss border status paths
  chk("gb nullsrc", ours_gauss_border(nullptr, sStep, 16, 16, 2, 2, d, dStep, 8, 8, 200, 2, 0, 1, 0),
      npp_gauss_border(nullptr, sStep, 16, 16, 2, 2, d, dStep, 8, 8, 200, 2, 0, 1, 0));
  chk("gb srcsize0", ours_gauss_border(s, sStep, 0, 16, 2, 2, d, dStep, 8, 8, 200, 2, 0, 1, 0),
      npp_gauss_border(s, sStep, 0, 16, 2, 2, d, dStep, 8, 8, 200, 2, 0, 1, 0));
  chk("gb offneg", ours_gauss_border(s, sStep, 16, 16, -1, 2, d, dStep, 8, 8, 200, 2, 0, 1, 0),
      npp_gauss_border(s, sStep, 16, 16, -1, 2, d, dStep, 8, 8, 200, 2, 0, 1, 0));
  chk("gb badborder", ours_gauss_border(s, sStep, 16, 16, 2, 2, d, dStep, 8, 8, 200, 0, 0, 1, 0),
      npp_gauss_border(s, sStep, 16, 16, 2, 2, d, dStep, 8, 8, 200, 0, 0, 1, 0));
  chk("gb badmask", ours_gauss_border(s, sStep, 16, 16, 2, 2, d, dStep, 8, 8, 0, 2, 0, 1, 0),
      npp_gauss_border(s, sStep, 16, 16, 2, 2, d, dStep, 8, 8, 0, 2, 0, 1, 0));
  chk("gb zroi badmask", ours_gauss_border(s, sStep, 16, 16, 2, 2, d, dStep, 0, 0, 0, 2, 0, 1, 0),
      npp_gauss_border(s, sStep, 16, 16, 2, 2, d, dStep, 0, 0, 0, 2, 0, 1, 0));
  chk("gb zroi offneg", ours_gauss_border(s, sStep, 16, 16, 0, -1, d, dStep, 0, 0, 200, 2, 0, 1, 0),
      npp_gauss_border(s, sStep, 16, 16, 0, -1, d, dStep, 0, 0, 200, 2, 0, 1, 0));
  chk("gb zroi badborder", ours_gauss_border(s, sStep, 16, 16, 2, 2, d, dStep, 0, 0, 200, 0, 0, 1, 0),
      npp_gauss_border(s, sStep, 16, 16, 2, 2, d, dStep, 0, 0, 200, 0, 0, 1, 0));
  chk("gb zroi badstep", ours_gauss_border(s, sStep, 16, 16, 2, 2, d, 3, 0, 0, 200, 2, 0, 1, 0),
      npp_gauss_border(s, sStep, 16, 16, 2, 2, d, 3, 0, 0, 200, 2, 0, 1, 0));
  chk("gb stepodd", ours_gauss_border(s, sStep, 16, 16, 2, 2, d, 17, 8, 8, 200, 2, 1, 1, 0),
      npp_gauss_border(s, sStep, 16, 16, 2, 2, d, 17, 8, 8, 200, 2, 1, 1, 0));
  cudaFree(ds); cudaFree(dd); cudaFree(nz); cudaFree(dk);
}

int main() {
  const int masks[] = {200, 201, 400, 500, 800};
  for (int dtype = 0; dtype < 4; ++dtype)
    for (int ch = 0; ch < 3; ++ch) {
      const int channels = ch == 0 ? 1 : (ch == 1 ? 3 : 4);
      for (int ac4 = 0; ac4 <= (channels == 4 ? 1 : 0); ++ac4) {
        char tag[128];
        for (int mi = 0; mi < 5; ++mi) {
          int mask = masks[mi];
          snprintf(tag, sizeof tag, "gauss d%d c%d a%d m%d", dtype, channels, ac4, mask);
          run(tag, true, false, dtype, channels, ac4, mask, 0, 0, 0, 0, 0, 0.f);
          snprintf(tag, sizeof tag, "gaussB d%d c%d a%d m%d", dtype, channels, ac4, mask);
          run(tag, false, false, dtype, channels, ac4, mask, 0, 0, 0, 0, 0, 0.f);
        }
        for (int taps : {3, 5, 7, 9}) {
          snprintf(tag, sizeof tag, "adv d%d c%d a%d t%d", dtype, channels, ac4, taps);
          run(tag, true, true, dtype, channels, ac4, 0, taps, 0, 0, 0, 0, 0.f);
          snprintf(tag, sizeof tag, "advB d%d c%d a%d t%d", dtype, channels, ac4, taps);
          run(tag, false, true, dtype, channels, ac4, 0, taps, 0, 0, 0, 0, 0.f);
        }
      }
    }
  // Wiener: dtype 0=8u, 1=16s, 2=32f; mask 0 selects the Wiener path.
  const int wdtypes[] = {0, 1, 3};
  for (int dtype : wdtypes)
    for (int ch = 1; ch <= 4; ++ch) {
      const int channels = ch == 1 ? 1 : (ch == 2 ? 3 : 4);
      for (int ac4 = 0; ac4 <= (channels == 4 ? 1 : 0); ++ac4) {
        struct WC { int mw, mh, ax, ay; float n; } cases[] = {
            {3, 3, 1, 1, 1.f}, {3, 3, 0, 0, 0.1f}, {5, 5, 2, 2, 0.5f}, {4, 4, 1, 1, 0.25f},
            {7, 7, 3, 3, 0.1f}, {1, 3, 0, 1, 1.f}, {3, 1, 1, 0, 0.5f}, {1, 1, 0, 0, 1.f},
            {2, 1, 0, 0, 0.f}, {3, 5, 1, 2, 0.5f},
        };
        for (auto &w : cases) {
          char tag[128];
          snprintf(tag, sizeof tag, "wiener d%d c%d a%d m%dx%d", dtype, channels, ac4, w.mw, w.mh);
          run(tag, false, false, dtype, channels, ac4, 0, 0, w.mw, w.mh, w.ax, w.ay, w.n);
        }
      }
    }
  error_matrix();
  printf("STATUS DIFFS: %d\n", errdiff);
  return 0;
}
