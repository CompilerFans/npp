// Probe J: unsharp remaining gaps:
//  1) GBS degenerate (r, sigma) statuses
//  2) r-sweep at sigma=1 (k vs fractional radius)
//  3) sigma-breakpoints per dtype via float-bit binary search (auto-radius rule)
//  4) exact hex tap dumps
//  5) border-type matrices (REPLICATE/WRAP/MIRROR/CONSTANT/0/5)
//  6) oSrcOffset semantics
//  7) validation matrix (null/negroi/zeroroi/step/buffer/kernel-vs-roi/in-place)
//  8) layout independence of GBS
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>

#define CK(x) do { cudaError_t e_=(x); if(e_){printf("CUDAERR %s\n", cudaGetErrorString(e_)); exit(1);} } while(0)

static NppStreamContext mkctx() {
  NppStreamContext c{};
  c.hStream = 0;
  int dev = 0; CK(cudaGetDevice(&dev)); c.nCudaDeviceId = dev;
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
  c.nMultiProcessorCount = prop.multiProcessorCount;
  c.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  c.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
  c.nSharedMemPerBlock = prop.sharedMemPerBlock;
  return c;
}

static int g_dtype = 0; // 0=8u 1=16u 2=16s 3=32f
static NppStatus gbs(float r, float s, int *n) {
  switch (g_dtype) {
  case 0: return nppiFilterUnsharpGetBufferSize_8u_C1R(r, s, n);
  case 1: return nppiFilterUnsharpGetBufferSize_16u_C1R(r, s, n);
  case 2: return nppiFilterUnsharpGetBufferSize_16s_C1R(r, s, n);
  default: return nppiFilterUnsharpGetBufferSize_32f_C1R(r, s, n);
  }
}
static const char *g_dname[] = {"8u", "16u", "16s", "32f"};

static Npp8u *g_buf = nullptr;

static NppStatus ush8(const Npp8u *s, int ss, NppiPoint off, Npp8u *d, int ds, NppiSize roi,
                      float r, float sg, float w, float thr, NppiBorderType b, Npp8u *buf) {
  return nppiFilterUnsharpBorder_8u_C1R_Ctx(s, ss, off, d, ds, roi, r, sg, w, thr, b, buf, mkctx());
}
static NppStatus ush32(const Npp32f *s, int ss, NppiPoint off, Npp32f *d, int ds, NppiSize roi,
                       float r, float sg, float w, float thr, NppiBorderType b, Npp8u *buf) {
  return nppiFilterUnsharpBorder_32f_C1R_Ctx(s, ss, off, d, ds, roi, r, sg, w, thr, b, buf, mkctx());
}

static void dump_taps(const char *tag, int k) {
  int n = k * k;
  float *h = (float *)malloc(n * 4);
  CK(cudaMemcpy(h, g_buf, n * 4, cudaMemcpyDeviceToHost));
  printf("TAPS %s k=%d\n", tag, k);
  int full = (k <= 15) ? k : 0;
  for (int y = 0; y < (full ? k : 2); ++y) {
    int yy = (full ? y : (y == 0 ? 0 : k / 2));
    printf("  T %d", yy);
    for (int x = 0; x < k; ++x) printf(" %a", h[yy * k + x]);
    printf("\n");
  }
  if (!full) {
    printf("  T %d", k - 1);
    for (int x = 0; x < k; ++x) printf(" %a", h[(k - 1) * k + x]);
    printf("\n");
  }
  free(h);
}

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  int part = (argc > 1) ? atoi(argv[1]) : 0;
  CK(cudaMalloc(&g_buf, 1 << 22));
  const int SW = 160, SH = 160;
  Npp8u *d8; Npp32f *d32;
  CK(cudaMalloc(&d8, SW * SH));
  CK(cudaMalloc(&d32, SW * SH * 4));
  CK(cudaMemset(d8, 100, SW * SH));
  CK(cudaMemset(d32, 0, SW * SH * 4));

  if (part == 0 || part == 1) {
    printf("== part1 gbs degenerates\n");
    float rs[] = {0.0f, -1.0f, -0.5f, 0.5f, 1.0f, 1e6f, 1e30f};
    float ss[] = {0.0f, -1.0f, -0.5f, 1e-6f, 1.0f, 1e6f, 1e30f};
    for (int dt = 0; dt < 4; ++dt) {
      g_dtype = dt;
      for (float r : rs) for (float s : ss) {
        int n = -1; NppStatus st = gbs(r, s, &n);
        printf("DEG %s r=%a s=%a status=%d size=%d\n", g_dname[dt], r, s, (int)st, n);
      }
    }
  }
  if (part == 0 || part == 2) {
    printf("== part2 r-sweep sigma=1\n");
    float rs[] = {0.5f, 0.9999999f, 1.0f, 1.0000001f, 1.5f, 2.0f, 2.5f, 3.0f, 10.5f, 100.5f, 1000.5f, 16383.5f, 1e6f};
    g_dtype = 0;
    for (float r : rs) { int n = -1; NppStatus st = gbs(r, 1.0f, &n); printf("RSW r=%a size=%d status=%d\n", r, n, (int)st); }
  }
  if (part == 0 || part == 3) {
    printf("== part3 sigma breakpoints\n");
    for (int dt = 0; dt < 4; ++dt) {
      g_dtype = dt;
      int kprev = 0;
      float lo = 1e-6f;
      for (int target = 3; target <= 161; target += 2) {
        uint32_t lb, hb;
        memcpy(&lb, &lo, 4);
        float hi = 64.0f; memcpy(&hb, &hi, 4);
        int n = -1; NppStatus st = gbs(0.0f, hi, &n);
        if (st != 0 || n < 4 * target * target) { printf("BP %s target=%d NOT_REACHED\n", g_dname[dt], target); break; }
        while (hb - lb > 1) {
          uint32_t mb = lb + (hb - lb) / 2;
          float ms; memcpy(&ms, &mb, 4);
          n = -1; st = gbs(0.0f, ms, &n);
          if (st == 0 && n >= 4 * target * target) hb = mb; else lb = mb;
        }
        float bs; memcpy(&bs, &hb, 4);
        n = -1; gbs(0.0f, bs, &n);
        if (n / 4 != kprev) {
          printf("BP %s sigma=%a size=%d k=%d\n", g_dname[dt], bs, n, (int)sqrtf((float)(n / 4)));
          kprev = n / 4;
        }
        lo = bs;
        if (bs >= 63.9f) break;
      }
    }
  }
  if (part == 0 || part == 4) {
    printf("== part4 hex taps\n");
    struct C { int dt; float r, s; };
    C cs[] = {
      {0, 2.0f, 1.0f}, {0, 3.0f, 1.0f}, {0, 4.0f, 1.0f}, {0, 2.0f, 0.5f}, {0, 2.0f, 2.0f},
      {0, 5.0f, 3.0f}, {0, 12.0f, 5.0f}, {0, 35.0f, 5.0f}, {0, 0.0f, 1.0f}, {0, 0.0f, 2.0f},
      {0, 0.0f, 5.0f}, {0, 0.0f, 10.0f},
      {1, 12.0f, 5.0f}, {1, 0.0f, 5.0f}, {1, 0.0f, 1.0f},
      {2, 12.0f, 5.0f}, {2, 0.0f, 5.0f},
      {3, 12.0f, 5.0f}, {3, 0.0f, 5.0f}, {3, 0.0f, 1.0f}, {3, 0.0f, 10.0f},
    };
    for (auto &c : cs) {
      int need = -1; g_dtype = c.dt;
      NppStatus st = gbs(c.r, c.s, &need);
      if (st != 0 || need <= 0) { printf("TAPS dt=%s r=%g s=%g gbs_status=%d need=%d\n", g_dname[c.dt], c.r, c.s, (int)st, need); continue; }
      int k = (int)(sqrtf((float)(need / 4)) + 0.5f);
      if ((long long)k * k * 4 != need) { printf("TAPS dt=%s r=%g s=%g need=%d NOT k^2*4\n", g_dname[c.dt], c.r, c.s, need); continue; }
      int W = k + 8, H = k + 8;
      if (W > SW || H > SH) { printf("TAPS dt=%s r=%g s=%g SKIP too big k=%d\n", g_dname[c.dt], c.r, c.s, k); continue; }
      CK(cudaMemset(g_buf, 0x5A, 1 << 22));
      char tag[64]; snprintf(tag, 64, "%s_r%g_s%g", g_dname[c.dt], c.r, c.s);
      NppStatus us;
      if (c.dt == 0) us = ush8(d8, SW, NppiPoint{0, 0}, d8 + SW * SH / 2, SW, NppiSize{W, H}, c.r, c.s, 1.0f, 0.0f, NPP_BORDER_REPLICATE, g_buf);
      else if (c.dt == 3) us = ush32(d32, SW * 4, NppiPoint{0, 0}, d32 + SW * SH / 2, SW * 4, NppiSize{W, H}, c.r, c.s, 1.0f, 0.0f, NPP_BORDER_REPLICATE, g_buf);
      else {
        Npp16u *s16u = (Npp16u *)d32; Npp16s *s16s = (Npp16s *)d32;
        if (c.dt == 1) us = nppiFilterUnsharpBorder_16u_C1R_Ctx(s16u, SW * 2, NppiPoint{0, 0}, s16u + SW * SH / 2, SW * 2, NppiSize{W, H}, c.r, c.s, 1.0f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
        else us = nppiFilterUnsharpBorder_16s_C1R_Ctx(s16s, SW * 2, NppiPoint{0, 0}, s16s + SW * SH / 2, SW * 2, NppiSize{W, H}, c.r, c.s, 1.0f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
      }
      printf("TAPSCALL %s status=%d\n", tag, (int)us);
      if (us == 0) dump_taps(tag, k);
    }
  }
  if (part == 0 || part == 5) {
    printf("== part5 border matrices (32f delta at corner)\n");
    const int BW = 31, BH = 31, W = 21, H = 21, OFF = 5;
    Npp32f hs[BW * BH], hd[W * H];
    Npp32f *ds, *dd; CK(cudaMalloc(&ds, sizeof(hs))); CK(cudaMalloc(&dd, sizeof(hd)));
    struct BC { const char *name; NppiBorderType b; };
    BC bs[] = {{"rep", NPP_BORDER_REPLICATE}, {"wra", NPP_BORDER_WRAP}, {"mir", NPP_BORDER_MIRROR},
               {"con", NPP_BORDER_CONSTANT}, {"zero", (NppiBorderType)0}, {"five", (NppiBorderType)5}};
    for (auto &b : bs) {
      for (int i = 0; i < BW * BH; ++i) hs[i] = 0.0f;
      hs[OFF + 0 * BW] = 1.0f; // delta at ROI corner (0,0)
      for (int i = 0; i < W * H; ++i) hd[i] = -777.0f;
      CK(cudaMemcpy(ds, hs, sizeof(hs), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dd, hd, sizeof(hd), cudaMemcpyHostToDevice));
      NppStatus st = ush32(ds + OFF, BW * 4, NppiPoint{0, 0}, dd, W * 4, NppiSize{W, H}, 2.0f, 1.0f, 1.0f, 0.0f, b.b, g_buf);
      CK(cudaMemcpy(hd, dd, sizeof(hd), cudaMemcpyDeviceToHost));
      printf("BORD %s status=%d\n", b.name, (int)st);
      if (st == 0) for (int y = 0; y < 6; ++y) { printf("  B");
        for (int x = 0; x < 6; ++x) printf(" %.9g", hd[y * W + x]); printf("\n"); }
    }
    cudaFree(ds); cudaFree(dd);
  }
  if (part == 0 || part == 6) {
    printf("== part6 offset semantics (32f delta at corner)\n");
    const int BW = 31, BH = 31, W = 21, H = 21, OFF = 5;
    Npp32f hs[BW * BH], hd[W * H];
    Npp32f *ds, *dd; CK(cudaMalloc(&ds, sizeof(hs))); CK(cudaMalloc(&dd, sizeof(hd)));
    struct OC { const char *name; NppiPoint o; };
    OC os[] = {{"o00", {0, 0}}, {"o22", {2, 2}}, {"o50", {50, 50}}, {"oneg", {-2, -2}}, {"o1010", {10, 10}}};
    for (auto &o : os) {
      for (int i = 0; i < BW * BH; ++i) hs[i] = 0.0f;
      hs[OFF + 0 * BW] = 1.0f;
      for (int i = 0; i < W * H; ++i) hd[i] = -777.0f;
      CK(cudaMemcpy(ds, hs, sizeof(hs), cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dd, hd, sizeof(hd), cudaMemcpyHostToDevice));
      NppStatus st = ush32(ds + OFF, BW * 4, o.o, dd, W * 4, NppiSize{W, H}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, g_buf);
      CK(cudaMemcpy(hd, dd, sizeof(hd), cudaMemcpyDeviceToHost));
      printf("OFF %s (%d,%d) status=%d\n", o.name, o.o.x, o.o.y, (int)st);
      if (st == 0) for (int y = 0; y < 6; ++y) { printf("  B");
        for (int x = 0; x < 6; ++x) printf(" %.9g", hd[y * W + x]); printf("\n"); }
    }
    cudaFree(ds); cudaFree(dd);
  }
  if (part == 0 || part == 7) {
    printf("== part7 validation matrix (8u C1 unless noted)\n");
    const int W = 16, H = 8;
    Npp8u *ds, *dd; CK(cudaMalloc(&ds, W * H * 32)); CK(cudaMalloc(&dd, W * H * 32));
    CK(cudaMemset(ds, 100, W * H * 32));
    int need = 0; g_dtype = 0; gbs(2.0f, 1.0f, &need);
    printf("VAL need=%d\n", need);
    auto p = [&](const char *tag, NppStatus s) { printf("VAL %-26s %d\n", tag, (int)s); };
    p("ok", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("nullsrc", ush8(nullptr, W, NppiPoint{0, 0}, dd, W, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("nulldst", ush8(ds, W, NppiPoint{0, 0}, nullptr, W, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("nullbuf", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, nullptr));
    p("negroi_w", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{-4, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("negroi_h", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{W, -4}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("zeroroi", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{0, 0}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("zeroroi_w", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{0, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("zeroroi_h", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{W, 0}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("zeroroi_nullbuf", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{0, 0}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, nullptr));
    p("zeroroi_negroi", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{-4, 0}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("negroi_nullbuf", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{-4, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, nullptr));
    p("negroi_radneg", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{-4, H}, -1.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("radneg", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{W, H}, -1.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("radfrac", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{W, H}, 2.5f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("signeg", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{W, H}, 2.0f, -1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("sigzero", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{W, H}, 2.0f, 0.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("border_none", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, (NppiBorderType)0, g_buf));
    p("border_five", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, (NppiBorderType)5, g_buf));
    p("border_con", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_CONSTANT, g_buf));
    p("border_wra", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_WRAP, g_buf));
    p("border_mir", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_MIRROR, g_buf));
    p("zeroroi_badborder", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{0, 0}, 2.0f, 1.0f, 1.2f, 0.0f, (NppiBorderType)0, g_buf));
    p("negroi_badborder", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{-4, H}, 2.0f, 1.0f, 1.2f, 0.0f, (NppiBorderType)0, g_buf));
    p("offneg", ush8(ds, W, NppiPoint{-1, 0}, dd, W, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("offbig", ush8(ds, W, NppiPoint{40, 0}, dd, W, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("stepsmall", ush8(ds, W, NppiPoint{0, 0}, dd, 15, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("stepodd8u_c1", ush8(ds, W, NppiPoint{0, 0}, dd, 17, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("srcstepodd", ush8(ds, 17, NppiPoint{0, 0}, dd, W, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("kernelGTroi", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{W, H}, 8.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("kernelEQroi", ush8(ds, 5, NppiPoint{0, 0}, dd, 5, NppiSize{5, 5}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("w100", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{W, H}, 2.0f, 1.0f, 100.0f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("thrneg", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, -5.0f, NPP_BORDER_REPLICATE, g_buf));
    { Npp8u *b2; CK(cudaMalloc(&b2, need)); p("bufexact", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, b2)); cudaFree(b2); }
    { Npp8u *b2; CK(cudaMalloc(&b2, need - 4)); p("bufminus4", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, b2)); cudaFree(b2); }
    { Npp8u *b2; CK(cudaMalloc(&b2, 4)); p("buf4", ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, b2)); cudaFree(b2); }
    p("32f_stepodd", ush32((Npp32f *)ds, W * 4, NppiPoint{0, 0}, (Npp32f *)dd, W * 4 + 1, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("32f_stepsmall", ush32((Npp32f *)ds, W * 4, NppiPoint{0, 0}, (Npp32f *)dd, W * 4 - 4, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("16u_stepodd", nppiFilterUnsharpBorder_16u_C1R_Ctx((Npp16u *)ds, W * 2, NppiPoint{0, 0}, (Npp16u *)dd, W * 2 + 1, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx()));
    p("8u_ac4_stepodd", nppiFilterUnsharpBorder_8u_AC4R_Ctx(ds, W * 4, NppiPoint{0, 0}, dd, W * 4 + 1, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx()));
    p("8u_ac4_stepsmall", nppiFilterUnsharpBorder_8u_AC4R_Ctx(ds, W * 4, NppiPoint{0, 0}, dd, W * 4 - 1, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx()));
    {
      Npp8u h[W * H]; for (int i = 0; i < W * H; ++i) h[i] = (Npp8u)(i * 7);
      CK(cudaMemcpy(ds, h, W * H, cudaMemcpyHostToDevice));
      NppStatus st = ush8(ds, W, NppiPoint{0, 0}, ds, W, NppiSize{W, H}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf);
      CK(cudaMemcpy(h, ds, W * H, cudaMemcpyDeviceToHost));
      printf("VAL inplace status=%d out0=%d out1=%d outW=%d\n", (int)st, h[0], h[1], h[W - 1]);
    }
    cudaFree(ds); cudaFree(dd);
  }
  if (part == 0 || part == 8) {
    printf("== part8 GBS layout independence\n");
    g_dtype = 0;
    for (float s : {5.0f, 2.0f}) {
      int a = -1, b = -1, c = -1, d = -1;
      nppiFilterUnsharpGetBufferSize_8u_C1R(0.0f, s, &a);
      nppiFilterUnsharpGetBufferSize_8u_C3R(0.0f, s, &b);
      nppiFilterUnsharpGetBufferSize_8u_C4R(0.0f, s, &c);
      nppiFilterUnsharpGetBufferSize_8u_AC4R(0.0f, s, &d);
      printf("LAY 8u s=%g c1=%d c3=%d c4=%d ac4=%d\n", s, a, b, c, d);
    }
  }
  cudaFree(d8); cudaFree(d32); cudaFree(g_buf);
  return 0;
}
