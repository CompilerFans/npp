// Probe L: unsharp final gaps:
//  1) GBS degenerates (r<0, sigma<=0 with explicit r, tiny sigma auto) + exact k1->k3 float
//  2) emission dumps per dtype (rounding/saturation evidence): base w=1.2, w=100, w=-0.7, const
//  3) threshold semantics (32f ramp thr=0.5, 8u ramp thr=3)
//  4) Border auto(r=0) equivalence vs explicit radius; tiny-sigma auto behavior
//  5) AC4 alpha handling
//  6) tap dtype independence (r=2 sigma=1)
//  7) offset semantics full dump
//  8) extra validation-order combos (sigma/radius vs border/step/kernel-roi)
//  9) Border r=0 with sigma<=0 (hang risk: run with timeout, last)
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
static const char *g_dname[] = {"8u", "16u", "16s", "32f"};
static NppStatus gbs(float r, float s, int *n) {
  switch (g_dtype) {
  case 0: return nppiFilterUnsharpGetBufferSize_8u_C1R(r, s, n);
  case 1: return nppiFilterUnsharpGetBufferSize_16u_C1R(r, s, n);
  case 2: return nppiFilterUnsharpGetBufferSize_16s_C1R(r, s, n);
  default: return nppiFilterUnsharpGetBufferSize_32f_C1R(r, s, n);
  }
}
static Npp8u *g_buf = nullptr;

static NppStatus ush8(const Npp8u *s, int ss, NppiPoint off, Npp8u *d, int ds, NppiSize roi,
                      float r, float sg, float w, float thr, NppiBorderType b, Npp8u *buf) {
  return nppiFilterUnsharpBorder_8u_C1R_Ctx(s, ss, off, d, ds, roi, r, sg, w, thr, b, buf, mkctx());
}
static NppStatus ush32(const Npp32f *s, int ss, NppiPoint off, Npp32f *d, int ds, NppiSize roi,
                       float r, float sg, float w, float thr, NppiBorderType b, Npp8u *buf) {
  return nppiFilterUnsharpBorder_32f_C1R_Ctx(s, ss, off, d, ds, roi, r, sg, w, thr, b, buf, mkctx());
}
static uint32_t lcg_state = 12345;
static uint32_t lcg() { lcg_state = lcg_state * 1664525u + 1013904223u; return lcg_state; }

static void print_irow(const char *tag, const void *h, int dt, int row, int w) {
  printf("%s%d", tag, row);
  for (int x = 0; x < w; ++x) {
    long long v = 0;
    if (dt == 0) v = ((const Npp8u *)h)[row * w + x];
    else if (dt == 1) v = ((const Npp16u *)h)[row * w + x];
    else if (dt == 2) v = ((const Npp16s *)h)[row * w + x];
    else { float f = ((const Npp32f *)h)[row * w + x]; printf(" %a", f); continue; }
    printf(" %lld", v);
  }
  printf("\n");
}

// emission dump: WxH image of dtype dt, kernel r/sigma, weight w, threshold thr
static void emit_case(const char *tag, int dt, int W, int H, float r, float sg, float w, float thr,
                      int constant) {
  const int es = (dt == 0) ? 1 : (dt == 3) ? 4 : 2;
  const size_t n = (size_t)W * H;
  void *hs = malloc(n * es), *hd = malloc(n * es);
  lcg_state = 777;
  for (size_t i = 0; i < n; ++i) {
    uint32_t x = lcg();
    if (constant) x = 0x00404000; // -> 64/16384/-16384/1.71875-ish below
    if (dt == 0) ((Npp8u *)hs)[i] = constant ? (Npp8u)128 : (Npp8u)((x >> 16) & 0xFF);
    else if (dt == 1) ((Npp16u *)hs)[i] = constant ? (Npp16u)30000 : (Npp16u)((x >> 8) & 0xFFFF);
    else if (dt == 2) ((Npp16s *)hs)[i] = constant ? (Npp16s)-1000 : (Npp16s)((x >> 8) & 0xFFFF);
    else ((Npp32f *)hs)[i] = constant ? 1.5f : (float)((int)((x >> 8) & 0xFFFF) - 32768) * 0.003f;
  }
  memset(hd, 0, n * es);
  void *ds, *dd; CK(cudaMalloc(&ds, n * es)); CK(cudaMalloc(&dd, n * es));
  CK(cudaMemcpy(ds, hs, n * es, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dd, hd, n * es, cudaMemcpyHostToDevice));
  int step = W * es;
  NppStatus st;
  if (dt == 0) st = ush8((Npp8u *)ds, step, NppiPoint{0, 0}, (Npp8u *)dd, step, NppiSize{W, H}, r, sg, w, thr, NPP_BORDER_REPLICATE, g_buf);
  else if (dt == 1) st = nppiFilterUnsharpBorder_16u_C1R_Ctx((Npp16u *)ds, step, NppiPoint{0, 0}, (Npp16u *)dd, step, NppiSize{W, H}, r, sg, w, thr, NPP_BORDER_REPLICATE, g_buf, mkctx());
  else if (dt == 2) st = nppiFilterUnsharpBorder_16s_C1R_Ctx((Npp16s *)ds, step, NppiPoint{0, 0}, (Npp16s *)dd, step, NppiSize{W, H}, r, sg, w, thr, NPP_BORDER_REPLICATE, g_buf, mkctx());
  else st = ush32((Npp32f *)ds, step, NppiPoint{0, 0}, (Npp32f *)dd, step, NppiSize{W, H}, r, sg, w, thr, NPP_BORDER_REPLICATE, g_buf);
  CK(cudaMemcpy(hd, dd, n * es, cudaMemcpyDeviceToHost));
  printf("EMIT %s dt=%s W=%d H=%d r=%a s=%a w=%a thr=%a status=%d\n", tag, g_dname[dt], W, H, r, sg, w, thr, (int)st);
  for (int y = 0; y < H; ++y) print_irow("  S", hs, dt, y, W);
  for (int y = 0; y < H; ++y) print_irow("  D", hd, dt, y, W);
  // taps (dtype-independent? dump for reference)
  {
    float *t = (float *)malloc((size_t)(2 * r + 3) * (2 * r + 3) * 4);
    CK(cudaMemcpy(t, g_buf, (size_t)(2 * (int)r + 3) * (2 * (int)r + 3) * 4, cudaMemcpyDeviceToHost));
    int k = 2 * (int)ceilf(r) + 1;
    printf("  TAPS k=%d\n", k);
    for (int y = 0; y < k; ++y) {
      printf("  T%d", y);
      for (int x = 0; x < k; ++x) printf(" %a", t[y * k + x]);
      printf("\n");
    }
    free(t);
  }
  cudaFree(ds); cudaFree(dd); free(hs); free(hd);
}

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  cudaFree(0);
  CK(cudaMalloc(&g_buf, 1 << 22));
  int part = (argc > 1) ? atoi(argv[1]) : 1;
  const int W = 16, H = 8;

  if (part == 1) {
    printf("== part1 GBS degenerates + tiny sigma\n");
    for (int dt = 0; dt < 4; ++dt) {
      g_dtype = dt;
      struct { float r, s; const char *t; } cs[] = {
        {-1.0f, 1.0f, "rneg"}, {-0.5f, 1.0f, "rnegfrac"}, {2.0f, -1.0f, "rexpsigneg"},
        {2.0f, -0.5f, "rexpsignegfrac"},
      };
      for (auto &c : cs) { int n = -1; NppStatus st = gbs(c.r, c.s, &n); printf("DEG %s %s r=%a s=%a status=%d size=%d\n", g_dname[dt], c.t, c.r, c.s, (int)st, n); }
      float ss[] = {1e-5f, 1.01e-6f, 1.0000002e-6f, 1.0000001e-6f, 1e-6f, 9.9e-7f, 1e-7f, 1e-9f, 1e-20f, 1e-30f};
      for (float s : ss) { int n = -1; NppStatus st = gbs(0.0f, s, &n); printf("DEG %s auto s=%a status=%d size=%d\n", g_dname[dt], s, (int)st, n); }
    }
    // exact k1->k3 transition float (8u only; dtype-independent per pj_part3)
    g_dtype = 0;
    float lo = 1e-30f, hi = 1e-5f;
    uint32_t lb, hb; memcpy(&lb, &lo, 4); memcpy(&hb, &hi, 4);
    int klo = -1, khi = -1; gbs(0.0f, lo, &klo); gbs(0.0f, hi, &khi);
    printf("K13 lo=%a k=%d hi=%a k=%d\n", lo, klo, hi, khi);
    if (klo >= 0 && khi >= 36) {
      while (hb - lb > 1) {
        uint32_t mb = lb + (hb - lb) / 2; float ms; memcpy(&ms, &mb, 4);
        int n = -1; gbs(0.0f, ms, &n);
        if (n >= 36) hb = mb; else lb = mb;
      }
      float bs; memcpy(&bs, &hb, 4);
      int n = -1; gbs(0.0f, bs, &n);
      printf("K13 first_k3_sigma=%a size=%d\n", bs, n);
    }
  }
  if (part == 2) {
    printf("== part2 emission dumps (r=2 s=1 k=5)\n");
    for (int dt = 0; dt < 4; ++dt) {
      emit_case("base", dt, 24, 12, 2.0f, 1.0f, 1.2f, 0.0f, 0);
      emit_case("w100", dt, 24, 12, 2.0f, 1.0f, 100.0f, 0.0f, 0);
      emit_case("wneg", dt, 24, 12, 2.0f, 1.0f, -0.7f, 0.0f, 0);
      emit_case("const", dt, 12, 12, 2.0f, 1.0f, 1.2f, 0.0f, 1);
    }
  }
  if (part == 3) {
    printf("== part3 threshold semantics (ramp src)\n");
    // 32f ramp: src = x*0.25 - 3 -> blur smooth; delta sweep
    {
      const int Rw = 32, Rh = 8;
      Npp32f *hs = (Npp32f *)malloc(Rw * Rh * 4);
      for (int y = 0; y < Rh; ++y) for (int x = 0; x < Rw; ++x) hs[y * Rw + x] = x * 0.25f - 3.0f + (y & 1) * 0.05f;
      Npp32f *ds, *dd; CK(cudaMalloc(&ds, Rw * Rh * 4)); CK(cudaMalloc(&dd, Rw * Rh * 4));
      CK(cudaMemcpy(ds, hs, Rw * Rh * 4, cudaMemcpyHostToDevice));
      NppStatus st = ush32(ds, Rw * 4, NppiPoint{0, 0}, dd, Rw * 4, NppiSize{Rw, Rh}, 2.0f, 1.0f, 1.0f, 0.5f, NPP_BORDER_REPLICATE, g_buf);
      Npp32f *hd = (Npp32f *)malloc(Rw * Rh * 4);
      CK(cudaMemcpy(hd, dd, Rw * Rh * 4, cudaMemcpyDeviceToHost));
      printf("THR 32f w=1 thr=0.5 status=%d\n", (int)st);
      for (int y = 0; y < Rh; ++y) print_irow("  S", hs, 3, y, Rw);
      for (int y = 0; y < Rh; ++y) print_irow("  D", hd, 3, y, Rw);
      cudaFree(ds); cudaFree(dd); free(hs); free(hd);
    }
    // 8u ramp 0..255, thr=3 and thr=0
    {
      const int Rw = 32, Rh = 8;
      Npp8u *hs = (Npp8u *)malloc(Rw * Rh);
      for (int y = 0; y < Rh; ++y) for (int x = 0; x < Rw; ++x) hs[y * Rw + x] = (Npp8u)(x * 8 + y);
      Npp8u *ds, *dd; CK(cudaMalloc(&ds, Rw * Rh)); CK(cudaMalloc(&dd, Rw * Rh));
      CK(cudaMemcpy(ds, hs, Rw * Rh, cudaMemcpyHostToDevice));
      NppStatus st = ush8(ds, Rw, NppiPoint{0, 0}, dd, Rw, NppiSize{Rw, Rh}, 2.0f, 1.0f, 1.0f, 3.0f, NPP_BORDER_REPLICATE, g_buf);
      Npp8u *hd = (Npp8u *)malloc(Rw * Rh);
      CK(cudaMemcpy(hd, dd, Rw * Rh, cudaMemcpyDeviceToHost));
      printf("THR 8u w=1 thr=3 status=%d\n", (int)st);
      for (int y = 0; y < Rh; ++y) print_irow("  S", hs, 0, y, Rw);
      for (int y = 0; y < Rh; ++y) print_irow("  D", hd, 0, y, Rw);
      cudaFree(ds); cudaFree(dd); free(hs); free(hd);
    }
  }
  if (part == 4) {
    printf("== part4 Border auto(r=0) equivalence\n");
    for (int dt = 0; dt < 4; ++dt) {
      g_dtype = dt;
      for (float s : {1.3f, 5.0f, 0.05f}) {
        int n = -1; NppStatus st = gbs(0.0f, s, &n);
        if (st != 0 || n < 4) { printf("AUTOEQ %s s=%a gbs_status=%d size=%d SKIP\n", g_dname[dt], s, (int)st, n); continue; }
        int k = (int)(sqrtf((float)(n / 4)) + 0.5f);
        const int Wq = 40, Hq = 20;
        const int es = (dt == 0) ? 1 : (dt == 3) ? 4 : 2;
        size_t bytes = (size_t)Wq * Hq * es;
        void *hs = malloc(bytes); void *hdA = malloc(bytes); void *hdB = malloc(bytes);
        lcg_state = 999;
        for (size_t i = 0; i < Wq * Hq; ++i) {
          uint32_t x = lcg();
          if (dt == 0) ((Npp8u *)hs)[i] = (Npp8u)((x >> 16) & 0xFF);
          else if (dt == 1) ((Npp16u *)hs)[i] = (Npp16u)((x >> 8) & 0xFFFF);
          else if (dt == 2) ((Npp16s *)hs)[i] = (Npp16s)((x >> 8) & 0xFFFF);
          else ((Npp32f *)hs)[i] = (float)((int)((x >> 8) & 0xFFFF) - 32768) * 0.003f;
        }
        void *ds, *dA, *dB; CK(cudaMalloc(&ds, bytes)); CK(cudaMalloc(&dA, bytes)); CK(cudaMalloc(&dB, bytes));
        CK(cudaMemcpy(ds, hs, bytes, cudaMemcpyHostToDevice));
        int step = Wq * es;
        NppStatus sA, sB;
        if (dt == 0) {
          sA = ush8((Npp8u *)ds, step, NppiPoint{0, 0}, (Npp8u *)dA, step, NppiSize{Wq, Hq}, 0.0f, s, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf);
          sB = ush8((Npp8u *)ds, step, NppiPoint{0, 0}, (Npp8u *)dB, step, NppiSize{Wq, Hq}, (float)(k / 2), s, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf);
        } else if (dt == 1) {
          sA = nppiFilterUnsharpBorder_16u_C1R_Ctx((Npp16u *)ds, step, NppiPoint{0, 0}, (Npp16u *)dA, step, NppiSize{Wq, Hq}, 0.0f, s, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
          sB = nppiFilterUnsharpBorder_16u_C1R_Ctx((Npp16u *)ds, step, NppiPoint{0, 0}, (Npp16u *)dB, step, NppiSize{Wq, Hq}, (float)(k / 2), s, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
        } else if (dt == 2) {
          sA = nppiFilterUnsharpBorder_16s_C1R_Ctx((Npp16s *)ds, step, NppiPoint{0, 0}, (Npp16s *)dA, step, NppiSize{Wq, Hq}, 0.0f, s, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
          sB = nppiFilterUnsharpBorder_16s_C1R_Ctx((Npp16s *)ds, step, NppiPoint{0, 0}, (Npp16s *)dB, step, NppiSize{Wq, Hq}, (float)(k / 2), s, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
        } else {
          sA = ush32((Npp32f *)ds, step, NppiPoint{0, 0}, (Npp32f *)dA, step, NppiSize{Wq, Hq}, 0.0f, s, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf);
          sB = ush32((Npp32f *)ds, step, NppiPoint{0, 0}, (Npp32f *)dB, step, NppiSize{Wq, Hq}, (float)(k / 2), s, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf);
        }
        CK(cudaMemcpy(hdA, dA, bytes, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hdB, dB, bytes, cudaMemcpyDeviceToHost));
        int eq = memcmp(hdA, hdB, bytes) == 0;
        printf("AUTOEQ %s s=%a k=%d stA=%d stB=%d eq=%d\n", g_dname[dt], s, k, (int)sA, (int)sB, eq);
        cudaFree(ds); cudaFree(dA); cudaFree(dB); free(hs); free(hdA); free(hdB);
      }
    }
    // tiny sigma: auto k=1?  (large buffer first, then exact-size)
    for (float s : {1e-7f, 1e-6f, 1.0000001e-6f, 1.0000002e-6f, 1.01e-6f}) {
      g_dtype = 0;
      int n = -1; NppStatus st = gbs(0.0f, s, &n);
      const int Wq = 8, Hq = 8;
      Npp8u hs[Wq * Hq], hdA[Wq * Hq];
      for (int i = 0; i < Wq * Hq; ++i) hs[i] = (Npp8u)(i * 37 + 5);
      Npp8u *ds, *dA; CK(cudaMalloc(&ds, Wq * Hq)); CK(cudaMalloc(&dA, Wq * Hq));
      CK(cudaMemcpy(ds, hs, Wq * Hq, cudaMemcpyHostToDevice));
      NppStatus sA = ush8(ds, Wq, NppiPoint{0, 0}, dA, Wq, NppiSize{Wq, Hq}, 0.0f, s, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf);
      CK(cudaMemcpy(hdA, dA, Wq * Hq, cudaMemcpyDeviceToHost));
      int eqsrc = memcmp(hs, hdA, Wq * Hq) == 0;
      printf("TINY s=%a gbs_status=%d size=%d border_status=%d out_eq_src=%d\n", s, (int)st, n, (int)sA, eqsrc);
      if (st == 0 && n > 0 && n <= (1 << 22)) {
        Npp8u *b2; CK(cudaMalloc(&b2, n));
        NppStatus sB = ush8(ds, Wq, NppiPoint{0, 0}, dA, Wq, NppiSize{Wq, Hq}, 0.0f, s, 1.2f, 0.0f, NPP_BORDER_REPLICATE, b2);
        printf("TINY exact_buf=%d status=%d\n", n, (int)sB);
        cudaFree(b2);
      }
      cudaFree(ds); cudaFree(dA);
    }
  }
  if (part == 5) {
    printf("== part5 AC4 alpha\n");
    const int Wq = 10, Hq = 8;
    Npp8u hs[Wq * Hq * 4], hd[Wq * Hq * 4];
    lcg_state = 4242;
    for (int i = 0; i < Wq * Hq * 4; ++i) hs[i] = (Npp8u)((lcg() >> 16) & 0xFF);
    Npp8u *ds, *dd; CK(cudaMalloc(&ds, sizeof(hs))); CK(cudaMalloc(&dd, sizeof(hs)));
    CK(cudaMemcpy(ds, hs, sizeof(hs), cudaMemcpyHostToDevice));
    NppStatus st = nppiFilterUnsharpBorder_8u_AC4R_Ctx(ds, Wq * 4, NppiPoint{0, 0}, dd, Wq * 4, NppiSize{Wq, Hq}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
    CK(cudaMemcpy(hd, dd, sizeof(hs), cudaMemcpyDeviceToHost));
    printf("AC4 status=%d\n", (int)st);
    for (int y = 0; y < Hq; ++y) {
      printf("  S%d", y);
      for (int x = 0; x < Wq; ++x) printf(" (%d,%d,%d,%d)", hs[(y * Wq + x) * 4], hs[(y * Wq + x) * 4 + 1], hs[(y * Wq + x) * 4 + 2], hs[(y * Wq + x) * 4 + 3]);
      printf("\n");
    }
    for (int y = 0; y < Hq; ++y) {
      printf("  D%d", y);
      for (int x = 0; x < Wq; ++x) printf(" (%d,%d,%d,%d)", hd[(y * Wq + x) * 4], hd[(y * Wq + x) * 4 + 1], hd[(y * Wq + x) * 4 + 2], hd[(y * Wq + x) * 4 + 3]);
      printf("\n");
    }
    cudaFree(ds); cudaFree(dd);
  }
  if (part == 6) {
    printf("== part6 taps dtype independence (r=2 s=1)\n");
    for (int dt = 1; dt < 4; ++dt) {
      g_dtype = dt;
      int n = -1; gbs(2.0f, 1.0f, &n);
      const int Wq = 8, Hq = 8, es = (dt == 3) ? 4 : 2;
      void *ds, *dd; CK(cudaMalloc(&ds, Wq * Hq * es)); CK(cudaMalloc(&dd, Wq * Hq * es));
      CK(cudaMemset(ds, 64, Wq * Hq * es));
      NppStatus st;
      if (dt == 1) st = nppiFilterUnsharpBorder_16u_C1R_Ctx((Npp16u *)ds, Wq * 2, NppiPoint{0, 0}, (Npp16u *)dd, Wq * 2, NppiSize{Wq, Hq}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
      else if (dt == 2) st = nppiFilterUnsharpBorder_16s_C1R_Ctx((Npp16s *)ds, Wq * 2, NppiPoint{0, 0}, (Npp16s *)dd, Wq * 2, NppiSize{Wq, Hq}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx());
      else st = ush32((Npp32f *)ds, Wq * 4, NppiPoint{0, 0}, (Npp32f *)dd, Wq * 4, NppiSize{Wq, Hq}, 2.0f, 1.0f, 1.0f, 0.0f, NPP_BORDER_REPLICATE, g_buf);
      float t[25];
      CK(cudaMemcpy(t, g_buf, 100, cudaMemcpyDeviceToHost));
      printf("TAPDT %s status=%d\n", g_dname[dt], (int)st);
      for (int y = 0; y < 5; ++y) {
        printf("  T%d", y);
        for (int x = 0; x < 5; ++x) printf(" %a", t[y * 5 + x]);
        printf("\n");
      }
      cudaFree(ds); cudaFree(dd);
    }
  }
  if (part == 7) {
    printf("== part7 offset semantics (8u ROI 8x8 at off (2,1), src 14x12)\n");
    const int SW = 14, SH = 12, Wq = 8, Hq = 8;
    Npp8u hs[SW * SH], hd[Wq * Hq];
    lcg_state = 31337;
    for (int i = 0; i < SW * SH; ++i) hs[i] = (Npp8u)((lcg() >> 16) & 0xFF);
    Npp8u *ds, *dd; CK(cudaMalloc(&ds, sizeof(hs))); CK(cudaMalloc(&dd, sizeof(hd)));
    CK(cudaMemcpy(ds, hs, sizeof(hs), cudaMemcpyHostToDevice));
    NppStatus st = ush8(ds, SW, NppiPoint{2, 1}, dd, Wq, NppiSize{Wq, Hq}, 2.0f, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf);
    CK(cudaMemcpy(hd, dd, sizeof(hd), cudaMemcpyDeviceToHost));
    printf("OFF status=%d\n", (int)st);
    for (int y = 0; y < SH; ++y) print_irow("  S", hs, 0, y, SW);
    for (int y = 0; y < Hq; ++y) print_irow("  D", hd, 0, y, Wq);
    cudaFree(ds); cudaFree(dd);
  }
  if (part == 8) {
    printf("== part8 extra validation combos (8u C1, W=16 H=8)\n");
    Npp8u *ds, *dd; CK(cudaMalloc(&ds, W * H * 32)); CK(cudaMalloc(&dd, W * H * 32));
    CK(cudaMemset(ds, 100, W * H * 32));
    auto p = [&](const char *tag, NppStatus s) { printf("VAL %-24s %d\n", tag, (int)s); };
    auto u = [&](float r, float sg, NppiSize roi, int dstep, NppiBorderType b) {
      return ush8(ds, W, NppiPoint{0, 0}, dd, dstep, roi, r, sg, 1.2f, 0.0f, b, g_buf);
    };
    NppiSize roi{W, H};
    p("radneg_badborder", u(-1.0f, 1.0f, roi, W, (NppiBorderType)0));
    p("signeg_badborder", u(2.0f, -1.0f, roi, W, (NppiBorderType)0));
    p("signeg_stepsmall", u(2.0f, -1.0f, roi, 15, NPP_BORDER_REPLICATE));
    p("radneg_stepsmall", u(-1.0f, 1.0f, roi, 15, NPP_BORDER_REPLICATE));
    p("kgtroi_badborder", u(8.0f, 1.0f, roi, W, (NppiBorderType)0));
    p("kgtroi_stepsmall", u(8.0f, 1.0f, roi, 15, NPP_BORDER_REPLICATE));
    p("kgtroi_signeg", u(8.0f, -1.0f, roi, W, NPP_BORDER_REPLICATE));
    p("zeroroi_radneg", u(-1.0f, 1.0f, NppiSize{0, 0}, W, NPP_BORDER_REPLICATE));
    p("signeg_stepodd_ac4", nppiFilterUnsharpBorder_8u_AC4R_Ctx(ds, W * 4, NppiPoint{0, 0}, dd, W * 4 + 1, roi, 2.0f, -1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf, mkctx()));
    p("thr_nan", ush8(ds, W, NppiPoint{0, 0}, dd, W, roi, 2.0f, 1.0f, 1.2f, NAN, NPP_BORDER_REPLICATE, g_buf));
    p("w_nan", ush8(ds, W, NppiPoint{0, 0}, dd, W, roi, 2.0f, 1.0f, NAN, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("r_nan", ush8(ds, W, NppiPoint{0, 0}, dd, W, roi, NAN, 1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf));
    p("gbs_rnan_s1", [&]{ int n=-1; g_dtype=0; return gbs(NAN, 1.0f, &n); }());
    p("gbs_r1_snan", [&]{ int n=-1; g_dtype=0; NppStatus s = gbs(1.0f, NAN, &n); printf("VAL gbs_r1_snan_size=%d\n", n); return s; }());
    cudaFree(ds); cudaFree(dd);
  }
  if (part == 9) {
    printf("== part9 Border auto with sigma<=0 (HANG RISK)\n");
    Npp8u *ds, *dd; CK(cudaMalloc(&ds, W * H)); CK(cudaMalloc(&dd, W * H));
    CK(cudaMemset(ds, 100, W * H));
    printf("P9 before signeg r=0\n");
    NppStatus s1 = ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{W, H}, 0.0f, -1.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf);
    printf("P9 signeg_auto status=%d\n", (int)s1);
    NppStatus s2 = ush8(ds, W, NppiPoint{0, 0}, dd, W, NppiSize{W, H}, 0.0f, 0.0f, 1.2f, 0.0f, NPP_BORDER_REPLICATE, g_buf);
    printf("P9 sigzero_auto status=%d\n", (int)s2);
    cudaFree(ds); cudaFree(dd);
  }
  cudaFree(g_buf);
  return 0;
}
