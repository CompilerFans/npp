// Probe K: unsharp auto-radius (nRadius=0) rule at LARGE sigma.
// Background: at small sigma the rule looks like "largest r with t1(r)/S1(r) >= eps"
// (eps = 2^-8 / 2^-16 / 2^-15 / 1e-12 for 8u/16u/16s/32f) but the 8u breakpoint
// sequence diverges from that model for sigma >~ 20 (measured r bigger than model).
// part1 <dt> <lo> <hi> <n>: dense log-spaced sigma sweep -> "SW <dt> s=<hex> k=<k>"
// part2: exact f32-bit sigma breakpoints for large k targets (bracketed search)
// part3: timing of one GBS call at increasing sigma (cost scaling)
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>

static int g_dtype = 0;
static const char *g_dname[] = {"8u", "16u", "16s", "32f"};

static NppStatus gbs(float r, float s, int *n) {
  switch (g_dtype) {
  case 0: return nppiFilterUnsharpGetBufferSize_8u_C1R(r, s, n);
  case 1: return nppiFilterUnsharpGetBufferSize_16u_C1R(r, s, n);
  case 2: return nppiFilterUnsharpGetBufferSize_16s_C1R(r, s, n);
  default: return nppiFilterUnsharpGetBufferSize_32f_C1R(r, s, n);
  }
}
// returns k (kernel width) if size == 4*k*k, else negative
static int k_of(float sigma) {
  int n = -1;
  NppStatus st = gbs(0.0f, sigma, &n);
  if (st != 0) return -(100 + (int)st);
  if (n <= 0) return -1;
  long long k = (long long)(sqrt((double)n / 4.0) + 0.5);
  if (4 * k * k != n) return -2;
  return (int)k;
}
static double now_s() {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char **argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  cudaFree(0); // init ctx (GBS is host-side but be safe)
  int part = (argc > 1) ? atoi(argv[1]) : 3;

  if (part == 1) {
    int dt = (argc > 2) ? atoi(argv[2]) : 0;
    double lo = (argc > 3) ? atof(argv[3]) : 0.6;
    double hi = (argc > 4) ? atof(argv[4]) : 2000.0;
    int n = (argc > 5) ? atoi(argv[5]) : 400;
    g_dtype = dt;
    if (n < 1) n = 1;
    double lls = log(lo), lhs = log(hi);
    for (int i = 0; i < n; ++i) {
      double sg = (n == 1) ? lo : exp(lls + (lhs - lls) * i / (n - 1));
      float s = (float)sg;
      double t0 = now_s();
      int k = k_of(s);
      double dtms = (now_s() - t0) * 1e3;
      printf("SW %s s=%a k=%d ms=%.1f\n", g_dname[dt], s, k, dtms);
    }
  }
  if (part == 2) {
    struct T { int dt; int ktarget; double lo, hi; };
    // NOTE: NPP's auto-radius loop costs ~4.5ns*r^3 per call; keep targets small
    // enough that bracketed search (<= ~30 f32-bit steps) stays fast.
    T ts[] = {
      {0, 171, 58.0, 400.0},  {0, 201, 60.0, 700.0},  {0, 241, 70.0, 1300.0},
      {0, 301, 100.0, 2500.0},{0, 401, 150.0, 5000.0},{0, 801, 300.0, 20000.0},
      {1, 321, 30.0, 100.0},  {1, 641, 70.0, 220.0},  {1, 1281, 150.0, 700.0},
      {2, 321, 35.0, 120.0},  {2, 641, 75.0, 280.0},
      {3, 321, 18.0, 80.0},   {3, 641, 35.0, 160.0},
    };
    for (auto &t : ts) {
      g_dtype = t.dt;
      float lo = (float)t.lo, hi = (float)t.hi;
      int klo = k_of(lo), khi = k_of(hi);
      printf("BPK %s target=%d lo_s=%a k_lo=%d hi_s=%a k_hi=%d\n",
             g_dname[t.dt], t.ktarget, lo, klo, hi, khi);
      if (klo < 0 || khi < 0 || klo >= t.ktarget || khi < t.ktarget) { printf("BPK BADBRACKET\n"); continue; }
      uint32_t lb, hb; memcpy(&lb, &lo, 4); memcpy(&hb, &hi, 4);
      while (hb - lb > 1) {
        uint32_t mb = lb + (hb - lb) / 2;
        float ms; memcpy(&ms, &mb, 4);
        int k = k_of(ms);
        if (k >= t.ktarget) hb = mb; else lb = mb;
      }
      float bs; memcpy(&bs, &hb, 4);
      printf("BP2 %s ktarget=%d sigma=%a k=%d\n", g_dname[t.dt], t.ktarget, bs, k_of(bs));
    }
  }
  if (part == 3) {
    double sig[] = {10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120};
    for (int dt = 0; dt < 4; ++dt) {
      g_dtype = dt;
      for (double v : sig) {
        float s = (float)v;
        double t0 = now_s();
        int k = k_of(s);
        double ms = (now_s() - t0) * 1e3;
        printf("TIM %s s=%g k=%d ms=%.1f\n", g_dname[dt], v, k, ms);
      }
    }
  }
  return 0;
}
