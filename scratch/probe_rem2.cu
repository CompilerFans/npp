// Probe 2: exhaustive semantics for Normalize_16s, Div_32s16s,
// MulC_64f64s_ISfs, complex threshold rounding, SumLn buffer usage.
#include <npp.h>
#include <nppcore.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

extern "C" { NppStatus nppGetStreamContext(NppStreamContext *); }
static NppStreamContext G;

template <class T> struct DB {
  T *p; size_t n;
  DB(const std::vector<T> &h) : n(h.size()) {
    cudaMalloc(&p, n * sizeof(T));
    cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
  }
  ~DB() { cudaFree(p); }
  std::vector<T> get() const {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
  }
};

static void normalize() {
  std::vector<Npp16s> vals = {-1516, -110, -10, -3, -2, 0, 1, 7, 90, 100, 32767, -32768};
  size_t n = vals.size();
  for (int div : {1, 2, 3, 4, 5, 7, 8}) {
    for (int sub : {0, 10, -10}) {
      for (int sf : {0, 1, 2, 3}) {
        DB<Npp16s> da(vals); std::vector<Npp16s> d(n); DB<Npp16s> dd(d);
        NppStatus st = nppsNormalize_16s_Sfs_Ctx(da.p, dd.p, n, (Npp16s)sub, div, sf, G);
        auto o = dd.get();
        printf("Norm16s div=%d sub=%d sf=%d st=%d:", div, sub, sf, (int)st);
        for (auto e : o) printf(" %d", (int)e);
        printf("\n");
        if (st != 0) return;
      }
    }
  }
  // complex
  std::vector<Npp16sc> cvals;
  for (int re : {100, -1506, -10, 7, -100, 3, 50})
    for (int im : {50, 1506, -4, -7, 4}) cvals.push_back(Npp16sc{(Npp16s)re, (Npp16s)im});
  size_t m = cvals.size();
  for (int div : {1, 2, 4, 5, 7}) {
    for (int sf : {0, 1, 2, 3}) {
      DB<Npp16sc> da(cvals); std::vector<Npp16sc> d(m); DB<Npp16sc> dd(d);
      NppStatus st = nppsNormalize_16sc_Sfs_Ctx(da.p, dd.p, m, Npp16sc{10, -4}, div, sf, G);
      auto o = dd.get();
      printf("Norm16sc div=%d sf=%d st=%d:", div, sf, (int)st);
      for (size_t i = 0; i < o.size(); ++i) {
        if (i && i % 8 == 0) printf("\n   ");
        printf(" (%d,%d)", (int)o[i].re, (int)o[i].im);
      }
      printf("\n");
      if (st != 0) return;
    }
  }
}

static void div32s16s() {
  std::vector<int> as = {-4, -3, -2, -1, 1, 2, 3, 4, 5, 7, 8};
  std::vector<int> bs = {-100, -37, -10, -7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7, 10, 37, 100, 1000, -1000};
  for (int sf : {0, 1, 2, 3}) {
    printf("Div32s16s sf=%d\n", sf);
    for (int a : as) {
      printf("  a=%4d:", a);
      std::vector<Npp16s> av(1, (Npp16s)a);
      std::vector<Npp16s> out;
      for (int b : bs) {
        std::vector<Npp32s> bv(1, (Npp32s)b);
        std::vector<Npp16s> d(1);
        DB<Npp16s> da(av); DB<Npp32s> db(bv); DB<Npp16s> dd(d);
        NppStatus st = nppsDiv_32s16s_Sfs_Ctx(da.p, db.p, dd.p, 1, sf, G);
        int r = (int)dd.get()[0];
        out.push_back((Npp16s)(st == 0 ? r : 9999));
      }
      for (auto e : out) printf(" %6d", (int)e);
      printf("\n");
    }
  }
  printf("Div32s16s b-labels:");
  for (auto b : bs) printf(" %6d", b);
  printf("\n");
}

static void mulc64() {
  std::vector<Npp64s> vals = {-1000000000000LL, -7, -5, -3, -1, 0, 1, 2, 3, 5, 7, 1000000000000LL};
  for (double v : {0.5, 1.5, 2.5, -0.5, -1.5, 0.25, 0.1}) {
    for (int sf : {0, 1, 2, 3}) {
      DB<Npp64s> da(vals);
      NppStatus st = nppsMulC_64f64s_ISfs_Ctx(v, da.p, vals.size(), sf, G);
      auto o = da.get();
      printf("MulC64 v=%.3g sf=%d st=%d:", v, sf, (int)st);
      for (auto e : o) printf(" %lld", (long long)e);
      printf("\n");
      if (st != 0) return;
    }
  }
}

static void thr16sc() {
  std::vector<Npp16sc> in;
  for (int re : {-20, -12, -5, -3, -1, 0, 1, 3, 5, 12, 20, 30000, -30000})
    for (int im : {-12, -5, -2, 0, 2, 5, 12}) in.push_back(Npp16sc{(Npp16s)re, (Npp16s)im});
  size_t n = in.size();
  for (int lvl : {3, 5, 9, 100}) {
    DB<Npp16sc> s(in); std::vector<Npp16sc> d(n); DB<Npp16sc> dd(d);
    NppStatus st = nppsThreshold_LT_16sc_Ctx(s.p, dd.p, n, (Npp16s)lvl, G);
    auto o = dd.get();
    printf("ThrLT16sc lvl=%d st=%d\n", lvl, (int)st);
    for (size_t i = 0; i < o.size(); ++i) {
      if (i % 7 == 0) printf("  re=%5d:", (int)in[i].re);
      printf(" (%d,%d)", (int)o[i].re, (int)o[i].im);
      if (i % 7 == 6) printf("\n");
    }
    if (st != 0) return;
  }
  for (int lvl : {3, 5, 9}) {
    DB<Npp16sc> s(in); std::vector<Npp16sc> d(n); DB<Npp16sc> dd(d);
    NppStatus st = nppsThreshold_GT_16sc_Ctx(s.p, dd.p, n, (Npp16s)lvl, G);
    auto o = dd.get();
    printf("ThrGT16sc lvl=%d st=%d\n", lvl, (int)st);
    for (size_t i = 0; i < o.size(); ++i) {
      if (i % 7 == 0) printf("  re=%5d:", (int)in[i].re);
      printf(" (%d,%d)", (int)o[i].re, (int)o[i].im);
      if (i % 7 == 6) printf("\n");
    }
    if (st != 0) return;
  }
  // 64fc scaling precision
  std::vector<Npp64fc> din = {{12,-5},{3,4},{-1,2},{7,7},{0.5,-0.25}};
  for (double lvl : {5.0, 3.0}) {
    DB<Npp64fc> s(din); std::vector<Npp64fc> d(din.size()); DB<Npp64fc> dd(d);
    NppStatus st = nppsThreshold_LT_64fc_Ctx(s.p, dd.p, din.size(), lvl, G);
    auto o = dd.get();
    printf("ThrLT64fc lvl=%g st=%d:", lvl, (int)st);
    for (auto &e : o) printf(" (%.17g,%.17g)", e.re, e.im);
    printf("\n");
  }
}

static void sumln() {
  std::vector<float> a = {1.0f, 2.0f, 3.0f, 0.5f};
  size_t bs = 0;
  NppStatus stb = nppsSumLnGetBufferSize_32f_Ctx(a.size(), &bs, G);
  printf("SumLn_32f bufsz st=%d bs=%zu\n", (int)stb, bs);
  void *buf = nullptr; cudaMalloc(&buf, bs ? bs : 1);
  DB<float> da(a); std::vector<float> o1(1); DB<float> do1(o1);
  NppStatus st = nppsSumLn_32f_Ctx(da.p, a.size(), do1.p, (Npp8u *)buf, G);
  auto r = do1.get();
  printf("SumLn_32f st=%d val=%.9g\n", (int)st, r[0]);
  cudaFree(buf);
  // zero/negative/NaN
  std::vector<float> z = {0.0f, -1.0f, 2.0f};
  size_t bz = 0; nppsSumLnGetBufferSize_32f_Ctx(z.size(), &bz, G);
  void *bufz = nullptr; cudaMalloc(&bufz, bz ? bz : 1);
  DB<float> dz(z); std::vector<float> oz(1); DB<float> doz(oz);
  st = nppsSumLn_32f_Ctx(dz.p, z.size(), doz.p, (Npp8u *)bufz, G);
  auto rz = doz.get();
  printf("SumLn_32f zero/neg st=%d val=%g\n", (int)st, rz[0]);
  cudaFree(bufz);
}

int main() {
  nppGetStreamContext(&G);
  normalize();
  div32s16s();
  mulc64();
  thr16sc();
  sumln();
  return 0;
}
