// Probe NPP 13.4 semantics for the remaining npps arithmetic families.
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

static void pc16sc(const char *tag, const std::vector<Npp16sc> &v) {
  printf("%s:", tag);
  for (auto &e : v) printf(" (%d,%d)", (int)e.re, (int)e.im);
  printf("\n");
}
static void pf(const char *tag, const std::vector<float> &v) {
  printf("%s:", tag);
  for (auto e : v) printf(" %.6g", e);
  printf("\n");
}
static void pd(const char *tag, const std::vector<double> &v) {
  printf("%s:", tag);
  for (auto e : v) printf(" %.10g", e);
  printf("\n");
}
static void pi(const char *tag, const std::vector<int> &v) {
  printf("%s:", tag);
  for (auto e : v) printf(" %d", e);
  printf("\n");
}

static void threshold16sc() {
  std::vector<Npp16sc> in = {{10,0},{0,10},{6,8},{8,6},{-10,0},{0,-10},
                             {-6,8},{6,-8},{3,4},{4,3},{0,0},{-1,0},{1,0},
                             {5,0},{-5,0},{0,5},{0,-5},{12,-5},{-12,5}};
  size_t n = in.size();
  DB<Npp16sc> s(in);
  std::vector<Npp16sc> o(n);
  DB<Npp16sc> d(o);
  Npp16s lvl = 5;
  Npp16sc val{77, 88};
  NppStatus st;
#define SHOW(name) do { o = d.get(); pc16sc(name, o); } while (0)
  st = nppsThreshold_16sc_Ctx(s.p, d.p, n, lvl, NPP_CMP_LESS, G);
  printf("Threshold_16sc LESS lvl=5 st=%d\n", (int)st); SHOW("  out");
  st = nppsThreshold_16sc_Ctx(s.p, d.p, n, lvl, NPP_CMP_GREATER, G);
  printf("Threshold_16sc GREATER lvl=5 st=%d\n", (int)st); SHOW("  out");
  st = nppsThreshold_GT_16sc_Ctx(s.p, d.p, n, lvl, G);
  printf("Threshold_GT_16sc lvl=5 st=%d\n", (int)st); SHOW("  out");
  st = nppsThreshold_LT_16sc_Ctx(s.p, d.p, n, lvl, G);
  printf("Threshold_LT_16sc lvl=5 st=%d\n", (int)st); SHOW("  out");
  st = nppsThreshold_GTVal_16sc_Ctx(s.p, d.p, n, lvl, val, G);
  printf("Threshold_GTVal_16sc lvl=5 val=(77,88) st=%d\n", (int)st); SHOW("  out");
  st = nppsThreshold_LTVal_16sc_Ctx(s.p, d.p, n, lvl, val, G);
  printf("Threshold_LTVal_16sc lvl=5 val=(77,88) st=%d\n", (int)st); SHOW("  out");
  // in-place
  DB<Npp16sc> s2(in);
  st = nppsThreshold_16sc_I_Ctx(s2.p, n, lvl, NPP_CMP_LESS, G);
  o = s2.get();
  printf("Threshold_16sc_I LESS lvl=5 st=%d\n", (int)st); pc16sc("  out", o);
#undef SHOW
}

static void threshold32fc() {
  std::vector<Npp32fc> in = {{10,0},{0,10},{6,8},{8,6},{-10,0},{0,-10},
                             {-6,8},{6,-8},{3,4},{4,3},{0,0},{-1,0},{1,0},
                             {5,0},{-5,0},{0,5},{0,-5},{12,-5},{-12,5}};
  size_t n = in.size();
  DB<Npp32fc> s(in);
  std::vector<Npp32fc> o(n);
  DB<Npp32fc> d(o);
  Npp32f lvl = 5.0f;
  Npp32fc val{77.0f, 88.0f};
  NppStatus st = nppsThreshold_32fc_Ctx(s.p, d.p, n, lvl, NPP_CMP_LESS, G);
  printf("Threshold_32fc LESS lvl=5 st=%d\n", (int)st);
  o = d.get(); printf("  out:"); for (auto &e : o) printf(" (%.4g,%.4g)", e.re, e.im); printf("\n");
  st = nppsThreshold_GT_32fc_Ctx(s.p, d.p, n, lvl, G);
  printf("Threshold_GT_32fc lvl=5 st=%d\n", (int)st);
  o = d.get(); printf("  out:"); for (auto &e : o) printf(" (%.4g,%.4g)", e.re, e.im); printf("\n");
  st = nppsThreshold_LTVal_32fc_Ctx(s.p, d.p, n, lvl, val, G);
  printf("Threshold_LTVal_32fc lvl=5 val=(77,88) st=%d\n", (int)st);
  o = d.get(); printf("  out:"); for (auto &e : o) printf(" (%.4g,%.4g)", e.re, e.im); printf("\n");
}

static void addProduct() {
  // 16s -> 16s
  {
    std::vector<Npp16s> a = {30000, 1000, 1000, -1000, 100, 7, 0, -30000};
    std::vector<Npp16s> b = {2, 100, -100, 100, 100, 100, 0, 2};
    std::vector<Npp16s> d = {7, 7, 7, 7, 7, 7, 7, 7};
    size_t n = a.size();
    for (int sf : {0, 1, 2}) {
      DB<Npp16s> da(a), db(b), dd(d);
      NppStatus st = nppsAddProduct_16s_Sfs_Ctx(da.p, db.p, dd.p, n, sf, G);
      auto o = dd.get();
      char t[64]; snprintf(t, 64, "AddProduct_16s_Sfs sf=%d st=%d", sf, (int)st);
      printf("%s:", t); for (auto e : o) printf(" %d", (int)e); printf("\n");
    }
  }
  // 16s -> 32s
  {
    std::vector<Npp16s> a = {30000, 1000, 1000, -1000, 100, 7, 0, -30000};
    std::vector<Npp16s> b = {2, 100, -100, 100, 100, 100, 0, 2};
    std::vector<Npp32s> d = {7, 100000, -100000, 7, 0, 7, 7, 7};
    size_t n = a.size();
    for (int sf : {0, 1, 2}) {
      DB<Npp16s> da(a), db(b); DB<Npp32s> dd(d);
      NppStatus st = nppsAddProduct_16s32s_Sfs_Ctx(da.p, db.p, dd.p, n, sf, G);
      auto o = dd.get();
      char t[64]; snprintf(t, 64, "AddProduct_16s32s_Sfs sf=%d st=%d", sf, (int)st);
      printf("%s:", t); for (auto e : o) printf(" %d", (int)e); printf("\n");
    }
  }
  // 32s -> 32s
  {
    std::vector<Npp32s> a = {30000, 100000, -100000, 46341, 1, 0};
    std::vector<Npp32s> b = {2, 100000, 100000, 46341, 1, 0};
    std::vector<Npp32s> d = {7, 7, 7, 7, 7, 7};
    size_t n = a.size();
    for (int sf : {0, 1, 2}) {
      DB<Npp32s> da(a), db(b), dd(d);
      NppStatus st = nppsAddProduct_32s_Sfs_Ctx(da.p, db.p, dd.p, n, sf, G);
      auto o = dd.get();
      char t[64]; snprintf(t, 64, "AddProduct_32s_Sfs sf=%d st=%d", sf, (int)st);
      printf("%s:", t); for (auto e : o) printf(" %d", (int)e); printf("\n");
    }
  }
  // 32f
  {
    std::vector<float> a = {1.5f, -2.25f, 0.0f}; std::vector<float> b = {4.0f, 3.0f, 5.0f};
    std::vector<float> d = {0.5f, 100.0f, -1.0f};
    DB<float> da(a), db(b), dd(d);
    NppStatus st = nppsAddProduct_32f_Ctx(da.p, db.p, dd.p, a.size(), G);
    auto o = dd.get(); printf("AddProduct_32f st=%d:", (int)st); for (auto e : o) printf(" %.6g", e); printf("\n");
  }
  // 64f
  {
    std::vector<double> a = {1.5, -2.25, 0.0}; std::vector<double> b = {4.0, 3.0, 5.0};
    std::vector<double> d = {0.5, 100.0, -1.0};
    DB<double> da(a), db(b), dd(d);
    NppStatus st = nppsAddProduct_64f_Ctx(da.p, db.p, dd.p, a.size(), G);
    auto o = dd.get(); printf("AddProduct_64f st=%d:", (int)st); for (auto e : o) printf(" %.10g", e); printf("\n");
  }
  // 32fc
  {
    std::vector<Npp32fc> a = {{1.5f,-2.0f},{-2.25f,1.0f},{0.0f,0.0f}};
    std::vector<Npp32fc> b = {{4.0f,0.5f},{3.0f,-1.0f},{5.0f,2.0f}};
    std::vector<Npp32fc> d = {{0.5f,1.0f},{100.0f,-3.0f},{-1.0f,2.0f}};
    DB<Npp32fc> da(a), db(b), dd(d);
    NppStatus st = nppsAddProduct_32fc_Ctx(da.p, db.p, dd.p, a.size(), G);
    auto o = dd.get(); printf("AddProduct_32fc st=%d:", (int)st);
    for (auto &e : o) printf(" (%.6g,%.6g)", e.re, e.im); printf("\n");
  }
  // 64fc
  {
    std::vector<Npp64fc> a = {{1.5,-2.0},{-2.25,1.0},{0.0,0.0}};
    std::vector<Npp64fc> b = {{4.0,0.5},{3.0,-1.0},{5.0,2.0}};
    std::vector<Npp64fc> d = {{0.5,1.0},{100.0,-3.0},{-1.0,2.0}};
    DB<Npp64fc> da(a), db(b), dd(d);
    NppStatus st = nppsAddProduct_64fc_Ctx(da.p, db.p, dd.p, a.size(), G);
    auto o = dd.get(); printf("AddProduct_64fc st=%d:", (int)st);
    for (auto &e : o) printf(" (%.10g,%.10g)", e.re, e.im); printf("\n");
  }
  // AddProductC_32f
  {
    std::vector<float> a = {1.5f, -2.25f, 0.0f, 1e30f}; std::vector<float> d = {0.5f, 100.0f, -1.0f, 0.0f};
    DB<float> da(a), dd(d);
    NppStatus st = nppsAddProductC_32f_Ctx(da.p, 2.5f, dd.p, a.size(), G);
    auto o = dd.get(); printf("AddProductC_32f st=%d:", (int)st); for (auto e : o) printf(" %.6g", e); printf("\n");
  }
}

static void divRound() {
  std::vector<Npp16s> a = {2, 2, 2, 2, 2, 2, 4, 4, 1, 3, 3, 3, -2, -2, -2, 0};
  std::vector<Npp16s> b = {1, 3, 5, 7, -1, -3, 2, 6, 1, 2, 4, 8, -1, -3, 5, 3};
  size_t n = a.size();
  const char *mn[3] = {"FIN", "NEAR", "ZERO"};
  for (int sf : {0, 1, 2}) {
    for (int mi = 0; mi < 3; ++mi) {
      DB<Npp16s> da(a), db(b); std::vector<Npp16s> d(n); DB<Npp16s> dd(d);
      NppStatus st = nppsDiv_Round_16s_Sfs_Ctx(da.p, db.p, dd.p, n,
                                               (NppRoundMode)mi, sf, G);
      auto o = dd.get();
      printf("Div_Round_16s sf=%d %s st=%d:", sf, mn[mi], (int)st);
      for (auto e : o) printf(" %d", (int)e); printf("\n");
    }
  }
  {
    std::vector<Npp8u> a = {2, 2, 2, 4, 4, 1, 3, 3};
    std::vector<Npp8u> b = {1, 3, 5, 2, 6, 1, 2, 4};
    for (int sf : {0, 1, 2}) {
      for (int mi = 0; mi < 3; ++mi) {
        DB<Npp8u> da(a), db(b); std::vector<Npp8u> d(a.size()); DB<Npp8u> dd(d);
        NppStatus st = nppsDiv_Round_8u_Sfs_Ctx(da.p, db.p, dd.p, a.size(),
                                                (NppRoundMode)mi, sf, G);
        auto o = dd.get();
        printf("Div_Round_8u sf=%d %s st=%d:", sf, mn[mi], (int)st);
        for (auto e : o) printf(" %d", (int)e); printf("\n");
      }
    }
  }
  {
    std::vector<Npp16s> a = {2, 2, 2, 4, 4, 3, -2, -3};
    std::vector<Npp32s> b = {5, -5, 10, 10, -10, 10, 5, 10};
    for (int sf : {0, 1, 2}) {
      DB<Npp16s> da(a); DB<Npp32s> db(b); std::vector<Npp16s> d(a.size()); DB<Npp16s> dd(d);
      NppStatus st = nppsDiv_32s16s_Sfs_Ctx(da.p, db.p, dd.p, a.size(), sf, G);
      auto o = dd.get();
      printf("Div_32s16s sf=%d st=%d:", sf, (int)st);
      for (auto e : o) printf(" %d", (int)e); printf("\n");
    }
  }
}

static void mulLow() {
  {
    std::vector<Npp32s> a = {70000, 46341, 100000, -70000, 65536, 3};
    std::vector<Npp32s> b = {70000, 46341, 100000, 70000, 65536, 4};
    for (int sf : {0, 1, 2}) {
      DB<Npp32s> da(a), db(b); std::vector<Npp32s> d(a.size()); DB<Npp32s> dd(d);
      NppStatus st = nppsMul_Low_32s_Sfs_Ctx(da.p, db.p, dd.p, a.size(), sf, G);
      auto o = dd.get();
      printf("Mul_Low_32s sf=%d st=%d:", sf, (int)st);
      for (auto e : o) printf(" %d", (int)e); printf("\n");
    }
  }
  {
    std::vector<float> a = {1.5f, 3.5f, 40000.0f, -3.5f, 70000.0f, -70000.0f, 2.0f, 0.0f};
    std::vector<Npp16s> d(a.size());
    for (float v : {1.0f, 2.0f, 0.5f, -1.0f, 0.0f}) {
      DB<float> da(a); DB<Npp16s> dd(d);
      NppStatus st = nppsMulC_Low_32f16s_Ctx(da.p, v, dd.p, a.size(), G);
      auto o = dd.get();
      printf("MulC_Low_32f16s v=%g st=%d:", v, (int)st);
      for (auto e : o) printf(" %d", (int)e); printf("\n");
    }
  }
  {
    std::vector<Npp32f> a = {1.5f, -2.25f}; std::vector<Npp32fc> b = {{4.0f,0.5f},{3.0f,-1.0f}};
    std::vector<Npp32fc> d(2);
    DB<float> da(a); DB<Npp32fc> db(b); DB<Npp32fc> dd(d);
    NppStatus st = nppsMul_32f32fc_Ctx(da.p, db.p, dd.p, 2, G);
    auto o = dd.get(); printf("Mul_32f32fc st=%d:", (int)st);
    for (auto &e : o) printf(" (%.6g,%.6g)", e.re, e.im); printf("\n");
  }
  {
    std::vector<Npp32s> a = {3, -3, 100000, 7}; std::vector<Npp32sc> b = {{4,5},{-4,5},{3,-2},{0,1}};
    for (int sf : {0, 1, 2}) {
      DB<Npp32s> da(a); DB<Npp32sc> db(b); std::vector<Npp32sc> d(4); DB<Npp32sc> dd(d);
      NppStatus st = nppsMul_32s32sc_Sfs_Ctx(da.p, db.p, dd.p, 4, sf, G);
      auto o = dd.get(); printf("Mul_32s32sc sf=%d st=%d:", sf, (int)st);
      for (auto &e : o) printf(" (%d,%d)", (int)e.re, (int)e.im); printf("\n");
    }
  }
  {
    std::vector<Npp64s> a = {3, -3, 5, 7};
    for (int sf : {0, 1, 2}) {
      DB<Npp64s> da(a);
      NppStatus st = nppsMulC_64f64s_ISfs_Ctx(1.5, da.p, 4, sf, G);
      auto o = da.get(); printf("MulC_64f64s_ISfs v=1.5 sf=%d st=%d:", sf, (int)st);
      for (auto e : o) printf(" %lld", (long long)e); printf("\n");
    }
    // negative/rounding boundaries
    std::vector<Npp64s> c = {-1, 1, -3, 2};
    for (int sf : {0, 1}) {
      DB<Npp64s> dc(c);
      NppStatus st = nppsMulC_64f64s_ISfs_Ctx(0.5, dc.p, 4, sf, G);
      auto o = dc.get(); printf("MulC_64f64s_ISfs v=0.5 sf=%d st=%d:", sf, (int)st);
      for (auto e : o) printf(" %lld", (long long)e); printf("\n");
    }
  }
}

static void normalize() {
  {
    std::vector<Npp16s> a = {100, -1506, 0, 7, -100};
    std::vector<Npp16s> d(a.size());
    for (int sf : {0, 1, 2}) {
      DB<Npp16s> da(a); DB<Npp16s> dd(d);
      NppStatus st = nppsNormalize_16s_Sfs_Ctx(da.p, dd.p, a.size(), 10, 4, sf, G);
      auto o = dd.get(); printf("Normalize_16s sub=10 div=4 sf=%d st=%d:", sf, (int)st);
      for (auto e : o) printf(" %d", (int)e); printf("\n");
    }
  }
  {
    std::vector<Npp16sc> a = {{100,50},{-1506,1506},{0,0},{7,-7},{-100,4}};
    std::vector<Npp16sc> d(a.size());
    for (int sf : {0, 1, 2}) {
      DB<Npp16sc> da(a); DB<Npp16sc> dd(d);
      NppStatus st = nppsNormalize_16sc_Sfs_Ctx(da.p, dd.p, a.size(), Npp16sc{10,-4}, 4, sf, G);
      auto o = dd.get(); printf("Normalize_16sc sub=(10,-4) div=4 sf=%d st=%d:", sf, (int)st);
      for (auto &e : o) printf(" (%d,%d)", (int)e.re, (int)e.im); printf("\n");
    }
  }
  {
    std::vector<float> a = {100.0f, -1506.0f, 0.0f};
    DB<float> da(a); std::vector<float> d(a.size()); DB<float> dd(d);
    NppStatus st = nppsNormalize_32f_Ctx(da.p, dd.p, a.size(), 10.0f, 4.0f, G);
    auto o = dd.get(); printf("Normalize_32f st=%d:", (int)st);
    for (auto e : o) printf(" %.6g", e); printf("\n");
  }
  {
    std::vector<Npp32fc> a = {{100.0f,50.0f},{-1506.0f,1506.0f},{0.0f,0.0f}};
    DB<Npp32fc> da(a); std::vector<Npp32fc> d(a.size()); DB<Npp32fc> dd(d);
    NppStatus st = nppsNormalize_32fc_Ctx(da.p, dd.p, a.size(), Npp32fc{10.0f,-4.0f}, 4.0f, G);
    auto o = dd.get(); printf("Normalize_32fc st=%d:", (int)st);
    for (auto &e : o) printf(" (%.6g,%.6g)", e.re, e.im); printf("\n");
  }
  {
    std::vector<double> a = {100.0, -1506.0, 0.0};
    DB<double> da(a); std::vector<double> d(a.size()); DB<double> dd(d);
    NppStatus st = nppsNormalize_64f_Ctx(da.p, dd.p, a.size(), 10.0, 4.0, G);
    auto o = dd.get(); printf("Normalize_64f st=%d:", (int)st);
    for (auto e : o) printf(" %.10g", e); printf("\n");
  }
  {
    std::vector<Npp64fc> a = {{100.0,50.0},{-1506.0,1506.0},{0.0,0.0}};
    DB<Npp64fc> da(a); std::vector<Npp64fc> d(a.size()); DB<Npp64fc> dd(d);
    NppStatus st = nppsNormalize_64fc_Ctx(da.p, dd.p, a.size(), Npp64fc{10.0,-4.0}, 4.0, G);
    auto o = dd.get(); printf("Normalize_64fc st=%d:", (int)st);
    for (auto &e : o) printf(" (%.10g,%.10g)", e.re, e.im); printf("\n");
  }
}

static void sumln() {
  std::vector<float> a = {1.0f, 2.0f, 3.0f, 0.5f};
  DB<float> da(a); std::vector<float> o1(1); DB<float> do1(o1);
  size_t bs = (size_t)-1;
  NppStatus stb = nppsSumLnGetBufferSize_32f_Ctx(a.size(), &bs, G);
  NppStatus st = nppsSumLn_32f_Ctx(da.p, a.size(), do1.p, nullptr, G);
  auto r = do1.get();
  printf("SumLn_32f st=%d bufsz(st=%d)=%zu val=%.9g\n", (int)st, (int)stb, bs, r[0]);
  std::vector<double> o2(1); DB<double> do2(o2);
  st = nppsSumLn_32f64f_Ctx(da.p, a.size(), do2.p, nullptr, G);
  auto r2 = do2.get();
  printf("SumLn_32f64f st=%d val=%.15g\n", (int)st, r2[0]);
  std::vector<Npp16s> b = {1, 2, 3, 4};
  DB<Npp16s> db(b); std::vector<float> o3(1); DB<float> do3(o3);
  size_t bs3 = (size_t)-1; stb = nppsSumLnGetBufferSize_16s32f_Ctx(b.size(), &bs3, G);
  st = nppsSumLn_16s32f_Ctx(db.p, b.size(), do3.p, nullptr, G);
  auto r3 = do3.get();
  printf("SumLn_16s32f st=%d bufsz(st=%d)=%zu val=%.9g\n", (int)st, (int)stb, bs3, r3[0]);
  std::vector<double> c = {1.0, 2.0, 3.0, 0.5};
  DB<double> dc(c); std::vector<double> o4(1); DB<double> do4(o4);
  size_t bs4 = (size_t)-1; stb = nppsSumLnGetBufferSize_64f_Ctx(c.size(), &bs4, G);
  st = nppsSumLn_64f_Ctx(dc.p, c.size(), do4.p, nullptr, G);
  auto r4 = do4.get();
  printf("SumLn_64f st=%d bufsz(st=%d)=%zu val=%.15g\n", (int)st, (int)stb, bs4, r4[0]);
  // zeros / negatives
  std::vector<float> z = {0.0f, -1.0f};
  DB<float> dz(z); std::vector<float> oz(1); DB<float> doz(oz);
  st = nppsSumLn_32f_Ctx(dz.p, z.size(), doz.p, nullptr, G);
  auto rz = doz.get();
  printf("SumLn_32f zero/neg st=%d val=%g\n", (int)st, rz[0]);
}

static void cauchy() {
  std::vector<float> a = {0.5f, 1.0f, 2.0f, -1.0f, 10.0f, 0.0f};
  for (float p : {1.0f, 0.5f, 2.0f}) {
    DB<float> da(a);
    NppStatus st = nppsCauchy_32f_I_Ctx(da.p, a.size(), p, G);
    auto o = da.get(); printf("Cauchy p=%g st=%d:", p, (int)st);
    for (auto e : o) printf(" %.7g", e); printf("\n");
    DB<float> db(a);
    st = nppsCauchyD_32f_I_Ctx(db.p, a.size(), p, G);
    auto o2 = db.get(); printf("CauchyD p=%g st=%d:", p, (int)st);
    for (auto e : o2) printf(" %.7g", e); printf("\n");
    DB<float> dc(a); std::vector<float> d2(a.size()); DB<float> dd(d2);
    st = nppsCauchyDD2_32f_I_Ctx(dc.p, dd.p, a.size(), p, G);
    auto o3 = dc.get(); auto o4 = dd.get();
    printf("CauchyDD2 p=%g st=%d:\n  d :", p, (int)st);
    for (auto e : o3) printf(" %.7g", e);
    printf("\n  d2:"); for (auto e : o4) printf(" %.7g", e); printf("\n");
  }
}

int main() {
  NppStatus st = nppGetStreamContext(&G);
  printf("ctx=%d\n", (int)st);
  threshold16sc();
  threshold32fc();
  addProduct();
  divRound();
  mulLow();
  normalize();
  sumln();
  cauchy();
  return 0;
}
