// Probe 4: Normalize with negative and non-finite divisors.
#include <npp.h>
#include <nppcore.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <limits>
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

int main() {
  nppGetStreamContext(&G);
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::nanf("");
  // 16sc with negative divisors
  std::vector<Npp16sc> c = {{100,50},{-1506,1506},{0,0},{7,-7},{-1,2},{5,-5},{-1,0},{0,-1}};
  for (int div : {2, -2, 3, -3, 1, -1, 5, -5}) {
    for (int sf : {0, 1, 2}) {
      DB<Npp16sc> da(c);
      std::vector<Npp16sc> d(c.size());
      DB<Npp16sc> dd(d);
      NppStatus st = nppsNormalize_16sc_Sfs_Ctx(da.p, dd.p, c.size(), Npp16sc{10, -4}, div, sf, G);
      auto o = dd.get();
      printf("Norm16sc div=%d sf=%d st=%d:", div, sf, (int)st);
      for (auto &e : o) printf(" (%d,%d)", (int)e.re, (int)e.im);
      printf("\n");
    }
  }
  // 16s real with negative divisors
  std::vector<Npp16s> r = {100, -1506, 0, 7, -1, 5, 1, -5};
  for (int div : {2, -2, 3, -3}) {
    for (int sf : {0, 1}) {
      DB<Npp16s> da(r);
      std::vector<Npp16s> d(r.size());
      DB<Npp16s> dd(d);
      NppStatus st = nppsNormalize_16s_Sfs_Ctx(da.p, dd.p, r.size(), 10, div, sf, G);
      auto o = dd.get();
      printf("Norm16s div=%d sf=%d st=%d:", div, sf, (int)st);
      for (auto e : o) printf(" %d", (int)e);
      printf("\n");
    }
  }
  // 32fc with non-finite inputs / sub / div
  std::vector<Npp32fc> fc = {{inf, 2.0f}, {-inf, 3.0f}, {nan, 4.0f}, {6.0f, inf},
                             {7.0f, -inf}, {8.0f, nan}, {1.0f, 2.0f}};
  struct SV { float sub, div; } sds[] = {{0.0f, 2.0f}, {inf, 2.0f}, {-inf, 2.0f},
                                        {0.0f, inf}, {0.0f, -inf}, {inf, inf},
                                        {1.0f, 0.0f}, {nan, 2.0f}};
  for (auto sd : sds) {
    DB<Npp32fc> da(fc);
    std::vector<Npp32fc> d(fc.size());
    DB<Npp32fc> dd(d);
    NppStatus st = nppsNormalize_32fc_Ctx(da.p, dd.p, fc.size(),
                                          Npp32fc{sd.sub, sd.sub}, sd.div, G);
    auto o = dd.get();
    printf("Norm32fc sub=%g div=%g st=%d:", sd.sub, sd.div, (int)st);
    for (auto &e : o) printf(" (%.7g,%.7g)", e.re, e.im);
    printf("\n");
  }
  return 0;
}
