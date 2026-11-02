// Probe 3: MulC_Low tie/extreme rounding and Div_Round zero-divisor signs.
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
  const float fltmax = std::numeric_limits<float>::max();
  std::vector<float> src = {2.5f, -2.5f, 1.5f, 0.5f, -0.5f, 3.5f, -3.5f,
                            inf, -inf, nan, fltmax, -fltmax, 0.0f};
  for (float v : {1.0f, 0.5f, 2.0f, 0.0f, -1.0f, inf}) {
    DB<float> da(src);
    std::vector<Npp16s> d(src.size());
    DB<Npp16s> dd(d);
    NppStatus st = nppsMulC_Low_32f16s_Ctx(da.p, v, dd.p, src.size(), G);
    auto o = dd.get();
    printf("MulC_Low v=%g st=%d:", v, (int)st);
    for (auto e : o) printf(" %d", (int)e);
    printf("\n");
  }
  // Div_Round zero divisor with negative / zero numerator
  std::vector<Npp16s> a = {0, 0, 0, 0, 1, -1};
  std::vector<Npp16s> b = {3, -3, 0, 1, 0, 0};
  for (int sf : {0, 1}) {
    for (int mode = 0; mode < 3; ++mode) {
      DB<Npp16s> da(a), db(b);
      std::vector<Npp16s> d(a.size());
      DB<Npp16s> dd(d);
      NppStatus st = nppsDiv_Round_16s_Sfs_Ctx(da.p, db.p, dd.p, a.size(),
                                               (NppRoundMode)mode, sf, G);
      auto o = dd.get();
      printf("DivRound_zero mode=%d sf=%d st=%d:", mode, sf, (int)st);
      for (auto e : o) printf(" %d", (int)e);
      printf("\n");
    }
  }
  std::vector<Npp8u> a8 = {0, 0, 1};
  std::vector<Npp8u> b8 = {3, 0, 0};
  for (int sf : {0, 1}) {
    for (int mode = 0; mode < 3; ++mode) {
      DB<Npp8u> da(a8), db(b8);
      std::vector<Npp8u> d(a8.size());
      DB<Npp8u> dd(d);
      NppStatus st = nppsDiv_Round_8u_Sfs_Ctx(da.p, db.p, dd.p, a8.size(),
                                              (NppRoundMode)mode, sf, G);
      auto o = dd.get();
      printf("DivRound8_zero mode=%d sf=%d st=%d:", mode, sf, (int)st);
      for (auto e : o) printf(" %d", (int)e);
      printf("\n");
    }
  }
  return 0;
}
