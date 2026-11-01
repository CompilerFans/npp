// NPP 13.4 MSSSIM vs WMSSSIM value probe (MSSSIM needs ROI >= 176).
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
NppStatus nppGetStreamContext(NppStreamContext *c);
}
static NppStreamContext ctx() {
  NppStreamContext c{};
  nppGetStreamContext(&c);
  return c;
}
template <typename T> struct Dev {
  T *p = nullptr;
  explicit Dev(size_t n) { cudaMalloc(&p, n * sizeof(T)); }
  ~Dev() { if (p) cudaFree(p); }
  void put(const std::vector<T> &h) { cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice); }
  T get() { T v; cudaMemcpy(&v, p, sizeof(T), cudaMemcpyDeviceToHost); return v; }
};

static void tex(int kind, int sz, std::vector<Npp8u> &a, std::vector<Npp8u> &b) {
  a.resize((size_t)sz * sz); b.resize((size_t)sz * sz);
  unsigned s = 12345u;
  for (int y = 0; y < sz; ++y)
    for (int x = 0; x < sz; ++x) {
      size_t i = (size_t)y * sz + x;
      switch (kind) {
        case 0: a[i] = (Npp8u)((x * 7 + y * 13 + 5) & 0xff); b[i] = (Npp8u)((x * 7 + y * 13 + 6) & 0xff); break;
        case 1: a[i] = 100; b[i] = 110; break;
        case 2: s = s * 1103515245u + 12345u; a[i] = (Npp8u)(s >> 16); s = s * 1103515245u + 12345u; b[i] = (Npp8u)(s >> 16); break;
        case 3: a[i] = (Npp8u)((x * 3 + y * 5) & 0xff); b[i] = a[i]; break;
        default: a[i] = (Npp8u)((x * 11 + y * 3) & 0xff); b[i] = (Npp8u)(((x * 11 + y * 3) * 3 / 2) & 0xff); break;
      }
    }
}

int main(int argc, char **argv) {
  NppStreamContext c = ctx();
  int kinds = argc > 1 ? atoi(argv[1]) : 4;
  Dev<Npp32f> out(8);
  Dev<Npp8u> buf(1);
  cudaMalloc((void **)&buf.p, 64u << 20);
  size_t gbsM = 0, gbsW = 0;
  for (int sz : {176, 180, 192, 224, 256, 320}) {
    nppiMSSSIMGetBufferHostSize_8u_C1R_Ctx(NppiSize{sz, sz}, &gbsM, c);
    nppiWMSSSIMGetBufferHostSize_8u_C1R_Ctx(NppiSize{sz, sz}, &gbsW, c);
    printf("gbs sz=%d msssim=%zu wmsssim=%zu\n", sz, gbsM, gbsW);
    for (int k = 0; k < kinds; ++k) {
      std::vector<Npp8u> a, b;
      tex(k, sz, a, b);
      Dev<Npp8u> s1(a.size()), s2(b.size());
      s1.put(a); s2.put(b);
      out.put(std::vector<Npp32f>(8, -1e30f));
      NppStatus sm = nppiMSSSIM_8u_C1R_Ctx(s1.p, sz, s2.p, sz, NppiSize{sz, sz}, out.p, buf.p, c);
      float vm = out.get();
      out.put(std::vector<Npp32f>(8, -1e30f));
      NppStatus sw = nppiWMSSSIM_8u_C1R_Ctx(s1.p, sz, s2.p, sz, NppiSize{sz, sz}, out.p, buf.p, c);
      float vw = out.get();
      out.put(std::vector<Npp32f>(8, -1e30f));
      NppStatus ss = nppiSSIM_8u_C1R_Ctx(s1.p, sz, s2.p, sz, NppiSize{sz, sz}, out.p, buf.p, c);
      float vs = out.get();
      printf("sz=%d tex=%d ssim st=%d %.9g | msssim st=%d %.9g | wmsssim st=%d %.9g\n", sz, k, (int)ss, vs, (int)sm,
             vm, (int)sw, vw);
      cudaDeviceSynchronize();
      cudaGetLastError();
    }
  }
  return 0;
}
