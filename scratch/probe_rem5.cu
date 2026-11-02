// Probe 5: labeled 16sc Normalize rounding across divisors / scale factors.
#include <npp.h>
#include <nppcore.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

extern "C" { NppStatus nppGetStreamContext(NppStreamContext *); }
static NppStreamContext G;

int main() {
  nppGetStreamContext(&G);
  std::vector<Npp16sc> in;
  for (int re = -12; re <= 12; re += 1)
    for (int im = -3; im <= 3; im += 1) in.push_back(Npp16sc{(Npp16s)re, (Npp16s)im});
  size_t n = in.size();
  std::vector<Npp16sc> d(n);
  Npp16sc *dp = nullptr;
  cudaMalloc(&dp, n * sizeof(Npp16sc));
  Npp16sc *sp = nullptr;
  cudaMalloc(&sp, n * sizeof(Npp16sc));
  for (int div : {1, 2, 3, 4, 5, 6, 7}) {
    for (int sf : {0, 1, 2}) {
      cudaMemcpy(sp, in.data(), n * sizeof(Npp16sc), cudaMemcpyHostToDevice);
      NppStatus st = nppsNormalize_16sc_Sfs_Ctx(sp, dp, n, Npp16sc{0, 0}, div, sf, G);
      cudaMemcpy(d.data(), dp, n * sizeof(Npp16sc), cudaMemcpyDeviceToHost);
      printf("div=%d sf=%d st=%d\n", div, sf, (int)st);
      for (int re = -12; re <= 12; ++re) {
        printf("  re=%3d:", re);
        for (int im = -3; im <= 3; ++im) {
          const Npp16sc &o = d[(size_t)(re + 12) * 7 + (im + 3)];
          printf(" (%3d,%3d)", (int)o.re, (int)o.im);
        }
        printf("\n");
      }
    }
  }
  cudaFree(sp); cudaFree(dp);
  return 0;
}
