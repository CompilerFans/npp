// f13: isolated Advanced batch label markers statuses.
#include "npp.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv) {
  const int W = 10, H = 8;
  int which = argc > 1 ? atoi(argv[1]) : 0;
  std::vector<Npp8u> hs((size_t)W * H, 0);
  for (size_t i = 0; i < hs.size(); ++i) hs[i] = (Npp8u)((i % 3) * 60);
  Npp8u *s = nullptr;
  Npp32u *d = nullptr;
  cudaMalloc(&s, hs.size() * 2);
  cudaMalloc(&d, hs.size() * 4 * 2);
  cudaMemcpy(s, hs.data(), hs.size(), cudaMemcpyHostToDevice);
  cudaMemcpy((char *)s + hs.size(), hs.data(), hs.size(), cudaMemcpyHostToDevice);
  cudaMemset(d, 0, hs.size() * 4 * 2);
  NppStreamContext ctx{};
  ctx.hStream = 0;
  NppiImageDescriptor src[2] = {{(char *)s, W, {W, H}}, {(char *)s + hs.size(), W, {W, H}}};
  NppiImageDescriptor dst[2] = {{(char *)d, W * 4, {W, H}},
                                {(char *)d + hs.size() * 4, W * 4, {W, H}}};
  NppiImageDescriptor *ps = src;
  NppiImageDescriptor *pdst = dst;
  int nb = 2;
  NppiSize roi = {W, H};
  NppiNorm norm = nppiNormL1;
  const char *name = "ok";
  switch (which) {
    case 1: name = "null_src"; ps = nullptr; break;
    case 2: name = "null_dst"; pdst = nullptr; break;
    case 3: name = "batch0"; nb = 0; break;
    case 4: name = "batch1"; nb = 1; break;
    case 5: name = "roi_zero"; roi = {0, H}; break;
    case 6: name = "roi_neg"; roi = {-1, H}; break;
    case 7: name = "badnorm"; norm = (NppiNorm)7; break;
    case 8: name = "batch_neg"; nb = -1; break;
    case 9: name = "batch0_roi_neg"; nb = 0; roi = {-1, H}; break;
    default: break;
  }
  NppStatus st = ps ? nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx(ps, pdst, nb, roi, norm, ctx) : (NppStatus)-999;
  cudaError_t e = cudaDeviceSynchronize();
  printf("%-9s st=%d cuda=%s\n", name, (int)st, cudaGetErrorName(e));
  return 0;
}
