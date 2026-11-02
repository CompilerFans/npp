// f13: isolated statuses for label marker error inputs (one case per process).
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
  Npp8u *buf = nullptr;
  cudaMalloc(&s, hs.size());
  cudaMalloc(&d, hs.size() * 4);
  cudaMalloc(&buf, 1 << 20);
  cudaMemcpy(s, hs.data(), hs.size(), cudaMemcpyHostToDevice);
  cudaMemset(d, 0, hs.size() * 4);
  NppStreamContext ctx{};
  ctx.hStream = 0;
  Npp8u *ps = s;
  int ss = W;
  Npp32u *pd = d;
  int ds = W * 4;
  Npp8u *pb = buf;
  NppiSize roi = {W, H};
  NppiNorm norm = nppiNormL1;
  const char *name = "ok";
  switch (which) {
    case 1: name = "null_src"; ps = nullptr; break;
    case 2: name = "null_dst"; pd = nullptr; break;
    case 3: name = "null_buf"; pb = nullptr; break;
    case 4: name = "roi_zero"; roi = {0, H}; break;
    case 5: name = "roi_neg"; roi = {-1, H}; break;
    case 6: name = "srcstep0"; ss = 0; break;
    case 7: name = "dststep0"; ds = 0; break;
    case 8: name = "badnorm"; norm = (NppiNorm)7; break;
    default: break;
  }
  NppStatus st = nppiLabelMarkersUF_8u32u_C1R_Ctx(ps, ss, pd, ds, roi, norm, pb, ctx);
  cudaError_t e = cudaDeviceSynchronize();
  printf("%-10s st=%d cuda=%s\n", name, (int)st, cudaGetErrorName(e));
  return 0;
}
