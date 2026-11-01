// f13: one isolated Advanced compress-batch case per process (context must not be poisoned).
#include "npp.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv) {
  const int W = 10, H = 7;
  int which = argc > 1 ? atoi(argv[1]) : 0;
  std::vector<Npp32u> h((size_t)W * H * 2, 1);
  Npp32u *d = nullptr;
  cudaMalloc(&d, h.size() * 4);
  cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
  int gbs = 0;
  nppiCompressMarkerLabelsGetBufferSize_32u_C1R(W * H, &gbs);
  Npp8u *b0 = nullptr, *b1 = nullptr;
  cudaMalloc(&b0, gbs);
  cudaMalloc(&b1, gbs);
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppiImageDescriptor im[2] = {{(char *)d, W * 4, {W, H}}, {(char *)d + (size_t)W * H * 4, W * 4, {W, H}}};
  NppiBufferDescriptor bu[2] = {{b0, gbs}, {b1, gbs}};
  NppiSize roi = {W, H};
  int batch = 2, bufSize = gbs;
  const char *name = "";
  switch (which) {
    case 0: name = "ok"; break;
    case 1: name = "img0 ROI > max"; im[0].oSize = {W + 2, H}; roi = {W, H}; break;
    case 2: name = "img0 ROI 0"; im[0].oSize = {0, H}; break;
    case 3: name = "img0 ROI neg"; im[0].oSize = {-3, H}; break;
    case 4: name = "img0 step 0"; im[0].nStep = 0; break;
    case 5: name = "img0 step neg"; im[0].nStep = -4; break;
    case 6: name = "img0 null"; im[0].pData = nullptr; break;
    case 7: name = "buf0 null"; bu[0].pData = nullptr; break;
    case 8: name = "buf size 1"; bu[0].nBufferSize = 1; bu[1].nBufferSize = 1; break;
    case 9: name = "buf size 0"; bu[0].nBufferSize = 0; bu[1].nBufferSize = 0; break;
    case 10: name = "max roi 0"; roi = {0, H}; break;
    case 11: name = "max roi neg"; roi = {-1, H}; break;
    case 12: name = "batch 1"; batch = 1; break;
    case 13: name = "batch 0"; batch = 0; break;
    case 14: name = "batch -1"; batch = -1; break;
    case 15: name = "buf size 0 param"; bufSize = 0; break;
    case 16: name = "buf size neg param"; bufSize = -8; break;
    default: name = "?"; break;
  }
  Npp32u nm[2] = {0xAAAAu, 0xAAAAu};
  NppStatus st = nppiCompressMarkerLabelsUFBatch_32u_C1IR_Advanced_Ctx(im, bu, nm, batch, roi, bufSize, ctx);
  cudaError_t e = cudaDeviceSynchronize();
  printf("adv %-20s st=%d cuda=%s nm=%u,%u\n", name, (int)st, cudaGetErrorName(e), nm[0], nm[1]);
  return 0;
}
