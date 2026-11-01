// advanced dst-step rule: status vs ROI width / dstep / dtype
#include <cuda_runtime.h>
#include <npp.h>
#include "nppcore.h"
extern "C" { NppStatus nppGetStreamContext(NppStreamContext *); }
#include <cstdio>
static NppStreamContext g_ctx;
static const char *sn(NppStatus s) {
  switch (s) {
    case NPP_NO_ERROR: return "OK";
    case NPP_STEP_ERROR: return "STEP";
    case NPP_NOT_EVEN_STEP_ERROR: return "NOTEVEN";
    case NPP_SIZE_ERROR: return "SIZE";
    default: return "OTHER";
  }
}
int main() {
  nppGetStreamContext(&g_ctx);
  Npp8u *s = nullptr, *d = nullptr, *b = nullptr;
  cudaMalloc(&s, 1 << 20); cudaMemset(s, 1, 1 << 20);
  cudaMalloc(&d, 1 << 20); cudaMemset(d, 0, 1 << 20);
  cudaMalloc(&b, 1 << 20); cudaMemset(b, 0, 1 << 20);
  int Ws[] = {6, 7, 9, 10, 11, 14, 18, 20};
  int Ds[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 18, 20};
  for (int w : Ws) {
    printf("W=%2d 8u_C1R:", w);
    for (int ds : Ds)
      printf(" %d=%s", ds, sn(nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(s, 128, NppiSize{128, 64}, NppiPoint{0, 0}, d, ds, NppiSize{w, 12}, NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, b, g_ctx)));
    printf("\n");
  }
  for (int w : {1, 3, 13}) {
    printf("W=%2d 16u_C1R:", w);
    for (int ds : Ds)
      printf(" %d=%s", ds, sn(nppiFilterBoxBorderAdvanced_16u_C1R_Ctx((Npp16u *)s, 256, NppiSize{128, 64}, NppiPoint{0, 0}, (Npp16u *)d, ds, NppiSize{w, 12}, NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, b, g_ctx)));
    printf("\n");
  }
  for (int w : {1, 3, 13}) {
    printf("W=%2d 32f_C4R:", w);
    for (int ds : Ds)
      printf(" %d=%s", ds, sn(nppiFilterBoxBorderAdvanced_32f_C4R_Ctx((Npp32f *)s, 512, NppiSize{128, 64}, NppiPoint{0, 0}, (Npp32f *)d, ds, NppiSize{w, 12}, NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, b, g_ctx)));
    printf("\n");
  }
  return 0;
}
