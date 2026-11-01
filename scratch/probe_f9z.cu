// 64f_C1R advanced: does it ever succeed?
#include <cuda_runtime.h>
#include <npp.h>
#include "nppcore.h"
extern "C" { NppStatus nppGetStreamContext(NppStreamContext *); }
#include <cstdio>
static NppStreamContext g_ctx;
static const char *sn(NppStatus s) {
  switch (s) {
    case NPP_NO_ERROR: return "OK";
    case NPP_NULL_POINTER_ERROR: return "NULLPTR";
    case NPP_SIZE_ERROR: return "SIZE";
    case NPP_STEP_ERROR: return "STEP";
    case NPP_NOT_SUPPORTED_MODE_ERROR: return "NOTSUP";
    default: return "OTHER";
  }
}
int main() {
  nppGetStreamContext(&g_ctx);
  for (int W : {1, 3, 4, 8, 16, 64})
    for (int pad = 0; pad < 2; ++pad) {
      int H = W;
      int step = (W + pad * 4) * 8;
      Npp64f *s = nullptr, *d = nullptr;
      Npp8u *b = nullptr;
      cudaMalloc(&s, 1 << 20); cudaMemset(s, 0, 1 << 20);
      cudaMalloc(&d, 1 << 20); cudaMemset(d, 0, 1 << 20);
      int bs = -1;
      NppStatus bst = nppiFilterBoxBorderAdvancedGetDeviceBufferSize_64(NppiSize{W, H}, 1, &bs);
      cudaMalloc(&b, 1 << 20); cudaMemset(b, 0, 1 << 20);
      NppStatus st = nppiFilterBoxBorderAdvanced_64f_C1R_Ctx(s, step, NppiSize{W, H}, NppiPoint{0, 0},
          d, step, NppiSize{W, H}, NppiSize{3, 3}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, b, g_ctx);
      int bs2 = -1;
      NppStatus bst2 = nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{W, H}, 1, &bs2);
      printf("W=%2d pad=%d bsz64=%s/%d bsz32=%s/%d call=%s\n", W, pad, sn(bst), bs, sn(bst2), bs2, sn(st));
      cudaFree(s); cudaFree(d); cudaFree(b);
    }
  return 0;
}
