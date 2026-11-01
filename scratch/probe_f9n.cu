// nppiFilterBoxBorderAdvanced 8u_C1R sweep: which inputs give NULL_POINTER?
#include <cuda_runtime.h>
#include <npp.h>
#include "nppcore.h"
extern "C" { NppStatus nppGetStreamContext(NppStreamContext *); }
#include <cstdio>
#include <vector>

static NppStreamContext g_ctx;
static const char *sn(NppStatus s) {
  switch (s) {
    case NPP_NO_ERROR: return "OK";
    case NPP_NULL_POINTER_ERROR: return "NULLPTR";
    case NPP_SIZE_ERROR: return "SIZE";
    case NPP_STEP_ERROR: return "STEP";
    case NPP_NOT_EVEN_STEP_ERROR: return "NOTEVEVEN";
    case NPP_MASK_SIZE_ERROR: return "MASK";
    case NPP_ANCHOR_ERROR: return "ANCHOR";
    case NPP_NOT_SUPPORTED_MODE_ERROR: return "NOTSUP";
    default: return "OTHER";
  }
}

int main() {
  nppGetStreamContext(&g_ctx);
  int ws[] = {3, 7, 8, 16};
  int ss[] = {16, 20, 32, 64};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      for (int offi = 0; offi < 2; ++offi)
        for (int m = 0; m < 2; ++m) {
          int W = ws[i], H = W, sw = ss[j], sh = sw;
          int offx = offi ? 2 : 0, offy = offi ? 3 : 0;
          int mw = m ? 5 : 3, mh = mw, ax = mw / 2, ay = mh / 2;
          int aw = sw + 64;
          std::vector<Npp8u> h((size_t)aw * (sh + 64), 7);
          Npp8u *sp = nullptr, *dp = nullptr, *bp = nullptr;
          cudaMalloc(&sp, h.size());
          cudaMemcpy(sp, h.data(), h.size(), cudaMemcpyHostToDevice);
          const Npp8u *src = sp + (size_t)(16 + offy) * aw + 16 + offx;
          size_t dbytes = (size_t)W * H + 4096;
          cudaMalloc(&dp, dbytes);
          cudaMemset(dp, 0x61, dbytes);
          int bsz = -1;
          nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{W, H}, 1, &bsz);
          cudaMalloc(&bp, (size_t)bsz + 4096);
          cudaMemset(bp, 0, (size_t)bsz + 4096);
          NppStatus st = nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(
              src, aw, NppiSize{sw, sh}, NppiPoint{offx, offy}, dp, W, NppiSize{W, H},
              NppiSize{mw, mh}, NppiPoint{ax, ay}, NPP_BORDER_REPLICATE, bp, g_ctx);
          printf("W=%2d src=%2d off=(%d,%d) m=%dx%d bsz=%5d step=%2d dststep=%2d -> %s\n", W, sw,
                 offx, offy, mw, mh, bsz, aw, W, sn(st));
          cudaFree(sp); cudaFree(dp); cudaFree(bp);
        }
  return 0;
}
