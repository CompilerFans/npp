// isolate why 8u_C1R advanced reports NULL_POINTER in probe_f9m
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
    case NPP_MASK_SIZE_ERROR: return "MASK";
    case NPP_ANCHOR_ERROR: return "ANCHOR";
    default: return "OTHER";
  }
}

static NppStatus run(int W, int H, int sw, int sh, int offx, int offy, int mw, int mh,
                     bool bigdst, bool bigbuf, bool memfill) {
  int aw = sw + 64, ah = sh + 64;
  std::vector<Npp8u> h((size_t)aw * ah, 7);
  Npp8u *sp = nullptr;
  cudaMalloc(&sp, h.size());
  cudaMemcpy(sp, h.data(), h.size(), cudaMemcpyHostToDevice);
  const Npp8u *src = sp + (size_t)(16 + offy) * aw + 16 + offx;
  Npp8u *dp = nullptr;
  size_t dbytes = (size_t)W * H + (bigdst ? 4096 : 0);
  cudaMalloc(&dp, dbytes);
  cudaMemset(dp, 0x61, dbytes);
  int bsz = -1;
  nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{W, H}, 1, &bsz);
  Npp8u *bp = nullptr;
  size_t bbytes = (size_t)bsz + (bigbuf ? 4096 : 0);
  cudaMalloc(&bp, bbytes);
  if (memfill) cudaMemset(bp, 0, bbytes);
  NppStatus st = nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(
      src, aw, NppiSize{sw, sh}, NppiPoint{offx, offy}, dp, W, NppiSize{W, H},
      NppiSize{mw, mh}, NppiPoint{mw / 2, mh / 2}, NPP_BORDER_REPLICATE, bp, g_ctx);
  cudaFree(sp); cudaFree(dp); cudaFree(bp);
  return st;
}

int main() {
  nppGetStreamContext(&g_ctx);
  for (int big = 0; big < 2; ++big)
    for (int bb = 0; bb < 2; ++bb)
      for (int mf = 0; mf < 2; ++mf) {
        printf("bigdst=%d bigbuf=%d memfill=%d:", big, bb, mf);
        for (int W : {7, 8, 16, 32}) {
          NppStatus st = run(W, W, W + 9, W + 9, 2, 3, 3, 3, big, bb, mf);
          printf(" W%d=%s", W, sn(st));
        }
        printf("\n");
      }
  return 0;
}
