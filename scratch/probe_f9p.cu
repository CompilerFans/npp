// 8u advanced: print dst for ch1/3/4 on constant and gradient images (ROI 7x7)
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
    default: return "OTHER";
  }
}

static void run(int ch, int cst) {
  const int W = 7, H = 7, sw = 16, sh = 16, offx = 0, offy = 0, mw = 3, mh = 3;
  const int aw = sw + 64, ah = sh + 64;
  std::vector<Npp8u> h((size_t)aw * ah * ch, 0);
  for (int j = 0; j < sh; ++j)
    for (int i = 0; i < sw; ++i)
      for (int c = 0; c < ch; ++c) {
        int v = cst ? 100 : ((i * 3 + j * 5 + c * 7) % 97);
        h[((size_t)(16 + j) * aw + (size_t)(16 + i) * ch) + c] = (Npp8u)v;
      }
  Npp8u *sp = nullptr;
  cudaMalloc(&sp, h.size());
  cudaMemcpy(sp, h.data(), h.size(), cudaMemcpyHostToDevice);
  const Npp8u *src = sp + (size_t)16 * aw + 16 * ch;
  const int dw = W * ch;
  Npp8u *dp = nullptr;
  cudaMalloc(&dp, (size_t)dw * H);
  cudaMemset(dp, 0x61, (size_t)dw * H);
  int bsz = -1;
  nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{W, H}, ch, &bsz);
  Npp8u *bp = nullptr;
  cudaMalloc(&bp, (size_t)bsz);
  cudaMemset(bp, 0, (size_t)bsz);
  NppStatus st;
  if (ch == 1)
    st = nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(src, aw, NppiSize{sw, sh}, NppiPoint{offx, offy},
         dp, dw, NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, bp, g_ctx);
  else if (ch == 3)
    st = nppiFilterBoxBorderAdvanced_8u_C3R_Ctx(src, aw, NppiSize{sw, sh}, NppiPoint{offx, offy},
         dp, dw, NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, bp, g_ctx);
  else
    st = nppiFilterBoxBorderAdvanced_8u_C4R_Ctx(src, aw, NppiSize{sw, sh}, NppiPoint{offx, offy},
         dp, dw, NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{1, 1}, NPP_BORDER_REPLICATE, bp, g_ctx);
  std::vector<Npp8u> out((size_t)dw * H);
  cudaMemcpy(out.data(), dp, out.size(), cudaMemcpyDeviceToHost);
  printf("== ch%d img=%s st=%s bsz=%d\n", ch, cst ? "const100" : "gradient", sn(st), bsz);
  printf("   src row0(px0..6, ch0):");
  for (int x = 0; x < 7; ++x) printf(" %3d", (int)h[(size_t)16 * aw + (size_t)x * ch]);
  printf("\n");
  if (st == NPP_NO_ERROR)
    for (int y = 0; y < H; ++y) {
      printf("   ");
      for (int x = 0; x < W; ++x) printf(" %3d", (int)out[(size_t)y * dw + x * ch]);
      printf("\n");
    }
  cudaFree(sp); cudaFree(dp); cudaFree(bp);
  fflush(stdout);
}

int main() {
  nppGetStreamContext(&g_ctx);
  for (int ch : {1, 3, 4}) {
    run(ch, 1);
    run(ch, 0);
  }
  return 0;
}
