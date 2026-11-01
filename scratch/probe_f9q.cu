// Full impulse-response decode of nppiFilterBoxBorderAdvanced (32f_C1R).
#include <cuda_runtime.h>
#include <npp.h>
#include "nppcore.h"
extern "C" { NppStatus nppGetStreamContext(NppStreamContext *); }
#include <cstdio>
#include <cstring>
#include <vector>

static NppStreamContext g_ctx;
static const char *sn(NppStatus s) {
  switch (s) {
    case NPP_NO_ERROR: return "OK";
    case NPP_NULL_POINTER_ERROR: return "NULLPTR";
    case NPP_SIZE_ERROR: return "SIZE";
    default: return "OTHER";
  }
}

static void decode(int sw, int sh, int W, int H, int offx, int offy, int mw, int mh, int ax,
                   int ay, bool detail) {
  int aw = sw + 8, ah = sh + 8;
  int px = 4, py = 4;  // image origin inside the padded allocation
  Npp32f *sp = nullptr, *dp = nullptr;
  cudaMalloc(&sp, (size_t)aw * ah * sizeof(float));
  cudaMalloc(&dp, (size_t)W * H * sizeof(float));
  int bsz = -1;
  nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{W, H}, 1, &bsz);
  Npp8u *bp = nullptr;
  cudaMalloc(&bp, (size_t)bsz);
  const Npp32f *src = sp + (size_t)(py + offy) * aw + px + offx;
  std::vector<Npp32f> img((size_t)aw * ah, 0.f), out((size_t)W * H, 0.f);
  printf("-- src=%dx%d roi=%dx%d off=(%d,%d) mask=%dx%d a=(%d,%d) bsz=%d\n", sw, sh, W, H, offx,
         offy, mw, mh, ax, ay, bsz);
  std::vector<std::vector<std::pair<int,int>>> taps((size_t)W * H);
  for (int q = 0; q < sh; ++q)
    for (int p = 0; p < sw; ++p) {
      std::fill(img.begin(), img.end(), 0.f);
      img[(size_t)(py + q) * aw + px + p] = 1.f;
      cudaMemcpy(sp, img.data(), img.size() * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemset(dp, 0, (size_t)W * H * sizeof(float));
      cudaMemset(bp, 0, (size_t)bsz);
      NppStatus st = nppiFilterBoxBorderAdvanced_32f_C1R_Ctx(
          src, aw * (int)sizeof(float), NppiSize{sw, sh}, NppiPoint{offx, offy}, dp, W * (int)sizeof(float),
          NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay}, NPP_BORDER_REPLICATE, bp, g_ctx);
      if (st != NPP_NO_ERROR) { printf("   (status %s at impulse %d,%d)\n", sn(st), p, q); return; }
      cudaMemcpy(out.data(), dp, out.size() * sizeof(float), cudaMemcpyDeviceToHost);
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
          if (out[(size_t)y * W + x] != 0.f) taps[(size_t)y * W + x].push_back({p, q});
    }
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      auto &t = taps[(size_t)y * W + x];
      if (!detail && !(y < 4 && x < 4)) continue;
      printf("   d(%2d,%2d) n=%2zu:", x, y, t.size());
      for (auto &pq : t) printf(" (%d,%d)", pq.first, pq.second);
      printf("\n");
    }
  }
  cudaFree(sp); cudaFree(dp); cudaFree(bp);
  fflush(stdout);
}

int main() {
  nppGetStreamContext(&g_ctx);
  decode(16, 16, 7, 7, 0, 0, 3, 3, 1, 1, true);
  decode(16, 16, 7, 7, 2, 3, 3, 3, 1, 1, true);
  decode(16, 16, 16, 16, 0, 0, 3, 3, 1, 1, false);
  decode(16, 16, 16, 16, 0, 0, 5, 5, 2, 2, false);
  decode(16, 16, 7, 7, 0, 0, 3, 3, 0, 0, true);
  return 0;
}
