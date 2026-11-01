// Advanced box: compare with the standard replicate-box model on gradient images.
// usage: probe_f9w sw sh W H offx offy mw mh ax ay pad
#include <cuda_runtime.h>
#include <npp.h>
#include "nppcore.h"
extern "C" { NppStatus nppGetStreamContext(NppStreamContext *); }
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>

static NppStreamContext g_ctx;
static int clampv(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

int main(int argc, char **argv) {
  nppGetStreamContext(&g_ctx);
  int sw = atoi(argv[1]), sh = atoi(argv[2]), W = atoi(argv[3]), H = atoi(argv[4]);
  int offx = atoi(argv[5]), offy = atoi(argv[6]), mw = atoi(argv[7]), mh = atoi(argv[8]);
  int ax = atoi(argv[9]), ay = atoi(argv[10]), pad = argc > 11 ? atoi(argv[11]) : 8;
  int aw = sw + 2 * pad, ah = sh + 2 * pad, px = pad, py = pad;
  Npp8u *sp = nullptr, *dp = nullptr, *bp = nullptr;
  std::vector<Npp8u> img((size_t)aw * ah, 0);
  for (int j = 0; j < sh; ++j)
    for (int i = 0; i < sw; ++i) img[(size_t)(py + j) * aw + px + i] = (Npp8u)((7 * i + 13 * j) % 251);
  cudaMalloc(&sp, img.size());
  cudaMemcpy(sp, img.data(), img.size(), cudaMemcpyHostToDevice);
  cudaMalloc(&dp, (size_t)W * H);
  cudaMemset(dp, 0x55, (size_t)W * H);
  int bsz = -1;
  nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{W, H}, 1, &bsz);
  cudaMalloc(&bp, (size_t)bsz);
  cudaMemset(bp, 0, (size_t)bsz);
  const Npp8u *src = sp + (size_t)(py + offy) * aw + px + offx;
  NppStatus st = nppiFilterBoxBorderAdvanced_8u_C1R_Ctx(src, aw, NppiSize{sw, sh},
      NppiPoint{offx, offy}, dp, W, NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay},
      NPP_BORDER_REPLICATE, bp, g_ctx);
  std::vector<Npp8u> out((size_t)W * H);
  cudaMemcpy(out.data(), dp, out.size(), cudaMemcpyDeviceToHost);
  int mism = 0; std::string ex;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      int s = 0;
      for (int j = 0; j < mh; ++j)
        for (int i = 0; i < mw; ++i) {
          int ix = clampv(offx + x - ax + i, 0, sw - 1);
          int iy = clampv(offy + y - ay + j, 0, sh - 1);
          s += (int)img[(size_t)(py + iy) * aw + px + ix];
        }
      int e = s / (mw * mh);
      int g = out[(size_t)y * W + x];
      if (e != g) { ++mism; if (mism <= 3) { char b[64]; snprintf(b, sizeof b, " (%d,%d)%d/%d", x, y, g, e); ex += b; } }
    }
  printf("sw=%d sh=%d roi=%dx%d off=%d,%d m=%dx%d a=%d,%d pad=%d st=%d bsz=%d mism=%d/%d%s\n",
         sw, sh, W, H, offx, offy, mw, mh, ax, ay, pad, (int)st, bsz, mism, W * H, ex.c_str());
  return 0;
}
