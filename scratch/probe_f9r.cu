// Impulse-response dump of nppiFilterBoxBorderAdvanced (32f_C1R) for one config.
// usage: probe_f9r sw sh W H offx offy mw mh ax ay [pad]
// prints: D x y n f0 f1 ... (flat offsets relative to the image origin, row stride = sw+pad)
#include <cuda_runtime.h>
#include <npp.h>
#include "nppcore.h"
extern "C" { NppStatus nppGetStreamContext(NppStreamContext *); }
#include <cstdio>
#include <cstdlib>
#include <vector>

static NppStreamContext g_ctx;

int main(int argc, char **argv) {
  nppGetStreamContext(&g_ctx);
  if (argc < 11) { printf("usage\n"); return 1; }
  int sw = atoi(argv[1]), sh = atoi(argv[2]), W = atoi(argv[3]), H = atoi(argv[4]);
  int offx = atoi(argv[5]), offy = atoi(argv[6]), mw = atoi(argv[7]), mh = atoi(argv[8]);
  int ax = atoi(argv[9]), ay = atoi(argv[10]);
  int pad = argc > 11 ? atoi(argv[11]) : 8;
  int dstep = argc > 12 ? atoi(argv[12]) : W;
  int aw = sw + 2 * pad, ah = sh + 2 * pad;
  int dw = dstep > W ? dstep : W;
  int px = pad, py = pad;
  Npp32f *sp = nullptr, *dp = nullptr;
  cudaMalloc(&sp, (size_t)aw * ah * sizeof(float));
  cudaMalloc(&dp, ((size_t)dw * H) * sizeof(float));
  int bsz = -1;
  nppiFilterBoxBorderAdvancedGetDeviceBufferSize(NppiSize{W, H}, 1, &bsz);
  Npp8u *bp = nullptr;
  cudaMalloc(&bp, (size_t)bsz);
  const Npp32f *src = sp + (size_t)(py + offy) * aw + px + offx;
  std::vector<Npp32f> img((size_t)aw * ah, 0.f), out((size_t)dw * H, 0.f);
  printf("C sw=%d sh=%d W=%d H=%d off=%d,%d mask=%dx%d a=%d,%d pad=%d dstep=%d aw=%d bsz=%d\n", sw, sh, W, H,
         offx, offy, mw, mh, ax, ay, pad, dstep, aw, bsz);
  for (int q = 0; q < sh; ++q)
    for (int p = 0; p < sw; ++p) {
      std::fill(img.begin(), img.end(), 0.f);
      img[(size_t)(py + q) * aw + px + p] = 1.f;
      cudaMemcpy(sp, img.data(), img.size() * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemset(dp, 0, ((size_t)dw * H) * sizeof(float));
      cudaMemset(bp, 0, (size_t)bsz);
      NppStatus st = nppiFilterBoxBorderAdvanced_32f_C1R_Ctx(
          src, aw * (int)sizeof(float), NppiSize{sw, sh}, NppiPoint{offx, offy},
          dp, dstep * (int)sizeof(float), NppiSize{W, H}, NppiSize{mw, mh}, NppiPoint{ax, ay},
          NPP_BORDER_REPLICATE, bp, g_ctx);
      if (st != NPP_NO_ERROR) { printf("STATUS %d at %d,%d\n", (int)st, p, q); return 2; }
      cudaMemcpy(out.data(), dp, ((size_t)dw * H) * sizeof(float), cudaMemcpyDeviceToHost);
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
          if (out[(size_t)y * W + x] != 0.f)
            printf("D %d %d %d %d %.9g %d\n", x, y, p, q, out[(size_t)y * dstep + x], q * aw + p);
    }
  cudaFree(sp); cudaFree(dp); cudaFree(bp);
  return 0;
}
