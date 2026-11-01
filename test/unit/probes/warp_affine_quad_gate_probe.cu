// Quad gate probe: identity quads, configurable ROIs.  Prints which destination
// pixels the 5090 writes (dst pre-filled with 0xEE).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <nppi.h>
#include <nppi_geometry_transforms.h>
extern "C" {
NppStatus nppGetStreamContext(NppStreamContext *c);
}
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("cuda err %s %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

int main(int argc, char **argv) {
  const int sx = argc > 1 ? atoi(argv[1]) : 0, sy = argc > 2 ? atoi(argv[2]) : 0;
  const int sw = argc > 3 ? atoi(argv[3]) : 17, sh = argc > 4 ? atoi(argv[4]) : 13;
  const int dx = argc > 5 ? atoi(argv[5]) : 0, dy = argc > 6 ? atoi(argv[6]) : 0;
  const int dw = argc > 7 ? atoi(argv[7]) : 17, dh = argc > 8 ? atoi(argv[8]) : 13;
  const int wx = argc > 9 ? atoi(argv[9]) : 2, wy = argc > 10 ? atoi(argv[10]) : 3;
  const int ww = argc > 11 ? atoi(argv[11]) : 13, wh = argc > 12 ? atoi(argv[12]) : 8;
  const int W = 17, H = 13, step = 33;
  unsigned char hs[33 * 13], hd[33 * 13];
  for (int i = 0; i < 33 * 13; ++i) { hs[i] = (unsigned char)(10 + (i * 7) % 90); hd[i] = 0xEE; }
  unsigned char *ds, *dd;
  CK(cudaMalloc(&ds, sizeof(hs))); CK(cudaMalloc(&dd, sizeof(hs)));
  CK(cudaMemcpy(ds, hs, sizeof(hs), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dd, hd, sizeof(hs), cudaMemcpyHostToDevice));
  // identity quad pair at (wx, wy)
  double q[4][2] = {{(double)wx, (double)wy}, {(double)(wx + ww), (double)wy},
                    {(double)(wx + ww), (double)(wy + wh)}, {(double)wx, (double)(wy + wh)}};
  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  NppStatus st = nppiWarpAffineQuad_8u_C1R_Ctx(ds, NppiSize{W, H}, step, NppiRect{sx, sy, sw, sh}, q, dd, step,
                                               NppiRect{dx, dy, dw, dh}, q, NPPI_INTER_NN, ctx);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hd, dd, sizeof(hs), cudaMemcpyDeviceToHost));
  printf("srcROI=(%d,%d,%d,%d) dstROI=(%d,%d,%d,%d) quad origin=(%d,%d) size=(%d,%d) status=%d\n", sx, sy, sw, sh, dx,
         dy, dw, dh, wx, wy, ww, wh, (int)st);
  for (int y = 0; y < H; ++y) {
    printf("%2d ", y);
    for (int x = 0; x < W; ++x) printf("%c", hd[y * step + x] == 0xEE ? '.' : '#');
    printf("\n");
  }
  return 0;
}
