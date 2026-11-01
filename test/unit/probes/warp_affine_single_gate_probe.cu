// Single-image WarpAffineBack gate probe: which mapped float coordinates are
// written?  8u C1 7x5 image (step 16), dst pre-filled with 0xEE.
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
  const double tx = argc > 1 ? atof(argv[1]) : 0.0;
  const double ty = argc > 2 ? atof(argv[2]) : 0.0;
  const int interp = argc > 3 ? atoi(argv[3]) : NPPI_INTER_NN;
  const int w = 7, hgt = 5, step = 16;
  unsigned char hs[16 * 5], hd[16 * 5];
  for (int i = 0; i < 16 * 5; ++i) { hs[i] = (unsigned char)(10 + (i * 7) % 90); hd[i] = 0xEE; }
  unsigned char *ds, *dd;
  CK(cudaMalloc(&ds, 16 * 5)); CK(cudaMalloc(&dd, 16 * 5));
  CK(cudaMemcpy(ds, hs, 80, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dd, hd, 80, cudaMemcpyHostToDevice));
  // dst -> src map (WarpAffineBack takes it directly).
  double m[2][3] = {{1, 0, tx}, {0, 1, ty}};
  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  NppStatus st = nppiWarpAffineBack_8u_C1R_Ctx(ds, NppiSize{w, hgt}, step, NppiRect{0, 0, w, hgt}, dd, step,
                                               NppiRect{0, 0, w, hgt}, m, interp, ctx);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hd, dd, 80, cudaMemcpyDeviceToHost));
  printf("map=(1,0,%g;0,1,%g) interp=%d status=%d written=", tx, ty, interp, (int)st);
  for (int y = 0; y < hgt; ++y) { printf("|"); for (int x = 0; x < w; ++x) printf("%c", hd[y * step + x] == 0xEE ? '.' : '#'); }
  printf("|\n");
  for (int y = 0; y < hgt; ++y) { for (int x = 0; x < w; ++x) printf("%3d ", (int)hd[y * step + x]); printf("\n"); }
  return 0;
}
