// Batch sampling-gate probe: which mapped float coordinates are written?
// 8u C1 7x5 image (step 16), dst pre-filled with 0xEE, two identical images.
// Coeffs are the already-inverted dst -> src map.
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <nppi.h>
#include <nppi_geometry_transforms.h>
extern "C" {
NppStatus nppGetStreamContext(NppStreamContext *c);
}
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("cuda err %s %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

static void dump(const char *tag, const unsigned char *h, int w, int hgt, int step) {
  printf("%s\n", tag);
  for (int y = 0; y < hgt; ++y) {
    for (int x = 0; x < w; ++x) printf("%3d ", (int)h[y * step + x]);
    printf("\n");
  }
}

int main(int argc, char **argv) {
  const double tx = argc > 1 ? atof(argv[1]) : 0.0;
  const double ty = argc > 2 ? atof(argv[2]) : 0.0;
  const double sx = argc > 3 ? atof(argv[3]) : 1.0;
  const double sy = argc > 4 ? atof(argv[4]) : 1.0;
  const int interp = argc > 5 ? atoi(argv[5]) : NPPI_INTER_NN;
  const int scenario = argc > 6 ? atoi(argv[6]) : 0;
  const double a01 = argc > 7 ? atof(argv[7]) : 0.0;
  const double a10 = argc > 8 ? atof(argv[8]) : 0.0;
  const int w = 7, hgt = 5, step = 16, n = 2;
  unsigned char hs[16 * 5], hd[16 * 5];
  for (int i = 0; i < 16 * 5; ++i) { hs[i] = (unsigned char)(10 + (i * 7) % 90); hd[i] = 0xEE; }
  unsigned char *ds, *dd;
  CK(cudaMalloc(&ds, 16 * 5));
  CK(cudaMalloc(&dd, 16 * 5));
  CK(cudaMemcpy(ds, hs, 80, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dd, hd, 80, cudaMemcpyHostToDevice));
  Npp64f hc[6] = {sx, a01, tx, a10, sy, ty};
  Npp64f *dc;
  CK(cudaMalloc(&dc, 48));
  CK(cudaMemcpy(dc, hc, 48, cudaMemcpyHostToDevice));
  NppiWarpAffineBatchCXR hl[2];
  memset(hl, 0, sizeof(hl));
  for (int i = 0; i < n; ++i) {
    hl[i].pSrc = ds; hl[i].nSrcStep = step; hl[i].pDst = dd; hl[i].nDstStep = step; hl[i].pCoeffs = dc;
  }
  NppiWarpAffineBatchCXR *dl;
  CK(cudaMalloc(&dl, sizeof(hl)));
  CK(cudaMemcpy(dl, hl, sizeof(hl), cudaMemcpyHostToDevice));
  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  NppStatus st = nppiWarpAffineBatchInit_Ctx(dl, n, ctx);
  if (st != NPP_NO_ERROR) { printf("init %d\n", (int)st); return 1; }
  {
    NppiWarpAffineBatchCXR back[2];
    CK(cudaMemcpy(back, dl, sizeof(back), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
      printf("init[%d] = {%g %g %g; %g %g %g}\n", i, back[i].aTransformedCoeffs[0][0], back[i].aTransformedCoeffs[0][1],
             back[i].aTransformedCoeffs[0][2], back[i].aTransformedCoeffs[1][0], back[i].aTransformedCoeffs[1][1],
             back[i].aTransformedCoeffs[1][2]);
    }
  }
  NppiRect srcRoi{0, 0, w, hgt};
  NppiRect dstRoi{0, 0, w, hgt};
  if (scenario == 1) { srcRoi = NppiRect{2, 1, 4, 3}; }
  if (scenario == 2) { dstRoi = NppiRect{0, 1, w, hgt - 1}; }
  st = nppiWarpAffineBatch_8u_C1R_Ctx(NppiSize{w, hgt}, srcRoi, dstRoi, interp, dl, n, ctx);
  CK(cudaDeviceSynchronize());
  printf("fwd={%g %g %g; %g %g %g} interp=%d scenario=%d status %d\n", sx, a01, tx, a10, sy, ty, interp, scenario, (int)st);
  CK(cudaMemcpy(hd, dd, 80, cudaMemcpyDeviceToHost));
  dump("dst (0xEE = untouched)", hd, w, hgt, step);
  printf("written: ");
  for (int y = 0; y < hgt; ++y) { printf("|"); for (int x = 0; x < w; ++x) printf("%c", hd[y * step + x] == 0xEE ? '.' : '#'); }
  printf("|\n");
  return 0;
}
