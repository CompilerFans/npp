#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <nppi.h>
#include <nppi_geometry_transforms.h>
extern "C" {
NppStatus nppGetStreamContext(NppStreamContext *c);
}

static void dump(const char *tag, const Npp8u *h, int w, int hgt, int step) {
  printf("%s\n", tag);
  for (int y = 0; y < hgt; ++y) {
    for (int x = 0; x < w; ++x) printf("%3d ", (int)h[y * step + x]);
    printf("\n");
  }
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("cuda err %s %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

struct Batch {
  static const int w = 7, hgt = 5, step = 8, n = 2;
  Npp8u hs[n][8 * 5], hd[n][8 * 5];
  Npp8u *ds[n], *dd[n];
  Npp64f hc[n][6];
  Npp64f *dc[n];
  NppiWarpAffineBatchCXR hl[n];
  NppiWarpAffineBatchCXR *dl;
  NppStreamContext ctx;

  int setup(const double (*coeffs)[6], Npp8u srcBase) {
    (void)srcBase;
    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < 8 * 5; ++k) { hs[i][k] = (Npp8u)(10 + k + 20 * i); hd[i][k] = 0xEE; }
      for (int k = 0; k < 6; ++k) hc[i][k] = coeffs[i][k];
      CK(cudaMalloc(&ds[i], 8 * 5));
      CK(cudaMalloc(&dd[i], 8 * 5));
      CK(cudaMemcpy(ds[i], hs[i], 40, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dd[i], hd[i], 40, cudaMemcpyHostToDevice));
      CK(cudaMalloc(&dc[i], 48));
      CK(cudaMemcpy(dc[i], hc[i], 48, cudaMemcpyHostToDevice));
      memset(&hl[i], 0, sizeof(hl[i]));
      hl[i].pSrc = ds[i]; hl[i].nSrcStep = step; hl[i].pDst = dd[i]; hl[i].nDstStep = step; hl[i].pCoeffs = dc[i];
    }
    CK(cudaMalloc(&dl, sizeof(hl)));
    CK(cudaMemcpy(dl, hl, sizeof(hl), cudaMemcpyHostToDevice));
    nppGetStreamContext(&ctx);
    NppStatus st = nppiWarpAffineBatchInit_Ctx(dl, n, ctx);
    if (st != NPP_NO_ERROR) printf("init status %d\n", (int)st);
    return 0;
  }
  int run(const char *tag, NppiRect srcRoi, NppiRect dstRoi, int interp) {
    NppStatus st = nppiWarpAffineBatch_8u_C1R_Ctx(NppiSize{w, hgt}, srcRoi, dstRoi, interp, dl, n, ctx);
    CK(cudaDeviceSynchronize());
    printf("%s status %d\n", tag, (int)st);
    for (int i = 0; i < n; ++i) {
      CK(cudaMemcpy(hd[i], dd[i], 40, cudaMemcpyDeviceToHost));
      char b[128]; snprintf(b, sizeof b, "%s img%d", tag, i);
      dump(b, hd[i], w, hgt, step);
    }
    return 0;
  }
};

int main() {
  const double ident[2][6] = {{1, 0, 0, 0, 1, 0}, {1, 0, 0, 0, 1, 0}};
  Batch b;
  if (b.setup(ident, 10)) return 1;
  b.run("identity srcROI(2,1,5,4) dstROI(0,0,7,5)", NppiRect{2, 1, 5, 4}, NppiRect{0, 0, 7, 5}, NPPI_INTER_NN);
  b.run("identity full ROIs", NppiRect{0, 0, 7, 5}, NppiRect{0, 0, 7, 5}, NPPI_INTER_NN);
  b.run("identity srcROI(2,1,5,4) dstROI(2,1,5,4)", NppiRect{2, 1, 5, 4}, NppiRect{2, 1, 5, 4}, NPPI_INTER_NN);
  return 0;
}
