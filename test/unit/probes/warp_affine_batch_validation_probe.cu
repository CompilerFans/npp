// Batch validation probe: run one case per process (argv[1]) because some bad
// inputs make NVIDIA NPP fault.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <nppi.h>
#include <nppi_geometry_transforms.h>
extern "C" {
NppStatus nppGetStreamContext(NppStreamContext *c);
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("cuda err %s %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

int main(int argc, char **argv) {
  const int c = argc > 1 ? atoi(argv[1]) : 0;
  const int w = 7, hgt = 5, step = 8, n = 2;
  const int ch = (c == 20 || c == 21) ? 3 : 1;  // step cases use C3 8u
  std::vector<unsigned char> hs(n * 8 * 5 * 3), hd(n * 8 * 5 * 3, 0xEE);
  for (size_t i = 0; i < hs.size(); ++i) hs[i] = (unsigned char)(10 + (i % 97));
  void *ds[2], *dd[2];
  Npp64f hc[2][6] = {{1, 0, 0, 0, 1, 0}, {1, 0, 0, 0, 1, 0}};
  Npp64f *dc[2];
  NppiWarpAffineBatchCXR hl[2];
  for (int i = 0; i < n; ++i) {
    CK(cudaMalloc(&ds[i], 8 * 5 * 3)); CK(cudaMalloc(&dd[i], 8 * 5 * 3));
    CK(cudaMemcpy(ds[i], &hs[i * 8 * 5 * 3], 8 * 5 * 3, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dd[i], &hd[i * 8 * 5 * 3], 8 * 5 * 3, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dc[i], 48)); CK(cudaMemcpy(dc[i], hc[i], 48, cudaMemcpyHostToDevice));
    memset(&hl[i], 0, sizeof(hl[i]));
    hl[i].pSrc = ds[i];
    hl[i].nSrcStep = (ch == 3) ? 24 : step;
    hl[i].pDst = dd[i];
    hl[i].nDstStep = (ch == 3) ? 24 : step;
    hl[i].pCoeffs = dc[i];
  }
  NppiRect srcRoi{0, 0, w, hgt}, dstRoi{0, 0, w, hgt};
  NppiSize srcSize{w, hgt};
  int interp = NPPI_INTER_NN;
  unsigned batch = (unsigned)n;
  int mode = 0;
  if (c == 0) { /* baseline */ }
  else if (c == 1) { batch = 1; }
  else if (c == 2) { batch = 0; }
  else if (c == 3) { srcSize = NppiSize{1, 5}; srcRoi = NppiRect{0, 0, 1, 5}; }
  else if (c == 4) { srcRoi = NppiRect{0, 0, 1, 5}; }
  else if (c == 5) { srcRoi = NppiRect{20, 20, 4, 4}; }
  else if (c == 6) { srcRoi = NppiRect{-1, 0, 7, 5}; }
  else if (c == 7) { dstRoi = NppiRect{0, 0, 7, 0}; }
  else if (c == 8) { dstRoi = NppiRect{0, 0, 7, -3}; }
  else if (c == 9) { interp = NPPI_INTER_SUPER; }
  else if (c == 10) { srcRoi = NppiRect{6, 0, 4, 5}; }
  else if (c == 11) { srcSize = NppiSize{0, 5}; srcRoi = NppiRect{0, 0, 0, 5}; }
  else if (c == 12) { srcRoi = NppiRect{0, 0, 7, 1}; }
  else if (c == 13) { for (int i = 0; i < n; ++i) hl[i].pSrc = nullptr; }
  else if (c == 14) { for (int i = 0; i < n; ++i) hl[i].pDst = nullptr; }
  else if (c == 15) { for (int i = 0; i < n; ++i) hl[i].pCoeffs = nullptr; }
  else if (c == 20) { mode = 1; }  // C3, src step too small (below ROI row)
  else if (c == 21) { mode = 2; }  // C3, dst step too small
  else if (c == 22) { mode = 3; }  // C3, src step not multiple of 3
  else if (c == 23) { mode = 4; }  // C3, dst step not multiple of 3
  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  if (mode == 0) {
    for (int i = 0; i < n; ++i) hl[i].nSrcStep = (ch == 3) ? 24 : step;
    for (int i = 0; i < n; ++i) hl[i].nDstStep = (ch == 3) ? 24 : step;
  } else if (mode == 1) { for (int i = 0; i < n; ++i) hl[i].nSrcStep = 4; }
  else if (mode == 2) { for (int i = 0; i < n; ++i) hl[i].nDstStep = 4; }
  else if (mode == 3) { for (int i = 0; i < n; ++i) hl[i].nSrcStep = 25; }
  else if (mode == 4) { for (int i = 0; i < n; ++i) hl[i].nDstStep = 25; }
  NppiWarpAffineBatchCXR *dl;
  CK(cudaMalloc(&dl, sizeof(hl)));
  CK(cudaMemcpy(dl, hl, sizeof(hl), cudaMemcpyHostToDevice));
  NppStatus st = nppiWarpAffineBatchInit_Ctx(dl, batch, ctx);
  printf("case %d init %d\n", c, (int)st);
  if (st != NPP_NO_ERROR) return 0;
  if (ch == 3) {
    st = nppiWarpAffineBatch_8u_C3R_Ctx(srcSize, srcRoi, dstRoi, interp, dl, batch, ctx);
  } else {
    st = nppiWarpAffineBatch_8u_C1R_Ctx(srcSize, srcRoi, dstRoi, interp, dl, batch, ctx);
  }
  cudaError_t e = cudaDeviceSynchronize();
  printf("case %d batch %d err %s\n", c, (int)st, cudaGetErrorString(e));
  return 0;
}
