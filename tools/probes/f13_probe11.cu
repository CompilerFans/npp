// f13 probe 11: which GradientVector error-matrix call kills the CUDA
// context?  Each case runs in its own process (argv[1] selects); after the
// call we sync and probe cudaMalloc to detect the sticky fault.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const char *es(NppStatus s) { static char b[32]; snprintf(b, 32, "%d", (int)s); return b; }

int main(int argc, char **argv) {
  int which = argc > 1 ? atoi(argv[1]) : 0;
  const int W = 9, H = 9, R = 4;
  const int sstep = 16, dstep = R * 2, astep = R * 4;
  std::vector<Npp8u> h((size_t)W * H, 0);
  for (size_t i = 0; i < h.size(); ++i) h[i] = (Npp8u)((i * 41 + 7) & 0xff);
  Npp8u *d; cudaMalloc(&d, 4096); cudaMemcpy(d, h.data(), W * H, cudaMemcpyHostToDevice);
  Npp16s *dx, *dy, *dm; Npp32f *da;
  cudaMalloc(&dx, 4096); cudaMalloc(&dy, 4096); cudaMalloc(&dm, 4096); cudaMalloc(&da, 4096);
  NppStreamContext c{}; c.hStream = 0;
  const Npp8u *pSrc = d + 1 * sstep + 1;
  NppiSize sz{W, H}; NppiPoint off{1, 1}; NppiSize roi{R, R};
  typedef NppStatus (*F8)(const Npp8u*, int, NppiSize, NppiPoint, Npp16s*, int, Npp16s*, int, Npp16s*, int, Npp32f*, int, NppiSize, NppiMaskSize, NppiNorm, NppiBorderType, NppStreamContext);
  F8 f = nppiGradientVectorSobelBorder_8u16s_C1R_Ctx;
  typedef NppStatus (*F3)(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, Npp32f*, int, Npp32f*, int, Npp32f*, int, NppiSize, NppiMaskSize, NppiNorm, NppiBorderType, NppStreamContext);
  F3 g = nppiGradientVectorSobelBorder_32f_C1R_Ctx;
  Npp32f *fbuf; cudaMalloc(&fbuf, 4096);
  { std::vector<Npp32f> hf(1024); for (int i = 0; i < 1024; ++i) hf[i] = (Npp32f)((i * 37 + 5) % 25); cudaMemcpy(fbuf, hf.data(), 4096, cudaMemcpyHostToDevice); }
  const Npp32f *fs = fbuf + 1 * 4 + 1;  // 20-byte offset: 4-byte aligned like the diff tests
  NppStatus st = NPP_NO_ERROR;
  switch (which) {
    case 1: st = f(pSrc, sstep, sz, off, dx, dstep, dy, dstep, dm, dstep, da, astep, roi, (NppiMaskSize)0, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 2: st = f(pSrc, sstep, sz, off, dx, dstep, dy, dstep, dm, dstep, da, astep, roi, NPP_MASK_SIZE_7_X_7, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 3: st = f(pSrc, sstep, sz, off, dx, dstep, dy, dstep, dm, dstep, da, astep, roi, (NppiMaskSize)0, nppiNormInf, NPP_BORDER_REPLICATE, c); break;
    case 4: st = f(pSrc, sstep, sz, off, nullptr, 0, dy, 16, dm, 6, da, 16, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 5: st = f(pSrc, sstep, sz, off, nullptr, 0, nullptr, 0, dm, 16, da, 12, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 6: st = f(pSrc, sstep, sz, off, nullptr, 0, nullptr, 0, nullptr, 0, da, 3, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 7: st = f(pSrc, sstep, sz, off, dx, 3, dy, dstep, dm, dstep, da, astep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 8: st = f(pSrc, sstep, sz, off, dx, dstep, dy, dstep, dm, dstep, da, astep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_CONSTANT, c); break;
    case 9: st = f(pSrc, sstep, sz, off, dx, dstep, dy, dstep, dm, dstep, da, astep, roi, NPP_MASK_SIZE_3_X_3, nppiNormInf, NPP_BORDER_REPLICATE, c); break;
    case 11: st = g(fs, sstep, sz, off, (Npp32f *)dx, 16, (Npp32f *)dy, 16, (Npp32f *)dm, 16, da, 16, roi, (NppiMaskSize)0, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 12: st = g(fs, sstep, sz, off, (Npp32f *)dx, 16, (Npp32f *)dy, 16, (Npp32f *)dm, 16, da, 16, roi, NPP_MASK_SIZE_7_X_7, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 13: st = g(fs, sstep, sz, off, (Npp32f *)dx, 16, (Npp32f *)dy, 16, (Npp32f *)dm, 16, da, 16, roi, NPP_MASK_SIZE_5_X_5, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 14: st = g(fs, sstep, sz, off, nullptr, 0, (Npp32f *)dy, 16, (Npp32f *)dm, 12, da, 16, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 15: st = g(fs, sstep, sz, off, nullptr, 0, nullptr, 0, (Npp32f *)dm, 16, da, 12, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 16: st = g(fs, sstep, sz, off, nullptr, 0, nullptr, 0, nullptr, 0, da, 3, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 17: st = g(fs, sstep, sz, off, (Npp32f *)dx, 16, (Npp32f *)dy, 16, (Npp32f *)dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3, nppiNormInf, NPP_BORDER_REPLICATE, c); break;
    case 18: st = g(fs, sstep, sz, off, (Npp32f *)dx, 16, (Npp32f *)dy, 16, (Npp32f *)dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_CONSTANT, c); break;
    case 19: st = g(fs, sstep, sz, off, (Npp32f *)dx, 15, (Npp32f *)dy, 16, (Npp32f *)dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 20: st = g(fs, sstep, sz, off, (Npp32f *)dx, 17, (Npp32f *)dy, 16, (Npp32f *)dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 21: st = g(fs, sstep, sz, NppiPoint{9, 1}, (Npp32f *)dx, 16, (Npp32f *)dy, 16, (Npp32f *)dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 22: st = g(fs, sstep, NppiSize{0, 8}, off, (Npp32f *)dx, 16, (Npp32f *)dy, 16, (Npp32f *)dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 23: st = g(fs, sstep, sz, off, (Npp32f *)dx, 16, (Npp32f *)dy, 16, nullptr, 0, da, 16, roi, NPP_MASK_SIZE_3_X_3, nppiNormInf, NPP_BORDER_REPLICATE, c); break;
    case 24: st = g(fs, sstep, sz, off, (Npp32f *)dx, 16, (Npp32f *)dy, 16, (Npp32f *)dm, 16, da, 16, NppiSize{0, 4}, (NppiMaskSize)0, nppiNormInf, NPP_BORDER_CONSTANT, c); break;
    case 25: st = g(fs, sstep, sz, off, (Npp32f *)dx, 16, (Npp32f *)dy, 16, (Npp32f *)dm, 16, da, 16, NppiSize{4, -1}, (NppiMaskSize)0, nppiNormInf, NPP_BORDER_CONSTANT, c); break;
    case 26: st = g(fs, sstep, sz, off, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, roi, (NppiMaskSize)0, nppiNormInf, NPP_BORDER_CONSTANT, c); break;
    case 27: st = g(fs, sstep, NppiSize{0, 8}, off, (Npp32f *)dx, 3, (Npp32f *)dy, 16, (Npp32f *)dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 28: st = g(fs, sstep, sz, NppiPoint{0, -1}, (Npp32f *)dx, 3, (Npp32f *)dy, 16, (Npp32f *)dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 29: st = g(nullptr, sstep, sz, off, (Npp32f *)dx, 16, (Npp32f *)dy, 16, (Npp32f *)dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 30: st = g(fs, sstep, sz, off, nullptr, 0, (Npp32f *)dy, 3, (Npp32f *)dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 31: st = g(fs, sstep, sz, off, nullptr, 0, (Npp32f *)dy, 16, (Npp32f *)dm, 12, da, 16, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 32: st = g(fs, sstep, sz, off, nullptr, 0, nullptr, 0, (Npp32f *)dm, 16, da, 12, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 33: st = g(fs, sstep, sz, off, nullptr, 0, (Npp32f *)dy, 16, (Npp32f *)dm, 0, da, 16, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 34: st = g(fs, sstep, sz, off, nullptr, 0, nullptr, 0, (Npp32f *)dm, 16, da, 0, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 35: st = g(fs, sstep, sz, off, nullptr, 0, (Npp32f *)dy, 16, (Npp32f *)dm, 8, da, 16, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 36: st = g(fs, sstep, sz, off, nullptr, 0, nullptr, 0, (Npp32f *)dm, 16, da, 8, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 37: st = g(fs, sstep, sz, off, (Npp32f *)dx, 16, (Npp32f *)dy, 16, (Npp32f *)dm, 16, da, 16, roi, NPP_MASK_SIZE_5_X_5, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    case 38: st = g(fs, sstep, sz, off, (Npp32f *)dx, 16, (Npp32f *)dy, 16, (Npp32f *)dm, 16, da, 16, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); break;
    default: printf("no case\n"); return 0;
  }
  cudaError_t e1 = cudaDeviceSynchronize();
  void *probe = nullptr;
  cudaError_t e2 = cudaMalloc(&probe, 1024);
  printf("case %d: status=%s sync=%s malloc=%s\n", which, es(st), cudaGetErrorString(e1), cudaGetErrorString(e2));
  return 0;
}
