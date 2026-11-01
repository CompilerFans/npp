// f13 probe 12: reproduce the GradientVector record-run context death.
// Attempt-2 symptom: running all three families in one process kills the
// context at the end of the Sobel test (Prewitt's first cudaMalloc fails
// with "misaligned address"), but each ok_statusonly case passes in a
// fresh process (probe11).  This probe replays the tail of the Sobel test
// in one process - argv[1] selects how much of the sequence runs - and
// then checks context health.
#include <npp.h>
#include <nppcore.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const char *he(NppStatus s) { static char b[32]; snprintf(b, 32, "%d", (int)s); return b; }

static bool healthy(const char *tag) {
  cudaError_t e1 = cudaDeviceSynchronize();
  void *p = nullptr;
  cudaError_t e2 = cudaMalloc(&p, 1024);
  if (p) cudaFree(p);
  (void)cudaGetLastError();
  printf("%-28s sync=%s malloc=%s\n", tag, cudaGetErrorString(e1), cudaGetErrorString(e2));
  return e1 == cudaSuccess && e2 == cudaSuccess;
}

int main(int argc, char **argv) {
  int upto = argc > 1 ? atoi(argv[1]) : 0;  // how many sequence steps to run
  typedef NppStatus (*F3)(const Npp32f *, int, NppiSize, NppiPoint, Npp32f *, int, Npp32f *, int,
                          Npp32f *, int, Npp32f *, int, NppiSize, NppiMaskSize, NppiNorm,
                          NppiBorderType, NppStreamContext);
  F3 g = nppiGradientVectorSobelBorder_32f_C1R_Ctx;
  NppStreamContext c{};
  if (argc > 2 && atoi(argv[2]) == 1) {
    // real stream context, like the diff tests use (nppGetStreamContext
    // fills device attributes that NPP may use for kernel launch config)
    NppStatus cst = nppGetStreamContext(&c);
    printf("ctx st=%s\n", he(cst));
  } else {
    c.hStream = 0;
  }
  Npp32f *s, *dx, *dy, *dm, *da;
  cudaMalloc(&s, 4096); cudaMalloc(&dx, 4096); cudaMalloc(&dy, 4096); cudaMalloc(&dm, 4096);
  cudaMalloc(&da, 4096);
  { std::vector<Npp32f> hf(1024); for (int i = 0; i < 1024; ++i) hf[i] = (Npp32f)((i * 37 + 5) % 25); cudaMemcpy(s, hf.data(), 4096, cudaMemcpyHostToDevice); }
  Npp32f *pSrc = s + 1 * 4 + 1;  // byte 20: 4-byte aligned, like the diff test
  NppiSize sz{9, 9}; NppiPoint off{1, 1}; NppiSize roi{4, 4};
  NppStatus st = NPP_NO_ERROR;
  int step = 0;
  // Sequence 1..2: the two ok_statusonly launches from the 32f error matrix,
  // in test order (xnull_mbadstep, then magonly_abadstep).
  if (upto >= ++step) { st = g(pSrc, 16, sz, off, nullptr, 0, dy, 16, dm, 12, da, 16, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); printf("xnull_mbadstep  st=%s\n", he(st)); }
  if (upto >= ++step) { st = g(pSrc, 16, sz, off, nullptr, 0, nullptr, 0, dm, 16, da, 12, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); printf("magonly_abadstep st=%s\n", he(st)); }
  // Sequence 3..5: a full 32f gv_ok happy path (5x5, like the Sobel-only
  // 5x5 ok cases) and then the two bad-stride launches.
  if (upto >= ++step) { st = g(pSrc, 16, sz, off, dx, 16, dy, 16, dm, 16, da, 16, roi, NPP_MASK_SIZE_5_X_5, nppiNormL2, NPP_BORDER_REPLICATE, c); printf("ok5x5           st=%s\n", he(st)); }
  if (upto >= ++step) { st = g(pSrc, 16, sz, off, nullptr, 0, dy, 16, dm, 12, da, 16, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); printf("xnull_mbadstep  st=%s\n", he(st)); }
  if (upto >= ++step) { st = g(pSrc, 16, sz, off, nullptr, 0, nullptr, 0, dm, 16, da, 12, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c); printf("magonly_abadstep st=%s\n", he(st)); }
  healthy("after sequence");
  return 0;
}
