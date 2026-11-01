// f13 probe 3: why do 32f/16s32f GradientVector calls return NULL_POINTER?
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
extern "C" NppStatus nppGetStreamContext(NppStreamContext *);

static const char *es(NppStatus s) { switch (s) {
  case NPP_NO_ERROR: return "0"; case NPP_NULL_POINTER_ERROR: return "-8 null";
  case NPP_SIZE_ERROR: return "-6 size"; case NPP_OUT_OFF_RANGE_ERROR: return "-11 offrange";
  case NPP_NOT_SUPPORTED_MODE_ERROR: return "-14 mode"; case NPP_STEP_ERROR: return "-16 step";
  case NPP_NOT_EVEN_STEP_ERROR: return "-108 evenstep"; case NPP_MASK_SIZE_ERROR: return "-33 mask";
  case NPP_BAD_ARGUMENT_ERROR: return "-5 badarg"; default: { static char b[32]; snprintf(b,32,"%d",(int)s); return b; } } }

int main() {
  const int W = 9, H = 9, R = 4;
  const int sstep = W * 4, dstep = R * 4;
  std::vector<Npp32f> h((size_t)W * H, 0.f);
  for (size_t i = 0; i < h.size(); ++i) h[i] = (float)((i * 29 + 3) & 0xff);
  Npp32f *d; cudaMalloc(&d, sstep * H); cudaMemcpy(d, h.data(), sstep * H, cudaMemcpyHostToDevice);
  Npp32f *dx, *dy, *dm, *da;
  cudaMalloc(&dx, dstep * R); cudaMalloc(&dy, dstep * R); cudaMalloc(&dm, dstep * R); cudaMalloc(&da, dstep * R);
  printf("malloc dx=%p dy=%p dm=%p da=%p d=%p\n", (void*)dx, (void*)dy, (void*)dm, (void*)da, (void*)d);
  NppStreamContext c{}; NppStatus gs = nppGetStreamContext(&c);
  printf("getctx=%s stream=%p\n", es(gs), (void*)c.hStream);
  const Npp32f *pSrc = d + 1 * (sstep / 4) + 1;
  NppiSize sz{W, H}; NppiPoint off{1, 1}; NppiSize roi{R, R};
  // 1: zeroed ctx, all outputs
  NppStreamContext z{}; z.hStream = 0;
  printf("32f zeroed-ctx  all4 3x3 L2: %s\n", es(nppiGradientVectorSobelBorder_32f_C1R_Ctx(pSrc, sstep, sz, off, dx, dstep, dy, dstep, dm, dstep, da, dstep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, z)));
  // 2: real ctx
  printf("32f real-ctx    all4 3x3 L2: %s\n", es(nppiGradientVectorSobelBorder_32f_C1R_Ctx(pSrc, sstep, sz, off, dx, dstep, dy, dstep, dm, dstep, da, dstep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c)));
  // 3: mag/angle null
  printf("32f real-ctx    XY   3x3 L2: %s\n", es(nppiGradientVectorSobelBorder_32f_C1R_Ctx(pSrc, sstep, sz, off, dx, dstep, dy, dstep, nullptr, 0, nullptr, 0, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c)));
  // 4: L1
  printf("32f real-ctx    all4 3x3 L1: %s\n", es(nppiGradientVectorSobelBorder_32f_C1R_Ctx(pSrc, sstep, sz, off, dx, dstep, dy, dstep, dm, dstep, da, dstep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL1, NPP_BORDER_REPLICATE, c)));
  // 5: 5x5
  printf("32f real-ctx    all4 5x5 L2: %s\n", es(nppiGradientVectorSobelBorder_32f_C1R_Ctx(pSrc, sstep, sz, off, dx, dstep, dy, dstep, dm, dstep, da, dstep, roi, NPP_MASK_SIZE_5_X_5, nppiNormL2, NPP_BORDER_REPLICATE, c)));
  // 16s32f / 16u32f quick checks with real ctx
  {
    const int ss = W * 2;
    std::vector<Npp16s> h16((size_t)W * H, 0);
    for (size_t i = 0; i < h16.size(); ++i) h16[i] = (Npp16s)((int)(i * 37) % 251 - 125);
    Npp16s *d16; cudaMalloc(&d16, ss * H); cudaMemcpy(d16, h16.data(), ss * H, cudaMemcpyHostToDevice);
    const Npp16s *ps = d16 + 1 * (ss / 2) + 1;
    printf("16s32f real-ctx all4 3x3 L2: %s\n", es(nppiGradientVectorSobelBorder_16s32f_C1R_Ctx(ps, ss, sz, off, dx, dstep, dy, dstep, dm, dstep, da, dstep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c)));
    printf("16s32f real-ctx all4 5x5 L2: %s\n", es(nppiGradientVectorSobelBorder_16s32f_C1R_Ctx(ps, ss, sz, off, dx, dstep, dy, dstep, dm, dstep, da, dstep, roi, NPP_MASK_SIZE_5_X_5, nppiNormL2, NPP_BORDER_REPLICATE, c)));
    cudaFree(d16);
  }
  {
    const int ss = W * 2;
    std::vector<Npp16u> h16((size_t)W * H, 0);
    for (size_t i = 0; i < h16.size(); ++i) h16[i] = (Npp16u)((i * 53) % 65536);
    Npp16u *d16; cudaMalloc(&d16, ss * H); cudaMemcpy(d16, h16.data(), ss * H, cudaMemcpyHostToDevice);
    const Npp16u *ps = d16 + 1 * (ss / 2) + 1;
    printf("16u32f real-ctx all4 3x3 L2: %s\n", es(nppiGradientVectorSobelBorder_16u32f_C1R_Ctx(ps, ss, sz, off, dx, dstep, dy, dstep, dm, dstep, da, dstep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c)));
    cudaFree(d16);
  }
  // C3C1R 32f
  {
    const int ss = W * 3 * 4;
    std::vector<Npp32f> h3((size_t)W * H * 3, 0.f);
    for (size_t i = 0; i < h3.size(); ++i) h3[i] = (float)((i * 31 + 7) & 0xff);
    Npp32f *d3; cudaMalloc(&d3, ss * H); cudaMemcpy(d3, h3.data(), ss * H, cudaMemcpyHostToDevice);
    const Npp32f *ps = d3 + 1 * (ss / 4) + 1 * 3;
    printf("32f C3C1R real-ctx all4 3x3 L2: %s\n", es(nppiGradientVectorSobelBorder_32f_C3C1R_Ctx(ps, ss, sz, off, dx, dstep, dy, dstep, dm, dstep, da, dstep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c)));
    cudaFree(d3);
  }
  // numeric spot for the working path
  NppStatus st = nppiGradientVectorSobelBorder_32f_C1R_Ctx(pSrc, sstep, sz, off, dx, dstep, dy, dstep, dm, dstep, da, dstep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c);
  if (st == NPP_NO_ERROR) {
    std::vector<float> hx((size_t)dstep / 4 * R), hy = hx, hm = hx, ha = hx;
    cudaMemcpy(hx.data(), dx, dstep * R, cudaMemcpyDeviceToHost);
    cudaMemcpy(hy.data(), dy, dstep * R, cudaMemcpyDeviceToHost);
    cudaMemcpy(hm.data(), dm, dstep * R, cudaMemcpyDeviceToHost);
    cudaMemcpy(ha.data(), da, dstep * R, cudaMemcpyDeviceToHost);
    printf("spot X0=%.1f Y0=%.1f M0=%.2f A0=%.4f\n", hx[0], hy[0], hm[0], ha[0]);
  }
  printf("done\n");
  return 0;
}
