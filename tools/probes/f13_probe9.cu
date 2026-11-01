// f13 probe 9: dump NPP 32f/16s32f/16u32f GV X/Y/Mag/Angle matrices for kernel fitting.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
  NppStreamContext c{}; c.hStream = 0;
  const int W = 9, H = 9, OFF = 2, R = 5;
  const int sstep = W * 4, dstep = R * 4;
  std::vector<Npp32f> imf((size_t)W * H);
  for (size_t i = 0; i < imf.size(); ++i) imf[i] = (float)((int)(i * 53 % 211) - 100) * 0.375f;
  printf("IMAGE 32f (9x9):\n");
  for (int y = 0; y < H; ++y) { for (int x = 0; x < W; ++x) printf(" %.4f", imf[(size_t)y * W + x]); printf("\n"); }
  Npp32f *s; cudaMalloc(&s, sstep * H); cudaMemcpy(s, imf.data(), sstep * H, cudaMemcpyHostToDevice);
  Npp32f *fx, *fy, *fm, *fa;
  cudaMalloc(&fx, dstep * H); cudaMalloc(&fy, dstep * H); cudaMalloc(&fm, dstep * H); cudaMalloc(&fa, dstep * H);
  const Npp32f *ps = s + OFF * (sstep / 4) + OFF;
  const NppiSize sz{W, H}; const NppiPoint off{OFF, OFF}; const NppiSize roi{R, R};
  auto dump = [&](const char *name, NppStatus st) {
    printf("== %s status %d\nX:\n", name, (int)st);
    std::vector<Npp32f> hx((size_t)dstep / 4 * R), hy((size_t)dstep / 4 * R), hm((size_t)dstep / 4 * R), ha((size_t)dstep / 4 * R);
    cudaMemcpy(hx.data(), fx, dstep * R, cudaMemcpyDeviceToHost);
    cudaMemcpy(hy.data(), fy, dstep * R, cudaMemcpyDeviceToHost);
    cudaMemcpy(hm.data(), fm, dstep * R, cudaMemcpyDeviceToHost);
    cudaMemcpy(ha.data(), fa, dstep * R, cudaMemcpyDeviceToHost);
    for (int y = 0; y < R; ++y) { for (int x = 0; x < R; ++x) printf(" %.4f", hx[(size_t)y * (dstep / 4) + x]); printf("\n"); }
    printf("Y:\n");
    for (int y = 0; y < R; ++y) { for (int x = 0; x < R; ++x) printf(" %.4f", hy[(size_t)y * (dstep / 4) + x]); printf("\n"); }
    printf("MAG(L2):\n");
    for (int y = 0; y < R; ++y) { for (int x = 0; x < R; ++x) printf(" %.4f", hm[(size_t)y * (dstep / 4) + x]); printf("\n"); }
    printf("ANGLE:\n");
    for (int y = 0; y < R; ++y) { for (int x = 0; x < R; ++x) printf(" %.6f", ha[(size_t)y * (dstep / 4) + x]); printf("\n"); }
  };
  dump("sobel 32f C1R 3x3 L2", nppiGradientVectorSobelBorder_32f_C1R_Ctx(ps, sstep, sz, off, fx, dstep, fy, dstep, fm, dstep, fa, dstep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c));
  dump("sobel 32f C1R 5x5 L2", nppiGradientVectorSobelBorder_32f_C1R_Ctx(ps, sstep, sz, off, fx, dstep, fy, dstep, fm, dstep, fa, dstep, roi, NPP_MASK_SIZE_5_X_5, nppiNormL2, NPP_BORDER_REPLICATE, c));
  dump("prewitt 32f C1R 3x3 L2", nppiGradientVectorPrewittBorder_32f_C1R_Ctx(ps, sstep, sz, off, fx, dstep, fy, dstep, fm, dstep, fa, dstep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c));
  dump("scharr 32f C1R 3x3 L2", nppiGradientVectorScharrBorder_32f_C1R_Ctx(ps, sstep, sz, off, fx, dstep, fy, dstep, fm, dstep, fa, dstep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c));
  // 16s32f
  {
    std::vector<Npp16s> ims((size_t)W * H);
    for (size_t i = 0; i < ims.size(); ++i) ims[i] = (Npp16s)((int)(i * 53 % 211) - 100);
    printf("IMAGE 16s (9x9):\n");
    for (int y = 0; y < H; ++y) { for (int x = 0; x < W; ++x) printf(" %d", ims[(size_t)y * W + x]); printf("\n"); }
    Npp16s *ss; cudaMalloc(&ss, sstep * H); cudaMemcpy(ss, ims.data(), sstep * H, cudaMemcpyHostToDevice);
    const Npp16s *pss = ss + OFF * (sstep / 4) + OFF;
    dump("sobel 16s32f C1R 3x3 L2", nppiGradientVectorSobelBorder_16s32f_C1R_Ctx(pss, sstep, sz, off, fx, dstep, fy, dstep, fm, dstep, fa, dstep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c));
  }
  printf("done\n");
  return 0;
}
