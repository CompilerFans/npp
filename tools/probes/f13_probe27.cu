// f13 probe 27: status matrices for BilateralGaussBorder,
// ThresholdAdaptiveBoxBorder and HarrisCornersBorder.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const char *es(NppStatus s) { static char b[32]; snprintf(b, 32, "%d", (int)s); return b; }

int main() {
  const int W = 12, H = 8, SP = 12, DP = 12;
  NppStreamContext c{};
  c.hStream = 0;
  Npp8u *d8, *o8, *buf;
  Npp32f *of;
  cudaMalloc(&d8, 4096);
  cudaMalloc(&o8, 4096);
  cudaMalloc(&of, 4096);
  cudaMalloc(&buf, 1 << 16);
  cudaMemset(d8, 5, 4096);
  cudaMemset(buf, 0, 1 << 16);

  struct B { const char *n; Npp8u *s; Npp8u *d2; int ss, ds; NppiSize sz, roi;
             NppiPoint off; int r, step; float vs, ps; NppiBorderType bt; };
  B bs[] = {
      {"ok", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, 1, 1, 10.f, 2.f, NPP_BORDER_REPLICATE},
      {"null src", nullptr, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, 1, 1, 10.f, 2.f, NPP_BORDER_REPLICATE},
      {"null dst", d8, nullptr, SP, DP, {W, H}, {8, 6}, {0, 0}, 1, 1, 10.f, 2.f, NPP_BORDER_REPLICATE},
      {"r0", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, 0, 1, 10.f, 2.f, NPP_BORDER_REPLICATE},
      {"r33", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, 33, 1, 10.f, 2.f, NPP_BORDER_REPLICATE},
      {"rneg", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, -1, 1, 10.f, 2.f, NPP_BORDER_REPLICATE},
      {"step0", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, 1, 0, 10.f, 2.f, NPP_BORDER_REPLICATE},
      {"stepneg", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, 1, -1, 10.f, 2.f, NPP_BORDER_REPLICATE},
      {"step33", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, 1, 33, 10.f, 2.f, NPP_BORDER_REPLICATE},
      {"step4", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, 1, 4, 10.f, 2.f, NPP_BORDER_REPLICATE},
      {"vs0", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, 1, 1, 0.f, 2.f, NPP_BORDER_REPLICATE},
      {"ps0", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, 1, 1, 10.f, 0.f, NPP_BORDER_REPLICATE},
      {"vsneg", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, 1, 1, -1.f, 2.f, NPP_BORDER_REPLICATE},
      {"mirror", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, 1, 1, 10.f, 2.f, NPP_BORDER_MIRROR},
      {"constant", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, 1, 1, 10.f, 2.f, NPP_BORDER_CONSTANT},
      {"roineg", d8, o8, SP, DP, {W, H}, {-1, 6}, {0, 0}, 1, 1, 10.f, 2.f, NPP_BORDER_REPLICATE},
      {"roizero", d8, o8, SP, 1, {W, H}, {0, 6}, {0, 0}, 1, 1, 10.f, 2.f, NPP_BORDER_REPLICATE},
      {"srcsize0", d8, o8, SP, DP, {0, H}, {8, 6}, {0, 0}, 1, 1, 10.f, 2.f, NPP_BORDER_REPLICATE},
      {"offneg", d8, o8, SP, DP, {W, H}, {8, 6}, {-1, 0}, 1, 1, 10.f, 2.f, NPP_BORDER_REPLICATE},
      {"offeq", d8, o8, SP, DP, {W, H}, {8, 6}, {W, 0}, 1, 1, 10.f, 2.f, NPP_BORDER_REPLICATE},
      {"steps", d8, o8, SP, 4, {W, H}, {8, 6}, {0, 0}, 1, 1, 10.f, 2.f, NPP_BORDER_REPLICATE},
      {"stepunalign", d8, o8, SP, 13, {W, H}, {8, 6}, {0, 0}, 1, 1, 10.f, 2.f, NPP_BORDER_REPLICATE},
      {"srcstep0", d8, o8, 0, DP, {W, H}, {8, 6}, {0, 0}, 1, 1, 10.f, 2.f, NPP_BORDER_REPLICATE},
      {"r0+badborder", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, 0, 1, 10.f, 2.f, NPP_BORDER_CONSTANT},
      {"step0+badborder", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, 1, 0, 10.f, 2.f, NPP_BORDER_CONSTANT},
      {"roineg+badborder", d8, o8, SP, DP, {W, H}, {-1, 6}, {0, 0}, 1, 1, 10.f, 2.f, NPP_BORDER_CONSTANT},
  };
  for (auto &b : bs) {
    NppStatus st = nppiFilterBilateralGaussBorder_8u_C1R_Ctx(b.s, b.ss, b.sz, b.off, b.d2, b.ds,
                                                             b.roi, b.r, b.step, b.vs, b.ps, b.bt, c);
    printf("bil %-18s st=%s\n", b.n, es(st));
  }

  struct A { const char *n; Npp8u *s; Npp8u *d2; int ss, ds; NppiSize sz, roi;
             NppiPoint off; NppiSize m; float delta; NppiBorderType bt; };
  A as[] = {
      {"ok", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, {3, 3}, 0.f, NPP_BORDER_REPLICATE},
      {"null src", nullptr, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, {3, 3}, 0.f, NPP_BORDER_REPLICATE},
      {"null dst", d8, nullptr, SP, DP, {W, H}, {8, 6}, {0, 0}, {3, 3}, 0.f, NPP_BORDER_REPLICATE},
      {"m0", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, {0, 0}, 0.f, NPP_BORDER_REPLICATE},
      {"m1", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, {1, 1}, 0.f, NPP_BORDER_REPLICATE},
      {"m2", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, {2, 2}, 0.f, NPP_BORDER_REPLICATE},
      {"m9", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, {9, 9}, 0.f, NPP_BORDER_REPLICATE},
      {"m3x5", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, {3, 5}, 0.f, NPP_BORDER_REPLICATE},
      {"mneg", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, {-3, 3}, 0.f, NPP_BORDER_REPLICATE},
      {"mirror", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, {3, 3}, 0.f, NPP_BORDER_MIRROR},
      {"roineg", d8, o8, SP, DP, {W, H}, {-1, 6}, {0, 0}, {3, 3}, 0.f, NPP_BORDER_REPLICATE},
      {"roizero", d8, o8, SP, 1, {W, H}, {0, 6}, {0, 0}, {3, 3}, 0.f, NPP_BORDER_REPLICATE},
      {"srcsize0", d8, o8, SP, DP, {0, H}, {8, 6}, {0, 0}, {3, 3}, 0.f, NPP_BORDER_REPLICATE},
      {"offneg", d8, o8, SP, DP, {W, H}, {8, 6}, {-1, 0}, {3, 3}, 0.f, NPP_BORDER_REPLICATE},
      {"offeq", d8, o8, SP, DP, {W, H}, {8, 6}, {W, 0}, {3, 3}, 0.f, NPP_BORDER_REPLICATE},
      {"stepsmall", d8, o8, SP, 4, {W, H}, {8, 6}, {0, 0}, {3, 3}, 0.f, NPP_BORDER_REPLICATE},
      {"stepunalign", d8, o8, SP, 13, {W, H}, {8, 6}, {0, 0}, {3, 3}, 0.f, NPP_BORDER_REPLICATE},
      {"srcstep0", d8, o8, 0, DP, {W, H}, {8, 6}, {0, 0}, {3, 3}, 0.f, NPP_BORDER_REPLICATE},
      {"m2+badborder", d8, o8, SP, DP, {W, H}, {8, 6}, {0, 0}, {2, 2}, 0.f, NPP_BORDER_CONSTANT},
  };
  for (auto &a : as) {
    NppStatus st = nppiFilterThresholdAdaptiveBoxBorder_8u_C1R_Ctx(
        a.s, a.ss, a.sz, a.off, a.d2, a.ds, a.roi, a.m, a.delta, 1, 0, a.bt, c);
    printf("at  %-18s st=%s\n", a.n, es(st));
  }

  struct Hc { const char *n; Npp8u *s; int ss; NppiSize sz, roi; NppiPoint off; int ds;
              NppiDifferentialKernel f; NppiMaskSize m, aw; float k, sc; NppiBorderType bt;
              Npp8u *buf; };
  Hc hs[] = {
      {"ok", d8, SP, {W, H}, {8, 6}, {0, 0}, 32, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, .04f, 1.f, NPP_BORDER_REPLICATE, buf},
      {"null src", nullptr, SP, {W, H}, {8, 6}, {0, 0}, 32, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, .04f, 1.f, NPP_BORDER_REPLICATE, buf},
      {"null dst", d8, SP, {W, H}, {8, 6}, {0, 0}, 32, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, .04f, 1.f, NPP_BORDER_REPLICATE, buf},
      {"null buf", d8, SP, {W, H}, {8, 6}, {0, 0}, 32, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, .04f, 1.f, NPP_BORDER_REPLICATE, nullptr},
      {"badsize", d8, SP, {W, H}, {8, 6}, {0, 0}, 32, (NppiDifferentialKernel)9, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, .04f, 1.f, NPP_BORDER_REPLICATE, buf},
      {"mask1", d8, SP, {W, H}, {8, 6}, {0, 0}, 32, NPP_FILTER_SOBEL, NPP_MASK_SIZE_1_X_3, NPP_MASK_SIZE_3_X_3, .04f, 1.f, NPP_BORDER_REPLICATE, buf},
      {"mask5", d8, SP, {W, H}, {8, 6}, {0, 0}, 32, NPP_FILTER_SOBEL, NPP_MASK_SIZE_5_X_5, NPP_MASK_SIZE_3_X_3, .04f, 1.f, NPP_BORDER_REPLICATE, buf},
      {"aw1", d8, SP, {W, H}, {8, 6}, {0, 0}, 32, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_1_X_3, .04f, 1.f, NPP_BORDER_REPLICATE, buf},
      {"aw7", d8, SP, {W, H}, {8, 6}, {0, 0}, 32, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_7_X_7, .04f, 1.f, NPP_BORDER_REPLICATE, buf},
      {"mirror", d8, SP, {W, H}, {8, 6}, {0, 0}, 32, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, .04f, 1.f, NPP_BORDER_MIRROR, buf},
      {"roineg", d8, SP, {W, H}, {-1, 6}, {0, 0}, 32, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, .04f, 1.f, NPP_BORDER_REPLICATE, buf},
      {"roizero", d8, SP, {W, H}, {0, 6}, {0, 0}, 32, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, .04f, 1.f, NPP_BORDER_REPLICATE, buf},
      {"srcsize0", d8, SP, {0, H}, {8, 6}, {0, 0}, 32, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, .04f, 1.f, NPP_BORDER_REPLICATE, buf},
      {"offneg", d8, SP, {W, H}, {8, 6}, {-1, 0}, 32, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, .04f, 1.f, NPP_BORDER_REPLICATE, buf},
      {"offeq", d8, SP, {W, H}, {8, 6}, {W, 0}, 32, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, .04f, 1.f, NPP_BORDER_REPLICATE, buf},
      {"stepsmall", d8, SP, {W, H}, {8, 6}, {0, 0}, 28, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, .04f, 1.f, NPP_BORDER_REPLICATE, buf},
      {"stepunalign", d8, SP, {W, H}, {8, 6}, {0, 0}, 34, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, .04f, 1.f, NPP_BORDER_REPLICATE, buf},
      {"srcstep0", d8, 0, {W, H}, {8, 6}, {0, 0}, 32, NPP_FILTER_SOBEL, NPP_MASK_SIZE_3_X_3, NPP_MASK_SIZE_3_X_3, .04f, 1.f, NPP_BORDER_REPLICATE, buf},
  };
  for (auto &h : hs) {
    NppStatus st = nppiFilterHarrisCornersBorder_8u32f_C1R_Ctx(
        h.s, h.ss, h.sz, h.off, of, h.ds, h.roi, h.f, h.m, h.aw, h.k, h.sc, h.bt, h.buf, c);
    printf("hc  %-18s st=%s\n", h.n, es(st));
  }
  cudaFree(d8);
  cudaFree(o8);
  cudaFree(of);
  cudaFree(buf);
  return 0;
}
