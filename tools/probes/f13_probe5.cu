// f13 probe 5: remaining GradientVector validation-order pairs + 5x5 mask support.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>

static const char *es(NppStatus s) { switch (s) {
  case NPP_NO_ERROR: return "0"; case NPP_NULL_POINTER_ERROR: return "-8 null";
  case NPP_SIZE_ERROR: return "-6 size"; case NPP_OUT_OFF_RANGE_ERROR: return "-11 offrange";
  case NPP_NOT_SUPPORTED_MODE_ERROR: return "-14 mode"; case NPP_STEP_ERROR: return "-16 step";
  case NPP_NOT_EVEN_STEP_ERROR: return "-108 evenstep"; case NPP_MASK_SIZE_ERROR: return "-33 mask";
  case NPP_BAD_ARGUMENT_ERROR: return "-5 badarg"; default: { static char b[32]; snprintf(b,32,"%d",(int)s); return b; } } }

static NppStreamContext mkctx() { NppStreamContext c{}; c.hStream = 0; return c; }

int main() {
  NppStreamContext c = mkctx();
  const int W = 9, H = 9, R = 4;
  const int sstep = W, dstep = R * 2, astep = R * 4;  // 8u src / 16s dst
  std::vector<Npp8u> h((size_t)W * H, 0);
  for (size_t i = 0; i < h.size(); ++i) h[i] = (Npp8u)((i * 73 + 11) & 0xff);
  Npp8u *d; cudaMalloc(&d, sstep * H); cudaMemcpy(d, h.data(), sstep * H, cudaMemcpyHostToDevice);
  Npp16s *dx, *dy, *dm; Npp32f *da;
  cudaMalloc(&dx, dstep * H); cudaMalloc(&dy, dstep * H); cudaMalloc(&dm, dstep * H); cudaMalloc(&da, astep * H);
  const Npp8u *pSrc = d + 1 * sstep + 1;
  typedef NppStatus (*F)(const Npp8u*, int, NppiSize, NppiPoint, Npp16s*, int, Npp16s*, int, Npp16s*, int, Npp32f*, int, NppiSize, NppiMaskSize, NppiNorm, NppiBorderType, NppStreamContext);
  F sob = nppiGradientVectorSobelBorder_8u16s_C1R_Ctx;
  F pre = nppiGradientVectorPrewittBorder_8u16s_C1R_Ctx;
  F sch = nppiGradientVectorScharrBorder_8u16s_C1R_Ctx;
  auto call = [&](F f, Npp16s *x, int xs, Npp16s *y, int ys, Npp16s *m, int ms, Npp32f *a, int as,
                  NppiSize rs = NppiSize{4, 4}, NppiMaskSize mk = NPP_MASK_SIZE_3_X_3, NppiNorm nm = nppiNormL2,
                  NppiBorderType bt = NPP_BORDER_REPLICATE, NppiSize sz_ = NppiSize{9, 9}, NppiPoint off_ = NppiPoint{1, 1}) {
    return f(pSrc, sstep, sz_, off_, x, xs, y, ys, m, ms, a, as, rs, mk, nm, bt, c);
  };
  printf("== order pairs\n");
  printf("  srcsize0_badborder  %s\n", es(call(sob, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_CONSTANT, NppiSize{0, 8})));
  printf("  offneg_badborder    %s\n", es(call(sob, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_CONSTANT, NppiSize{9,9}, NppiPoint{-1,1})));
  printf("  srcsize0_offneg     %s\n", es(call(sob, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, NppiSize{0, 8}, NppiPoint{-1,1})));
  printf("  badborder_badnorm   %s\n", es(call(sob, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_3_X_3, nppiNormInf, NPP_BORDER_CONSTANT)));
  printf("  badborder_badmask   %s\n", es(call(sob, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, (NppiMaskSize)7, nppiNormL2, NPP_BORDER_CONSTANT)));
  printf("  offneg_badnorm      %s\n", es(call(sob, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_3_X_3, nppiNormInf, NPP_BORDER_REPLICATE, NppiSize{9,9}, NppiPoint{-1,1})));
  printf("  srcsize0_badnorm    %s\n", es(call(sob, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_3_X_3, nppiNormInf, NPP_BORDER_REPLICATE, NppiSize{0, 8})));
  printf("== normInf vs null mag\n");
  printf("  normInf_magnull     %s\n", es(call(sob, dx, dstep, dy, dstep, nullptr, 0, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_3_X_3, nppiNormInf)));
  printf("  normInf_magok       %s\n", es(call(sob, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_3_X_3, nppiNormInf)));
  printf("  badnorm6_magok      %s\n", es(call(sob, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_3_X_3, (NppiNorm)6)));
  printf("== null X + bad ystep / magonly badstep\n");
  printf("  xnull_ybadstep      %s\n", es(call(sob, nullptr, 0, dy, 3, dm, dstep, da, astep)));
  printf("  xnull_ystep0        %s\n", es(call(sob, nullptr, 0, dy, 0, dm, dstep, da, astep)));
  printf("  magonly_badstep     %s\n", es(call(sob, nullptr, 0, nullptr, 0, dm, 3, da, astep)));
  printf("  angleonly           %s\n", es(call(sob, nullptr, 0, nullptr, 0, nullptr, 0, da, astep)));
  printf("== 5x5 mask support\n");
  printf("  prewitt 5x5         %s\n", es(call(pre, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_5_X_5)));
  printf("  scharr  5x5         %s\n", es(call(sch, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_5_X_5)));
  printf("== offset/roi vs src bounds (replicate)\n");
  printf("  off77_roi4x4        %s\n", es(call(sob, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, NppiSize{9,9}, NppiPoint{7,7})));
  printf("  off9x1              %s\n", es(call(sob, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, NppiSize{9,9}, NppiPoint{9,1})));
  printf("  off8x8_roi1x1       %s\n", es(call(sob, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{1,1}, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, NppiSize{9,9}, NppiPoint{8,8})));
  printf("  off0x0_roi1x1       %s\n", es(call(sob, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{1,1})));
  // 32f x-step alignment
  {
    const int W2 = 9, H2 = 9, R2 = 4, ss2 = W2 * 4, ds2 = R2 * 4;
    std::vector<Npp32f> h2((size_t)W2 * H2, 0.f);
    for (size_t i = 0; i < h2.size(); ++i) h2[i] = (float)((int)(i * 13 % 97) - 48) * 0.25f;
    Npp32f *s2; cudaMalloc(&s2, ss2 * H2); cudaMemcpy(s2, h2.data(), ss2 * H2, cudaMemcpyHostToDevice);
    Npp32f *x2, *y2, *m2, *a2;
    cudaMalloc(&x2, ds2 * H2); cudaMalloc(&y2, ds2 * H2); cudaMalloc(&m2, ds2 * H2); cudaMalloc(&a2, ds2 * H2);
    typedef NppStatus (*F2)(const Npp32f*, int, NppiSize, NppiPoint, Npp32f*, int, Npp32f*, int, Npp32f*, int, Npp32f*, int, NppiSize, NppiMaskSize, NppiNorm, NppiBorderType, NppStreamContext);
    F2 f2 = nppiGradientVectorSobelBorder_32f_C1R_Ctx;
    auto c2 = [&](Npp32f *x, int xs, Npp32f *y, int ys, Npp32f *m, int ms, Npp32f *a, int as,
                  NppiSize rs = NppiSize{4,4}, NppiMaskSize mk = NPP_MASK_SIZE_3_X_3, NppiNorm nm = nppiNormL2,
                  NppiBorderType bt = NPP_BORDER_REPLICATE, NppiSize sz_ = NppiSize{9,9}, NppiPoint off_ = NppiPoint{1,1}) {
      return f2(s2 + 1 * (ss2/4) + 1, ss2, sz_, off_, x, xs, y, ys, m, ms, a, as, rs, mk, nm, bt, c);
    };
    printf("== 32f dst steps (roiW=4 needs 16, align 4)\n");
    printf("  32f xstep15    %s\n", es(c2(x2, 15, y2, ds2, m2, ds2, a2, ds2)));
    printf("  32f xstep17    %s\n", es(c2(x2, 17, y2, ds2, m2, ds2, a2, ds2)));
    printf("  32f xstep20    %s\n", es(c2(x2, 20, y2, ds2, m2, ds2, a2, ds2)));
    printf("  32f 5x5 ok     %s\n", es(c2(x2, ds2, y2, ds2, m2, ds2, a2, ds2, NppiSize{4,4}, NPP_MASK_SIZE_5_X_5)));
    printf("  32f prew5x5    %s\n", es(nppiGradientVectorPrewittBorder_32f_C1R_Ctx(s2 + (ss2/4) + 1, ss2, NppiSize{9,9}, NppiPoint{1,1}, x2, ds2, y2, ds2, m2, ds2, a2, ds2, NppiSize{4,4}, NPP_MASK_SIZE_5_X_5, nppiNormL2, NPP_BORDER_REPLICATE, c)));
    printf("  32f scharr5x5  %s\n", es(nppiGradientVectorScharrBorder_32f_C1R_Ctx(s2 + (ss2/4) + 1, ss2, NppiSize{9,9}, NppiPoint{1,1}, x2, ds2, y2, ds2, m2, ds2, a2, ds2, NppiSize{4,4}, NPP_MASK_SIZE_5_X_5, nppiNormL2, NPP_BORDER_REPLICATE, c)));
    // 16s32f step check
    typedef NppStatus (*F3)(const Npp16s*, int, NppiSize, NppiPoint, Npp32f*, int, Npp32f*, int, Npp32f*, int, Npp32f*, int, NppiSize, NppiMaskSize, NppiNorm, NppiBorderType, NppStreamContext);
    F3 f3 = nppiGradientVectorSobelBorder_16s32f_C1R_Ctx;
    printf("== 16s32f dst xstep (needs 16)\n");
    printf("  16s32f xstep15 %s\n", es(f3((const Npp16s*)(s2 + 1 * (ss2/4) + 1), ss2, NppiSize{9,9}, NppiPoint{1,1}, x2, 15, y2, ds2, m2, ds2, a2, ds2, NppiSize{4,4}, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, c)));
  }
  printf("alive check %s\n", es(call(sob, dx, dstep, dy, dstep, dm, dstep, da, astep)));
  printf("done\n");
  return 0;
}
