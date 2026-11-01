// f13 probe 6: first-non-null step model, border types, offset rules, C3C1R validation.
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

int main() {
  NppStreamContext c{}; c.hStream = 0;
  const int W = 9, H = 9, R = 4;
  const int sstep = W, dstep = R * 2, astep = R * 4;
  std::vector<Npp8u> h((size_t)W * H, 0);
  for (size_t i = 0; i < h.size(); ++i) h[i] = (Npp8u)((i * 73 + 11) & 0xff);
  Npp8u *d; cudaMalloc(&d, sstep * H); cudaMemcpy(d, h.data(), sstep * H, cudaMemcpyHostToDevice);
  Npp16s *dx, *dy, *dm; Npp32f *da;
  cudaMalloc(&dx, dstep * H); cudaMalloc(&dy, dstep * H); cudaMalloc(&dm, dstep * H); cudaMalloc(&da, astep * H);
  typedef NppStatus (*F)(const Npp8u*, int, NppiSize, NppiPoint, Npp16s*, int, Npp16s*, int, Npp16s*, int, Npp32f*, int, NppiSize, NppiMaskSize, NppiNorm, NppiBorderType, NppStreamContext);
  F sob = nppiGradientVectorSobelBorder_8u16s_C1R_Ctx;
  auto call = [&](F f, const Npp8u *ps, int ss, Npp16s *x, int xs, Npp16s *y, int ys, Npp16s *m, int ms, Npp32f *a, int as,
                  NppiSize rs = NppiSize{4, 4}, NppiMaskSize mk = NPP_MASK_SIZE_3_X_3, NppiNorm nm = nppiNormL2,
                  NppiBorderType bt = NPP_BORDER_REPLICATE, NppiSize sz_ = NppiSize{9, 9}, NppiPoint off_ = NppiPoint{1, 1}) {
    return f(ps, ss, sz_, off_, x, xs, y, ys, m, ms, a, as, rs, mk, nm, bt, c);
  };
  const Npp8u *pSrc = d + sstep + 1;
  printf("== first-non-null step model\n");
  printf("  xnull_yvalid_mbad    %s\n", es(call(sob, pSrc, sstep, nullptr, 0, dy, dstep, dm, 3, da, astep)));
  printf("  xnull_ynull_mvalid_abad %s\n", es(call(sob, pSrc, sstep, nullptr, 0, nullptr, 0, dm, dstep, da, 3)));
  printf("  angleonly_badstep    %s\n", es(call(sob, pSrc, sstep, nullptr, 0, nullptr, 0, nullptr, 0, da, 3)));
  printf("  xnull_yvalid_astep0  %s\n", es(call(sob, pSrc, sstep, nullptr, 0, dy, dstep, nullptr, 0, da, 0)));
  printf("  xbad_yvalid          %s\n", es(call(sob, pSrc, sstep, dx, 3, dy, dstep, dm, dstep, da, astep)));
  printf("== allnull / combos\n");
  printf("  allnull_negroi       %s\n", es(call(sob, pSrc, sstep, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, NppiSize{4,-1})));
  printf("  allnull_badnorm      %s\n", es(call(sob, pSrc, sstep, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, NppiSize{4,4}, NPP_MASK_SIZE_3_X_3, nppiNormInf)));
  printf("  allnull_badmask      %s\n", es(call(sob, pSrc, sstep, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0, NppiSize{4,4}, (NppiMaskSize)7)));
  printf("== border types (8u16s C1R)\n");
  printf("  wrap                 %s\n", es(call(sob, pSrc, sstep, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_WRAP)));
  printf("  mirror               %s\n", es(call(sob, pSrc, sstep, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_MIRROR)));
  printf("  constBorder          %s\n", es(call(sob, pSrc, sstep, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_CONSTANT)));
  printf("  replicate            %s\n", es(call(sob, pSrc, sstep, dx, dstep, dy, dstep, dm, dstep, da, astep)));
  printf("== offset rules\n");
  printf("  off10x1              %s\n", es(call(sob, pSrc, sstep, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, NppiSize{9,9}, NppiPoint{10,1})));
  printf("  off1x10              %s\n", es(call(sob, pSrc, sstep, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, NppiSize{9,9}, NppiPoint{1,10})));
  printf("  off9x9_zroi          %s\n", es(call(sob, pSrc, sstep, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{0,4}, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, NppiSize{9,9}, NppiPoint{9,9})));
  printf("  off8x1_roi2          %s\n", es(call(sob, pSrc, sstep, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{2,4}, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, NppiSize{9,9}, NppiPoint{8,1})));
  printf("== srcStep oddities\n");
  printf("  srcstep0             %s\n", es(call(sob, pSrc, 0, dx, dstep, dy, dstep, dm, dstep, da, astep)));
  printf("  srcstepneg           %s\n", es(call(sob, pSrc, -8, dx, dstep, dy, dstep, dm, dstep, da, astep)));
  printf("  srcstep3             %s\n", es(call(sob, pSrc, 3, dx, dstep, dy, dstep, dm, dstep, da, astep)));
  printf("== C3C1R validation (8u16s)\n");
  {
    const int sstep3 = W * 3;
    Npp8u *d3; cudaMalloc(&d3, sstep3 * H); cudaMemcpy(d3, h.data(), sstep * H, cudaMemcpyHostToDevice);
    typedef NppStatus (*G)(const Npp8u*, int, NppiSize, NppiPoint, Npp16s*, int, Npp16s*, int, Npp16s*, int, Npp32f*, int, NppiSize, NppiMaskSize, NppiNorm, NppiBorderType, NppStreamContext);
    G g = nppiGradientVectorSobelBorder_8u16s_C3C1R_Ctx;
    auto cg = [&](const Npp8u *ps, int ss, Npp16s *x, int xs, Npp16s *y, int ys, Npp16s *m, int ms, Npp32f *a, int as,
                  NppiSize rs = NppiSize{4,4}, NppiMaskSize mk = NPP_MASK_SIZE_3_X_3, NppiNorm nm = nppiNormL2,
                  NppiBorderType bt = NPP_BORDER_REPLICATE, NppiSize sz_ = NppiSize{9,9}, NppiPoint off_ = NppiPoint{1,1}) {
      return g(ps, ss, sz_, off_, x, xs, y, ys, m, ms, a, as, rs, mk, nm, bt, c);
    };
    const Npp8u *ps3 = d3 + sstep3 + 3;
    printf("  c3c1 ok              %s\n", es(cg(ps3, sstep3, dx, dstep, dy, dstep, dm, dstep, da, astep)));
    printf("  c3c1 allnull         %s\n", es(cg(ps3, sstep3, nullptr, 0, nullptr, 0, nullptr, 0, nullptr, 0)));
    printf("  c3c1 nullsrc         %s\n", es(cg(nullptr, sstep3, dx, dstep, dy, dstep, dm, dstep, da, astep)));
    printf("  c3c1 xbadstep        %s\n", es(cg(ps3, sstep3, dx, 3, dy, dstep, dm, dstep, da, astep)));
    printf("  c3c1 xnull_ybad      %s\n", es(cg(ps3, sstep3, nullptr, 0, dy, 3, dm, dstep, da, astep)));
    printf("  c3c1 negroi          %s\n", es(cg(ps3, sstep3, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,-1})));
    printf("  c3c1 zroi            %s\n", es(cg(ps3, sstep3, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{0,4})));
    printf("  c3c1 srcsize0        %s\n", es(cg(ps3, sstep3, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, NppiSize{0,8}, NppiPoint{1,1})));
    printf("  c3c1 offneg          %s\n", es(cg(ps3, sstep3, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, NppiSize{9,9}, NppiPoint{-1,1})));
    printf("  c3c1 badborder       %s\n", es(cg(ps3, sstep3, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_CONSTANT)));
    printf("  c3c1 badnorm         %s\n", es(cg(ps3, sstep3, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_3_X_3, nppiNormInf)));
    printf("  c3c1 badmask         %s\n", es(cg(ps3, sstep3, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, (NppiMaskSize)7)));
    printf("  c3c1 srcstep9        %s\n", es(cg(ps3, 9, dx, dstep, dy, dstep, dm, dstep, da, astep)));
    printf("  c3c1 5x5             %s\n", es(cg(ps3, sstep3, dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4,4}, NPP_MASK_SIZE_5_X_5)));
  }
  printf("alive %s\n", es(call(sob, pSrc, sstep, dx, dstep, dy, dstep, dm, dstep, da, astep)));
  printf("done\n");
  return 0;
}
