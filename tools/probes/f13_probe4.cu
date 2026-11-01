// f13 probe 4: GradientVector step validation order + optional-output writes.
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
  const int W = 9, H = 9, R = 4;
  const int sstep = W, dstep = R * 2, astep = R * 4;  // 8u src
  std::vector<Npp8u> h((size_t)W * H, 0);
  for (size_t i = 0; i < h.size(); ++i) h[i] = (Npp8u)((i * 41 + 7) & 0xff);
  Npp8u *d; cudaMalloc(&d, sstep * H); cudaMemcpy(d, h.data(), sstep * H, cudaMemcpyHostToDevice);
  Npp16s *dx, *dy, *dm; Npp32f *da;
  cudaMalloc(&dx, dstep * H); cudaMalloc(&dy, dstep * H); cudaMalloc(&dm, dstep * H); cudaMalloc(&da, astep * H);
  printf("alloc ok dx=%p\n", (void*)dx);
  NppStreamContext c{}; c.hStream = 0;
  const Npp8u *pSrc = d + 1 * sstep + 1;
  NppiSize sz{W, H}; NppiPoint off{1, 1}; NppiSize roi{R, R};
  typedef NppStatus (*F)(const Npp8u*, int, NppiSize, NppiPoint, Npp16s*, int, Npp16s*, int, Npp16s*, int, Npp32f*, int, NppiSize, NppiMaskSize, NppiNorm, NppiBorderType, NppStreamContext);
  F f = nppiGradientVectorSobelBorder_8u16s_C1R_Ctx;
  auto call = [&](Npp16s *x, int xs, Npp16s *y, int ys, Npp16s *m, int ms, Npp32f *a, int as,
                  NppiSize rs = NppiSize{4, 4}, NppiMaskSize mk = NPP_MASK_SIZE_3_X_3, NppiNorm nm = nppiNormL2,
                  NppiBorderType bt = NPP_BORDER_REPLICATE, NppiSize sz_ = NppiSize{9, 9}, NppiPoint off_ = NppiPoint{1, 1}) {
    return f(pSrc, sstep, sz_, off_, x, xs, y, ys, m, ms, a, as, rs, mk, nm, bt, c);
  };
  printf("== xstep checks (16s dst, roiW=4 needs >=8, align 2)\n");
  printf("  xstep_7      %s\n", es(call(dx, 7, dy, dstep, dm, dstep, da, astep)));
  printf("  xstep_8      %s\n", es(call(dx, 8, dy, dstep, dm, dstep, da, astep)));
  printf("  xstep_9(odd) %s\n", es(call(dx, 9, dy, dstep, dm, dstep, da, astep)));
  printf("  xstep_10     %s\n", es(call(dx, 10, dy, dstep, dm, dstep, da, astep)));
  printf("== xstep vs other checks\n");
  printf("  negroi_xbad      %s\n", es(call(dx, 3, dy, dstep, dm, dstep, da, astep, NppiSize{4, -1})));
  printf("  zroi_xbad        %s\n", es(call(dx, 3, dy, dstep, dm, dstep, da, astep, NppiSize{0, 0})));
  printf("  srcsize0_xbad    %s\n", es(call(dx, 3, dy, dstep, dm, dstep, da, astep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, NppiSize{0, 8}, off)));
  printf("  offneg_xbad      %s\n", es(call(dx, 3, dy, dstep, dm, dstep, da, astep, roi, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, sz, NppiPoint{-1, 1})));
  printf("  srcsize0_negroi  %s\n", es(call(dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4, -1}, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, NppiSize{0, 8}, off)));
  printf("  offneg_negroi    %s\n", es(call(dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4, -1}, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE, sz, NppiPoint{-1, 1})));
  printf("  badborder_negroi %s\n", es(call(dx, dstep, dy, dstep, dm, dstep, da, astep, NppiSize{4, -1}, NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_CONSTANT)));
  printf("== optional outputs: writes?\n");
  cudaMemset(dx, 0xAA, dstep * H); cudaMemset(dy, 0xAA, dstep * H); cudaMemset(dm, 0xAA, dstep * H); cudaMemset(da, 0xAA, astep * H);
  printf("  xonly        %s\n", es(call(dx, dstep, nullptr, 0, nullptr, 0, nullptr, 0)));
  std::vector<unsigned char> ry(dstep * H), ra(astep * H);
  cudaMemcpy(ry.data(), dy, dstep * H, cudaMemcpyDeviceToHost);
  cudaMemcpy(ra.data(), da, astep * H, cudaMemcpyDeviceToHost);
  bool yuntouched = true, auntouched = true;
  for (unsigned char b : ry) if (b != 0xAA) yuntouched = false;
  for (unsigned char b : ra) if (b != 0xAA) auntouched = false;
  printf("  y_untouched=%d a_untouched=%d\n", yuntouched, auntouched);
  printf("  still alive: %s\n", es(call(dx, dstep, dy, dstep, dm, dstep, da, astep)));
  printf("done\n");
  return 0;
}
