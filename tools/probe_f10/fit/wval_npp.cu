// NVIDIA NPP reference side of the f10 executor validation harness.
#include <npp.h>
extern "C" {
cudaStream_t nppGetStream(void);
NppStatus nppGetStreamContext(NppStreamContext *p);
}
#include <nppi_filtering_functions.h>
#include "wval_common.h"

static int canon(NppStatus s) {
  if (s == NPP_NO_ERROR) return 0;
  if (s == NPP_NULL_POINTER_ERROR) return 1;
  if (s == NPP_SIZE_ERROR) return 2;
  if (s == NPP_MASK_SIZE_ERROR) return 3;
  if (s == NPP_STEP_ERROR) return 4;
  if (s == NPP_NOT_EVEN_STEP_ERROR) return 5;
  if (s == NPP_OUT_OFF_RANGE_ERROR) return 6;
  if (s == NPP_NOT_SUPPORTED_MODE_ERROR) return 7;
  if (s == NPP_BAD_ARGUMENT_ERROR) return 8;
  return 99;
}
extern "C" int npp_canon(int status) { return canon((NppStatus)status); }

static NppStreamContext ctx() {
  NppStreamContext c{};
  nppGetStreamContext(&c);
  return c;
}

extern "C" int npp_wiener(const void *src, int ss, int srcW, int srcH, int ox, int oy, void *dst, int ds,
                          int W, int H, int mw, int mh, int ax, int ay, const float *nz, int border,
                          int dtype, int channels, int ac4) {
  NppiSize sz{srcW, srcH}, roi{W, H}, mask{mw, mh};
  NppiPoint off{ox, oy}, anc{ax, ay};
  NppStreamContext c = ctx();
  if (dtype == 0) {
    if (channels == 1) return nppiFilterWienerBorder_8u_C1R_Ctx((const Npp8u *)src, ss, sz, off, (Npp8u *)dst, ds, roi, mask, anc, (Npp32f *)nz, (NppiBorderType)border, c);
    if (channels == 3) return nppiFilterWienerBorder_8u_C3R_Ctx((const Npp8u *)src, ss, sz, off, (Npp8u *)dst, ds, roi, mask, anc, (Npp32f *)nz, (NppiBorderType)border, c);
    if (ac4) return nppiFilterWienerBorder_8u_AC4R_Ctx((const Npp8u *)src, ss, sz, off, (Npp8u *)dst, ds, roi, mask, anc, (Npp32f *)nz, (NppiBorderType)border, c);
    return nppiFilterWienerBorder_8u_C4R_Ctx((const Npp8u *)src, ss, sz, off, (Npp8u *)dst, ds, roi, mask, anc, (Npp32f *)nz, (NppiBorderType)border, c);
  }
  if (dtype == 1) {
    if (channels == 1) return nppiFilterWienerBorder_16s_C1R_Ctx((const Npp16s *)src, ss, sz, off, (Npp16s *)dst, ds, roi, mask, anc, (Npp32f *)nz, (NppiBorderType)border, c);
    if (channels == 3) return nppiFilterWienerBorder_16s_C3R_Ctx((const Npp16s *)src, ss, sz, off, (Npp16s *)dst, ds, roi, mask, anc, (Npp32f *)nz, (NppiBorderType)border, c);
    if (ac4) return nppiFilterWienerBorder_16s_AC4R_Ctx((const Npp16s *)src, ss, sz, off, (Npp16s *)dst, ds, roi, mask, anc, (Npp32f *)nz, (NppiBorderType)border, c);
    return nppiFilterWienerBorder_16s_C4R_Ctx((const Npp16s *)src, ss, sz, off, (Npp16s *)dst, ds, roi, mask, anc, (Npp32f *)nz, (NppiBorderType)border, c);
  }
  if (channels == 1) return nppiFilterWienerBorder_32f_C1R_Ctx((const Npp32f *)src, ss, sz, off, (Npp32f *)dst, ds, roi, mask, anc, (Npp32f *)nz, (NppiBorderType)border, c);
  if (channels == 3) return nppiFilterWienerBorder_32f_C3R_Ctx((const Npp32f *)src, ss, sz, off, (Npp32f *)dst, ds, roi, mask, anc, (Npp32f *)nz, (NppiBorderType)border, c);
  if (ac4) return nppiFilterWienerBorder_32f_AC4R_Ctx((const Npp32f *)src, ss, sz, off, (Npp32f *)dst, ds, roi, mask, anc, (Npp32f *)nz, (NppiBorderType)border, c);
  return nppiFilterWienerBorder_32f_C4R_Ctx((const Npp32f *)src, ss, sz, off, (Npp32f *)dst, ds, roi, mask, anc, (Npp32f *)nz, (NppiBorderType)border, c);
}

extern "C" int npp_gauss(const void *src, int ss, void *dst, int ds, int W, int H, int mask, int dtype,
                         int channels, int ac4) {
  NppStreamContext c = ctx();
  NppiSize roi{W, H};
  const NppiMaskSize ms = (NppiMaskSize)mask;
  if (dtype == 0) {
    if (channels == 1) return nppiFilterGauss_8u_C1R_Ctx((const Npp8u *)src, ss, (Npp8u *)dst, ds, roi, ms, c);
    if (channels == 3) return nppiFilterGauss_8u_C3R_Ctx((const Npp8u *)src, ss, (Npp8u *)dst, ds, roi, ms, c);
    if (ac4) return nppiFilterGauss_8u_AC4R_Ctx((const Npp8u *)src, ss, (Npp8u *)dst, ds, roi, ms, c);
    return nppiFilterGauss_8u_C4R_Ctx((const Npp8u *)src, ss, (Npp8u *)dst, ds, roi, ms, c);
  }
  if (dtype == 1) {
    if (channels == 1) return nppiFilterGauss_16u_C1R_Ctx((const Npp16u *)src, ss, (Npp16u *)dst, ds, roi, ms, c);
    if (channels == 3) return nppiFilterGauss_16u_C3R_Ctx((const Npp16u *)src, ss, (Npp16u *)dst, ds, roi, ms, c);
    if (ac4) return nppiFilterGauss_16u_AC4R_Ctx((const Npp16u *)src, ss, (Npp16u *)dst, ds, roi, ms, c);
    return nppiFilterGauss_16u_C4R_Ctx((const Npp16u *)src, ss, (Npp16u *)dst, ds, roi, ms, c);
  }
  if (dtype == 2) {
    if (channels == 1) return nppiFilterGauss_16s_C1R_Ctx((const Npp16s *)src, ss, (Npp16s *)dst, ds, roi, ms, c);
    if (channels == 3) return nppiFilterGauss_16s_C3R_Ctx((const Npp16s *)src, ss, (Npp16s *)dst, ds, roi, ms, c);
    if (ac4) return nppiFilterGauss_16s_AC4R_Ctx((const Npp16s *)src, ss, (Npp16s *)dst, ds, roi, ms, c);
    return nppiFilterGauss_16s_C4R_Ctx((const Npp16s *)src, ss, (Npp16s *)dst, ds, roi, ms, c);
  }
  if (channels == 1) return nppiFilterGauss_32f_C1R_Ctx((const Npp32f *)src, ss, (Npp32f *)dst, ds, roi, ms, c);
  if (channels == 3) return nppiFilterGauss_32f_C3R_Ctx((const Npp32f *)src, ss, (Npp32f *)dst, ds, roi, ms, c);
  if (ac4) return nppiFilterGauss_32f_AC4R_Ctx((const Npp32f *)src, ss, (Npp32f *)dst, ds, roi, ms, c);
  return nppiFilterGauss_32f_C4R_Ctx((const Npp32f *)src, ss, (Npp32f *)dst, ds, roi, ms, c);
}

extern "C" int npp_gauss_border(const void *src, int ss, int srcW, int srcH, int ox, int oy, void *dst,
                                int ds, int W, int H, int mask, int border, int dtype, int channels, int ac4) {
  NppStreamContext c = ctx();
  NppiSize sz{srcW, srcH}, roi{W, H};
  NppiPoint off{ox, oy};
  if (dtype == 0) {
    if (channels == 1) return nppiFilterGaussBorder_8u_C1R_Ctx((const Npp8u *)src, ss, sz, off, (Npp8u *)dst, ds, roi, (NppiMaskSize)mask, (NppiBorderType)border, c);
    if (channels == 3) return nppiFilterGaussBorder_8u_C3R_Ctx((const Npp8u *)src, ss, sz, off, (Npp8u *)dst, ds, roi, (NppiMaskSize)mask, (NppiBorderType)border, c);
    if (ac4) return nppiFilterGaussBorder_8u_AC4R_Ctx((const Npp8u *)src, ss, sz, off, (Npp8u *)dst, ds, roi, (NppiMaskSize)mask, (NppiBorderType)border, c);
    return nppiFilterGaussBorder_8u_C4R_Ctx((const Npp8u *)src, ss, sz, off, (Npp8u *)dst, ds, roi, (NppiMaskSize)mask, (NppiBorderType)border, c);
  }
  if (dtype == 1) {
    if (channels == 1) return nppiFilterGaussBorder_16u_C1R_Ctx((const Npp16u *)src, ss, sz, off, (Npp16u *)dst, ds, roi, (NppiMaskSize)mask, (NppiBorderType)border, c);
    if (channels == 3) return nppiFilterGaussBorder_16u_C3R_Ctx((const Npp16u *)src, ss, sz, off, (Npp16u *)dst, ds, roi, (NppiMaskSize)mask, (NppiBorderType)border, c);
    if (ac4) return nppiFilterGaussBorder_16u_AC4R_Ctx((const Npp16u *)src, ss, sz, off, (Npp16u *)dst, ds, roi, (NppiMaskSize)mask, (NppiBorderType)border, c);
    return nppiFilterGaussBorder_16u_C4R_Ctx((const Npp16u *)src, ss, sz, off, (Npp16u *)dst, ds, roi, (NppiMaskSize)mask, (NppiBorderType)border, c);
  }
  if (dtype == 2) {
    if (channels == 1) return nppiFilterGaussBorder_16s_C1R_Ctx((const Npp16s *)src, ss, sz, off, (Npp16s *)dst, ds, roi, (NppiMaskSize)mask, (NppiBorderType)border, c);
    if (channels == 3) return nppiFilterGaussBorder_16s_C3R_Ctx((const Npp16s *)src, ss, sz, off, (Npp16s *)dst, ds, roi, (NppiMaskSize)mask, (NppiBorderType)border, c);
    if (ac4) return nppiFilterGaussBorder_16s_AC4R_Ctx((const Npp16s *)src, ss, sz, off, (Npp16s *)dst, ds, roi, (NppiMaskSize)mask, (NppiBorderType)border, c);
    return nppiFilterGaussBorder_16s_C4R_Ctx((const Npp16s *)src, ss, sz, off, (Npp16s *)dst, ds, roi, (NppiMaskSize)mask, (NppiBorderType)border, c);
  }
  if (channels == 1) return nppiFilterGaussBorder_32f_C1R_Ctx((const Npp32f *)src, ss, sz, off, (Npp32f *)dst, ds, roi, (NppiMaskSize)mask, (NppiBorderType)border, c);
  if (channels == 3) return nppiFilterGaussBorder_32f_C3R_Ctx((const Npp32f *)src, ss, sz, off, (Npp32f *)dst, ds, roi, (NppiMaskSize)mask, (NppiBorderType)border, c);
  if (ac4) return nppiFilterGaussBorder_32f_AC4R_Ctx((const Npp32f *)src, ss, sz, off, (Npp32f *)dst, ds, roi, (NppiMaskSize)mask, (NppiBorderType)border, c);
  return nppiFilterGaussBorder_32f_C4R_Ctx((const Npp32f *)src, ss, sz, off, (Npp32f *)dst, ds, roi, (NppiMaskSize)mask, (NppiBorderType)border, c);
}

extern "C" int npp_adv(const void *src, int ss, void *dst, int ds, int W, int H, int taps, const float *kern,
                       int dtype, int channels, int ac4) {
  NppStreamContext c = ctx();
  NppiSize roi{W, H};
  const int t = taps;
  const Npp32f *k = kern;
  if (dtype == 0) {
    if (channels == 1) return nppiFilterGaussAdvanced_8u_C1R_Ctx((const Npp8u *)src, ss, (Npp8u *)dst, ds, roi, t, k, c);
    if (channels == 3) return nppiFilterGaussAdvanced_8u_C3R_Ctx((const Npp8u *)src, ss, (Npp8u *)dst, ds, roi, t, k, c);
    if (ac4) return nppiFilterGaussAdvanced_8u_AC4R_Ctx((const Npp8u *)src, ss, (Npp8u *)dst, ds, roi, t, k, c);
    return nppiFilterGaussAdvanced_8u_C4R_Ctx((const Npp8u *)src, ss, (Npp8u *)dst, ds, roi, t, k, c);
  }
  if (dtype == 1) {
    if (channels == 1) return nppiFilterGaussAdvanced_16u_C1R_Ctx((const Npp16u *)src, ss, (Npp16u *)dst, ds, roi, t, k, c);
    if (channels == 3) return nppiFilterGaussAdvanced_16u_C3R_Ctx((const Npp16u *)src, ss, (Npp16u *)dst, ds, roi, t, k, c);
    if (ac4) return nppiFilterGaussAdvanced_16u_AC4R_Ctx((const Npp16u *)src, ss, (Npp16u *)dst, ds, roi, t, k, c);
    return nppiFilterGaussAdvanced_16u_C4R_Ctx((const Npp16u *)src, ss, (Npp16u *)dst, ds, roi, t, k, c);
  }
  if (dtype == 2) {
    if (channels == 1) return nppiFilterGaussAdvanced_16s_C1R_Ctx((const Npp16s *)src, ss, (Npp16s *)dst, ds, roi, t, k, c);
    if (channels == 3) return nppiFilterGaussAdvanced_16s_C3R_Ctx((const Npp16s *)src, ss, (Npp16s *)dst, ds, roi, t, k, c);
    if (ac4) return nppiFilterGaussAdvanced_16s_AC4R_Ctx((const Npp16s *)src, ss, (Npp16s *)dst, ds, roi, t, k, c);
    return nppiFilterGaussAdvanced_16s_C4R_Ctx((const Npp16s *)src, ss, (Npp16s *)dst, ds, roi, t, k, c);
  }
  if (channels == 1) return nppiFilterGaussAdvanced_32f_C1R_Ctx((const Npp32f *)src, ss, (Npp32f *)dst, ds, roi, t, k, c);
  if (channels == 3) return nppiFilterGaussAdvanced_32f_C3R_Ctx((const Npp32f *)src, ss, (Npp32f *)dst, ds, roi, t, k, c);
  if (ac4) return nppiFilterGaussAdvanced_32f_AC4R_Ctx((const Npp32f *)src, ss, (Npp32f *)dst, ds, roi, t, k, c);
  return nppiFilterGaussAdvanced_32f_C4R_Ctx((const Npp32f *)src, ss, (Npp32f *)dst, ds, roi, t, k, c);
}

extern "C" int npp_adv_border(const void *src, int ss, int srcW, int srcH, int ox, int oy, void *dst, int ds,
                              int W, int H, int taps, const float *kern, int border, int dtype, int channels, int ac4) {
  NppStreamContext c = ctx();
  NppiSize sz{srcW, srcH}, roi{W, H};
  NppiPoint off{ox, oy};
  const int t = taps;
  const Npp32f *k = kern;
  if (dtype == 0) {
    if (channels == 1) return nppiFilterGaussAdvancedBorder_8u_C1R_Ctx((const Npp8u *)src, ss, sz, off, (Npp8u *)dst, ds, roi, t, k, (NppiBorderType)border, c);
    if (channels == 3) return nppiFilterGaussAdvancedBorder_8u_C3R_Ctx((const Npp8u *)src, ss, sz, off, (Npp8u *)dst, ds, roi, t, k, (NppiBorderType)border, c);
    if (ac4) return nppiFilterGaussAdvancedBorder_8u_AC4R_Ctx((const Npp8u *)src, ss, sz, off, (Npp8u *)dst, ds, roi, t, k, (NppiBorderType)border, c);
    return nppiFilterGaussAdvancedBorder_8u_C4R_Ctx((const Npp8u *)src, ss, sz, off, (Npp8u *)dst, ds, roi, t, k, (NppiBorderType)border, c);
  }
  if (dtype == 1) {
    if (channels == 1) return nppiFilterGaussAdvancedBorder_16u_C1R_Ctx((const Npp16u *)src, ss, sz, off, (Npp16u *)dst, ds, roi, t, k, (NppiBorderType)border, c);
    if (channels == 3) return nppiFilterGaussAdvancedBorder_16u_C3R_Ctx((const Npp16u *)src, ss, sz, off, (Npp16u *)dst, ds, roi, t, k, (NppiBorderType)border, c);
    if (ac4) return nppiFilterGaussAdvancedBorder_16u_AC4R_Ctx((const Npp16u *)src, ss, sz, off, (Npp16u *)dst, ds, roi, t, k, (NppiBorderType)border, c);
    return nppiFilterGaussAdvancedBorder_16u_C4R_Ctx((const Npp16u *)src, ss, sz, off, (Npp16u *)dst, ds, roi, t, k, (NppiBorderType)border, c);
  }
  if (dtype == 2) {
    if (channels == 1) return nppiFilterGaussAdvancedBorder_16s_C1R_Ctx((const Npp16s *)src, ss, sz, off, (Npp16s *)dst, ds, roi, t, k, (NppiBorderType)border, c);
    if (channels == 3) return nppiFilterGaussAdvancedBorder_16s_C3R_Ctx((const Npp16s *)src, ss, sz, off, (Npp16s *)dst, ds, roi, t, k, (NppiBorderType)border, c);
    if (ac4) return nppiFilterGaussAdvancedBorder_16s_AC4R_Ctx((const Npp16s *)src, ss, sz, off, (Npp16s *)dst, ds, roi, t, k, (NppiBorderType)border, c);
    return nppiFilterGaussAdvancedBorder_16s_C4R_Ctx((const Npp16s *)src, ss, sz, off, (Npp16s *)dst, ds, roi, t, k, (NppiBorderType)border, c);
  }
  if (channels == 1) return nppiFilterGaussAdvancedBorder_32f_C1R_Ctx((const Npp32f *)src, ss, sz, off, (Npp32f *)dst, ds, roi, t, k, (NppiBorderType)border, c);
  if (channels == 3) return nppiFilterGaussAdvancedBorder_32f_C3R_Ctx((const Npp32f *)src, ss, sz, off, (Npp32f *)dst, ds, roi, t, k, (NppiBorderType)border, c);
  if (ac4) return nppiFilterGaussAdvancedBorder_32f_AC4R_Ctx((const Npp32f *)src, ss, sz, off, (Npp32f *)dst, ds, roi, t, k, (NppiBorderType)border, c);
  return nppiFilterGaussAdvancedBorder_32f_C4R_Ctx((const Npp32f *)src, ss, sz, off, (Npp32f *)dst, ds, roi, t, k, (NppiBorderType)border, c);
}
