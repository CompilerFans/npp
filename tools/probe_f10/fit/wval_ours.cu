// Compiles the real mcMPP executors with the repository headers.
#include "npp.h"
#include "nppi_filter_gauss_executor.h"
#include "nppi_filter_wiener_executor.h"
#include "wval_common.h"

using namespace nppi::filter;

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
extern "C" int ours_canon(int status) { return canon((NppStatus)status); }

template <typename T>
static NppStatus wiener_call(const void *src, int ss, int srcW, int srcH, int ox, int oy, void *dst, int ds,
                             int W, int H, int mw, int mh, int ax, int ay, const float *nz, int border,
                             int channels, int ac4) {
  NppiSize sz{srcW, srcH}, roi{W, H}, mask{mw, mh};
  NppiPoint off{ox, oy}, anc{ax, ay};
  if (channels == 1) return filterWienerBorder<T, 1, false>((const T *)src, ss, sz, off, (T *)dst, ds, roi, mask, anc, nz, (NppiBorderType)border, nullptr);
  if (channels == 3) return filterWienerBorder<T, 3, false>((const T *)src, ss, sz, off, (T *)dst, ds, roi, mask, anc, nz, (NppiBorderType)border, nullptr);
  if (ac4) return filterWienerBorder<T, 4, true>((const T *)src, ss, sz, off, (T *)dst, ds, roi, mask, anc, nz, (NppiBorderType)border, nullptr);
  return filterWienerBorder<T, 4, false>((const T *)src, ss, sz, off, (T *)dst, ds, roi, mask, anc, nz, (NppiBorderType)border, nullptr);
}

extern "C" int ours_wiener(const void *src, int ss, int srcW, int srcH, int ox, int oy, void *dst, int ds,
                           int W, int H, int mw, int mh, int ax, int ay, const float *nz, int border,
                           int dtype, int channels, int ac4) {
  NppStatus st;
  if (dtype == 0) st = wiener_call<Npp8u>(src, ss, srcW, srcH, ox, oy, dst, ds, W, H, mw, mh, ax, ay, nz, border, channels, ac4);
  else if (dtype == 1) st = wiener_call<Npp16s>(src, ss, srcW, srcH, ox, oy, dst, ds, W, H, mw, mh, ax, ay, nz, border, channels, ac4);
  else st = wiener_call<Npp32f>(src, ss, srcW, srcH, ox, oy, dst, ds, W, H, mw, mh, ax, ay, nz, border, channels, ac4);
  return (int)st;
}

template <typename T, bool Border>
static NppStatus gauss_call(const void *src, int ss, int srcW, int srcH, int ox, int oy, void *dst, int ds,
                            int W, int H, int mask, int border, int channels, int ac4) {
  (void)srcW; (void)srcH; (void)ox; (void)oy; (void)border;
  NppiSize sz{srcW, srcH}, roi{W, H};
  NppiPoint off{ox, oy};
  if constexpr (Border) {
    if (channels == 1) return filterGaussBorder<T, T, 1, false>((const T *)src, ss, sz, off, (T *)dst, ds, roi, (NppiMaskSize)mask, (NppiBorderType)border, nullptr);
    if (channels == 3) return filterGaussBorder<T, T, 3, false>((const T *)src, ss, sz, off, (T *)dst, ds, roi, (NppiMaskSize)mask, (NppiBorderType)border, nullptr);
    if (ac4) return filterGaussBorder<T, T, 4, true>((const T *)src, ss, sz, off, (T *)dst, ds, roi, (NppiMaskSize)mask, (NppiBorderType)border, nullptr);
    return filterGaussBorder<T, T, 4, false>((const T *)src, ss, sz, off, (T *)dst, ds, roi, (NppiMaskSize)mask, (NppiBorderType)border, nullptr);
  } else {
    if (channels == 1) return filterGauss<T, T, 1, false>((const T *)src, ss, (T *)dst, ds, roi, (NppiMaskSize)mask, nullptr);
    if (channels == 3) return filterGauss<T, T, 3, false>((const T *)src, ss, (T *)dst, ds, roi, (NppiMaskSize)mask, nullptr);
    if (ac4) return filterGauss<T, T, 4, true>((const T *)src, ss, (T *)dst, ds, roi, (NppiMaskSize)mask, nullptr);
    return filterGauss<T, T, 4, false>((const T *)src, ss, (T *)dst, ds, roi, (NppiMaskSize)mask, nullptr);
  }
}

extern "C" int ours_gauss(const void *src, int ss, void *dst, int ds, int W, int H, int mask, int dtype,
                          int channels, int ac4) {
  if (dtype == 0) return (int)gauss_call<Npp8u, false>(src, ss, 0, 0, 0, 0, dst, ds, W, H, mask, 2, channels, ac4);
  if (dtype == 1) return (int)gauss_call<Npp16u, false>(src, ss, 0, 0, 0, 0, dst, ds, W, H, mask, 2, channels, ac4);
  if (dtype == 2) return (int)gauss_call<Npp16s, false>(src, ss, 0, 0, 0, 0, dst, ds, W, H, mask, 2, channels, ac4);
  return (int)gauss_call<Npp32f, false>(src, ss, 0, 0, 0, 0, dst, ds, W, H, mask, 2, channels, ac4);
}

extern "C" int ours_gauss_border(const void *src, int ss, int srcW, int srcH, int ox, int oy, void *dst,
                                 int ds, int W, int H, int mask, int border, int dtype, int channels, int ac4) {
  if (dtype == 0) return (int)gauss_call<Npp8u, true>(src, ss, srcW, srcH, ox, oy, dst, ds, W, H, mask, border, channels, ac4);
  if (dtype == 1) return (int)gauss_call<Npp16u, true>(src, ss, srcW, srcH, ox, oy, dst, ds, W, H, mask, border, channels, ac4);
  if (dtype == 2) return (int)gauss_call<Npp16s, true>(src, ss, srcW, srcH, ox, oy, dst, ds, W, H, mask, border, channels, ac4);
  return (int)gauss_call<Npp32f, true>(src, ss, srcW, srcH, ox, oy, dst, ds, W, H, mask, border, channels, ac4);
}

template <typename T, bool Border>
static NppStatus adv_call(const void *src, int ss, int srcW, int srcH, int ox, int oy, void *dst, int ds,
                          int W, int H, int taps, const float *kern, int border, int channels, int ac4) {
  NppiSize sz{srcW, srcH}, roi{W, H};
  NppiPoint off{ox, oy};
  if constexpr (Border) {
    if (channels == 1) return filterGaussAdvancedBorder<T, T, 1, false>((const T *)src, ss, sz, off, (T *)dst, ds, roi, taps, kern, (NppiBorderType)border, nullptr);
    if (channels == 3) return filterGaussAdvancedBorder<T, T, 3, false>((const T *)src, ss, sz, off, (T *)dst, ds, roi, taps, kern, (NppiBorderType)border, nullptr);
    if (ac4) return filterGaussAdvancedBorder<T, T, 4, true>((const T *)src, ss, sz, off, (T *)dst, ds, roi, taps, kern, (NppiBorderType)border, nullptr);
    return filterGaussAdvancedBorder<T, T, 4, false>((const T *)src, ss, sz, off, (T *)dst, ds, roi, taps, kern, (NppiBorderType)border, nullptr);
  } else {
    if (channels == 1) return filterGaussAdvanced<T, T, 1, false>((const T *)src, ss, (T *)dst, ds, roi, taps, kern, nullptr);
    if (channels == 3) return filterGaussAdvanced<T, T, 3, false>((const T *)src, ss, (T *)dst, ds, roi, taps, kern, nullptr);
    if (ac4) return filterGaussAdvanced<T, T, 4, true>((const T *)src, ss, (T *)dst, ds, roi, taps, kern, nullptr);
    return filterGaussAdvanced<T, T, 4, false>((const T *)src, ss, (T *)dst, ds, roi, taps, kern, nullptr);
  }
}

extern "C" int ours_adv(const void *src, int ss, void *dst, int ds, int W, int H, int taps, const float *kern,
                        int dtype, int channels, int ac4) {
  if (dtype == 0) return (int)adv_call<Npp8u, false>(src, ss, 0, 0, 0, 0, dst, ds, W, H, taps, kern, 2, channels, ac4);
  if (dtype == 1) return (int)adv_call<Npp16u, false>(src, ss, 0, 0, 0, 0, dst, ds, W, H, taps, kern, 2, channels, ac4);
  if (dtype == 2) return (int)adv_call<Npp16s, false>(src, ss, 0, 0, 0, 0, dst, ds, W, H, taps, kern, 2, channels, ac4);
  return (int)adv_call<Npp32f, false>(src, ss, 0, 0, 0, 0, dst, ds, W, H, taps, kern, 2, channels, ac4);
}

extern "C" int ours_adv_border(const void *src, int ss, int srcW, int srcH, int ox, int oy, void *dst, int ds,
                               int W, int H, int taps, const float *kern, int border, int dtype, int channels, int ac4) {
  if (dtype == 0) return (int)adv_call<Npp8u, true>(src, ss, srcW, srcH, ox, oy, dst, ds, W, H, taps, kern, border, channels, ac4);
  if (dtype == 1) return (int)adv_call<Npp16u, true>(src, ss, srcW, srcH, ox, oy, dst, ds, W, H, taps, kern, border, channels, ac4);
  if (dtype == 2) return (int)adv_call<Npp16s, true>(src, ss, srcW, srcH, ox, oy, dst, ds, W, H, taps, kern, border, channels, ac4);
  return (int)adv_call<Npp32f, true>(src, ss, srcW, srcH, ox, oy, dst, ds, W, H, taps, kern, border, channels, ac4);
}
