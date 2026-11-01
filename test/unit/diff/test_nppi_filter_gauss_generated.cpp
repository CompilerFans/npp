// Differential tests for package f10: Gauss filter families
// (nppiFilterGauss / GaussBorder / GaussAdvanced / GaussAdvancedBorder).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiFilterGaussDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests \
//     --gtest_filter='NppiFilterGaussDiff.*'
//
// Measured semantics (5090, NPP 13.4; tools/probe_f10/fit/gauss_arith.cu):
//  - fixed masks are the square NppiMaskSize values 3,5,7,9,11,13,15 only
//    (1x3/1x5/3x1/5x1 are rejected with NPP_MASK_SIZE_ERROR);
//  - the 1D kernel is a normalised Gaussian with
//    sigma = 0.6 + 0.4 * (taps/2); the 2D kernel is its outer product and the
//    convolution is a correlation with the kernel origin at (taps-1)/2;
//  - GaussAdvanced takes an odd nFilterTaps >= 3 plus a user 1D kernel, outer
//    product, same geometry (no normalisation);
//  - plain variants read raw pixels around pSrc (caller provides the margin);
//    border variants accept only NPP_BORDER_REPLICATE and clamp reads to the
//    image bounds relative to pSrc: [-off, srcSize-1-off];
//  - float32 accumulation with FMA; integer destinations truncate toward zero
//    then saturate (8u [0,255], 16u [0,65535], 16s [-32768,32767]);
//  - AC4 computes channels 0..2 only and never writes the alpha byte; C4
//    computes all four channels;
//  - validation order plain Gauss: null -> negative ROI -> mask size ->
//    zero-size ROI -> dst step.  Border adds oSrcSize, oSrcOffset and border
//    type checks before the mask size.  Advanced plain: null -> negative ROI ->
//    zero-ROI tap quirks (taps <= 0 -> SIZE, 1/2 -> MASK, >=3 passes) ->
//    negative taps (OUT_OFF_RANGE) -> taps < 3 or even (MASK).  AdvancedBorder:
//    null -> negative ROI -> srcSize -> offset -> border type -> zero-ROI tap
//    quirks (taps < 3 -> MASK_SIZE_ERROR, >=3 passes regardless of parity) ->
//    taps (MASK_SIZE_ERROR) -> dst step;
//  - dst step is checked unsigned (negative passes) and must be a multiple of
//    the alignment (element size for C1/C3, pixel size for C4/AC4); the source
//    step is never validated.
//
// Integer outputs use a 1-LSB tolerance: the Gaussian kernel is generated in
// float32 with expf, and a 1-ulp kernel difference can move a truncation
// boundary by one (measured on the 5090 at <=0.5% of pixels for 16u; 8u/16s
// were exact in the probes).

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cmath>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;
namespace {

template <typename T> DeviceMemory<T> &arena(std::size_t n) {
  static std::vector<std::unique_ptr<DeviceMemory<T>>> v;
  v.push_back(std::make_unique<DeviceMemory<T>>(n));
  return *v.back();
}

NppStreamContext current_ctx() {
  NppStreamContext c{};
  NppStatus st = nppGetStreamContext(&c);
  EXPECT_EQ(st, NPP_NO_ERROR);
  return c;
}

int align_step(int step, int es) { return ((step + es - 1) / es) * es; }

template <typename T> ValueKind kind_of();
template <> ValueKind kind_of<Npp8u>() { return ValueKind::U8Tol; }
template <> ValueKind kind_of<Npp16u>() { return ValueKind::U16Tol; }
template <> ValueKind kind_of<Npp16s>() { return ValueKind::I16Tol; }
template <> ValueKind kind_of<Npp32f>() { return ValueKind::F32; }

template <typename T> void tolerance(double &rtol, double &atol) {
  if constexpr (std::is_floating_point_v<T>) {
    rtol = 1e-3;
    atol = 1e-3;
  } else {
    rtol = 0.0;
    atol = 1.0;
  }
}

int taps_of(NppiMaskSize m) {
  switch (static_cast<int>(m)) {
    case NPP_MASK_SIZE_3_X_3: return 3;
    case NPP_MASK_SIZE_5_X_5: return 5;
    case NPP_MASK_SIZE_7_X_7: return 7;
    case NPP_MASK_SIZE_9_X_9: return 9;
    case NPP_MASK_SIZE_11_X_11: return 11;
    case NPP_MASK_SIZE_13_X_13: return 13;
    case NPP_MASK_SIZE_15_X_15: return 15;
    default: return 1;
  }
}

std::vector<float> gauss_taps(int taps) {
  std::vector<float> k(taps);
  const int r = taps / 2;
  const double sigma = 0.6 + 0.4 * r;
  double sum = 0;
  for (int i = 0; i < taps; ++i) {
    const double d = i - r;
    k[i] = static_cast<float>(std::exp(-(d * d) / (2.0 * sigma * sigma)));
    sum += k[i];
  }
  for (float &v : k) v = static_cast<float>(v / sum);
  return k;
}

DeviceMemory<float> &kernel_buf(const std::vector<float> &k, uint64_t seed) {
  DeviceMemory<float> &mem = arena<float>(k.size() + 4);
  std::vector<float> host = k;
  host.resize(k.size() + 4, 0.f);
  mem.copyFromHost(host);
  (void)seed;
  return mem;
}

// ---------------------------------------------------------------- helpers

// Plain Gauss: source buffer is (W+2m)x(H+2m), pSrc inset by m pixels.
template <typename S, typename F>
void plain_ok(const std::string &fname, const std::string &key, int channels, int W, int H, int margin,
              int pad, uint64_t seed, F f) {
  const int spx = channels * (int)sizeof(S);
  const int bw = W + 2 * margin, bh = H + 2 * margin;
  const int sStep = align_step(bw * spx, spx);
  const int dpx = channels * (int)sizeof(S);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(S);
  const int dStep = align_step(W * dpx + pad, dAlign);
  const size_t se = (size_t)bh * sStep / sizeof(S);
  const size_t de = (size_t)H * dStep / sizeof(S);
  std::vector<S> hs = make_uniform<S>(se, seed + 11);
  std::vector<S> hd = make_uniform<S>(de, seed + 29);
  DeviceMemory<S> &s = arena<S>(se);
  DeviceMemory<S> &d = arena<S>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  const S *pSrc = s.get() + (size_t)margin * (sStep / sizeof(S)) + (size_t)margin * channels;
  clear_cuda_error();
  NppStatus st = f(pSrc, sStep, d.get(), dStep, NppiSize{W, H}, current_ctx());
  double rtol, atol;
  tolerance<S>(rtol, atol);
  Case c("nppiFilterGauss/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.status(st);
  c.device_output<S>(d.get(), de, kind_of<S>(), rtol, atol);
  c.finish();
}

// Border Gauss: image srcW x srcH, pSrc = image + offset pixels.
template <typename S, typename F>
void border_ok(const std::string &fname, const std::string &key, int channels, int srcW, int srcH, int offX,
               int offY, int W, int H, int pad, uint64_t seed, F f) {
  const int spx = channels * (int)sizeof(S);
  const int sStep = align_step(srcW * spx + pad, spx);
  const int dpx = channels * (int)sizeof(S);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(S);
  const int dStep = align_step(W * dpx + pad, dAlign);
  const size_t se = (size_t)srcH * sStep / sizeof(S);
  const size_t de = (size_t)H * dStep / sizeof(S);
  std::vector<S> hs = make_uniform<S>(se, seed + 11);
  std::vector<S> hd = make_uniform<S>(de, seed + 29);
  DeviceMemory<S> &s = arena<S>(se);
  DeviceMemory<S> &d = arena<S>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  const S *pSrc = s.get() + (size_t)offY * (sStep / sizeof(S)) + (size_t)offX * channels;
  clear_cuda_error();
  NppStatus st =
      f(pSrc, sStep, NppiSize{srcW, srcH}, NppiPoint{offX, offY}, d.get(), dStep, NppiSize{W, H},
        NPP_BORDER_REPLICATE, current_ctx());
  double rtol, atol;
  tolerance<S>(rtol, atol);
  Case c("nppiFilterGauss/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.status(st);
  c.device_output<S>(d.get(), de, kind_of<S>(), rtol, atol);
  c.finish();
}

// Plain fixed-mask wrapper: binds eMaskSize.
template <typename S, typename F>
void mask_ok(const std::string &fname, const std::string &key, int channels, int W, int H, uint64_t seed,
             NppiMaskSize ms, F f) {
  plain_ok<S>(fname, key, channels, W, H, taps_of(ms) / 2 + 1, 3, seed,
              [&](const S *p, int ss, S *q, int ds, NppiSize roi, NppStreamContext ctx) {
                return f(p, ss, q, ds, roi, ms, ctx);
              });
}

template <typename S, typename F>
void mask_border_ok(const std::string &fname, const std::string &key, int channels, int srcW, int srcH,
                    int offX, int offY, int W, int H, uint64_t seed, NppiMaskSize ms, F f) {
  border_ok<S>(fname, key, channels, srcW, srcH, offX, offY, W, H, 3, seed,
               [&](const S *p, int ss, NppiSize sz, NppiPoint o, S *q, int ds, NppiSize roi,
                   NppiBorderType b, NppStreamContext ctx) {
                 return f(p, ss, sz, o, q, ds, roi, ms, b, ctx);
               });
}

// Plain Advanced wrapper: binds the taps.
template <typename S, typename F>
void adv_ok(const std::string &fname, const std::string &key, int channels, int W, int H,
            const std::vector<float> &kern, uint64_t seed, F f) {
  const int taps = (int)kern.size();
  DeviceMemory<float> &kb = kernel_buf(kern, seed);
  plain_ok<S>(fname, key, channels, W, H, taps / 2 + 1, 3, seed,
              [&](const S *p, int ss, S *q, int ds, NppiSize roi, NppStreamContext ctx) {
                return f(p, ss, q, ds, roi, taps, kb.get(), ctx);
              });
}

template <typename S, typename F>
void adv_border_ok(const std::string &fname, const std::string &key, int channels, int srcW, int srcH,
                   int offX, int offY, int W, int H, const std::vector<float> &kern, uint64_t seed, F f) {
  const int taps = (int)kern.size();
  DeviceMemory<float> &kb = kernel_buf(kern, seed);
  border_ok<S>(fname, key, channels, srcW, srcH, offX, offY, W, H, 3, seed,
               [&](const S *p, int ss, NppiSize sz, NppiPoint o, S *q, int ds, NppiSize roi,
                   NppiBorderType b, NppStreamContext ctx) {
                 return f(p, ss, sz, o, q, ds, roi, taps, kb.get(), b, ctx);
               });
}

template <typename F> void only_status(const std::string &fname, const std::string &key, F f) {
  clear_cuda_error();
  NppStatus st = f();
  Case c("nppiFilterGauss/" + key, fname);
  c.status(st);
  c.finish();
}

// --------------------------------------------------------- error matrices

template <typename S, typename F>
void gauss_plain_errors(const std::string &fam, const std::string &suffix, int channels, S *s, S *d, F f) {
  NppStreamContext ctx = current_ctx();
  const std::string fn = fam + suffix;
  const NppiSize roi{4, 4};
  const int dpx = channels * (int)sizeof(S);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(S);
  const NppiMaskSize bad = (NppiMaskSize)0;
  const NppiMaskSize good = NPP_MASK_SIZE_3_X_3;
  only_status(fn, fam + "/err/nullsrc" + suffix, [&] { return f(nullptr, 16, d, 16, roi, bad, ctx); });
  only_status(fn, fam + "/err/nulldst" + suffix, [&] { return f(s, 16, nullptr, 16, roi, bad, ctx); });
  only_status(fn, fam + "/err/negroi_h" + suffix, [&] { return f(s, 16, d, 16, NppiSize{4, -1}, good, ctx); });
  only_status(fn, fam + "/err/negroi_badmask" + suffix,
              [&] { return f(s, 16, d, 16, NppiSize{4, -1}, bad, ctx); });
  only_status(fn, fam + "/err/badmask" + suffix, [&] { return f(s, 16, d, 16, roi, bad, ctx); });
  only_status(fn, fam + "/err/badmask900" + suffix,
              [&] { return f(s, 16, d, 16, roi, (NppiMaskSize)900, ctx); });
  only_status(fn, fam + "/err/badmask_1x3" + suffix,
              [&] { return f(s, 16, d, 16, roi, NPP_MASK_SIZE_1_X_3, ctx); });
  only_status(fn, fam + "/err/badmask_3x1" + suffix,
              [&] { return f(s, 16, d, 16, roi, NPP_MASK_SIZE_3_X_1, ctx); });
  only_status(fn, fam + "/err/roizero_w" + suffix, [&] { return f(s, 16, d, 16, NppiSize{0, 4}, good, ctx); });
  only_status(fn, fam + "/err/roizero_badmask" + suffix,
              [&] { return f(s, 16, d, 16, NppiSize{0, 4}, bad, ctx); });
  only_status(fn, fam + "/err/roizero_badstep" + suffix,
              [&] { return f(s, 16, d, 3, NppiSize{0, 0}, good, ctx); });
  only_status(fn, fam + "/err/badmask_stepsmall" + suffix,
              [&] { return f(s, 16, d, 4 * dpx - 1, roi, bad, ctx); });
  only_status(fn, fam + "/err/stepsmall" + suffix, [&] { return f(s, 16, d, 4 * dpx - 1, roi, good, ctx); });
  if (dAlign > 1)
    only_status(fn, fam + "/err/stepodd" + suffix, [&] { return f(s, 16, d, 4 * dpx + 1, roi, good, ctx); });
}

template <typename S, typename F>
void gauss_border_errors(const std::string &fam, const std::string &suffix, int channels, S *s, S *d, F f) {
  NppStreamContext ctx = current_ctx();
  const std::string fn = fam + suffix;
  const NppiSize sz{7, 7};
  const NppiPoint off{2, 2};
  const NppiSize roi{4, 4};
  const int dpx = channels * (int)sizeof(S);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(S);
  const NppiBorderType badB = NPP_BORDER_CONSTANT;
  const NppiBorderType rep = NPP_BORDER_REPLICATE;
  const NppiMaskSize badM = (NppiMaskSize)0;
  const NppiMaskSize goodM = NPP_MASK_SIZE_3_X_3;
  only_status(fn, fam + "/err/nullsrc" + suffix, [&] { return f(nullptr, 16, sz, off, d, 16, roi, goodM, rep, ctx); });
  only_status(fn, fam + "/err/nulldst" + suffix, [&] { return f(s, 16, sz, off, nullptr, 16, roi, goodM, rep, ctx); });
  only_status(fn, fam + "/err/negroi" + suffix, [&] { return f(s, 16, sz, off, d, 16, NppiSize{4, -1}, goodM, rep, ctx); });
  only_status(fn, fam + "/err/srcsize_w0" + suffix, [&] { return f(s, 16, NppiSize{0, 5}, off, d, 16, roi, goodM, rep, ctx); });
  only_status(fn, fam + "/err/srcsize_neg" + suffix, [&] { return f(s, 16, NppiSize{-1, 5}, off, d, 16, roi, goodM, rep, ctx); });
  only_status(fn, fam + "/err/offneg" + suffix, [&] { return f(s, 16, sz, NppiPoint{-1, 0}, d, 16, roi, goodM, rep, ctx); });
  only_status(fn, fam + "/err/off_eq_w" + suffix, [&] { return f(s, 16, sz, NppiPoint{7, 2}, d, 16, roi, goodM, rep, ctx); });
  only_status(fn, fam + "/err/off_w_minus1" + suffix, [&] { return f(s, 16, sz, NppiPoint{6, 2}, d, 16, roi, goodM, rep, ctx); });
  only_status(fn, fam + "/err/off_eq_h" + suffix, [&] { return f(s, 16, sz, NppiPoint{2, 7}, d, 16, roi, goodM, rep, ctx); });
  only_status(fn, fam + "/err/badborder" + suffix, [&] { return f(s, 16, sz, off, d, 16, roi, goodM, badB, ctx); });
  only_status(fn, fam + "/err/badborder_badmask" + suffix, [&] { return f(s, 16, sz, off, d, 16, roi, badM, badB, ctx); });
  only_status(fn, fam + "/err/badmask" + suffix, [&] { return f(s, 16, sz, off, d, 16, roi, badM, rep, ctx); });
  only_status(fn, fam + "/err/roizero" + suffix, [&] { return f(s, 16, sz, off, d, 16, NppiSize{0, 0}, goodM, rep, ctx); });
  only_status(fn, fam + "/err/roizero_badmask" + suffix, [&] { return f(s, 16, sz, off, d, 16, NppiSize{0, 0}, badM, rep, ctx); });
  only_status(fn, fam + "/err/roizero_badborder" + suffix, [&] { return f(s, 16, sz, off, d, 16, NppiSize{0, 0}, goodM, badB, ctx); });
  only_status(fn, fam + "/err/roizero_offneg" + suffix, [&] { return f(s, 16, sz, NppiPoint{0, -1}, d, 16, NppiSize{0, 0}, goodM, rep, ctx); });
  only_status(fn, fam + "/err/roizero_badstep" + suffix, [&] { return f(s, 16, sz, off, d, 3, NppiSize{0, 0}, goodM, rep, ctx); });
  only_status(fn, fam + "/err/badmask_stepsmall" + suffix, [&] { return f(s, 16, sz, off, d, 4 * dpx - 1, roi, badM, rep, ctx); });
  only_status(fn, fam + "/err/stepsmall" + suffix, [&] { return f(s, 16, sz, off, d, 4 * dpx - 1, roi, goodM, rep, ctx); });
  if (dAlign > 1)
    only_status(fn, fam + "/err/stepodd" + suffix, [&] { return f(s, 16, sz, off, d, 4 * dpx + 1, roi, goodM, rep, ctx); });
}

template <typename S, typename F>
void adv_plain_errors(const std::string &fam, const std::string &suffix, int channels, S *s, S *d, float *k,
                      F f) {
  NppStreamContext ctx = current_ctx();
  const std::string fn = fam + suffix;
  const NppiSize roi{4, 4};
  const NppiSize zroi{0, 0};
  const int dpx = channels * (int)sizeof(S);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(S);
  only_status(fn, fam + "/err/nullsrc" + suffix, [&] { return f(nullptr, 16, d, 16, roi, 3, k, ctx); });
  only_status(fn, fam + "/err/nulldst" + suffix, [&] { return f(s, 16, nullptr, 16, roi, 3, k, ctx); });
  only_status(fn, fam + "/err/nullkernel" + suffix, [&] { return f(s, 16, d, 16, roi, 3, nullptr, ctx); });
  only_status(fn, fam + "/err/negroi_nullk" + suffix, [&] { return f(s, 16, d, 16, NppiSize{4, -1}, 3, nullptr, ctx); });
  only_status(fn, fam + "/err/negroi" + suffix, [&] { return f(s, 16, d, 16, NppiSize{4, -1}, 3, k, ctx); });
  only_status(fn, fam + "/err/negroi_badtaps" + suffix, [&] { return f(s, 16, d, 16, NppiSize{4, -1}, 4, k, ctx); });
  only_status(fn, fam + "/err/taps0" + suffix, [&] { return f(s, 16, d, 16, roi, 0, k, ctx); });
  only_status(fn, fam + "/err/tapsneg" + suffix, [&] { return f(s, 16, d, 16, roi, -1, k, ctx); });
  only_status(fn, fam + "/err/taps1" + suffix, [&] { return f(s, 16, d, 16, roi, 1, k, ctx); });
  only_status(fn, fam + "/err/taps2" + suffix, [&] { return f(s, 16, d, 16, roi, 2, k, ctx); });
  only_status(fn, fam + "/err/taps4" + suffix, [&] { return f(s, 16, d, 16, roi, 4, k, ctx); });
  only_status(fn, fam + "/err/taps6" + suffix, [&] { return f(s, 16, d, 16, roi, 6, k, ctx); });
  only_status(fn, fam + "/err/taps33" + suffix, [&] { return f(s, 16, d, 16, roi, 33, k, ctx); });
  only_status(fn, fam + "/err/taps65" + suffix, [&] { return f(s, 16, d, 16, roi, 65, k, ctx); });
  // 5090 zero-ROI quirks: taps <= 0 -> SIZE, 1/2 -> MASK, >= 3 passes (even).
  only_status(fn, fam + "/err/roizero_taps0" + suffix, [&] { return f(s, 16, d, 16, zroi, 0, k, ctx); });
  only_status(fn, fam + "/err/roizero_tapsneg" + suffix, [&] { return f(s, 16, d, 16, zroi, -1, k, ctx); });
  only_status(fn, fam + "/err/roizero_taps1" + suffix, [&] { return f(s, 16, d, 16, zroi, 1, k, ctx); });
  only_status(fn, fam + "/err/roizero_taps2" + suffix, [&] { return f(s, 16, d, 16, zroi, 2, k, ctx); });
  only_status(fn, fam + "/err/roizero_taps3" + suffix, [&] { return f(s, 16, d, 16, zroi, 3, k, ctx); });
  only_status(fn, fam + "/err/roizero_taps4" + suffix, [&] { return f(s, 16, d, 16, zroi, 4, k, ctx); });
  only_status(fn, fam + "/err/roizero_nullk" + suffix, [&] { return f(s, 16, d, 16, zroi, 3, nullptr, ctx); });
  only_status(fn, fam + "/err/roizero_badstep" + suffix, [&] { return f(s, 16, d, 3, zroi, 3, k, ctx); });
  only_status(fn, fam + "/err/taps0_stepsmall" + suffix, [&] { return f(s, 16, d, 4 * dpx - 1, roi, 0, k, ctx); });
  only_status(fn, fam + "/err/stepsmall" + suffix, [&] { return f(s, 16, d, 4 * dpx - 1, roi, 3, k, ctx); });
  if (dAlign > 1)
    only_status(fn, fam + "/err/stepodd" + suffix, [&] { return f(s, 16, d, 4 * dpx + 1, roi, 3, k, ctx); });
}

template <typename S, typename F>
void adv_border_errors(const std::string &fam, const std::string &suffix, int channels, S *s, S *d, float *k,
                       F f) {
  NppStreamContext ctx = current_ctx();
  const std::string fn = fam + suffix;
  const NppiSize sz{7, 7};
  const NppiPoint off{2, 2};
  const NppiSize roi{4, 4};
  const NppiSize zroi{0, 0};
  const int dpx = channels * (int)sizeof(S);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(S);
  const NppiBorderType rep = NPP_BORDER_REPLICATE;
  const NppiBorderType badB = NPP_BORDER_CONSTANT;
  only_status(fn, fam + "/err/nullsrc" + suffix, [&] { return f(nullptr, 16, sz, off, d, 16, roi, 3, k, rep, ctx); });
  only_status(fn, fam + "/err/nulldst" + suffix, [&] { return f(s, 16, sz, off, nullptr, 16, roi, 3, k, rep, ctx); });
  only_status(fn, fam + "/err/nullkernel" + suffix, [&] { return f(s, 16, sz, off, d, 16, roi, 3, nullptr, rep, ctx); });
  only_status(fn, fam + "/err/negroi" + suffix, [&] { return f(s, 16, sz, off, d, 16, NppiSize{4, -1}, 3, k, rep, ctx); });
  only_status(fn, fam + "/err/srcsize_w0" + suffix, [&] { return f(s, 16, NppiSize{0, 5}, off, d, 16, roi, 3, k, rep, ctx); });
  only_status(fn, fam + "/err/srcsize0_taps0" + suffix, [&] { return f(s, 16, NppiSize{0, 5}, off, d, 16, roi, 0, k, rep, ctx); });
  only_status(fn, fam + "/err/offneg" + suffix, [&] { return f(s, 16, sz, NppiPoint{-1, 0}, d, 16, roi, 3, k, rep, ctx); });
  only_status(fn, fam + "/err/offneg_taps0" + suffix, [&] { return f(s, 16, sz, NppiPoint{-1, 0}, d, 16, roi, 0, k, rep, ctx); });
  only_status(fn, fam + "/err/off_eq_w" + suffix, [&] { return f(s, 16, sz, NppiPoint{7, 2}, d, 16, roi, 3, k, rep, ctx); });
  only_status(fn, fam + "/err/badborder" + suffix, [&] { return f(s, 16, sz, off, d, 16, roi, 3, k, badB, ctx); });
  only_status(fn, fam + "/err/badborder_taps0" + suffix, [&] { return f(s, 16, sz, off, d, 16, roi, 0, k, badB, ctx); });
  only_status(fn, fam + "/err/badborder_nullk" + suffix, [&] { return f(s, 16, sz, off, d, 16, roi, 3, nullptr, badB, ctx); });
  only_status(fn, fam + "/err/taps0" + suffix, [&] { return f(s, 16, sz, off, d, 16, roi, 0, k, rep, ctx); });
  only_status(fn, fam + "/err/tapsneg" + suffix, [&] { return f(s, 16, sz, off, d, 16, roi, -1, k, rep, ctx); });
  only_status(fn, fam + "/err/taps2" + suffix, [&] { return f(s, 16, sz, off, d, 16, roi, 2, k, rep, ctx); });
  only_status(fn, fam + "/err/taps4" + suffix, [&] { return f(s, 16, sz, off, d, 16, roi, 4, k, rep, ctx); });
  only_status(fn, fam + "/err/taps6" + suffix, [&] { return f(s, 16, sz, off, d, 16, roi, 6, k, rep, ctx); });
  only_status(fn, fam + "/err/roizero" + suffix, [&] { return f(s, 16, sz, off, d, 16, zroi, 3, k, rep, ctx); });
  only_status(fn, fam + "/err/roizero_taps0" + suffix, [&] { return f(s, 16, sz, off, d, 16, zroi, 0, k, rep, ctx); });
  only_status(fn, fam + "/err/roizero_taps2" + suffix, [&] { return f(s, 16, sz, off, d, 16, zroi, 2, k, rep, ctx); });
  only_status(fn, fam + "/err/roizero_taps4" + suffix, [&] { return f(s, 16, sz, off, d, 16, zroi, 4, k, rep, ctx); });
  only_status(fn, fam + "/err/roizero_taps6" + suffix, [&] { return f(s, 16, sz, off, d, 16, zroi, 6, k, rep, ctx); });
  only_status(fn, fam + "/err/roizero_tapsneg" + suffix, [&] { return f(s, 16, sz, off, d, 16, zroi, -1, k, rep, ctx); });
  only_status(fn, fam + "/err/roizero_badborder" + suffix, [&] { return f(s, 16, sz, off, d, 16, zroi, 3, k, badB, ctx); });
  only_status(fn, fam + "/err/roizero_offneg" + suffix, [&] { return f(s, 16, sz, NppiPoint{0, -1}, d, 16, zroi, 3, k, rep, ctx); });
  only_status(fn, fam + "/err/roizero_badstep" + suffix, [&] { return f(s, 16, sz, off, d, 3, zroi, 3, k, rep, ctx); });
  only_status(fn, fam + "/err/taps0_stepsmall" + suffix, [&] { return f(s, 16, sz, off, d, 4 * dpx - 1, roi, 0, k, rep, ctx); });
  only_status(fn, fam + "/err/stepsmall" + suffix, [&] { return f(s, 16, sz, off, d, 4 * dpx - 1, roi, 3, k, rep, ctx); });
  if (dAlign > 1)
    only_status(fn, fam + "/err/stepodd" + suffix, [&] { return f(s, 16, sz, off, d, 4 * dpx + 1, roi, 3, k, rep, ctx); });
}

// ------------------------------------------------------------ call macros

#define GX_PLAIN(FAM, S, TYP, MS, MGN)                                                                     \
  mask_ok<S>(#FAM "_" #TYP "_C1R_Ctx", #FAM "/" #TYP "/C1", 1, 11, 7,     \
                                               __LINE__, MS, FAM##_##TYP##_C1R_Ctx);                        \
  mask_ok<S>(#FAM "_" #TYP "_C3R_Ctx", #FAM "/" #TYP "/C3", 3, 11, 7,     \
                                               __LINE__, MS, FAM##_##TYP##_C3R_Ctx);                        \
  mask_ok<S>(#FAM "_" #TYP "_C4R_Ctx", #FAM "/" #TYP "/C4", 4, 11, 7,     \
                                               __LINE__, MS, FAM##_##TYP##_C4R_Ctx);                        \
  mask_ok<S>(#FAM "_" #TYP "_AC4R_Ctx", #FAM "/" #TYP "/AC4", 4, 11, 7,   \
                                                __LINE__, MS, FAM##_##TYP##_AC4R_Ctx)

#define GX_BORDER(FAM, S, TYP, MS)                                                                          \
  mask_border_ok<S>(#FAM "_" #TYP "_C1R_Ctx", #FAM "/" #TYP "/C1", 1, 13,   \
                                                     13, 2, 3, 9, 9, __LINE__, MS, FAM##_##TYP##_C1R_Ctx);   \
  mask_border_ok<S>(#FAM "_" #TYP "_C3R_Ctx", #FAM "/" #TYP "/C3", 3, 13,   \
                                                     13, 2, 3, 9, 9, __LINE__, MS, FAM##_##TYP##_C3R_Ctx);   \
  mask_border_ok<S>(#FAM "_" #TYP "_C4R_Ctx", #FAM "/" #TYP "/C4", 4, 13,   \
                                                     13, 2, 3, 9, 9, __LINE__, MS, FAM##_##TYP##_C4R_Ctx);   \
  mask_border_ok<S>(#FAM "_" #TYP "_AC4R_Ctx", #FAM "/" #TYP "/AC4", 4,    \
                                                      13, 13, 2, 3, 9, 9, __LINE__, MS,                     \
                                                      FAM##_##TYP##_AC4R_Ctx)

#define GX_ADV(FAM, S, TYP, KERN, TAG)                                                                       \
  adv_ok<S>(#FAM "_" #TYP "_C1R_Ctx", #FAM "/" #TYP "/C1/" TAG, 1, 9, 9,    \
                                             KERN, __LINE__, FAM##_##TYP##_C1R_Ctx);                         \
  adv_ok<S>(#FAM "_" #TYP "_C3R_Ctx", #FAM "/" #TYP "/C3/" TAG, 3, 9, 9,    \
                                             KERN, __LINE__, FAM##_##TYP##_C3R_Ctx);                         \
  adv_ok<S>(#FAM "_" #TYP "_C4R_Ctx", #FAM "/" #TYP "/C4/" TAG, 4, 9, 9,    \
                                             KERN, __LINE__, FAM##_##TYP##_C4R_Ctx);                         \
  adv_ok<S>(#FAM "_" #TYP "_AC4R_Ctx", #FAM "/" #TYP "/AC4/" TAG, 4, 9, 9, \
                                              KERN, __LINE__, FAM##_##TYP##_AC4R_Ctx)

#define GX_ADVB(FAM, S, TYP, KERN, TAG)                                                                      \
  adv_border_ok<S>(#FAM "_" #TYP "_C1R_Ctx", #FAM "/" #TYP "/C1/" TAG, 1,   \
                                                    13, 13, 2, 3, 9, 9, KERN, __LINE__,                     \
                                                    FAM##_##TYP##_C1R_Ctx);                                  \
  adv_border_ok<S>(#FAM "_" #TYP "_C3R_Ctx", #FAM "/" #TYP "/C3/" TAG, 3,   \
                                                    13, 13, 2, 3, 9, 9, KERN, __LINE__,                     \
                                                    FAM##_##TYP##_C3R_Ctx);                                  \
  adv_border_ok<S>(#FAM "_" #TYP "_C4R_Ctx", #FAM "/" #TYP "/C4/" TAG, 4,   \
                                                    13, 13, 2, 3, 9, 9, KERN, __LINE__,                     \
                                                    FAM##_##TYP##_C4R_Ctx);                                  \
  adv_border_ok<S>(#FAM "_" #TYP "_AC4R_Ctx", #FAM "/" #TYP "/AC4/" TAG,   \
                                                     4, 13, 13, 2, 3, 9, 9, KERN, __LINE__,                 \
                                                     FAM##_##TYP##_AC4R_Ctx)

#define GX_ERR(FAM, ERRS)                                                                              \
  ERRS(#FAM, "_8u_C1R_Ctx", 1, arena<Npp8u>(64).get(), arena<Npp8u>(64).get(), FAM##_8u_C1R_Ctx);      \
  ERRS(#FAM, "_16u_C1R_Ctx", 1, arena<Npp16u>(64).get(), arena<Npp16u>(64).get(), FAM##_16u_C1R_Ctx);  \
  ERRS(#FAM, "_32f_C1R_Ctx", 1, arena<Npp32f>(64).get(), arena<Npp32f>(64).get(), FAM##_32f_C1R_Ctx);  \
  ERRS(#FAM, "_8u_AC4R_Ctx", 4, arena<Npp8u>(64).get(), arena<Npp8u>(64).get(), FAM##_8u_AC4R_Ctx);    \
  ERRS(#FAM, "_16u_AC4R_Ctx", 4, arena<Npp16u>(64).get(), arena<Npp16u>(64).get(), FAM##_16u_AC4R_Ctx)

#define GX_AERR(FAM, ERRS)                                                                             \
  ERRS(#FAM, "_8u_C1R_Ctx", 1, arena<Npp8u>(64).get(), arena<Npp8u>(64).get(), advk.get(),              \
       FAM##_8u_C1R_Ctx);                                                                              \
  ERRS(#FAM, "_16u_C1R_Ctx", 1, arena<Npp16u>(64).get(), arena<Npp16u>(64).get(), advk.get(),           \
       FAM##_16u_C1R_Ctx);                                                                             \
  ERRS(#FAM, "_32f_C1R_Ctx", 1, arena<Npp32f>(64).get(), arena<Npp32f>(64).get(), advk.get(),           \
       FAM##_32f_C1R_Ctx);                                                                             \
  ERRS(#FAM, "_8u_AC4R_Ctx", 4, arena<Npp8u>(64).get(), arena<Npp8u>(64).get(), advk.get(),             \
       FAM##_8u_AC4R_Ctx);                                                                             \
  ERRS(#FAM, "_16u_AC4R_Ctx", 4, arena<Npp16u>(64).get(), arena<Npp16u>(64).get(), advk.get(),          \
       FAM##_16u_AC4R_Ctx)

}  // namespace

class NppiFilterGaussDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ============================ nppiFilterGauss =============================
TEST_F(NppiFilterGaussDiff, Gauss) {
  const NppiMaskSize masks[] = {NPP_MASK_SIZE_3_X_3,  NPP_MASK_SIZE_5_X_5,  NPP_MASK_SIZE_7_X_7,
                                NPP_MASK_SIZE_9_X_9,  NPP_MASK_SIZE_11_X_11, NPP_MASK_SIZE_13_X_13,
                                NPP_MASK_SIZE_15_X_15};
  for (NppiMaskSize m : masks)
    mask_ok<Npp8u>(
        "nppiFilterGauss_8u_C1R_Ctx", "nppiFilterGauss/8u/C1/m" + std::to_string(taps_of(m)), 1, 13, 11,
        __LINE__, m, nppiFilterGauss_8u_C1R_Ctx);
  GX_PLAIN(nppiFilterGauss, Npp8u, 8u, NPP_MASK_SIZE_3_X_3, 2);
  GX_PLAIN(nppiFilterGauss, Npp16u, 16u, NPP_MASK_SIZE_3_X_3, 2);
  GX_PLAIN(nppiFilterGauss, Npp16s, 16s, NPP_MASK_SIZE_3_X_3, 2);
  GX_PLAIN(nppiFilterGauss, Npp32f, 32f, NPP_MASK_SIZE_3_X_3, 2);
  GX_PLAIN(nppiFilterGauss, Npp8u, 8u, NPP_MASK_SIZE_5_X_5, 3);
  GX_PLAIN(nppiFilterGauss, Npp16u, 16u, NPP_MASK_SIZE_5_X_5, 3);
  GX_PLAIN(nppiFilterGauss, Npp16s, 16s, NPP_MASK_SIZE_5_X_5, 3);
  GX_PLAIN(nppiFilterGauss, Npp32f, 32f, NPP_MASK_SIZE_5_X_5, 3);
  // tiny ROI and padded destination step
  mask_ok<Npp8u>("nppiFilterGauss_8u_C1R_Ctx",
                                                      "nppiFilterGauss/8u/C1/1x1/m7", 1, 1, 1, __LINE__,
                                                      NPP_MASK_SIZE_7_X_7, nppiFilterGauss_8u_C1R_Ctx);
  mask_ok<Npp32f>("nppiFilterGauss_32f_C3R_Ctx",
                                                         "nppiFilterGauss/32f/C3/pad/m15", 3, 5, 4, __LINE__,
                                                         NPP_MASK_SIZE_15_X_15, nppiFilterGauss_32f_C3R_Ctx);
  GX_ERR(nppiFilterGauss, gauss_plain_errors);
}

// ========================= nppiFilterGaussBorder ==========================
TEST_F(NppiFilterGaussDiff, GaussBorder) {
  const NppiMaskSize masks[] = {NPP_MASK_SIZE_3_X_3,  NPP_MASK_SIZE_5_X_5,  NPP_MASK_SIZE_7_X_7,
                                NPP_MASK_SIZE_9_X_9,  NPP_MASK_SIZE_11_X_11, NPP_MASK_SIZE_13_X_13,
                                NPP_MASK_SIZE_15_X_15};
  for (NppiMaskSize m : masks)
    mask_border_ok<Npp8u>(
        "nppiFilterGaussBorder_8u_C1R_Ctx", "nppiFilterGaussBorder/8u/C1/m" + std::to_string(taps_of(m)),
        1, 17, 17, 3, 4, 11, 9, __LINE__, m, nppiFilterGaussBorder_8u_C1R_Ctx);
  GX_BORDER(nppiFilterGaussBorder, Npp8u, 8u, NPP_MASK_SIZE_3_X_3);
  GX_BORDER(nppiFilterGaussBorder, Npp16u, 16u, NPP_MASK_SIZE_3_X_3);
  GX_BORDER(nppiFilterGaussBorder, Npp16s, 16s, NPP_MASK_SIZE_3_X_3);
  GX_BORDER(nppiFilterGaussBorder, Npp32f, 32f, NPP_MASK_SIZE_3_X_3);
  GX_BORDER(nppiFilterGaussBorder, Npp8u, 8u, NPP_MASK_SIZE_5_X_5);
  GX_BORDER(nppiFilterGaussBorder, Npp16u, 16u, NPP_MASK_SIZE_5_X_5);
  GX_BORDER(nppiFilterGaussBorder, Npp16s, 16s, NPP_MASK_SIZE_5_X_5);
  GX_BORDER(nppiFilterGaussBorder, Npp32f, 32f, NPP_MASK_SIZE_5_X_5);
  // corner offset, overhang past srcSize, tiny image, padded source step
  mask_border_ok<Npp8u>(
      "nppiFilterGaussBorder_8u_C1R_Ctx", "nppiFilterGaussBorder/8u/C1/corner/m5", 1, 13, 13, 0, 0, 9, 9,
      __LINE__, NPP_MASK_SIZE_5_X_5, nppiFilterGaussBorder_8u_C1R_Ctx);
  mask_border_ok<Npp8u>(
      "nppiFilterGaussBorder_8u_C1R_Ctx", "nppiFilterGaussBorder/8u/C1/overhang/m7", 1, 13, 13, 10, 10, 5,
      5, __LINE__, NPP_MASK_SIZE_7_X_7, nppiFilterGaussBorder_8u_C1R_Ctx);
  mask_border_ok<Npp32f>(
      "nppiFilterGaussBorder_32f_C1R_Ctx", "nppiFilterGaussBorder/32f/C1/tiny/m9", 1, 3, 3, 1, 1, 5, 5,
      __LINE__, NPP_MASK_SIZE_9_X_9, nppiFilterGaussBorder_32f_C1R_Ctx);
  GX_ERR(nppiFilterGaussBorder, gauss_border_errors);
}

// ======================== nppiFilterGaussAdvanced =========================
TEST_F(NppiFilterGaussDiff, GaussAdvanced) {
  DeviceMemory<float> &advk = kernel_buf(std::vector<float>(65, 1.f / 65.f), __LINE__);
  (void)advk;
  for (int taps : {3, 5, 7, 9, 11}) {
    const std::vector<float> k = gauss_taps(taps);
    adv_ok<Npp8u>(
        "nppiFilterGaussAdvanced_8u_C1R_Ctx", "nppiFilterGaussAdvanced/8u/C1/t" + std::to_string(taps), 1,
        13, 11, k, __LINE__, nppiFilterGaussAdvanced_8u_C1R_Ctx);
  }
  const std::vector<float> k3 = gauss_taps(3);
  const std::vector<float> k5 = gauss_taps(5);
  const std::vector<float> asym7 = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};
  const std::vector<float> unnorm3 = {1.f, 2.f, 1.f};
  const std::vector<float> taps33(33, 1.f / 33.f);
  const std::vector<float> taps65(65, 1.f / 65.f);
  GX_ADV(nppiFilterGaussAdvanced, Npp8u, 8u, k3, "t3");
  GX_ADV(nppiFilterGaussAdvanced, Npp16u, 16u, k3, "t3");
  GX_ADV(nppiFilterGaussAdvanced, Npp16s, 16s, k3, "t3");
  GX_ADV(nppiFilterGaussAdvanced, Npp32f, 32f, k3, "t3");
  GX_ADV(nppiFilterGaussAdvanced, Npp8u, 8u, k5, "t5");
  GX_ADV(nppiFilterGaussAdvanced, Npp16u, 16u, k5, "t5");
  GX_ADV(nppiFilterGaussAdvanced, Npp16s, 16s, k5, "t5");
  GX_ADV(nppiFilterGaussAdvanced, Npp32f, 32f, k5, "t5");
  GX_ADV(nppiFilterGaussAdvanced, Npp32f, 32f, asym7, "t7asym");
  GX_ADV(nppiFilterGaussAdvanced, Npp8u, 8u, asym7, "t7asym");
  GX_ADV(nppiFilterGaussAdvanced, Npp16s, 16s, unnorm3, "t3unnorm");
  adv_ok<Npp32f>(
      "nppiFilterGaussAdvanced_32f_C1R_Ctx", "nppiFilterGaussAdvanced/32f/C1/t33", 1, 41, 37, taps33,
      __LINE__, nppiFilterGaussAdvanced_32f_C1R_Ctx);
  adv_ok<Npp16u>(
      "nppiFilterGaussAdvanced_16u_C1R_Ctx", "nppiFilterGaussAdvanced/16u/C1/t65", 1, 41, 37, taps65,
      __LINE__, nppiFilterGaussAdvanced_16u_C1R_Ctx);
  GX_AERR(nppiFilterGaussAdvanced, adv_plain_errors);
}

// ===================== nppiFilterGaussAdvancedBorder ======================
TEST_F(NppiFilterGaussDiff, GaussAdvancedBorder) {
  DeviceMemory<float> &advk = kernel_buf(std::vector<float>(65, 1.f / 65.f), __LINE__);
  (void)advk;
  const std::vector<float> k3 = gauss_taps(3);
  const std::vector<float> k5 = gauss_taps(5);
  const std::vector<float> k7 = gauss_taps(7);
  const std::vector<float> k9 = gauss_taps(9);
  const std::vector<float> asym7 = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f};
  GX_ADVB(nppiFilterGaussAdvancedBorder, Npp8u, 8u, k3, "t3");
  GX_ADVB(nppiFilterGaussAdvancedBorder, Npp16u, 16u, k3, "t3");
  GX_ADVB(nppiFilterGaussAdvancedBorder, Npp16s, 16s, k3, "t3");
  GX_ADVB(nppiFilterGaussAdvancedBorder, Npp32f, 32f, k3, "t3");
  GX_ADVB(nppiFilterGaussAdvancedBorder, Npp8u, 8u, k5, "t5");
  GX_ADVB(nppiFilterGaussAdvancedBorder, Npp32f, 32f, k5, "t5");
  GX_ADVB(nppiFilterGaussAdvancedBorder, Npp32f, 32f, k7, "t7");
  GX_ADVB(nppiFilterGaussAdvancedBorder, Npp8u, 8u, asym7, "t7asym");
  adv_border_ok<Npp32f>(
      "nppiFilterGaussAdvancedBorder_32f_C1R_Ctx", "nppiFilterGaussAdvancedBorder/32f/C1/corner/t9", 1, 15,
      15, 0, 0, 11, 11, k9, __LINE__, nppiFilterGaussAdvancedBorder_32f_C1R_Ctx);
  adv_border_ok<Npp8u>(
      "nppiFilterGaussAdvancedBorder_8u_C1R_Ctx", "nppiFilterGaussAdvancedBorder/8u/C1/overhang/t5", 1,
      13, 13, 10, 10, 5, 5, k5, __LINE__, nppiFilterGaussAdvancedBorder_8u_C1R_Ctx);
  adv_border_ok<Npp16s>(
      "nppiFilterGaussAdvancedBorder_16s_C1R_Ctx", "nppiFilterGaussAdvancedBorder/16s/C1/tiny/t7", 1, 3,
      3, 1, 1, 5, 5, k7, __LINE__, nppiFilterGaussAdvancedBorder_16s_C1R_Ctx);
  GX_AERR(nppiFilterGaussAdvancedBorder, adv_border_errors);
}
