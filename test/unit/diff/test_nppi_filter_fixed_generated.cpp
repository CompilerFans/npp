// Differential tests for package f11: fixed-kernel filters
// (nppiFilterHighPass / LowPass / Sharpen / Laplace and their Border twins;
// UnsharpBorder lives in test_nppi_filter_unsharp_generated.cpp).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiFilterFixedDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests \
//     --gtest_filter='NppiFilterFixedDiff.*'
//
// Measured semantics (5090, NPP 13.4, probes pa/pb/pi):
//  - kernels: HighPass n*center - sum(n-1 window) (n = 9 / 25);
//    LowPass box sum with bias (sum + n/2) / n (C truncation!);
//    Sharpen (16*center - sum8 + 3) / 8 (C truncation);
//    Laplace exact (-1 ring / 8, 5x5 second-difference table);
//    32f runs float arithmetic without the integer rounding bias.
//  - integer destinations saturate: 8u [0,255], 16u [0,65535],
//    16s [-32768,32767].
//  - plain variants read a raw window around each ROI pixel (no bounds
//    check); the source buffer must provide a margin.
//  - border variants accept only NPP_BORDER_REPLICATE; reads clamp to the
//    image bounds relative to pSrc: [-off, srcSize-1-off]; offset+ROI may
//    overhang srcSize.  offset must satisfy 0 <= off < srcSize.
//  - AC4 computes channels 0..2 only; dst alpha is never written.
//  - plain validation order: null(-8) -> ROI neg(-6) -> mask size(-33) ->
//    ROI zero(0, a zero width or height alone suffices) -> dst step
//    (-16 too small, -108 not a multiple of the alignment).
//  - border validation order: null(-8) -> ROI neg(-6) -> srcSize<=0(-6) ->
//    offset(-11) -> border type(-14) -> mask size(-33) -> ROI zero(0) ->
//    dst step.
//  - dst step alignment: element size for C1/C3, pixel size for C4/AC4;
//    the comparison is unsigned, so a negative step passes validation (such
//    cases are UB and excluded from this suite).  The source step is never
//    validated.
//  - in-place operation returns 0 but is racy; excluded from this suite.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <memory>
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
template <> ValueKind kind_of<Npp8u>() { return ValueKind::U8; }
template <> ValueKind kind_of<Npp8s>() { return ValueKind::I8; }
template <> ValueKind kind_of<Npp16u>() { return ValueKind::U16; }
template <> ValueKind kind_of<Npp16s>() { return ValueKind::I16; }
template <> ValueKind kind_of<Npp32f>() { return ValueKind::F32; }

// ---------------------------------------------------------------- helpers

// Plain fixed filter: source buffer is (W+2m)x(H+2m), pSrc inset by m pixels.
template <typename S, typename D, typename F>
void plain_ok(const std::string &fname, const std::string &key, int channels, int W, int H,
              int margin, int pad, uint64_t seed, F f) {
  const int spx = channels * (int)sizeof(S);
  const int bw = W + 2 * margin, bh = H + 2 * margin;
  const int sStep = align_step(bw * spx, spx);
  const int dpx = channels * (int)sizeof(D);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(D);
  const int dStep = align_step(W * dpx + pad, dAlign);
  const size_t se = (size_t)bh * sStep / sizeof(S);
  const size_t de = (size_t)H * dStep / sizeof(D);
  std::vector<S> hs = make_uniform<S>(se, seed + 11);
  std::vector<D> hd = make_uniform<D>(de, seed + 29);
  DeviceMemory<S> &s = arena<S>(se);
  DeviceMemory<D> &d = arena<D>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  const S *pSrc = s.get() + (size_t)margin * (sStep / sizeof(S)) + (size_t)margin * channels;
  clear_cuda_error();
  NppStatus st = f(pSrc, sStep, d.get(), dStep, NppiSize{W, H}, current_ctx());
  Case c("nppiFilterFixed/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.status(st);
  c.device_output<D>(d.get(), de, kind_of<D>(), 1e-3, 1e-3);
  c.finish();
}

// Masked plain fixed filter (adds eMaskSize).
template <typename S, typename D, typename F>
void mask_ok(const std::string &fname, const std::string &key, int channels, int W, int H,
             int margin, uint64_t seed, NppiMaskSize ms, F f) {
  plain_ok<S, D>(fname, key, channels, W, H, margin, 3, seed,
                 [&](const S *p, int ss, D *q, int ds, NppiSize roi, NppStreamContext ctx) {
                   return f(p, ss, q, ds, roi, ms, ctx);
                 });
}

// Border fixed filter: image srcW x srcH, pSrc = image + offset (pixels).
template <typename S, typename D, typename F>
void border_ok(const std::string &fname, const std::string &key, int channels, int srcW, int srcH,
               int offX, int offY, int W, int H, int pad, uint64_t seed, F f) {
  const int spx = channels * (int)sizeof(S);
  const int sStep = align_step(srcW * spx + pad, spx);
  const int dpx = channels * (int)sizeof(D);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(D);
  const int dStep = align_step(W * dpx + pad, dAlign);
  const size_t se = (size_t)srcH * sStep / sizeof(S);
  const size_t de = (size_t)H * dStep / sizeof(D);
  std::vector<S> hs = make_uniform<S>(se, seed + 11);
  std::vector<D> hd = make_uniform<D>(de, seed + 29);
  DeviceMemory<S> &s = arena<S>(se);
  DeviceMemory<D> &d = arena<D>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  const S *pSrc = s.get() + (size_t)offY * (sStep / sizeof(S)) + (size_t)offX * channels;
  clear_cuda_error();
  NppStatus st = f(pSrc, sStep, NppiSize{srcW, srcH}, NppiPoint{offX, offY}, d.get(), dStep,
                   NppiSize{W, H}, NPP_BORDER_REPLICATE, current_ctx());
  Case c("nppiFilterFixed/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.status(st);
  c.device_output<D>(d.get(), de, kind_of<D>(), 1e-3, 1e-3);
  c.finish();
}

// Masked border fixed filter (adds eMaskSize).
template <typename S, typename D, typename F>
void mask_border_ok(const std::string &fname, const std::string &key, int channels, int srcW,
                    int srcH, int offX, int offY, int W, int H, uint64_t seed, NppiMaskSize ms,
                    F f) {
  border_ok<S, D>(fname, key, channels, srcW, srcH, offX, offY, W, H, 3, seed,
                  [&](const S *p, int ss, NppiSize sz, NppiPoint o, D *q, int ds, NppiSize roi,
                      NppiBorderType b, NppStreamContext ctx) {
                    return f(p, ss, sz, o, q, ds, roi, ms, b, ctx);
                  });
}

// Status-only case: the functor fully binds the call.
template <typename F> void only_status(const std::string &fname, const std::string &key, F f) {
  clear_cuda_error();
  NppStatus st = f();
  Case c("nppiFilterFixed/" + key, fname);
  c.status(st);
  c.finish();
}

// ---------------------------------------------------------- error matrices

// Plain unmasked (Sharpen): null -> ROI neg -> ROI zero -> step.
template <typename S, typename D, typename F>
void plain_errors(const std::string &fam, const std::string &suffix, int channels, S *s, D *d, F f) {
  NppStreamContext ctx = current_ctx();
  const std::string fn = fam + suffix;
  const int dpx = channels * (int)sizeof(D);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(D);
  only_status(fn, fam + "/err/nullsrc" + suffix,
              [&] { return f(nullptr, 16, d, 16, NppiSize{4, 4}, ctx); });
  only_status(fn, fam + "/err/nulldst" + suffix,
              [&] { return f(s, 16, nullptr, 16, NppiSize{4, 4}, ctx); });
  only_status(fn, fam + "/err/negroi_w" + suffix,
              [&] { return f(s, 16, d, 16, NppiSize{-1, 4}, ctx); });
  only_status(fn, fam + "/err/negroi_h" + suffix,
              [&] { return f(s, 16, d, 16, NppiSize{4, -1}, ctx); });
  // zero width or height alone returns 0 before the step check
  only_status(fn, fam + "/err/roizero_w" + suffix,
              [&] { return f(s, 16, d, 16, NppiSize{0, 4}, ctx); });
  only_status(fn, fam + "/err/roizero_h" + suffix,
              [&] { return f(s, 16, d, 16, NppiSize{4, 0}, ctx); });
  only_status(fn, fam + "/err/roizero_badstep" + suffix,
              [&] { return f(s, 16, d, 3, NppiSize{0, 0}, ctx); });
  only_status(fn, fam + "/err/stepsmall" + suffix,
              [&] { return f(s, 16, d, 4 * dpx - 1, NppiSize{4, 4}, ctx); });
  if (dAlign > 1)
    only_status(fn, fam + "/err/stepodd" + suffix,
                [&] { return f(s, 16, d, 4 * dpx + 1, NppiSize{4, 4}, ctx); });
}

// Plain masked (HighPass/LowPass/Laplace):
// null -> ROI neg -> mask size -> ROI zero -> step.
template <typename S, typename D, typename F>
void mask_errors(const std::string &fam, const std::string &suffix, int channels, S *s, D *d, F f) {
  NppStreamContext ctx = current_ctx();
  const std::string fn = fam + suffix;
  const NppiSize roi{4, 4};
  const int dpx = channels * (int)sizeof(D);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(D);
  const NppiMaskSize bad = (NppiMaskSize)0;
  const NppiMaskSize good = NPP_MASK_SIZE_3_X_3;
  only_status(fn, fam + "/err/nullsrc" + suffix,
              [&] { return f(nullptr, 16, d, 16, roi, bad, ctx); });
  only_status(fn, fam + "/err/nulldst" + suffix,
              [&] { return f(s, 16, nullptr, 16, roi, bad, ctx); });
  only_status(fn, fam + "/err/negroi_h" + suffix,
              [&] { return f(s, 16, d, 16, NppiSize{4, -1}, good, ctx); });
  // negative ROI wins over the mask size
  only_status(fn, fam + "/err/negroi_badmask" + suffix,
              [&] { return f(s, 16, d, 16, NppiSize{4, -1}, bad, ctx); });
  only_status(fn, fam + "/err/badmask" + suffix,
              [&] { return f(s, 16, d, 16, roi, bad, ctx); });
  // zero width or height alone returns 0 (after the mask check)
  only_status(fn, fam + "/err/roizero_w" + suffix,
              [&] { return f(s, 16, d, 16, NppiSize{0, 4}, good, ctx); });
  only_status(fn, fam + "/err/roizero_w_badmask" + suffix,
              [&] { return f(s, 16, d, 16, NppiSize{0, 4}, bad, ctx); });
  // zero ROI wins over the step check but not over the mask size
  only_status(fn, fam + "/err/roizero_badstep" + suffix,
              [&] { return f(s, 16, d, 3, NppiSize{0, 0}, good, ctx); });
  // bad mask wins over the step check
  only_status(fn, fam + "/err/badmask_stepsmall" + suffix,
              [&] { return f(s, 16, d, 4 * dpx - 1, roi, bad, ctx); });
  only_status(fn, fam + "/err/stepsmall" + suffix,
              [&] { return f(s, 16, d, 4 * dpx - 1, roi, good, ctx); });
  if (dAlign > 1)
    only_status(fn, fam + "/err/stepodd" + suffix,
                [&] { return f(s, 16, d, 4 * dpx + 1, roi, good, ctx); });
}

// Border unmasked (SharpenBorder):
// null -> ROI neg -> srcSize -> offset -> border type -> ROI zero -> step.
template <typename S, typename D, typename F>
void border_errors(const std::string &fam, const std::string &suffix, int channels, S *s, D *d,
                   F f) {
  NppStreamContext ctx = current_ctx();
  const std::string fn = fam + suffix;
  const NppiSize sz{7, 7};
  const NppiPoint off{2, 2};
  const NppiSize roi{4, 4};
  const int dpx = channels * (int)sizeof(D);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(D);
  const NppiBorderType bad = NPP_BORDER_CONSTANT;
  const NppiBorderType rep = NPP_BORDER_REPLICATE;
  only_status(fn, fam + "/err/nullsrc" + suffix,
              [&] { return f(nullptr, 16, sz, off, d, 16, roi, rep, ctx); });
  only_status(fn, fam + "/err/nulldst" + suffix,
              [&] { return f(s, 16, sz, off, nullptr, 16, roi, rep, ctx); });
  only_status(fn, fam + "/err/negroi" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, NppiSize{4, -1}, rep, ctx); });
  only_status(fn, fam + "/err/negroi_badborder" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, NppiSize{4, -1}, bad, ctx); });
  only_status(fn, fam + "/err/srcsize_w0" + suffix,
              [&] { return f(s, 16, NppiSize{0, 5}, off, d, 16, roi, rep, ctx); });
  only_status(fn, fam + "/err/srcsize_neg" + suffix,
              [&] { return f(s, 16, NppiSize{-1, 5}, off, d, 16, roi, rep, ctx); });
  only_status(fn, fam + "/err/offneg" + suffix,
              [&] { return f(s, 16, sz, NppiPoint{-1, 0}, d, 16, roi, rep, ctx); });
  // offset must satisfy off < srcSize: off == srcSize fails, srcSize-1 passes
  only_status(fn, fam + "/err/off_eq_w" + suffix,
              [&] { return f(s, 16, sz, NppiPoint{7, 2}, d, 16, roi, rep, ctx); });
  only_status(fn, fam + "/err/off_w_minus1" + suffix,
              [&] { return f(s, 16, sz, NppiPoint{6, 2}, d, 16, roi, rep, ctx); });
  only_status(fn, fam + "/err/off_eq_h" + suffix,
              [&] { return f(s, 16, sz, NppiPoint{2, 7}, d, 16, roi, rep, ctx); });
  only_status(fn, fam + "/err/badborder" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, roi, bad, ctx); });
  // zero ROI: honored after offset / border type, before the step check
  only_status(fn, fam + "/err/roizero" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, NppiSize{0, 0}, rep, ctx); });
  only_status(fn, fam + "/err/roizero_badborder" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, NppiSize{0, 0}, bad, ctx); });
  only_status(fn, fam + "/err/roizero_offneg" + suffix,
              [&] { return f(s, 16, sz, NppiPoint{0, -1}, d, 16, NppiSize{0, 0}, rep, ctx); });
  only_status(fn, fam + "/err/roizero_badstep" + suffix,
              [&] { return f(s, 16, sz, off, d, 3, NppiSize{0, 0}, rep, ctx); });
  only_status(fn, fam + "/err/stepsmall" + suffix,
              [&] { return f(s, 16, sz, off, d, 4 * dpx - 1, roi, rep, ctx); });
  if (dAlign > 1)
    only_status(fn, fam + "/err/stepodd" + suffix,
                [&] { return f(s, 16, sz, off, d, 4 * dpx + 1, roi, rep, ctx); });
}

// Border masked (HighPassBorder/LowPassBorder/LaplaceBorder):
// null -> ROI neg -> srcSize -> offset -> border type -> mask size ->
// ROI zero -> step.
template <typename S, typename D, typename F>
void mask_border_errors(const std::string &fam, const std::string &suffix, int channels, S *s,
                        D *d, F f) {
  NppStreamContext ctx = current_ctx();
  const std::string fn = fam + suffix;
  const NppiSize sz{7, 7};
  const NppiPoint off{2, 2};
  const NppiSize roi{4, 4};
  const int dpx = channels * (int)sizeof(D);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(D);
  const NppiBorderType badB = NPP_BORDER_CONSTANT;
  const NppiBorderType rep = NPP_BORDER_REPLICATE;
  const NppiMaskSize badM = (NppiMaskSize)0;
  const NppiMaskSize goodM = NPP_MASK_SIZE_3_X_3;
  only_status(fn, fam + "/err/nullsrc" + suffix,
              [&] { return f(nullptr, 16, sz, off, d, 16, roi, goodM, rep, ctx); });
  only_status(fn, fam + "/err/nulldst" + suffix,
              [&] { return f(s, 16, sz, off, nullptr, 16, roi, goodM, rep, ctx); });
  only_status(fn, fam + "/err/negroi" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, NppiSize{4, -1}, goodM, rep, ctx); });
  only_status(fn, fam + "/err/srcsize_w0" + suffix,
              [&] { return f(s, 16, NppiSize{0, 5}, off, d, 16, roi, goodM, rep, ctx); });
  only_status(fn, fam + "/err/offneg" + suffix,
              [&] { return f(s, 16, sz, NppiPoint{-1, 0}, d, 16, roi, goodM, rep, ctx); });
  only_status(fn, fam + "/err/off_eq_w" + suffix,
              [&] { return f(s, 16, sz, NppiPoint{7, 2}, d, 16, roi, goodM, rep, ctx); });
  // border type wins over mask size / step
  only_status(fn, fam + "/err/badborder" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, roi, goodM, badB, ctx); });
  only_status(fn, fam + "/err/badborder_badmask" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, roi, badM, badB, ctx); });
  // bad mask wins over ROI zero / step
  only_status(fn, fam + "/err/badmask" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, roi, badM, rep, ctx); });
  only_status(fn, fam + "/err/roizero_badmask" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, NppiSize{0, 0}, badM, rep, ctx); });
  // zero ROI wins over the step check
  only_status(fn, fam + "/err/roizero" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, NppiSize{0, 0}, goodM, rep, ctx); });
  only_status(fn, fam + "/err/roizero_badborder" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, NppiSize{0, 0}, goodM, badB, ctx); });
  only_status(fn, fam + "/err/roizero_offneg" + suffix,
              [&] { return f(s, 16, sz, NppiPoint{0, -1}, d, 16, NppiSize{0, 0}, goodM, rep, ctx); });
  only_status(fn, fam + "/err/roizero_badstep" + suffix,
              [&] { return f(s, 16, sz, off, d, 3, NppiSize{0, 0}, goodM, rep, ctx); });
  only_status(fn, fam + "/err/badmask_stepsmall" + suffix,
              [&] { return f(s, 16, sz, off, d, 4 * dpx - 1, roi, badM, rep, ctx); });
  only_status(fn, fam + "/err/stepsmall" + suffix,
              [&] { return f(s, 16, sz, off, d, 4 * dpx - 1, roi, goodM, rep, ctx); });
  if (dAlign > 1)
    only_status(fn, fam + "/err/stepodd" + suffix,
                [&] { return f(s, 16, sz, off, d, 4 * dpx + 1, roi, goodM, rep, ctx); });
}

// ------------------------------------------------------------ call macros

// masked plain layout block: one dtype, all four layouts, one mask size
#define FXMP(FAM, S, TYP, MSS, MS, MGN)                                                            \
  mask_ok<S, S>(#FAM "_" #TYP "_C1R_Ctx", #FAM "/" #TYP "/C1/" MSS, 1, 9, 9, MGN, __LINE__, MS,   \
                FAM##_##TYP##_C1R_Ctx);                                                            \
  mask_ok<S, S>(#FAM "_" #TYP "_C3R_Ctx", #FAM "/" #TYP "/C3/" MSS, 3, 9, 9, MGN, __LINE__, MS,   \
                FAM##_##TYP##_C3R_Ctx);                                                            \
  mask_ok<S, S>(#FAM "_" #TYP "_C4R_Ctx", #FAM "/" #TYP "/C4/" MSS, 4, 9, 9, MGN, __LINE__, MS,   \
                FAM##_##TYP##_C4R_Ctx);                                                            \
  mask_ok<S, S>(#FAM "_" #TYP "_AC4R_Ctx", #FAM "/" #TYP "/AC4/" MSS, 4, 9, 9, MGN, __LINE__, MS, \
                FAM##_##TYP##_AC4R_Ctx)

// unmasked plain layout block (Sharpen)
#define FXPP(FAM, S, TYP)                                                                          \
  plain_ok<S, S>(#FAM "_" #TYP "_C1R_Ctx", #FAM "/" #TYP "/C1", 1, 9, 9, 2, 3, __LINE__,           \
                 FAM##_##TYP##_C1R_Ctx);                                                           \
  plain_ok<S, S>(#FAM "_" #TYP "_C3R_Ctx", #FAM "/" #TYP "/C3", 3, 9, 9, 2, 3, __LINE__,           \
                 FAM##_##TYP##_C3R_Ctx);                                                           \
  plain_ok<S, S>(#FAM "_" #TYP "_C4R_Ctx", #FAM "/" #TYP "/C4", 4, 9, 9, 2, 3, __LINE__,           \
                 FAM##_##TYP##_C4R_Ctx);                                                           \
  plain_ok<S, S>(#FAM "_" #TYP "_AC4R_Ctx", #FAM "/" #TYP "/AC4", 4, 9, 9, 2, 3, __LINE__,         \
                 FAM##_##TYP##_AC4R_Ctx)

// masked border layout block
#define FXMB(FAM, S, TYP, MSS, MS)                                                                 \
  mask_border_ok<S, S>(#FAM "_" #TYP "_C1R_Ctx", #FAM "/" #TYP "/C1/" MSS, 1, 13, 13, 2, 3, 9, 9,  \
                       __LINE__, MS, FAM##_##TYP##_C1R_Ctx);                                       \
  mask_border_ok<S, S>(#FAM "_" #TYP "_C3R_Ctx", #FAM "/" #TYP "/C3/" MSS, 3, 13, 13, 2, 3, 9, 9,  \
                       __LINE__, MS, FAM##_##TYP##_C3R_Ctx);                                       \
  mask_border_ok<S, S>(#FAM "_" #TYP "_C4R_Ctx", #FAM "/" #TYP "/C4/" MSS, 4, 13, 13, 2, 3, 9, 9,  \
                       __LINE__, MS, FAM##_##TYP##_C4R_Ctx);                                       \
  mask_border_ok<S, S>(#FAM "_" #TYP "_AC4R_Ctx", #FAM "/" #TYP "/AC4/" MSS, 4, 13, 13, 2, 3, 9,   \
                       9, __LINE__, MS, FAM##_##TYP##_AC4R_Ctx)

// unmasked border layout block (SharpenBorder)
#define FXPB(FAM, S, TYP)                                                                          \
  border_ok<S, S>(#FAM "_" #TYP "_C1R_Ctx", #FAM "/" #TYP "/C1", 1, 13, 13, 2, 3, 9, 9, 3,         \
                  __LINE__, FAM##_##TYP##_C1R_Ctx);                                                \
  border_ok<S, S>(#FAM "_" #TYP "_C3R_Ctx", #FAM "/" #TYP "/C3", 3, 13, 13, 2, 3, 9, 9, 3,         \
                  __LINE__, FAM##_##TYP##_C3R_Ctx);                                                \
  border_ok<S, S>(#FAM "_" #TYP "_C4R_Ctx", #FAM "/" #TYP "/C4", 4, 13, 13, 2, 3, 9, 9, 3,         \
                  __LINE__, FAM##_##TYP##_C4R_Ctx);                                                \
  border_ok<S, S>(#FAM "_" #TYP "_AC4R_Ctx", #FAM "/" #TYP "/AC4", 4, 13, 13, 2, 3, 9, 9, 3,       \
                  __LINE__, FAM##_##TYP##_AC4R_Ctx)

// error-matrix blocks over the dtype/layout configs whose step alignment
// differs (8u C1: align 1, 16u C1: 2, 32f C1: 4, 8u AC4: 4, 16u AC4: 8)
#define FXMERR(FAM, ERRS)                                                                          \
  ERRS(#FAM, "_8u_C1R_Ctx", 1, arena<Npp8u>(64).get(), arena<Npp8u>(64).get(), FAM##_8u_C1R_Ctx);  \
  ERRS(#FAM, "_16u_C1R_Ctx", 1, arena<Npp16u>(64).get(), arena<Npp16u>(64).get(),                  \
       FAM##_16u_C1R_Ctx);                                                                         \
  ERRS(#FAM, "_32f_C1R_Ctx", 1, arena<Npp32f>(64).get(), arena<Npp32f>(64).get(),                  \
       FAM##_32f_C1R_Ctx);                                                                         \
  ERRS(#FAM, "_8u_AC4R_Ctx", 4, arena<Npp8u>(64).get(), arena<Npp8u>(64).get(), FAM##_8u_AC4R_Ctx);\
  ERRS(#FAM, "_16u_AC4R_Ctx", 4, arena<Npp16u>(64).get(), arena<Npp16u>(64).get(),                 \
       FAM##_16u_AC4R_Ctx)

}  // namespace

class NppiFilterFixedDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ==================== HighPass (masked plain) ====================
TEST_F(NppiFilterFixedDiff, HighPass) {
  FXMP(nppiFilterHighPass, Npp8u, 8u, "m3", NPP_MASK_SIZE_3_X_3, 2);
  FXMP(nppiFilterHighPass, Npp16u, 16u, "m3", NPP_MASK_SIZE_3_X_3, 2);
  FXMP(nppiFilterHighPass, Npp16s, 16s, "m3", NPP_MASK_SIZE_3_X_3, 2);
  FXMP(nppiFilterHighPass, Npp32f, 32f, "m3", NPP_MASK_SIZE_3_X_3, 2);
  FXMP(nppiFilterHighPass, Npp8u, 8u, "m5", NPP_MASK_SIZE_5_X_5, 3);
  FXMP(nppiFilterHighPass, Npp16u, 16u, "m5", NPP_MASK_SIZE_5_X_5, 3);
  FXMP(nppiFilterHighPass, Npp16s, 16s, "m5", NPP_MASK_SIZE_5_X_5, 3);
  FXMP(nppiFilterHighPass, Npp32f, 32f, "m5", NPP_MASK_SIZE_5_X_5, 3);
  // extra shapes
  mask_ok<Npp8u, Npp8u>("nppiFilterHighPass_8u_C1R_Ctx", "nppiFilterHighPass/8u/C1/11x7/m3", 1, 11,
                        7, 2, __LINE__, NPP_MASK_SIZE_3_X_3, nppiFilterHighPass_8u_C1R_Ctx);
  mask_ok<Npp8u, Npp8u>("nppiFilterHighPass_8u_C1R_Ctx", "nppiFilterHighPass/8u/C1/1x1/m5", 1, 1, 1,
                        3, __LINE__, NPP_MASK_SIZE_5_X_5, nppiFilterHighPass_8u_C1R_Ctx);
  mask_ok<Npp16s, Npp16s>("nppiFilterHighPass_16s_C1R_Ctx", "nppiFilterHighPass/16s/C1/1x1/m3", 1,
                          1, 1, 2, __LINE__, NPP_MASK_SIZE_3_X_3, nppiFilterHighPass_16s_C1R_Ctx);
  mask_ok<Npp32f, Npp32f>("nppiFilterHighPass_32f_C3R_Ctx", "nppiFilterHighPass/32f/C3/11x7/m5", 3,
                          11, 7, 3, __LINE__, NPP_MASK_SIZE_5_X_5, nppiFilterHighPass_32f_C3R_Ctx);
  mask_ok<Npp16u, Npp16u>("nppiFilterHighPass_16u_C4R_Ctx", "nppiFilterHighPass/16u/C4/11x7/m5", 4,
                          11, 7, 3, __LINE__, NPP_MASK_SIZE_5_X_5, nppiFilterHighPass_16u_C4R_Ctx);
  FXMERR(nppiFilterHighPass, mask_errors);
}

// ==================== HighPassBorder (masked border) ====================
TEST_F(NppiFilterFixedDiff, HighPassBorder) {
  FXMB(nppiFilterHighPassBorder, Npp8u, 8u, "m3", NPP_MASK_SIZE_3_X_3);
  FXMB(nppiFilterHighPassBorder, Npp16u, 16u, "m3", NPP_MASK_SIZE_3_X_3);
  FXMB(nppiFilterHighPassBorder, Npp16s, 16s, "m3", NPP_MASK_SIZE_3_X_3);
  FXMB(nppiFilterHighPassBorder, Npp32f, 32f, "m3", NPP_MASK_SIZE_3_X_3);
  FXMB(nppiFilterHighPassBorder, Npp8u, 8u, "m5", NPP_MASK_SIZE_5_X_5);
  FXMB(nppiFilterHighPassBorder, Npp16u, 16u, "m5", NPP_MASK_SIZE_5_X_5);
  FXMB(nppiFilterHighPassBorder, Npp16s, 16s, "m5", NPP_MASK_SIZE_5_X_5);
  FXMB(nppiFilterHighPassBorder, Npp32f, 32f, "m5", NPP_MASK_SIZE_5_X_5);
  // extra shapes: corner offset, overhang past srcSize, odd sizes, tiny image
  mask_border_ok<Npp8u, Npp8u>("nppiFilterHighPassBorder_8u_C1R_Ctx",
                               "nppiFilterHighPassBorder/8u/C1/corner/m3", 1, 13, 13, 0, 0, 9, 9,
                               __LINE__, NPP_MASK_SIZE_3_X_3, nppiFilterHighPassBorder_8u_C1R_Ctx);
  mask_border_ok<Npp8u, Npp8u>("nppiFilterHighPassBorder_8u_C1R_Ctx",
                               "nppiFilterHighPassBorder/8u/C1/overhang/m5", 1, 13, 13, 10, 10, 5,
                               5, __LINE__, NPP_MASK_SIZE_5_X_5, nppiFilterHighPassBorder_8u_C1R_Ctx);
  mask_border_ok<Npp16s, Npp16s>("nppiFilterHighPassBorder_16s_C3R_Ctx",
                                 "nppiFilterHighPassBorder/16s/C3/11x7/m3", 3, 11, 7, 1, 2, 7, 5,
                                 __LINE__, NPP_MASK_SIZE_3_X_3,
                                 nppiFilterHighPassBorder_16s_C3R_Ctx);
  mask_border_ok<Npp32f, Npp32f>("nppiFilterHighPassBorder_32f_C1R_Ctx",
                                 "nppiFilterHighPassBorder/32f/C1/tiny/m5", 1, 3, 3, 1, 1, 5, 5,
                                 __LINE__, NPP_MASK_SIZE_5_X_5,
                                 nppiFilterHighPassBorder_32f_C1R_Ctx);
  mask_border_ok<Npp8u, Npp8u>("nppiFilterHighPassBorder_8u_AC4R_Ctx",
                               "nppiFilterHighPassBorder/8u/AC4/allclamp/m5", 4, 1, 1, 0, 0, 2, 2,
                               __LINE__, NPP_MASK_SIZE_5_X_5,
                               nppiFilterHighPassBorder_8u_AC4R_Ctx);
  FXMERR(nppiFilterHighPassBorder, mask_border_errors);
}

// ==================== LowPass (masked plain) ====================
TEST_F(NppiFilterFixedDiff, LowPass) {
  FXMP(nppiFilterLowPass, Npp8u, 8u, "m3", NPP_MASK_SIZE_3_X_3, 2);
  FXMP(nppiFilterLowPass, Npp16u, 16u, "m3", NPP_MASK_SIZE_3_X_3, 2);
  FXMP(nppiFilterLowPass, Npp16s, 16s, "m3", NPP_MASK_SIZE_3_X_3, 2);
  FXMP(nppiFilterLowPass, Npp32f, 32f, "m3", NPP_MASK_SIZE_3_X_3, 2);
  FXMP(nppiFilterLowPass, Npp8u, 8u, "m5", NPP_MASK_SIZE_5_X_5, 3);
  FXMP(nppiFilterLowPass, Npp16u, 16u, "m5", NPP_MASK_SIZE_5_X_5, 3);
  FXMP(nppiFilterLowPass, Npp16s, 16s, "m5", NPP_MASK_SIZE_5_X_5, 3);
  FXMP(nppiFilterLowPass, Npp32f, 32f, "m5", NPP_MASK_SIZE_5_X_5, 3);
  mask_ok<Npp8u, Npp8u>("nppiFilterLowPass_8u_C1R_Ctx", "nppiFilterLowPass/8u/C1/11x7/m3", 1, 11, 7,
                        2, __LINE__, NPP_MASK_SIZE_3_X_3, nppiFilterLowPass_8u_C1R_Ctx);
  mask_ok<Npp8u, Npp8u>("nppiFilterLowPass_8u_C1R_Ctx", "nppiFilterLowPass/8u/C1/1x1/m5", 1, 1, 1,
                        3, __LINE__, NPP_MASK_SIZE_5_X_5, nppiFilterLowPass_8u_C1R_Ctx);
  mask_ok<Npp16s, Npp16s>("nppiFilterLowPass_16s_C1R_Ctx", "nppiFilterLowPass/16s/C1/1x1/m3", 1, 1,
                          1, 2, __LINE__, NPP_MASK_SIZE_3_X_3, nppiFilterLowPass_16s_C1R_Ctx);
  mask_ok<Npp32f, Npp32f>("nppiFilterLowPass_32f_C3R_Ctx", "nppiFilterLowPass/32f/C3/11x7/m5", 3,
                          11, 7, 3, __LINE__, NPP_MASK_SIZE_5_X_5, nppiFilterLowPass_32f_C3R_Ctx);
  mask_ok<Npp16u, Npp16u>("nppiFilterLowPass_16u_C4R_Ctx", "nppiFilterLowPass/16u/C4/11x7/m5", 4,
                          11, 7, 3, __LINE__, NPP_MASK_SIZE_5_X_5, nppiFilterLowPass_16u_C4R_Ctx);
  FXMERR(nppiFilterLowPass, mask_errors);
}

// ==================== LowPassBorder (masked border) ====================
TEST_F(NppiFilterFixedDiff, LowPassBorder) {
  FXMB(nppiFilterLowPassBorder, Npp8u, 8u, "m3", NPP_MASK_SIZE_3_X_3);
  FXMB(nppiFilterLowPassBorder, Npp16u, 16u, "m3", NPP_MASK_SIZE_3_X_3);
  FXMB(nppiFilterLowPassBorder, Npp16s, 16s, "m3", NPP_MASK_SIZE_3_X_3);
  FXMB(nppiFilterLowPassBorder, Npp32f, 32f, "m3", NPP_MASK_SIZE_3_X_3);
  FXMB(nppiFilterLowPassBorder, Npp8u, 8u, "m5", NPP_MASK_SIZE_5_X_5);
  FXMB(nppiFilterLowPassBorder, Npp16u, 16u, "m5", NPP_MASK_SIZE_5_X_5);
  FXMB(nppiFilterLowPassBorder, Npp16s, 16s, "m5", NPP_MASK_SIZE_5_X_5);
  FXMB(nppiFilterLowPassBorder, Npp32f, 32f, "m5", NPP_MASK_SIZE_5_X_5);
  mask_border_ok<Npp8u, Npp8u>("nppiFilterLowPassBorder_8u_C1R_Ctx",
                               "nppiFilterLowPassBorder/8u/C1/corner/m3", 1, 13, 13, 0, 0, 9, 9,
                               __LINE__, NPP_MASK_SIZE_3_X_3, nppiFilterLowPassBorder_8u_C1R_Ctx);
  mask_border_ok<Npp8u, Npp8u>("nppiFilterLowPassBorder_8u_C1R_Ctx",
                               "nppiFilterLowPassBorder/8u/C1/overhang/m5", 1, 13, 13, 10, 10, 5,
                               5, __LINE__, NPP_MASK_SIZE_5_X_5, nppiFilterLowPassBorder_8u_C1R_Ctx);
  mask_border_ok<Npp16s, Npp16s>("nppiFilterLowPassBorder_16s_C3R_Ctx",
                                 "nppiFilterLowPassBorder/16s/C3/11x7/m3", 3, 11, 7, 1, 2, 7, 5,
                                 __LINE__, NPP_MASK_SIZE_3_X_3,
                                 nppiFilterLowPassBorder_16s_C3R_Ctx);
  mask_border_ok<Npp32f, Npp32f>("nppiFilterLowPassBorder_32f_C1R_Ctx",
                                 "nppiFilterLowPassBorder/32f/C1/tiny/m5", 1, 3, 3, 1, 1, 5, 5,
                                 __LINE__, NPP_MASK_SIZE_5_X_5,
                                 nppiFilterLowPassBorder_32f_C1R_Ctx);
  mask_border_ok<Npp8u, Npp8u>("nppiFilterLowPassBorder_8u_AC4R_Ctx",
                               "nppiFilterLowPassBorder/8u/AC4/allclamp/m5", 4, 1, 1, 0, 0, 2, 2,
                               __LINE__, NPP_MASK_SIZE_5_X_5,
                               nppiFilterLowPassBorder_8u_AC4R_Ctx);
  FXMERR(nppiFilterLowPassBorder, mask_border_errors);
}

// ==================== Sharpen (unmasked plain) ====================
TEST_F(NppiFilterFixedDiff, Sharpen) {
  FXPP(nppiFilterSharpen, Npp8u, 8u);
  FXPP(nppiFilterSharpen, Npp16u, 16u);
  FXPP(nppiFilterSharpen, Npp16s, 16s);
  FXPP(nppiFilterSharpen, Npp32f, 32f);
  plain_ok<Npp8u, Npp8u>("nppiFilterSharpen_8u_C1R_Ctx", "nppiFilterSharpen/8u/C1/11x7", 1, 11, 7,
                         2, 5, __LINE__, nppiFilterSharpen_8u_C1R_Ctx);
  plain_ok<Npp16s, Npp16s>("nppiFilterSharpen_16s_C1R_Ctx", "nppiFilterSharpen/16s/C1/1x1", 1, 1,
                           1, 2, 3, __LINE__, nppiFilterSharpen_16s_C1R_Ctx);
  plain_ok<Npp32f, Npp32f>("nppiFilterSharpen_32f_C3R_Ctx", "nppiFilterSharpen/32f/C3/11x7", 3, 11,
                           7, 2, 5, __LINE__, nppiFilterSharpen_32f_C3R_Ctx);
  plain_ok<Npp16u, Npp16u>("nppiFilterSharpen_16u_AC4R_Ctx", "nppiFilterSharpen/16u/AC4/11x7", 4,
                           11, 7, 2, 4, __LINE__, nppiFilterSharpen_16u_AC4R_Ctx);
  FXMERR(nppiFilterSharpen, plain_errors);
}

// ==================== SharpenBorder (unmasked border) ====================
TEST_F(NppiFilterFixedDiff, SharpenBorder) {
  FXPB(nppiFilterSharpenBorder, Npp8u, 8u);
  FXPB(nppiFilterSharpenBorder, Npp16u, 16u);
  FXPB(nppiFilterSharpenBorder, Npp16s, 16s);
  FXPB(nppiFilterSharpenBorder, Npp32f, 32f);
  border_ok<Npp8u, Npp8u>("nppiFilterSharpenBorder_8u_C1R_Ctx", "nppiFilterSharpenBorder/8u/C1/corner",
                          1, 13, 13, 0, 0, 9, 9, 3, __LINE__, nppiFilterSharpenBorder_8u_C1R_Ctx);
  border_ok<Npp8u, Npp8u>("nppiFilterSharpenBorder_8u_C1R_Ctx",
                          "nppiFilterSharpenBorder/8u/C1/overhang", 1, 13, 13, 10, 10, 5, 5, 3,
                          __LINE__, nppiFilterSharpenBorder_8u_C1R_Ctx);
  border_ok<Npp16s, Npp16s>("nppiFilterSharpenBorder_16s_C3R_Ctx",
                            "nppiFilterSharpenBorder/16s/C3/11x7", 3, 11, 7, 1, 2, 7, 5, 3,
                            __LINE__, nppiFilterSharpenBorder_16s_C3R_Ctx);
  border_ok<Npp32f, Npp32f>("nppiFilterSharpenBorder_32f_C1R_Ctx",
                            "nppiFilterSharpenBorder/32f/C1/tiny", 1, 3, 3, 1, 1, 5, 5, 3, __LINE__,
                            nppiFilterSharpenBorder_32f_C1R_Ctx);
  FXMERR(nppiFilterSharpenBorder, border_errors);
}

// ==================== Laplace (masked plain, 8u/16s/32f + 8u16s/8s16s) ====
TEST_F(NppiFilterFixedDiff, Laplace) {
  FXMP(nppiFilterLaplace, Npp8u, 8u, "m3", NPP_MASK_SIZE_3_X_3, 2);
  FXMP(nppiFilterLaplace, Npp16s, 16s, "m3", NPP_MASK_SIZE_3_X_3, 2);
  FXMP(nppiFilterLaplace, Npp32f, 32f, "m3", NPP_MASK_SIZE_3_X_3, 2);
  FXMP(nppiFilterLaplace, Npp8u, 8u, "m5", NPP_MASK_SIZE_5_X_5, 3);
  FXMP(nppiFilterLaplace, Npp16s, 16s, "m5", NPP_MASK_SIZE_5_X_5, 3);
  FXMP(nppiFilterLaplace, Npp32f, 32f, "m5", NPP_MASK_SIZE_5_X_5, 3);
  // widening conversions
  mask_ok<Npp8u, Npp16s>("nppiFilterLaplace_8u16s_C1R_Ctx", "nppiFilterLaplace/8u16s/m3", 1, 9, 9,
                         2, __LINE__, NPP_MASK_SIZE_3_X_3, nppiFilterLaplace_8u16s_C1R_Ctx);
  mask_ok<Npp8u, Npp16s>("nppiFilterLaplace_8u16s_C1R_Ctx", "nppiFilterLaplace/8u16s/m5", 1, 9, 9,
                         3, __LINE__, NPP_MASK_SIZE_5_X_5, nppiFilterLaplace_8u16s_C1R_Ctx);
  mask_ok<Npp8u, Npp16s>("nppiFilterLaplace_8u16s_C1R_Ctx", "nppiFilterLaplace/8u16s/11x7/m5", 1,
                         11, 7, 3, __LINE__, NPP_MASK_SIZE_5_X_5, nppiFilterLaplace_8u16s_C1R_Ctx);
  mask_ok<Npp8s, Npp16s>("nppiFilterLaplace_8s16s_C1R_Ctx", "nppiFilterLaplace/8s16s/m3", 1, 9, 9,
                         2, __LINE__, NPP_MASK_SIZE_3_X_3, nppiFilterLaplace_8s16s_C1R_Ctx);
  mask_ok<Npp8s, Npp16s>("nppiFilterLaplace_8s16s_C1R_Ctx", "nppiFilterLaplace/8s16s/m5", 1, 9, 9,
                         3, __LINE__, NPP_MASK_SIZE_5_X_5, nppiFilterLaplace_8s16s_C1R_Ctx);
  // extra shapes
  mask_ok<Npp8u, Npp8u>("nppiFilterLaplace_8u_C1R_Ctx", "nppiFilterLaplace/8u/C1/11x7/m3", 1, 11, 7,
                        2, __LINE__, NPP_MASK_SIZE_3_X_3, nppiFilterLaplace_8u_C1R_Ctx);
  mask_ok<Npp8u, Npp8u>("nppiFilterLaplace_8u_C1R_Ctx", "nppiFilterLaplace/8u/C1/1x1/m5", 1, 1, 1,
                        3, __LINE__, NPP_MASK_SIZE_5_X_5, nppiFilterLaplace_8u_C1R_Ctx);
  mask_ok<Npp16s, Npp16s>("nppiFilterLaplace_16s_C1R_Ctx", "nppiFilterLaplace/16s/C1/1x1/m3", 1, 1,
                          1, 2, __LINE__, NPP_MASK_SIZE_3_X_3, nppiFilterLaplace_16s_C1R_Ctx);
  mask_ok<Npp32f, Npp32f>("nppiFilterLaplace_32f_C3R_Ctx", "nppiFilterLaplace/32f/C3/11x7/m5", 3,
                          11, 7, 3, __LINE__, NPP_MASK_SIZE_5_X_5, nppiFilterLaplace_32f_C3R_Ctx);
  mask_ok<Npp8u, Npp8u>("nppiFilterLaplace_8u_C4R_Ctx", "nppiFilterLaplace/8u/C4/11x7/m5", 4, 11,
                        7, 3, __LINE__, NPP_MASK_SIZE_5_X_5, nppiFilterLaplace_8u_C4R_Ctx);
  // error matrices (no 16u variant in this family)
  mask_errors("nppiFilterLaplace", "_8u_C1R_Ctx", 1, arena<Npp8u>(64).get(), arena<Npp8u>(64).get(),
              nppiFilterLaplace_8u_C1R_Ctx);
  mask_errors("nppiFilterLaplace", "_16s_C1R_Ctx", 1, arena<Npp16s>(64).get(),
              arena<Npp16s>(64).get(), nppiFilterLaplace_16s_C1R_Ctx);
  mask_errors("nppiFilterLaplace", "_32f_C1R_Ctx", 1, arena<Npp32f>(64).get(),
              arena<Npp32f>(64).get(), nppiFilterLaplace_32f_C1R_Ctx);
  mask_errors("nppiFilterLaplace", "_8u_AC4R_Ctx", 4, arena<Npp8u>(64).get(),
              arena<Npp8u>(64).get(), nppiFilterLaplace_8u_AC4R_Ctx);
  mask_errors("nppiFilterLaplace", "_8u16s_C1R_Ctx", 1, arena<Npp8u>(64).get(),
              arena<Npp16s>(64).get(), nppiFilterLaplace_8u16s_C1R_Ctx);
}

// ============ LaplaceBorder (masked border, 8u/16s/32f + conversions) =====
TEST_F(NppiFilterFixedDiff, LaplaceBorder) {
  FXMB(nppiFilterLaplaceBorder, Npp8u, 8u, "m3", NPP_MASK_SIZE_3_X_3);
  FXMB(nppiFilterLaplaceBorder, Npp16s, 16s, "m3", NPP_MASK_SIZE_3_X_3);
  FXMB(nppiFilterLaplaceBorder, Npp32f, 32f, "m3", NPP_MASK_SIZE_3_X_3);
  FXMB(nppiFilterLaplaceBorder, Npp8u, 8u, "m5", NPP_MASK_SIZE_5_X_5);
  FXMB(nppiFilterLaplaceBorder, Npp16s, 16s, "m5", NPP_MASK_SIZE_5_X_5);
  FXMB(nppiFilterLaplaceBorder, Npp32f, 32f, "m5", NPP_MASK_SIZE_5_X_5);
  mask_border_ok<Npp8u, Npp16s>("nppiFilterLaplaceBorder_8u16s_C1R_Ctx",
                                "nppiFilterLaplaceBorder/8u16s/m3", 1, 13, 13, 2, 3, 9, 9,
                                __LINE__, NPP_MASK_SIZE_3_X_3,
                                nppiFilterLaplaceBorder_8u16s_C1R_Ctx);
  mask_border_ok<Npp8u, Npp16s>("nppiFilterLaplaceBorder_8u16s_C1R_Ctx",
                                "nppiFilterLaplaceBorder/8u16s/m5", 1, 13, 13, 2, 3, 9, 9,
                                __LINE__, NPP_MASK_SIZE_5_X_5,
                                nppiFilterLaplaceBorder_8u16s_C1R_Ctx);
  mask_border_ok<Npp8s, Npp16s>("nppiFilterLaplaceBorder_8s16s_C1R_Ctx",
                                "nppiFilterLaplaceBorder/8s16s/m3", 1, 13, 13, 2, 3, 9, 9,
                                __LINE__, NPP_MASK_SIZE_3_X_3,
                                nppiFilterLaplaceBorder_8s16s_C1R_Ctx);
  mask_border_ok<Npp8s, Npp16s>("nppiFilterLaplaceBorder_8s16s_C1R_Ctx",
                                "nppiFilterLaplaceBorder/8s16s/m5", 1, 13, 13, 2, 3, 9, 9,
                                __LINE__, NPP_MASK_SIZE_5_X_5,
                                nppiFilterLaplaceBorder_8s16s_C1R_Ctx);
  mask_border_ok<Npp8u, Npp8u>("nppiFilterLaplaceBorder_8u_C1R_Ctx",
                               "nppiFilterLaplaceBorder/8u/C1/corner/m3", 1, 13, 13, 0, 0, 9, 9,
                               __LINE__, NPP_MASK_SIZE_3_X_3, nppiFilterLaplaceBorder_8u_C1R_Ctx);
  mask_border_ok<Npp8u, Npp8u>("nppiFilterLaplaceBorder_8u_C1R_Ctx",
                               "nppiFilterLaplaceBorder/8u/C1/overhang/m5", 1, 13, 13, 10, 10, 5,
                               5, __LINE__, NPP_MASK_SIZE_5_X_5, nppiFilterLaplaceBorder_8u_C1R_Ctx);
  mask_border_ok<Npp16s, Npp16s>("nppiFilterLaplaceBorder_16s_C3R_Ctx",
                                 "nppiFilterLaplaceBorder/16s/C3/11x7/m3", 3, 11, 7, 1, 2, 7, 5,
                                 __LINE__, NPP_MASK_SIZE_3_X_3,
                                 nppiFilterLaplaceBorder_16s_C3R_Ctx);
  mask_border_ok<Npp32f, Npp32f>("nppiFilterLaplaceBorder_32f_C1R_Ctx",
                                 "nppiFilterLaplaceBorder/32f/C1/tiny/m5", 1, 3, 3, 1, 1, 5, 5,
                                 __LINE__, NPP_MASK_SIZE_5_X_5,
                                 nppiFilterLaplaceBorder_32f_C1R_Ctx);
  mask_border_errors("nppiFilterLaplaceBorder", "_8u_C1R_Ctx", 1, arena<Npp8u>(64).get(),
                     arena<Npp8u>(64).get(), nppiFilterLaplaceBorder_8u_C1R_Ctx);
  mask_border_errors("nppiFilterLaplaceBorder", "_16s_C1R_Ctx", 1, arena<Npp16s>(64).get(),
                     arena<Npp16s>(64).get(), nppiFilterLaplaceBorder_16s_C1R_Ctx);
  mask_border_errors("nppiFilterLaplaceBorder", "_32f_C1R_Ctx", 1, arena<Npp32f>(64).get(),
                     arena<Npp32f>(64).get(), nppiFilterLaplaceBorder_32f_C1R_Ctx);
  mask_border_errors("nppiFilterLaplaceBorder", "_8u_AC4R_Ctx", 4, arena<Npp8u>(64).get(),
                     arena<Npp8u>(64).get(), nppiFilterLaplaceBorder_8u_AC4R_Ctx);
  mask_border_errors("nppiFilterLaplaceBorder", "_8u16s_C1R_Ctx", 1, arena<Npp8u>(64).get(),
                     arena<Npp16s>(64).get(), nppiFilterLaplaceBorder_8u16s_C1R_Ctx);
}
