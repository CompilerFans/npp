// Differential tests for package f4: derivative filters
// (nppiFilterPrewitt / Roberts / Sobel families, 22 families / 348 functions).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiFilterDerivDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiFilterDerivDiff.*'
//
// Measured semantics (5090, NPP 13.4):
//  - plain variants read a window around each ROI pixel; the source buffer
//    must provide a margin, the ROI itself is written exactly
//  - border variants accept only NPP_BORDER_REPLICATE; reads clamp to the
//    image bounds expressed relative to pSrc ([-offset, srcSize-1-offset]);
//    offset+ROI may overhang srcSize (clamped)
//  - AC4 computes channels 0..2 only, dst alpha is never written
//  - validation order: null(-8) -> ROI neg(-6) -> ROI zero(0) ->
//    srcSize/offset(-11) -> border type(-14) -> mask size(-33) ->
//    dst step too small(-16) / not aligned(-108); the source step is never
//    validated
//  - 8u saturates to [0,255], 16s to [-32768,32767], 32f passthrough

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
template <> ValueKind kind_of<Npp16s>() { return ValueKind::I16; }
template <> ValueKind kind_of<Npp32f>() { return ValueKind::F32; }

// ---------------------------------------------------------------- helpers

// Plain derivative: source buffer is (W+2m)x(H+2m), pSrc inset by m pixels.
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
  Case c("nppiFilterDeriv/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.status(st);
  c.device_output<D>(d.get(), de, kind_of<D>(), 1e-3, 1e-3);
  c.finish();
}

// Masked plain derivative (adds eMaskSize).
template <typename S, typename D, typename F>
void mask_ok(const std::string &fname, const std::string &key, int channels, int W, int H,
             uint64_t seed, NppiMaskSize ms, F f) {
  plain_ok<S, D>(fname, key, channels, W, H, 2, 3, seed,
                 [&](const S *p, int ss, D *q, int ds, NppiSize roi, NppStreamContext ctx) {
                   return f(p, ss, q, ds, roi, ms, ctx);
                 });
}

// Border derivative: image srcW x srcH, pSrc = image + offset (pixels).
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
  Case c("nppiFilterDeriv/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.status(st);
  c.device_output<D>(d.get(), de, kind_of<D>(), 1e-3, 1e-3);
  c.finish();
}

// Masked border derivative (adds eMaskSize).
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
  Case c("nppiFilterDeriv/" + key, fname);
  c.status(st);
  c.finish();
}

// Error matrix for plain variants (W = H = 4). `suffix` distinguishes the
// variant inside the family. dst pixel requirement: req = 4*channels*sizeof(D).
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
  only_status(fn, fam + "/err/negroi" + suffix,
              [&] { return f(s, 16, d, 16, NppiSize{4, -1}, ctx); });
  // zero ROI returns 0 before the step check
  only_status(fn, fam + "/err/zeroroi" + suffix,
              [&] { return f(s, 16, d, 3, NppiSize{0, 0}, ctx); });
  only_status(fn, fam + "/err/stepsmall" + suffix,
              [&] { return f(s, 16, d, 4 * dpx - 1, NppiSize{4, 4}, ctx); });
  if (dAlign > 1)
    only_status(fn, fam + "/err/stepodd" + suffix,
                [&] { return f(s, 16, d, 4 * dpx + 1, NppiSize{4, 4}, ctx); });
}

// Error matrix for border variants.
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
  only_status(fn, fam + "/err/nullsrc" + suffix,
              [&] { return f(nullptr, 16, sz, off, d, 16, roi, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, fam + "/err/nulldst" + suffix,
              [&] { return f(s, 16, sz, off, nullptr, 16, roi, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, fam + "/err/negroi" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, NppiSize{4, -1}, NPP_BORDER_REPLICATE, ctx); });
  // zero ROI wins over srcSize / border type / step
  only_status(fn, fam + "/err/zeroroi" + suffix, [&] {
    return f(s, 16, NppiSize{0, 5}, off, d, 3, NppiSize{0, 0}, bad, ctx);
  });
  only_status(fn, fam + "/err/srcsize" + suffix,
              [&] { return f(s, 16, NppiSize{0, 5}, off, d, 16, roi, bad, ctx); });
  only_status(fn, fam + "/err/offneg" + suffix,
              [&] { return f(s, 16, sz, NppiPoint{-1, 0}, d, 16, roi, bad, ctx); });
  only_status(fn, fam + "/err/badborder" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, roi, bad, ctx); });
  only_status(fn, fam + "/err/stepsmall" + suffix,
              [&] { return f(s, 16, sz, off, d, 4 * dpx - 1, roi, NPP_BORDER_REPLICATE, ctx); });
  if (dAlign > 1)
    only_status(fn, fam + "/err/stepodd" + suffix,
                [&] { return f(s, 16, sz, off, d, 4 * dpx + 1, roi, NPP_BORDER_REPLICATE, ctx); });
  // Precedence pins: negative ROI wins over the border type; zero-size ROI is
  // honored before the step check but (on the legacy Prewitt/Roberts Border
  // families) after the offset and border-type checks.
  only_status(fn, fam + "/err/negroi_badborder" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, NppiSize{4, -1}, bad, ctx); });
  only_status(fn, fam + "/err/zroi_offneg" + suffix,
              [&] { return f(s, 16, sz, NppiPoint{0, -1}, d, 16, NppiSize{0, 0}, NPP_BORDER_REPLICATE, ctx); });
  only_status(fn, fam + "/err/zroi_badborder" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, NppiSize{0, 0}, bad, ctx); });
  only_status(fn, fam + "/err/zroi_badstep" + suffix,
              [&] { return f(s, 16, sz, off, d, 3, NppiSize{0, 0}, NPP_BORDER_REPLICATE, ctx); });
}

// Error matrix for masked plain variants.
template <typename S, typename D, typename F>
void mask_errors(const std::string &fam, const std::string &suffix, int channels, S *s, D *d, F f) {
  NppStreamContext ctx = current_ctx();
  const std::string fn = fam + suffix;
  const NppiSize roi{4, 4};
  const int dpx = channels * (int)sizeof(D);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(D);
  const NppiMaskSize bad = (NppiMaskSize)0;
  only_status(fn, fam + "/err/nullsrc" + suffix,
              [&] { return f(nullptr, 16, d, 16, roi, bad, ctx); });
  only_status(fn, fam + "/err/nulldst" + suffix,
              [&] { return f(s, 16, nullptr, 16, roi, NPP_MASK_SIZE_3_X_3, ctx); });
  only_status(fn, fam + "/err/negroi" + suffix,
              [&] { return f(s, 16, d, 16, NppiSize{4, -1}, NPP_MASK_SIZE_3_X_3, ctx); });
  // zero ROI wins over mask / step
  only_status(fn, fam + "/err/zeroroi" + suffix,
              [&] { return f(s, 16, d, 3, NppiSize{0, 0}, bad, ctx); });
  // bad mask wins over step
  only_status(fn, fam + "/err/badmask" + suffix,
              [&] { return f(s, 16, d, 4 * dpx - 1, roi, bad, ctx); });
  only_status(fn, fam + "/err/stepsmall" + suffix,
              [&] { return f(s, 16, d, 4 * dpx - 1, roi, NPP_MASK_SIZE_3_X_3, ctx); });
  if (dAlign > 1)
    only_status(fn, fam + "/err/stepodd" + suffix,
                [&] { return f(s, 16, d, 4 * dpx + 1, roi, NPP_MASK_SIZE_3_X_3, ctx); });
}

// Error matrix for masked border variants.
template <typename S, typename D, typename F>
void mask_border_errors(const std::string &fam, const std::string &suffix, int channels, S *s, D *d,
                        F f) {
  NppStreamContext ctx = current_ctx();
  const std::string fn = fam + suffix;
  const NppiSize sz{7, 7};
  const NppiPoint off{2, 2};
  const NppiSize roi{4, 4};
  const int dpx = channels * (int)sizeof(D);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(D);
  const NppiBorderType badB = NPP_BORDER_CONSTANT;
  const NppiMaskSize badM = (NppiMaskSize)0;
  only_status(fn, fam + "/err/nullsrc" + suffix,
              [&] { return f(nullptr, 16, sz, off, d, 16, roi, NPP_MASK_SIZE_3_X_3, badB, ctx); });
  only_status(fn, fam + "/err/negroi" + suffix, [&] {
    return f(s, 16, sz, off, d, 16, NppiSize{4, -1}, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, ctx);
  });
  // zero ROI wins over srcSize / border / mask / step
  only_status(fn, fam + "/err/zeroroi" + suffix, [&] {
    return f(s, 16, NppiSize{0, 5}, off, d, 3, NppiSize{0, 0}, badM, badB, ctx);
  });
  only_status(fn, fam + "/err/srcsize" + suffix,
              [&] { return f(s, 16, NppiSize{0, 5}, off, d, 16, roi, badM, badB, ctx); });
  only_status(fn, fam + "/err/offneg" + suffix,
              [&] { return f(s, 16, sz, NppiPoint{-1, 0}, d, 16, roi, badM, badB, ctx); });
  // border type wins over mask / step
  only_status(fn, fam + "/err/badborder" + suffix,
              [&] { return f(s, 16, sz, off, d, 16, roi, badM, badB, ctx); });
  // bad mask wins over step
  only_status(fn, fam + "/err/badmask" + suffix, [&] {
    return f(s, 16, sz, off, d, 4 * dpx - 1, roi, badM, NPP_BORDER_REPLICATE, ctx);
  });
  only_status(fn, fam + "/err/stepsmall" + suffix, [&] {
    return f(s, 16, sz, off, d, 4 * dpx - 1, roi, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, ctx);
  });
  if (dAlign > 1)
    only_status(fn, fam + "/err/stepodd" + suffix, [&] {
      return f(s, 16, sz, off, d, 4 * dpx + 1, roi, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, ctx);
    });
  // Precedence pins: negative ROI and zero-size ROI win over the offset,
  // border type, mask size and step checks.
  only_status(fn, fam + "/err/negroi_badborder" + suffix, [&] {
    return f(s, 16, sz, off, d, 16, NppiSize{4, -1}, badM, badB, ctx);
  });
  only_status(fn, fam + "/err/zroi_offneg" + suffix, [&] {
    return f(s, 16, sz, NppiPoint{0, -1}, d, 16, NppiSize{0, 0}, NPP_MASK_SIZE_3_X_3, NPP_BORDER_REPLICATE, ctx);
  });
  only_status(fn, fam + "/err/zroi_badborder" + suffix, [&] {
    return f(s, 16, sz, off, d, 16, NppiSize{0, 0}, NPP_MASK_SIZE_3_X_3, badB, ctx);
  });
}

// ------------------------------------------------------------ call macros

#define DPL(FN, S, D, LAY, CH)                                                                     \
  plain_ok<S, D>(#FN "_Ctx", #FN "/" LAY, CH, 9, 9, 2, (CH) == 4 ? 4 : 3, __LINE__, FN##_Ctx)

#define DBR(FN, S, D, LAY, CH)                                                                     \
  border_ok<S, D>(#FN "_Ctx", #FN "/" LAY, CH, 13, 13, 2, 3, 9, 9, 3, __LINE__, FN##_Ctx)

#define DMC(FN, S, D, MSSTR, MS)                                                                   \
  mask_ok<S, D>(#FN "_Ctx", #FN "/" MSSTR, 1, 9, 9, __LINE__, MS, FN##_Ctx)

#define DMB(FN, S, D, MSSTR, MS)                                                                   \
  mask_border_ok<S, D>(#FN "_Ctx", #FN "/" MSSTR, 1, 13, 13, 2, 3, 9, 9, __LINE__, MS, FN##_Ctx)

// full layout matrix (3 dtypes x C1/C3/C4/AC4) for one plain family
#define PLAIN_MATRIX(FAM)                                                                          \
  do {                                                                                             \
    DPL(FAM##_8u_C1R, Npp8u, Npp8u, "C1", 1);                                                      \
    DPL(FAM##_8u_C3R, Npp8u, Npp8u, "C3", 3);                                                      \
    DPL(FAM##_8u_C4R, Npp8u, Npp8u, "C4", 4);                                                      \
    DPL(FAM##_8u_AC4R, Npp8u, Npp8u, "AC4", 4);                                                    \
    DPL(FAM##_16s_C1R, Npp16s, Npp16s, "C1", 1);                                                   \
    DPL(FAM##_16s_C3R, Npp16s, Npp16s, "C3", 3);                                                   \
    DPL(FAM##_16s_C4R, Npp16s, Npp16s, "C4", 4);                                                   \
    DPL(FAM##_16s_AC4R, Npp16s, Npp16s, "AC4", 4);                                                 \
    DPL(FAM##_32f_C1R, Npp32f, Npp32f, "C1", 1);                                                   \
    DPL(FAM##_32f_C3R, Npp32f, Npp32f, "C3", 3);                                                   \
    DPL(FAM##_32f_C4R, Npp32f, Npp32f, "C4", 4);                                                   \
    DPL(FAM##_32f_AC4R, Npp32f, Npp32f, "AC4", 4);                                                 \
    plain_ok<Npp8u, Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/11x7", 1, 11, 7, 2, 5, __LINE__,       \
                           FAM##_8u_C1R_Ctx);                                                      \
    plain_ok<Npp16s, Npp16s>(#FAM "_16s_C1R_Ctx", #FAM "/16s/C1/1x1", 1, 1, 1, 2, 3, __LINE__,     \
                             FAM##_16s_C1R_Ctx);                                                   \
    plain_errors(#FAM, "_8u_C1R_Ctx", 1, arena<Npp8u>(64).get(), arena<Npp8u>(64).get(),           \
                 FAM##_8u_C1R_Ctx);                                                                \
    plain_errors(#FAM, "_16s_C1R_Ctx", 1, arena<Npp16s>(64).get(), arena<Npp16s>(64).get(),        \
                 FAM##_16s_C1R_Ctx);                                                               \
    plain_errors(#FAM, "_8u_AC4R_Ctx", 4, arena<Npp8u>(64).get(), arena<Npp8u>(64).get(),          \
                 FAM##_8u_AC4R_Ctx);                                                               \
  } while (0)

#define BORDER_MATRIX(FAM)                                                                         \
  do {                                                                                             \
    DBR(FAM##_8u_C1R, Npp8u, Npp8u, "C1", 1);                                                      \
    DBR(FAM##_8u_C3R, Npp8u, Npp8u, "C3", 3);                                                      \
    DBR(FAM##_8u_C4R, Npp8u, Npp8u, "C4", 4);                                                      \
    DBR(FAM##_8u_AC4R, Npp8u, Npp8u, "AC4", 4);                                                    \
    DBR(FAM##_16s_C1R, Npp16s, Npp16s, "C1", 1);                                                   \
    DBR(FAM##_16s_C3R, Npp16s, Npp16s, "C3", 3);                                                   \
    DBR(FAM##_16s_C4R, Npp16s, Npp16s, "C4", 4);                                                   \
    DBR(FAM##_16s_AC4R, Npp16s, Npp16s, "AC4", 4);                                                 \
    DBR(FAM##_32f_C1R, Npp32f, Npp32f, "C1", 1);                                                   \
    DBR(FAM##_32f_C3R, Npp32f, Npp32f, "C3", 3);                                                   \
    DBR(FAM##_32f_C4R, Npp32f, Npp32f, "C4", 4);                                                   \
    DBR(FAM##_32f_AC4R, Npp32f, Npp32f, "AC4", 4);                                                 \
    border_ok<Npp8u, Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/corner", 1, 13, 13, 0, 0, 9, 9, 3,    \
                            __LINE__, FAM##_8u_C1R_Ctx);                                           \
    border_ok<Npp8u, Npp8u>(#FAM "_8u_C1R_Ctx", #FAM "/8u/C1/overhang", 1, 13, 13, 10, 10, 5, 5,   \
                            3, __LINE__, FAM##_8u_C1R_Ctx);                                        \
    border_ok<Npp16s, Npp16s>(#FAM "_16s_C3R_Ctx", #FAM "/16s/C3/11x7", 3, 11, 7, 1, 2, 7, 5, 3,   \
                              __LINE__, FAM##_16s_C3R_Ctx);                                        \
    border_errors(#FAM, "_8u_C1R_Ctx", 1, arena<Npp8u>(64).get(), arena<Npp8u>(64).get(),          \
                  FAM##_8u_C1R_Ctx);                                                               \
    border_errors(#FAM, "_16s_C1R_Ctx", 1, arena<Npp16s>(64).get(), arena<Npp16s>(64).get(),       \
                  FAM##_16s_C1R_Ctx);                                                              \
    border_errors(#FAM, "_8u_AC4R_Ctx", 4, arena<Npp8u>(64).get(), arena<Npp8u>(64).get(),         \
                  FAM##_8u_AC4R_Ctx);                                                              \
  } while (0)

}  // namespace

class NppiFilterDerivDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ==================== Prewitt ====================
TEST_F(NppiFilterDerivDiff, PrewittHoriz) { PLAIN_MATRIX(nppiFilterPrewittHoriz); }
TEST_F(NppiFilterDerivDiff, PrewittHorizBorder) { BORDER_MATRIX(nppiFilterPrewittHorizBorder); }
TEST_F(NppiFilterDerivDiff, PrewittVert) { PLAIN_MATRIX(nppiFilterPrewittVert); }
TEST_F(NppiFilterDerivDiff, PrewittVertBorder) { BORDER_MATRIX(nppiFilterPrewittVertBorder); }

// ==================== Roberts ====================
TEST_F(NppiFilterDerivDiff, RobertsDown) { PLAIN_MATRIX(nppiFilterRobertsDown); }
TEST_F(NppiFilterDerivDiff, RobertsDownBorder) { BORDER_MATRIX(nppiFilterRobertsDownBorder); }
TEST_F(NppiFilterDerivDiff, RobertsUp) { PLAIN_MATRIX(nppiFilterRobertsUp); }
TEST_F(NppiFilterDerivDiff, RobertsUpBorder) { BORDER_MATRIX(nppiFilterRobertsUpBorder); }

// ==================== SobelHoriz / SobelVert (base + 8u16s / 8s16s) ============
TEST_F(NppiFilterDerivDiff, SobelHoriz) {
  PLAIN_MATRIX(nppiFilterSobelHoriz);
  DMC(nppiFilterSobelHoriz_8u16s_C1R, Npp8u, Npp16s, "8u16s/3x3", NPP_MASK_SIZE_3_X_3);
  DMC(nppiFilterSobelHoriz_8u16s_C1R, Npp8u, Npp16s, "8u16s/5x5", NPP_MASK_SIZE_5_X_5);
  DMC(nppiFilterSobelHoriz_8s16s_C1R, Npp8s, Npp16s, "8s16s/3x3", NPP_MASK_SIZE_3_X_3);
  DMC(nppiFilterSobelHoriz_8s16s_C1R, Npp8s, Npp16s, "8s16s/5x5", NPP_MASK_SIZE_5_X_5);
  mask_errors("nppiFilterSobelHoriz", "_8u16s_C1R_Ctx", 1, arena<Npp8u>(64).get(),
              arena<Npp16s>(64).get(), nppiFilterSobelHoriz_8u16s_C1R_Ctx);
}

TEST_F(NppiFilterDerivDiff, SobelHorizBorder) {
  BORDER_MATRIX(nppiFilterSobelHorizBorder);
  DMB(nppiFilterSobelHorizBorder_8u16s_C1R, Npp8u, Npp16s, "8u16s/3x3", NPP_MASK_SIZE_3_X_3);
  DMB(nppiFilterSobelHorizBorder_8u16s_C1R, Npp8u, Npp16s, "8u16s/5x5", NPP_MASK_SIZE_5_X_5);
  DMB(nppiFilterSobelHorizBorder_8s16s_C1R, Npp8s, Npp16s, "8s16s/3x3", NPP_MASK_SIZE_3_X_3);
  DMB(nppiFilterSobelHorizBorder_8s16s_C1R, Npp8s, Npp16s, "8s16s/5x5", NPP_MASK_SIZE_5_X_5);
  mask_border_errors("nppiFilterSobelHorizBorder", "_8u16s_C1R_Ctx", 1, arena<Npp8u>(64).get(),
                     arena<Npp16s>(64).get(), nppiFilterSobelHorizBorder_8u16s_C1R_Ctx);
}

TEST_F(NppiFilterDerivDiff, SobelVert) {
  PLAIN_MATRIX(nppiFilterSobelVert);
  DMC(nppiFilterSobelVert_8u16s_C1R, Npp8u, Npp16s, "8u16s/3x3", NPP_MASK_SIZE_3_X_3);
  DMC(nppiFilterSobelVert_8u16s_C1R, Npp8u, Npp16s, "8u16s/5x5", NPP_MASK_SIZE_5_X_5);
  DMC(nppiFilterSobelVert_8s16s_C1R, Npp8s, Npp16s, "8s16s/3x3", NPP_MASK_SIZE_3_X_3);
  DMC(nppiFilterSobelVert_8s16s_C1R, Npp8s, Npp16s, "8s16s/5x5", NPP_MASK_SIZE_5_X_5);
  mask_errors("nppiFilterSobelVert", "_8u16s_C1R_Ctx", 1, arena<Npp8u>(64).get(),
              arena<Npp16s>(64).get(), nppiFilterSobelVert_8u16s_C1R_Ctx);
}

TEST_F(NppiFilterDerivDiff, SobelVertBorder) {
  BORDER_MATRIX(nppiFilterSobelVertBorder);
  DMB(nppiFilterSobelVertBorder_8u16s_C1R, Npp8u, Npp16s, "8u16s/3x3", NPP_MASK_SIZE_3_X_3);
  DMB(nppiFilterSobelVertBorder_8u16s_C1R, Npp8u, Npp16s, "8u16s/5x5", NPP_MASK_SIZE_5_X_5);
  DMB(nppiFilterSobelVertBorder_8s16s_C1R, Npp8s, Npp16s, "8s16s/3x3", NPP_MASK_SIZE_3_X_3);
  DMB(nppiFilterSobelVertBorder_8s16s_C1R, Npp8s, Npp16s, "8s16s/5x5", NPP_MASK_SIZE_5_X_5);
  mask_border_errors("nppiFilterSobelVertBorder", "_8u16s_C1R_Ctx", 1, arena<Npp8u>(64).get(),
                     arena<Npp16s>(64).get(), nppiFilterSobelVertBorder_8u16s_C1R_Ctx);
}

// ==================== SobelCross / Sobel*Second (8u16s / 8s16s / 32f) ==========
TEST_F(NppiFilterDerivDiff, SobelCross) {
  DMC(nppiFilterSobelCross_8u16s_C1R, Npp8u, Npp16s, "8u16s/3x3", NPP_MASK_SIZE_3_X_3);
  DMC(nppiFilterSobelCross_8u16s_C1R, Npp8u, Npp16s, "8u16s/5x5", NPP_MASK_SIZE_5_X_5);
  DMC(nppiFilterSobelCross_8s16s_C1R, Npp8s, Npp16s, "8s16s/3x3", NPP_MASK_SIZE_3_X_3);
  DMC(nppiFilterSobelCross_8s16s_C1R, Npp8s, Npp16s, "8s16s/5x5", NPP_MASK_SIZE_5_X_5);
  DMC(nppiFilterSobelCross_32f_C1R, Npp32f, Npp32f, "32f/3x3", NPP_MASK_SIZE_3_X_3);
  DMC(nppiFilterSobelCross_32f_C1R, Npp32f, Npp32f, "32f/5x5", NPP_MASK_SIZE_5_X_5);
  mask_ok<Npp32f, Npp32f>("nppiFilterSobelCross_32f_C1R_Ctx", "nppiFilterSobelCross_32f_C1R/32f/3x3/1x1",
                          1, 1, 1, __LINE__, NPP_MASK_SIZE_3_X_3, nppiFilterSobelCross_32f_C1R_Ctx);
  mask_errors("nppiFilterSobelCross", "_8u16s_C1R_Ctx", 1, arena<Npp8u>(64).get(),
              arena<Npp16s>(64).get(), nppiFilterSobelCross_8u16s_C1R_Ctx);
  mask_errors("nppiFilterSobelCross", "_32f_C1R_Ctx", 1, arena<Npp32f>(64).get(),
              arena<Npp32f>(64).get(), nppiFilterSobelCross_32f_C1R_Ctx);
}

TEST_F(NppiFilterDerivDiff, SobelCrossBorder) {
  DMB(nppiFilterSobelCrossBorder_8u16s_C1R, Npp8u, Npp16s, "8u16s/3x3", NPP_MASK_SIZE_3_X_3);
  DMB(nppiFilterSobelCrossBorder_8u16s_C1R, Npp8u, Npp16s, "8u16s/5x5", NPP_MASK_SIZE_5_X_5);
  DMB(nppiFilterSobelCrossBorder_8s16s_C1R, Npp8s, Npp16s, "8s16s/3x3", NPP_MASK_SIZE_3_X_3);
  DMB(nppiFilterSobelCrossBorder_8s16s_C1R, Npp8s, Npp16s, "8s16s/5x5", NPP_MASK_SIZE_5_X_5);
  DMB(nppiFilterSobelCrossBorder_32f_C1R, Npp32f, Npp32f, "32f/3x3", NPP_MASK_SIZE_3_X_3);
  DMB(nppiFilterSobelCrossBorder_32f_C1R, Npp32f, Npp32f, "32f/5x5", NPP_MASK_SIZE_5_X_5);
  mask_border_errors("nppiFilterSobelCrossBorder", "_8u16s_C1R_Ctx", 1, arena<Npp8u>(64).get(),
                     arena<Npp16s>(64).get(), nppiFilterSobelCrossBorder_8u16s_C1R_Ctx);
  mask_border_errors("nppiFilterSobelCrossBorder", "_32f_C1R_Ctx", 1, arena<Npp32f>(64).get(),
                     arena<Npp32f>(64).get(), nppiFilterSobelCrossBorder_32f_C1R_Ctx);
}

TEST_F(NppiFilterDerivDiff, SobelHorizSecond) {
  DMC(nppiFilterSobelHorizSecond_8u16s_C1R, Npp8u, Npp16s, "8u16s/3x3", NPP_MASK_SIZE_3_X_3);
  DMC(nppiFilterSobelHorizSecond_8u16s_C1R, Npp8u, Npp16s, "8u16s/5x5", NPP_MASK_SIZE_5_X_5);
  DMC(nppiFilterSobelHorizSecond_8s16s_C1R, Npp8s, Npp16s, "8s16s/3x3", NPP_MASK_SIZE_3_X_3);
  DMC(nppiFilterSobelHorizSecond_8s16s_C1R, Npp8s, Npp16s, "8s16s/5x5", NPP_MASK_SIZE_5_X_5);
  DMC(nppiFilterSobelHorizSecond_32f_C1R, Npp32f, Npp32f, "32f/3x3", NPP_MASK_SIZE_3_X_3);
  DMC(nppiFilterSobelHorizSecond_32f_C1R, Npp32f, Npp32f, "32f/5x5", NPP_MASK_SIZE_5_X_5);
  mask_errors("nppiFilterSobelHorizSecond", "_8u16s_C1R_Ctx", 1, arena<Npp8u>(64).get(),
              arena<Npp16s>(64).get(), nppiFilterSobelHorizSecond_8u16s_C1R_Ctx);
  mask_errors("nppiFilterSobelHorizSecond", "_32f_C1R_Ctx", 1, arena<Npp32f>(64).get(),
              arena<Npp32f>(64).get(), nppiFilterSobelHorizSecond_32f_C1R_Ctx);
}

TEST_F(NppiFilterDerivDiff, SobelHorizSecondBorder) {
  DMB(nppiFilterSobelHorizSecondBorder_8u16s_C1R, Npp8u, Npp16s, "8u16s/3x3", NPP_MASK_SIZE_3_X_3);
  DMB(nppiFilterSobelHorizSecondBorder_8u16s_C1R, Npp8u, Npp16s, "8u16s/5x5", NPP_MASK_SIZE_5_X_5);
  DMB(nppiFilterSobelHorizSecondBorder_8s16s_C1R, Npp8s, Npp16s, "8s16s/3x3", NPP_MASK_SIZE_3_X_3);
  DMB(nppiFilterSobelHorizSecondBorder_8s16s_C1R, Npp8s, Npp16s, "8s16s/5x5", NPP_MASK_SIZE_5_X_5);
  DMB(nppiFilterSobelHorizSecondBorder_32f_C1R, Npp32f, Npp32f, "32f/3x3", NPP_MASK_SIZE_3_X_3);
  DMB(nppiFilterSobelHorizSecondBorder_32f_C1R, Npp32f, Npp32f, "32f/5x5", NPP_MASK_SIZE_5_X_5);
  mask_border_errors("nppiFilterSobelHorizSecondBorder", "_8u16s_C1R_Ctx", 1, arena<Npp8u>(64).get(),
                     arena<Npp16s>(64).get(), nppiFilterSobelHorizSecondBorder_8u16s_C1R_Ctx);
  mask_border_errors("nppiFilterSobelHorizSecondBorder", "_32f_C1R_Ctx", 1, arena<Npp32f>(64).get(),
                     arena<Npp32f>(64).get(), nppiFilterSobelHorizSecondBorder_32f_C1R_Ctx);
}

TEST_F(NppiFilterDerivDiff, SobelVertSecond) {
  DMC(nppiFilterSobelVertSecond_8u16s_C1R, Npp8u, Npp16s, "8u16s/3x3", NPP_MASK_SIZE_3_X_3);
  DMC(nppiFilterSobelVertSecond_8u16s_C1R, Npp8u, Npp16s, "8u16s/5x5", NPP_MASK_SIZE_5_X_5);
  DMC(nppiFilterSobelVertSecond_8s16s_C1R, Npp8s, Npp16s, "8s16s/3x3", NPP_MASK_SIZE_3_X_3);
  DMC(nppiFilterSobelVertSecond_8s16s_C1R, Npp8s, Npp16s, "8s16s/5x5", NPP_MASK_SIZE_5_X_5);
  DMC(nppiFilterSobelVertSecond_32f_C1R, Npp32f, Npp32f, "32f/3x3", NPP_MASK_SIZE_3_X_3);
  DMC(nppiFilterSobelVertSecond_32f_C1R, Npp32f, Npp32f, "32f/5x5", NPP_MASK_SIZE_5_X_5);
  mask_errors("nppiFilterSobelVertSecond", "_8u16s_C1R_Ctx", 1, arena<Npp8u>(64).get(),
              arena<Npp16s>(64).get(), nppiFilterSobelVertSecond_8u16s_C1R_Ctx);
  mask_errors("nppiFilterSobelVertSecond", "_32f_C1R_Ctx", 1, arena<Npp32f>(64).get(),
              arena<Npp32f>(64).get(), nppiFilterSobelVertSecond_32f_C1R_Ctx);
}

TEST_F(NppiFilterDerivDiff, SobelVertSecondBorder) {
  DMB(nppiFilterSobelVertSecondBorder_8u16s_C1R, Npp8u, Npp16s, "8u16s/3x3", NPP_MASK_SIZE_3_X_3);
  DMB(nppiFilterSobelVertSecondBorder_8u16s_C1R, Npp8u, Npp16s, "8u16s/5x5", NPP_MASK_SIZE_5_X_5);
  DMB(nppiFilterSobelVertSecondBorder_8s16s_C1R, Npp8s, Npp16s, "8s16s/3x3", NPP_MASK_SIZE_3_X_3);
  DMB(nppiFilterSobelVertSecondBorder_8s16s_C1R, Npp8s, Npp16s, "8s16s/5x5", NPP_MASK_SIZE_5_X_5);
  DMB(nppiFilterSobelVertSecondBorder_32f_C1R, Npp32f, Npp32f, "32f/3x3", NPP_MASK_SIZE_3_X_3);
  DMB(nppiFilterSobelVertSecondBorder_32f_C1R, Npp32f, Npp32f, "32f/5x5", NPP_MASK_SIZE_5_X_5);
  mask_border_errors("nppiFilterSobelVertSecondBorder", "_8u16s_C1R_Ctx", 1, arena<Npp8u>(64).get(),
                     arena<Npp16s>(64).get(), nppiFilterSobelVertSecondBorder_8u16s_C1R_Ctx);
  mask_border_errors("nppiFilterSobelVertSecondBorder", "_32f_C1R_Ctx", 1, arena<Npp32f>(64).get(),
                     arena<Npp32f>(64).get(), nppiFilterSobelVertSecondBorder_32f_C1R_Ctx);
}

// ==================== Sobel*Mask (32f) ====================
TEST_F(NppiFilterDerivDiff, SobelHorizMask) {
  DMC(nppiFilterSobelHorizMask_32f_C1R, Npp32f, Npp32f, "32f/3x3", NPP_MASK_SIZE_3_X_3);
  DMC(nppiFilterSobelHorizMask_32f_C1R, Npp32f, Npp32f, "32f/5x5", NPP_MASK_SIZE_5_X_5);
  mask_ok<Npp32f, Npp32f>("nppiFilterSobelHorizMask_32f_C1R_Ctx",
                          "nppiFilterSobelHorizMask_32f_C1R/32f/3x3/1x1", 1, 1, 1, __LINE__,
                          NPP_MASK_SIZE_3_X_3, nppiFilterSobelHorizMask_32f_C1R_Ctx);
  mask_errors("nppiFilterSobelHorizMask", "_32f_C1R_Ctx", 1, arena<Npp32f>(64).get(),
              arena<Npp32f>(64).get(), nppiFilterSobelHorizMask_32f_C1R_Ctx);
}

TEST_F(NppiFilterDerivDiff, SobelHorizMaskBorder) {
  DMB(nppiFilterSobelHorizMaskBorder_32f_C1R, Npp32f, Npp32f, "32f/3x3", NPP_MASK_SIZE_3_X_3);
  DMB(nppiFilterSobelHorizMaskBorder_32f_C1R, Npp32f, Npp32f, "32f/5x5", NPP_MASK_SIZE_5_X_5);
  mask_border_ok<Npp32f, Npp32f>(
      "nppiFilterSobelHorizMaskBorder_32f_C1R_Ctx", "nppiFilterSobelHorizMaskBorder_32f_C1R/32f/5x5/overhang",
      1, 13, 13, 10, 10, 5, 5, __LINE__, NPP_MASK_SIZE_5_X_5,
      nppiFilterSobelHorizMaskBorder_32f_C1R_Ctx);
  mask_border_errors("nppiFilterSobelHorizMaskBorder", "_32f_C1R_Ctx", 1, arena<Npp32f>(64).get(),
                     arena<Npp32f>(64).get(), nppiFilterSobelHorizMaskBorder_32f_C1R_Ctx);
}

TEST_F(NppiFilterDerivDiff, SobelVertMask) {
  DMC(nppiFilterSobelVertMask_32f_C1R, Npp32f, Npp32f, "32f/3x3", NPP_MASK_SIZE_3_X_3);
  DMC(nppiFilterSobelVertMask_32f_C1R, Npp32f, Npp32f, "32f/5x5", NPP_MASK_SIZE_5_X_5);
  mask_errors("nppiFilterSobelVertMask", "_32f_C1R_Ctx", 1, arena<Npp32f>(64).get(),
              arena<Npp32f>(64).get(), nppiFilterSobelVertMask_32f_C1R_Ctx);
}

TEST_F(NppiFilterDerivDiff, SobelVertMaskBorder) {
  DMB(nppiFilterSobelVertMaskBorder_32f_C1R, Npp32f, Npp32f, "32f/3x3", NPP_MASK_SIZE_3_X_3);
  DMB(nppiFilterSobelVertMaskBorder_32f_C1R, Npp32f, Npp32f, "32f/5x5", NPP_MASK_SIZE_5_X_5);
  mask_border_ok<Npp32f, Npp32f>(
      "nppiFilterSobelVertMaskBorder_32f_C1R_Ctx", "nppiFilterSobelVertMaskBorder_32f_C1R/32f/5x5/overhang",
      1, 13, 13, 10, 10, 5, 5, __LINE__, NPP_MASK_SIZE_5_X_5,
      nppiFilterSobelVertMaskBorder_32f_C1R_Ctx);
  mask_border_errors("nppiFilterSobelVertMaskBorder", "_32f_C1R_Ctx", 1, arena<Npp32f>(64).get(),
                     arena<Npp32f>(64).get(), nppiFilterSobelVertMaskBorder_32f_C1R_Ctx);
}
