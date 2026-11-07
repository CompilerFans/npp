// Differential tests for package f11: the unsharp-mask filter families
// (nppiFilterUnsharpBorder and nppiFilterUnsharpGetBufferSize).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiFilterUnsharpDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests \
//     --gtest_filter='NppiFilterUnsharpDiff.*'
//
// Measured semantics (5090, NPP 13.4, probes in probe_f11/: probe_j/l/m/n/o/p):
//  - the kernel is a normalised 2-D Gaussian; k = 2*ceil(radius)+1 for an
//    explicit radius, and with radius == 0 the window is derived from sigma and
//    the destination dtype (sigma=1 -> k=7/9/9/15 for 8u/16u/16s/32f).
//  - dst = src + weight*(src - blur) in floating point; integer destinations
//    saturate after C truncation toward zero, 32f stores the float result.
//  - nThreshold > 0 copies the source when |src - blur| <= threshold*scale
//    (scale 128 / 32768 / 16384 / 2^32 for 8u / 16u / 16s / 32f), so a smooth
//    ramp with threshold 2 is reproduced exactly.
//  - only NPP_BORDER_REPLICATE is supported; reads (and the centre sample) are
//    clamped to [pSrc - oSrcOffset, pSrc - oSrcOffset + ROI - 1], so the test
//    source buffer is sized ROI+offset and pSrc sits oSrcOffset inside it.
//  - AC4 computes channels 0..2 only; the destination alpha is untouched (the
//    random destination fill is part of the golden).
//  - validation order: null src/dst/buffer -> negative ROI -> border type ->
//    radius < 0 / sigma <= 0 -> kernel larger than the ROI (a zero ROI is
//    caught here) -> dst step.  oSrcOffset and the source step are never
//    validated.
//  - GetBufferSize returns k*k*sizeof(Npp32f) for every dtype/layout and
//    validates only radius/sigma.
//  - the taps are copied into pDeviceBuffer; a buffer of the wrong size makes
//    the call fail with NPP_CUDA_KERNEL_EXECUTION_ERROR (such cases are UB and
//    are not part of this suite).

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <memory>
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

// The reference is evaluated in float32 inside NVIDIA NPP while the executor
// accumulates in double, so integer destinations may differ by one LSB when the
// un-truncated value sits within a float rounding step of an integer (measured:
// <= 1 LSB on every recorded case, most pixels bit-exact).
template <typename T> ValueKind kind_of();
template <> ValueKind kind_of<Npp8u>() { return ValueKind::U8Tol; }
template <> ValueKind kind_of<Npp16u>() { return ValueKind::U16Tol; }
template <> ValueKind kind_of<Npp16s>() { return ValueKind::I16Tol; }
template <> ValueKind kind_of<Npp32f>() { return ValueKind::F32; }

// Integer outputs: exact or +-1; float outputs: the family default tolerance.
template <typename D> double out_rtol() { return std::is_floating_point_v<D> ? 1e-3 : 0.0; }
template <typename D> double out_atol() { return std::is_floating_point_v<D> ? 1e-3 : 1.0; }

// ---------------------------------------------------------------- helpers

// Border unsharp filter case.  The source buffer holds an image of
// (W+offX) x (H+offY) pixels and pSrc is placed oSrcOffset pixels inside it, so
// every read the executor clamps into [pSrc-off, pSrc-off+ROI-1] is valid.
template <typename S, typename D, typename F>
void border_ok(const std::string &fname, const std::string &key, int channels, int offX, int offY,
               int W, int H, int pad, uint64_t seed, float radius, float sigma, float weight,
               float threshold, F f) {
  const int spx = channels * (int)sizeof(S);
  const int srcW = W + offX, srcH = H + offY;
  const int sStep = align_step(srcW * spx, spx);
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
  // taps buffer: the exact size the family reports
  int need = 0;
  NppStatus gbs = nppiFilterUnsharpGetBufferSize_32f_C1R(radius, sigma, &need);
  EXPECT_EQ(gbs, NPP_NO_ERROR);
  DeviceMemory<Npp8u> &buf = arena<Npp8u>((size_t)(need > 0 ? need : 4));
  clear_cuda_error();
  NppStatus st = f(pSrc, sStep, NppiPoint{offX, offY}, d.get(), dStep, NppiSize{W, H}, radius, sigma,
                   weight, threshold, NPP_BORDER_REPLICATE, buf.get(), current_ctx());
  Case c("nppiFilterUnsharp/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.status(st);
  c.device_output<D>(d.get(), de, kind_of<D>(), out_rtol<D>(), out_atol<D>());
  c.finish();
}

// Smooth ramp source (small |src - blur|): exercises the threshold path where
// every pixel must be copied unchanged.
template <typename S, typename D, typename F>
void ramp_ok(const std::string &fname, const std::string &key, int channels, int W, int H,
             float radius, float sigma, float weight, float threshold, F f) {
  const int spx = channels * (int)sizeof(S);
  const int sStep = align_step(W * spx, spx);
  const int dpx = channels * (int)sizeof(D);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(D);
  const int dStep = align_step(W * dpx, dAlign);
  const size_t se = (size_t)H * sStep / sizeof(S);
  const size_t de = (size_t)H * dStep / sizeof(D);
  std::vector<S> hs(se);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      for (int ch = 0; ch < channels; ++ch) {
        const size_t idx = (size_t)y * (sStep / sizeof(S)) + (size_t)x * channels + ch;
        hs[idx] = (S)((x * 7 + y * 3 + ch * 5) % 97);
      }
  std::vector<D> hd = make_uniform<D>(de, 0x5eed + (uint64_t)W);
  DeviceMemory<S> &s = arena<S>(se);
  DeviceMemory<D> &d = arena<D>(de);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  int need = 0;
  EXPECT_EQ(nppiFilterUnsharpGetBufferSize_32f_C1R(radius, sigma, &need), NPP_NO_ERROR);
  DeviceMemory<Npp8u> &buf = arena<Npp8u>((size_t)(need > 0 ? need : 4));
  clear_cuda_error();
  NppStatus st = f(s.get(), sStep, NppiPoint{0, 0}, d.get(), dStep, NppiSize{W, H}, radius, sigma,
                   weight, threshold, NPP_BORDER_REPLICATE, buf.get(), current_ctx());
  Case c("nppiFilterUnsharp/" + key, fname);
  c.input(hs.data(), hs.size() * sizeof(S));
  c.status(st);
  c.device_output<D>(d.get(), de, kind_of<D>(), out_rtol<D>(), out_atol<D>());
  c.finish();
}

// Status-only case: the functor fully binds the call.
template <typename F> void only_status(const std::string &fname, const std::string &key, F f) {
  clear_cuda_error();
  NppStatus st = f();
  Case c("nppiFilterUnsharp/" + key, fname);
  c.status(st);
  c.finish();
}

// Buffer-size case: records status plus the reported size.
template <typename F> void gbs_case(const std::string &fname, const std::string &key, F f) {
  int need = -12345;
  NppStatus st = f(&need);
  Case c("nppiFilterUnsharp/gbs/" + key, fname);
  c.status(st);
  c.host_output<int>(&need, 1, ValueKind::I32);
  c.finish();
}

// ---------------------------------------------------------- error matrices

// Border unsharp: null -> ROI neg -> border -> radius/sigma -> kernel > ROI
// (covers ROI zero) -> dst step.
template <typename S, typename D, typename F>
void border_errors(const std::string &fam, const std::string &suffix, int channels, S *s, D *d,
                   Npp8u *buf, F f) {
  NppStreamContext ctx = current_ctx();
  const std::string fn = fam + suffix;
  const NppiPoint off{2, 1};
  const NppiSize roi{8, 6};
  const int dpx = channels * (int)sizeof(D);
  const int dAlign = channels == 4 ? dpx : (int)sizeof(D);
  const NppiBorderType bad = NPP_BORDER_CONSTANT;
  const NppiBorderType rep = NPP_BORDER_REPLICATE;
  const int dstep = align_step(roi.width * dpx, dAlign);
  only_status(fn, fam + "/err/nullsrc" + suffix,
              [&] { return f(nullptr, 16, off, d, dstep, roi, 2.f, 1.f, 1.2f, 0.f, rep, buf, ctx); });
  only_status(fn, fam + "/err/nulldst" + suffix,
              [&] { return f(s, 16, off, nullptr, dstep, roi, 2.f, 1.f, 1.2f, 0.f, rep, buf, ctx); });
  only_status(fn, fam + "/err/nullbuf" + suffix,
              [&] { return f(s, 16, off, d, dstep, roi, 2.f, 1.f, 1.2f, 0.f, rep, nullptr, ctx); });
  only_status(fn, fam + "/err/negroi_w" + suffix,
              [&] { return f(s, 16, off, d, dstep, NppiSize{-1, 6}, 2.f, 1.f, 1.2f, 0.f, rep, buf, ctx); });
  // negative ROI wins over the border type
  only_status(fn, fam + "/err/negroi_badborder" + suffix,
              [&] { return f(s, 16, off, d, dstep, NppiSize{-1, 6}, 2.f, 1.f, 1.2f, 0.f, bad, buf, ctx); });
  // border type wins over the radius/sigma checks and over a zero ROI
  only_status(fn, fam + "/err/badborder" + suffix,
              [&] { return f(s, 16, off, d, dstep, roi, 2.f, 1.f, 1.2f, 0.f, bad, buf, ctx); });
  only_status(fn, fam + "/err/badborder_radneg" + suffix,
              [&] { return f(s, 16, off, d, dstep, roi, -1.f, 1.f, 1.2f, 0.f, bad, buf, ctx); });
  only_status(fn, fam + "/err/radneg" + suffix,
              [&] { return f(s, 16, off, d, dstep, roi, -1.f, 1.f, 1.2f, 0.f, rep, buf, ctx); });
  only_status(fn, fam + "/err/sigzero" + suffix,
              [&] { return f(s, 16, off, d, dstep, roi, 2.f, 0.f, 1.2f, 0.f, rep, buf, ctx); });
  only_status(fn, fam + "/err/signeg" + suffix,
              [&] { return f(s, 16, off, d, dstep, roi, 2.f, -1.f, 1.2f, 0.f, rep, buf, ctx); });
  // radius/sigma win over the dst step check
  only_status(fn, fam + "/err/radneg_stepsmall" + suffix,
              [&] { return f(s, 16, off, d, roi.width * dpx - 1, roi, -1.f, 1.f, 1.2f, 0.f, rep, buf, ctx); });
  // kernel larger than the ROI (radius 8 -> k = 17) and a zero ROI
  only_status(fn, fam + "/err/kernel_gt_roi" + suffix,
              [&] { return f(s, 16, off, d, dstep, roi, 8.f, 1.f, 1.2f, 0.f, rep, buf, ctx); });
  only_status(fn, fam + "/err/roizero" + suffix,
              [&] { return f(s, 16, off, d, dstep, NppiSize{0, 0}, 2.f, 1.f, 1.2f, 0.f, rep, buf, ctx); });
  only_status(fn, fam + "/err/roizero_w" + suffix,
              [&] { return f(s, 16, off, d, dstep, NppiSize{0, 6}, 2.f, 1.f, 1.2f, 0.f, rep, buf, ctx); });
  only_status(fn, fam + "/err/kernel_gt_roi_badstep" + suffix,
              [&] { return f(s, 16, off, d, roi.width * dpx - 1, roi, 8.f, 1.f, 1.2f, 0.f, rep, buf, ctx); });
  only_status(fn, fam + "/err/stepsmall" + suffix,
              [&] { return f(s, 16, off, d, roi.width * dpx - 1, roi, 2.f, 1.f, 1.2f, 0.f, rep, buf, ctx); });
  if (dAlign > 1)
    only_status(fn, fam + "/err/stepodd" + suffix,
                [&] { return f(s, 16, off, d, roi.width * dpx + 1, roi, 2.f, 1.f, 1.2f, 0.f, rep, buf, ctx); });
  // Offsets out of the assumed image are not validated (status only).
  only_status(fn, fam + "/ok/offneg" + suffix,
              [&] { return f(s, 16, NppiPoint{-1, 0}, d, dstep, roi, 2.f, 1.f, 1.2f, 0.f, rep, buf, ctx); });
  only_status(fn, fam + "/ok/offbig" + suffix,
              [&] { return f(s, 16, NppiPoint{40, 0}, d, dstep, roi, 2.f, 1.f, 1.2f, 0.f, rep, buf, ctx); });
  // An odd source step is not validated either (the call returns NO_ERROR),
  // but the reference kernel then issues an unaligned vector load and faults
  // asynchronously with a sticky misaligned-address error that kills the CUDA
  // context for every later test in the process.  That is undefined behaviour
  // outside the API contract (the docs require a step of at least
  // width*sizeof(pixel)), so it is not pinned here: measured on NPP 13.4 with
  // 16u/16s/32f C1 and 8u AC4, 8u C1 stays clean.
}

// GetBufferSize: null pointer -> negative radius -> non-positive sigma.
template <typename F> void gbs_errors(const std::string &fam, const std::string &suffix, F f) {
  int need = 0;
  // A null hpBufferSize crashes NVIDIA NPP (it is written unconditionally), so
  // that contract cannot be recorded here.
  only_status(fam, fam + "/err/radneg" + suffix, [&] { return f(-1.f, 1.f, &need); });
  // sigma == 0 is accepted with an explicit radius (recorded in the matrix);
  // with an auto radius NPP 13.4 never returns, so that case is not probed.
  only_status(fam, fam + "/err/signeg" + suffix, [&] { return f(2.f, -1.f, &need); });
  only_status(fam, fam + "/err/radneg_sig0" + suffix, [&] { return f(-0.5f, 0.f, &need); });
}

// Runs the standard parameter matrix for one source/destination type and
// channel count.
template <typename S, typename D>
void matrix_for(const std::string &fname, const char *tag, int channels,
                NppStatus (*f)(const S *, Npp32s, NppiPoint, D *, Npp32s, NppiSize, Npp32f, Npp32f,
                               Npp32f, Npp32f, NppiBorderType, Npp8u *, NppStreamContext)) {
  const std::string base = std::string(tag) + "/";
  uint64_t seed = 0x1000 + (uint64_t)channels * 7 + sizeof(S) * 13;
  // explicit radius, no offset
  border_ok<S, D>(fname, base + "r1s1", channels, 0, 0, 12, 9, 2, seed + 1, 1.f, 1.f, 1.2f, 0.f, f);
  // explicit radius, offset (2,1)
  border_ok<S, D>(fname, base + "r1s1/off21", channels, 2, 1, 12, 9, 3, seed + 2, 1.f, 1.f, 1.2f, 0.f, f);
  // 5x5 window, offset (1,0) and a padded destination step
  border_ok<S, D>(fname, base + "r2s1", channels, 0, 0, 13, 10, 2, seed + 3, 2.f, 1.f, 1.2f, 0.f, f);
  border_ok<S, D>(fname, base + "r2s1/off10", channels, 1, 0, 13, 10, 5, seed + 4, 2.f, 1.f, 0.8f, 0.f, f);
  // negative weight (blur only)
  border_ok<S, D>(fname, base + "r2s1/wneg", channels, 2, 2, 13, 10, 2, seed + 5, 2.f, 1.f, -0.7f,
                  0.f, f);
  // auto radius from sigma (8u sigma=1 -> k=7, 16u/16s -> k=9, 32f -> k=15)
  border_ok<S, D>(fname, base + "auto_s1", channels, 1, 1, 17, 17, 2, seed + 6, 0.f, 1.f, 1.2f, 0.f, f);
  border_ok<S, D>(fname, base + "auto_s05", channels, 0, 0, 12, 9, 2, seed + 7, 0.f, 0.5f, 1.2f, 0.f, f);
  // threshold: smooth ramp must be reproduced unchanged
  ramp_ok<S, D>(fname, base + "thr2_ramp", channels, 12, 9, 1.f, 1.f, 1.2f, 2.f, f);
}

}  // namespace

class NppiFilterUnsharpDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(NppiFilterUnsharpDiff, Border_8u_C1) {
  matrix_for<Npp8u, Npp8u>("nppiFilterUnsharpBorder_8u_C1R_Ctx", "8u/C1", 1,
                           nppiFilterUnsharpBorder_8u_C1R_Ctx);
}

TEST_F(NppiFilterUnsharpDiff, Border_8u_C3) {
  matrix_for<Npp8u, Npp8u>("nppiFilterUnsharpBorder_8u_C3R_Ctx", "8u/C3", 3,
                           nppiFilterUnsharpBorder_8u_C3R_Ctx);
}

TEST_F(NppiFilterUnsharpDiff, Border_8u_C4) {
  matrix_for<Npp8u, Npp8u>("nppiFilterUnsharpBorder_8u_C4R_Ctx", "8u/C4", 4,
                           nppiFilterUnsharpBorder_8u_C4R_Ctx);
}

TEST_F(NppiFilterUnsharpDiff, Border_8u_AC4) {
  matrix_for<Npp8u, Npp8u>("nppiFilterUnsharpBorder_8u_AC4R_Ctx", "8u/AC4", 4,
                           nppiFilterUnsharpBorder_8u_AC4R_Ctx);
}

TEST_F(NppiFilterUnsharpDiff, Border_16u_C1) {
  matrix_for<Npp16u, Npp16u>("nppiFilterUnsharpBorder_16u_C1R_Ctx", "16u/C1", 1,
                             nppiFilterUnsharpBorder_16u_C1R_Ctx);
}

TEST_F(NppiFilterUnsharpDiff, Border_16u_C3) {
  matrix_for<Npp16u, Npp16u>("nppiFilterUnsharpBorder_16u_C3R_Ctx", "16u/C3", 3,
                             nppiFilterUnsharpBorder_16u_C3R_Ctx);
}

TEST_F(NppiFilterUnsharpDiff, Border_16u_C4) {
  matrix_for<Npp16u, Npp16u>("nppiFilterUnsharpBorder_16u_C4R_Ctx", "16u/C4", 4,
                             nppiFilterUnsharpBorder_16u_C4R_Ctx);
}

TEST_F(NppiFilterUnsharpDiff, Border_16u_AC4) {
  matrix_for<Npp16u, Npp16u>("nppiFilterUnsharpBorder_16u_AC4R_Ctx", "16u/AC4", 4,
                             nppiFilterUnsharpBorder_16u_AC4R_Ctx);
}

TEST_F(NppiFilterUnsharpDiff, Border_16s_C1) {
  matrix_for<Npp16s, Npp16s>("nppiFilterUnsharpBorder_16s_C1R_Ctx", "16s/C1", 1,
                             nppiFilterUnsharpBorder_16s_C1R_Ctx);
}

TEST_F(NppiFilterUnsharpDiff, Border_16s_C3) {
  matrix_for<Npp16s, Npp16s>("nppiFilterUnsharpBorder_16s_C3R_Ctx", "16s/C3", 3,
                             nppiFilterUnsharpBorder_16s_C3R_Ctx);
}

TEST_F(NppiFilterUnsharpDiff, Border_16s_C4) {
  matrix_for<Npp16s, Npp16s>("nppiFilterUnsharpBorder_16s_C4R_Ctx", "16s/C4", 4,
                             nppiFilterUnsharpBorder_16s_C4R_Ctx);
}

TEST_F(NppiFilterUnsharpDiff, Border_16s_AC4) {
  matrix_for<Npp16s, Npp16s>("nppiFilterUnsharpBorder_16s_AC4R_Ctx", "16s/AC4", 4,
                             nppiFilterUnsharpBorder_16s_AC4R_Ctx);
}

TEST_F(NppiFilterUnsharpDiff, Border_32f_C1) {
  matrix_for<Npp32f, Npp32f>("nppiFilterUnsharpBorder_32f_C1R_Ctx", "32f/C1", 1,
                             nppiFilterUnsharpBorder_32f_C1R_Ctx);
}

TEST_F(NppiFilterUnsharpDiff, Border_32f_C3) {
  matrix_for<Npp32f, Npp32f>("nppiFilterUnsharpBorder_32f_C3R_Ctx", "32f/C3", 3,
                             nppiFilterUnsharpBorder_32f_C3R_Ctx);
}

TEST_F(NppiFilterUnsharpDiff, Border_32f_C4) {
  matrix_for<Npp32f, Npp32f>("nppiFilterUnsharpBorder_32f_C4R_Ctx", "32f/C4", 4,
                             nppiFilterUnsharpBorder_32f_C4R_Ctx);
}

TEST_F(NppiFilterUnsharpDiff, Border_32f_AC4) {
  matrix_for<Npp32f, Npp32f>("nppiFilterUnsharpBorder_32f_AC4R_Ctx", "32f/AC4", 4,
                             nppiFilterUnsharpBorder_32f_AC4R_Ctx);
}

// Small ROIs (kernel equal to the ROI) and a tight source image.
TEST_F(NppiFilterUnsharpDiff, Border_small_roi) {
  border_ok<Npp8u, Npp8u>("nppiFilterUnsharpBorder_8u_C1R_Ctx", "8u/C1/k7x7", 1, 0, 0, 7, 7, 1,
                          0x2001, 3.f, 1.f, 1.2f, 0.f, nppiFilterUnsharpBorder_8u_C1R_Ctx);
  border_ok<Npp16u, Npp16u>("nppiFilterUnsharpBorder_16u_C1R_Ctx", "16u/C1/k3x3", 1, 0, 0, 3, 4, 2,
                            0x2002, 1.f, 1.f, 1.2f, 0.f, nppiFilterUnsharpBorder_16u_C1R_Ctx);
  border_ok<Npp32f, Npp32f>("nppiFilterUnsharpBorder_32f_C1R_Ctx", "32f/C1/tall", 1, 1, 0, 5, 21, 2,
                            0x2003, 2.f, 1.f, 1.2f, 0.f, nppiFilterUnsharpBorder_32f_C1R_Ctx);
  border_ok<Npp16s, Npp16s>("nppiFilterUnsharpBorder_16s_C1R_Ctx", "16s/C1/wide", 1, 0, 1, 21, 5, 2,
                            0x2004, 2.f, 1.f, 1.2f, 0.f, nppiFilterUnsharpBorder_16s_C1R_Ctx);
}

// Threshold suppression with a strong delta (8u keeps the centre response and
// suppresses everything below threshold*128).
TEST_F(NppiFilterUnsharpDiff, Border_threshold_delta) {
  border_ok<Npp8u, Npp8u>("nppiFilterUnsharpBorder_8u_C1R_Ctx", "8u/C1/thr1_delta", 1, 0, 0, 17,
                          11, 2, 0x3001, 2.f, 1.f, 1.f, 1.f, nppiFilterUnsharpBorder_8u_C1R_Ctx);
  border_ok<Npp16u, Npp16u>("nppiFilterUnsharpBorder_16u_C1R_Ctx", "16u/C1/thr1_delta", 1, 0, 0,
                            17, 11, 2, 0x3002, 2.f, 1.f, 1.f, 1.f,
                            nppiFilterUnsharpBorder_16u_C1R_Ctx);
  border_ok<Npp32f, Npp32f>("nppiFilterUnsharpBorder_32f_C1R_Ctx", "32f/C1/thr1e-3", 1, 0, 0, 17,
                            11, 2, 0x3003, 2.f, 1.f, 1.f, 1e-3f,
                            nppiFilterUnsharpBorder_32f_C1R_Ctx);
}

TEST_F(NppiFilterUnsharpDiff, Border_errors) {
  border_errors("nppiFilterUnsharpBorder", "_8u_C1R_Ctx", 1, arena<Npp8u>(256).get(),
                arena<Npp8u>(256).get(), arena<Npp8u>(4096).get(),
                nppiFilterUnsharpBorder_8u_C1R_Ctx);
  border_errors("nppiFilterUnsharpBorder", "_16u_C1R_Ctx", 1, arena<Npp16u>(256).get(),
                arena<Npp16u>(256).get(), arena<Npp8u>(4096).get(),
                nppiFilterUnsharpBorder_16u_C1R_Ctx);
  border_errors("nppiFilterUnsharpBorder", "_16s_C1R_Ctx", 1, arena<Npp16s>(256).get(),
                arena<Npp16s>(256).get(), arena<Npp8u>(4096).get(),
                nppiFilterUnsharpBorder_16s_C1R_Ctx);
  border_errors("nppiFilterUnsharpBorder", "_32f_C1R_Ctx", 1, arena<Npp32f>(256).get(),
                arena<Npp32f>(256).get(), arena<Npp8u>(4096).get(),
                nppiFilterUnsharpBorder_32f_C1R_Ctx);
  border_errors("nppiFilterUnsharpBorder", "_8u_AC4R_Ctx", 4, arena<Npp8u>(1024).get(),
                arena<Npp8u>(1024).get(), arena<Npp8u>(4096).get(),
                nppiFilterUnsharpBorder_8u_AC4R_Ctx);
  border_errors("nppiFilterUnsharpBorder", "_32f_C4R_Ctx", 4, arena<Npp32f>(1024).get(),
                arena<Npp32f>(1024).get(), arena<Npp8u>(4096).get(),
                nppiFilterUnsharpBorder_32f_C4R_Ctx);
}

// ------------------------------------------------------------ buffer sizes

TEST_F(NppiFilterUnsharpDiff, GetBufferSize_matrix) {
  struct P { float r, s; const char *tag; } ps[] = {
      {1.f, 1.f, "r1s1"},     {2.f, 1.f, "r2s1"},     {3.f, 1.f, "r3s1"},
      {0.5f, 1.f, "r05s1"},   {1.5f, 0.8f, "r15s08"}, {4.f, 2.f, "r4s2"},
      {0.f, 1.f, "auto_s1"},  {0.f, 0.5f, "auto_s05"}, {0.f, 2.f, "auto_s2"},
      {0.f, 0.25f, "auto_s025"},
      // explicit radius ignores sigma (sigma == 0 is accepted, negative is not)
      {2.f, 0.f, "r2s0"},     {0.f, 1e-6f, "auto_tiny"}, {5.f, 4.f, "r5s4"},
  };
  for (const auto &p : ps) {
    const std::string t = p.tag;
    gbs_case("nppiFilterUnsharpGetBufferSize_8u_C1R", t + "/8u_C1",
             [&](int *n) { return nppiFilterUnsharpGetBufferSize_8u_C1R(p.r, p.s, n); });
    gbs_case("nppiFilterUnsharpGetBufferSize_8u_AC4R", t + "/8u_AC4",
             [&](int *n) { return nppiFilterUnsharpGetBufferSize_8u_AC4R(p.r, p.s, n); });
    gbs_case("nppiFilterUnsharpGetBufferSize_16u_C3R", t + "/16u_C3",
             [&](int *n) { return nppiFilterUnsharpGetBufferSize_16u_C3R(p.r, p.s, n); });
    gbs_case("nppiFilterUnsharpGetBufferSize_16s_C4R", t + "/16s_C4",
             [&](int *n) { return nppiFilterUnsharpGetBufferSize_16s_C4R(p.r, p.s, n); });
    gbs_case("nppiFilterUnsharpGetBufferSize_32f_C1R", t + "/32f_C1",
             [&](int *n) { return nppiFilterUnsharpGetBufferSize_32f_C1R(p.r, p.s, n); });
    gbs_case("nppiFilterUnsharpGetBufferSize_32f_AC4R", t + "/32f_AC4",
             [&](int *n) { return nppiFilterUnsharpGetBufferSize_32f_AC4R(p.r, p.s, n); });
  }
}

TEST_F(NppiFilterUnsharpDiff, GetBufferSize_errors) {
  gbs_errors("nppiFilterUnsharpGetBufferSize", "_8u_C1R",
             nppiFilterUnsharpGetBufferSize_8u_C1R);
  gbs_errors("nppiFilterUnsharpGetBufferSize", "_16u_AC4R",
             nppiFilterUnsharpGetBufferSize_16u_AC4R);
  gbs_errors("nppiFilterUnsharpGetBufferSize", "_16s_C3R",
             nppiFilterUnsharpGetBufferSize_16s_C3R);
  gbs_errors("nppiFilterUnsharpGetBufferSize", "_32f_C1R",
             nppiFilterUnsharpGetBufferSize_32f_C1R);
}
