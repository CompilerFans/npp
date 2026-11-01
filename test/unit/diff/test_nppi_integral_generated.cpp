// Differential tests for nppiIntegral / nppiSqrIntegral (package s1).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiIntegralDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests --gtest_filter='NppiIntegralDiff.*'
//
// The destination image is (W+1) x (H+1); both record and verify initialize the
// whole padded destination deterministically so padding bytes are comparable.
// Error-path cases pin the status contract; output comparison is skipped for
// them by the harness.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;

namespace {

template <typename T> DeviceMemory<T> &arena(std::size_t n) {
  static std::vector<std::unique_ptr<DeviceMemory<T>>> v;
  v.push_back(std::make_unique<DeviceMemory<T>>(n));
  return *v.back();
}

NppStreamContext current_context() {
  NppStreamContext ctx{};
  NppStatus status = nppGetStreamContext(&ctx);
  EXPECT_EQ(status, NPP_NO_ERROR) << "nppGetStreamContext failed";
  return ctx;
}

inline int align_step(int step, int esize) { return ((step + esize - 1) / esize) * esize; }

template <typename T> ValueKind out_kind();
template <> ValueKind out_kind<Npp32s>() { return ValueKind::I32; }
template <> ValueKind out_kind<Npp32f>() { return ValueKind::F32; }
template <> ValueKind out_kind<Npp64f>() { return ValueKind::F64; }

// nppiIntegral happy path / error probe.
template <typename SrcT, typename DstT>
void integral_case(const char *fname, const std::string &id, int width, int height, int pad, DstT nVal,
                   NppStatus (*f)(const SrcT *, int, DstT *, int, NppiSize, DstT, NppStreamContext), uint64_t seed) {
  const int srcStep = align_step(width * (int)sizeof(SrcT) + pad, (int)sizeof(SrcT));
  const int dstStep = align_step((width + 1) * (int)sizeof(DstT) + pad, (int)sizeof(DstT));
  const size_t srcElems = (size_t)height * srcStep / sizeof(SrcT);
  const size_t dstElems = (size_t)(height + 1) * dstStep / sizeof(DstT);
  std::vector<SrcT> hs = make_uniform<SrcT>(srcElems ? srcElems : 1, seed);
  std::vector<DstT> hd = make_uniform<DstT>(dstElems ? dstElems : 1, seed + 101);
  DeviceMemory<SrcT> &s = arena<SrcT>(srcElems ? srcElems : 1);
  DeviceMemory<DstT> &d = arena<DstT>(dstElems ? dstElems : 1);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  Case c(id, fname);
  clear_cuda_error();
  NppStatus st = f(s.get(), srcStep, d.get(), dstStep, NppiSize{width, height}, nVal, current_context());
  c.input(hs.data(), hs.size() * sizeof(SrcT));
  c.status(st);
  c.device_output<DstT>(d.get(), dstElems, out_kind<DstT>(), 1e-5, 1e-6);
  c.finish();
}

// nppiSqrIntegral happy path / error probe.
template <typename SrcT, typename DstT, typename SqrT>
void sqr_integral_case(const char *fname, const std::string &id, int width, int height, int pad, DstT nVal,
                       SqrT nValSqr,
                       NppStatus (*f)(const SrcT *, int, DstT *, int, SqrT *, int, NppiSize, DstT, SqrT,
                                      NppStreamContext),
                       uint64_t seed) {
  const int srcStep = align_step(width * (int)sizeof(SrcT) + pad, (int)sizeof(SrcT));
  const int dstStep = align_step((width + 1) * (int)sizeof(DstT) + pad, (int)sizeof(DstT));
  const int sqrStep = align_step((width + 1) * (int)sizeof(SqrT) + pad, (int)sizeof(SqrT));
  const size_t srcElems = (size_t)height * srcStep / sizeof(SrcT);
  const size_t dstElems = (size_t)(height + 1) * dstStep / sizeof(DstT);
  const size_t sqrElems = (size_t)(height + 1) * sqrStep / sizeof(SqrT);
  std::vector<SrcT> hs = make_uniform<SrcT>(srcElems ? srcElems : 1, seed);
  std::vector<DstT> hd = make_uniform<DstT>(dstElems ? dstElems : 1, seed + 101);
  std::vector<SqrT> hq = make_uniform<SqrT>(sqrElems ? sqrElems : 1, seed + 211);
  DeviceMemory<SrcT> &s = arena<SrcT>(srcElems ? srcElems : 1);
  DeviceMemory<DstT> &d = arena<DstT>(dstElems ? dstElems : 1);
  DeviceMemory<SqrT> &q = arena<SqrT>(sqrElems ? sqrElems : 1);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  q.copyFromHost(hq);
  Case c(id, fname);
  clear_cuda_error();
  NppStatus st = f(s.get(), srcStep, d.get(), dstStep, q.get(), sqrStep, NppiSize{width, height}, nVal, nValSqr,
                   current_context());
  c.input(hs.data(), hs.size() * sizeof(SrcT));
  c.status(st);
  c.device_output<DstT>(d.get(), dstElems, out_kind<DstT>(), 1e-5, 1e-6);
  c.device_output<SqrT>(q.get(), sqrElems, out_kind<SqrT>(), 1e-5, 1e-6);
  c.finish();
}

// Error-path probes that pass explicit pointers / steps / ROIs.
template <typename SrcT, typename DstT>
void integral_err(const char *fname, const std::string &id, const SrcT *pSrc, int nSrcStep, DstT *pDst, int nDstStep,
                  NppiSize roi, NppStatus (*f)(const SrcT *, int, DstT *, int, NppiSize, DstT, NppStreamContext)) {
  Case c(id, fname);
  clear_cuda_error();
  NppStatus st = f(pSrc, nSrcStep, pDst, nDstStep, roi, DstT(3), current_context());
  c.status(st);
  c.finish();
}

template <typename SrcT, typename DstT, typename SqrT>
void sqr_integral_err(const char *fname, const std::string &id, const SrcT *pSrc, int nSrcStep, DstT *pDst,
                      int nDstStep, SqrT *pSqr, int nSqrStep, NppiSize roi,
                      NppStatus (*f)(const SrcT *, int, DstT *, int, SqrT *, int, NppiSize, DstT, SqrT,
                                     NppStreamContext)) {
  Case c(id, fname);
  clear_cuda_error();
  NppStatus st = f(pSrc, nSrcStep, pDst, nDstStep, pSqr, nSqrStep, roi, DstT(3), SqrT(5), current_context());
  c.status(st);
  c.finish();
}

}  // namespace

// ---------------------------------------------------------------------------
// nppiIntegral_8u32s_C1R
// ---------------------------------------------------------------------------

TEST(NppiIntegralDiff, Integral8u32s) {
  const auto f = nppiIntegral_8u32s_C1R_Ctx;
  integral_case<Npp8u, Npp32s>("nppiIntegral_8u32s_C1R", "nppiIntegral/8u32s/17x13_v0", 17, 13, 4, 0, f, 1);
  integral_case<Npp8u, Npp32s>("nppiIntegral_8u32s_C1R", "nppiIntegral/8u32s/17x13_v7", 17, 13, 4, 7, f, 2);
  integral_case<Npp8u, Npp32s>("nppiIntegral_8u32s_C1R", "nppiIntegral/8u32s/1x1", 1, 1, 0, 3, f, 3);
  integral_case<Npp8u, Npp32s>("nppiIntegral_8u32s_C1R", "nppiIntegral/8u32s/64x32", 64, 32, 0, 0, f, 4);
}

// The null / step / ROI probes need direct control over the arguments, which
// the happy-path helper does not expose; they use dedicated bodies below.
TEST(NppiIntegralDiff, Integral8u32sErrors) {
  const auto f = nppiIntegral_8u32s_C1R_Ctx;
  const int W = 17, H = 13;
  const int srcStep = align_step(W, 1);
  const int dstStep = align_step((W + 1) * 4, 4);
  DeviceMemory<Npp8u> &s = arena<Npp8u>((size_t)H * srcStep);
  DeviceMemory<Npp32s> &d = arena<Npp32s>((size_t)(H + 1) * dstStep / 4);
  s.copyFromHost(make_uniform<Npp8u>((size_t)H * srcStep, 42));
  d.copyFromHost(make_uniform<Npp32s>((size_t)(H + 1) * dstStep / 4, 43));
  integral_err<Npp8u, Npp32s>("nppiIntegral_8u32s_C1R", "nppiIntegral/8u32s/err_null_src", nullptr, srcStep, d.get(),
                              dstStep, NppiSize{W, H}, f);
  integral_err<Npp8u, Npp32s>("nppiIntegral_8u32s_C1R", "nppiIntegral/8u32s/err_null_dst", s.get(), srcStep, nullptr,
                              dstStep, NppiSize{W, H}, f);
  integral_err<Npp8u, Npp32s>("nppiIntegral_8u32s_C1R", "nppiIntegral/8u32s/err_neg_roi", s.get(), srcStep, d.get(),
                              dstStep, NppiSize{-1, H}, f);
  integral_err<Npp8u, Npp32s>("nppiIntegral_8u32s_C1R", "nppiIntegral/8u32s/err_zero_w", s.get(), srcStep, d.get(),
                              dstStep, NppiSize{0, H}, f);
  integral_err<Npp8u, Npp32s>("nppiIntegral_8u32s_C1R", "nppiIntegral/8u32s/err_zero_h", s.get(), srcStep, d.get(),
                              dstStep, NppiSize{W, 0}, f);
  integral_err<Npp8u, Npp32s>("nppiIntegral_8u32s_C1R", "nppiIntegral/8u32s/err_src_step", s.get(), W - 1, d.get(),
                              dstStep, NppiSize{W, H}, f);
  integral_err<Npp8u, Npp32s>("nppiIntegral_8u32s_C1R", "nppiIntegral/8u32s/err_dst_step", s.get(), srcStep, d.get(),
                              W * 4, NppiSize{W, H}, f);
}

TEST(NppiIntegralDiff, Integral8u32f) {
  const auto f = nppiIntegral_8u32f_C1R_Ctx;
  integral_case<Npp8u, Npp32f>("nppiIntegral_8u32f_C1R", "nppiIntegral/8u32f/17x13_v0", 17, 13, 4, 0.0f, f, 11);
  integral_case<Npp8u, Npp32f>("nppiIntegral_8u32f_C1R", "nppiIntegral/8u32f/17x13_v5", 17, 13, 4, 5.5f, f, 12);
  integral_case<Npp8u, Npp32f>("nppiIntegral_8u32f_C1R", "nppiIntegral/8u32f/1x1", 1, 1, 0, 2.5f, f, 13);
  integral_case<Npp8u, Npp32f>("nppiIntegral_8u32f_C1R", "nppiIntegral/8u32f/64x32", 64, 32, 0, 0.0f, f, 14);
}

TEST(NppiIntegralDiff, Integral8u32fErrors) {
  const auto f = nppiIntegral_8u32f_C1R_Ctx;
  const int W = 17, H = 13;
  const int srcStep = align_step(W, 1);
  const int dstStep = align_step((W + 1) * 4, 4);
  DeviceMemory<Npp8u> &s = arena<Npp8u>((size_t)H * srcStep);
  DeviceMemory<Npp32f> &d = arena<Npp32f>((size_t)(H + 1) * dstStep / 4);
  s.copyFromHost(make_uniform<Npp8u>((size_t)H * srcStep, 52));
  d.copyFromHost(make_uniform<Npp32f>((size_t)(H + 1) * dstStep / 4, 53));
  integral_err<Npp8u, Npp32f>("nppiIntegral_8u32f_C1R", "nppiIntegral/8u32f/err_null_src", nullptr, srcStep, d.get(),
                              dstStep, NppiSize{W, H}, f);
  integral_err<Npp8u, Npp32f>("nppiIntegral_8u32f_C1R", "nppiIntegral/8u32f/err_null_dst", s.get(), srcStep, nullptr,
                              dstStep, NppiSize{W, H}, f);
  integral_err<Npp8u, Npp32f>("nppiIntegral_8u32f_C1R", "nppiIntegral/8u32f/err_neg_roi", s.get(), srcStep, d.get(),
                              dstStep, NppiSize{W, -2}, f);
  integral_err<Npp8u, Npp32f>("nppiIntegral_8u32f_C1R", "nppiIntegral/8u32f/err_zero_w", s.get(), srcStep, d.get(),
                              dstStep, NppiSize{0, H}, f);
  integral_err<Npp8u, Npp32f>("nppiIntegral_8u32f_C1R", "nppiIntegral/8u32f/err_src_step", s.get(), W - 1, d.get(),
                              dstStep, NppiSize{W, H}, f);
  integral_err<Npp8u, Npp32f>("nppiIntegral_8u32f_C1R", "nppiIntegral/8u32f/err_dst_step", s.get(), srcStep, d.get(),
                              W * 4, NppiSize{W, H}, f);
}

// ---------------------------------------------------------------------------
// nppiSqrIntegral variants
// ---------------------------------------------------------------------------

TEST(NppiIntegralDiff, SqrIntegral8u32s) {
  const auto f = nppiSqrIntegral_8u32s_C1R_Ctx;
  sqr_integral_case<Npp8u, Npp32s, Npp32s>("nppiSqrIntegral_8u32s_C1R", "nppiSqrIntegral/8u32s/17x13", 17, 13, 4, 0,
                                           0, f, 21);
  sqr_integral_case<Npp8u, Npp32s, Npp32s>("nppiSqrIntegral_8u32s_C1R", "nppiSqrIntegral/8u32s/17x13_v", 17, 13, 4, 3,
                                           7, f, 22);
  sqr_integral_case<Npp8u, Npp32s, Npp32s>("nppiSqrIntegral_8u32s_C1R", "nppiSqrIntegral/8u32s/1x1", 1, 1, 0, 1, 2, f,
                                           23);
  sqr_integral_case<Npp8u, Npp32s, Npp32s>("nppiSqrIntegral_8u32s_C1R", "nppiSqrIntegral/8u32s/64x32", 64, 32, 0, 0,
                                           0, f, 24);
}

TEST(NppiIntegralDiff, SqrIntegral8u32sErrors) {
  const auto f = nppiSqrIntegral_8u32s_C1R_Ctx;
  const int W = 17, H = 13;
  const int srcStep = align_step(W, 1);
  const int dstStep = align_step((W + 1) * 4, 4);
  DeviceMemory<Npp8u> &s = arena<Npp8u>((size_t)H * srcStep);
  DeviceMemory<Npp32s> &d = arena<Npp32s>((size_t)(H + 1) * dstStep / 4);
  DeviceMemory<Npp32s> &q = arena<Npp32s>((size_t)(H + 1) * dstStep / 4);
  s.copyFromHost(make_uniform<Npp8u>((size_t)H * srcStep, 62));
  d.copyFromHost(make_uniform<Npp32s>((size_t)(H + 1) * dstStep / 4, 63));
  q.copyFromHost(make_uniform<Npp32s>((size_t)(H + 1) * dstStep / 4, 64));
  sqr_integral_err<Npp8u, Npp32s, Npp32s>("nppiSqrIntegral_8u32s_C1R", "nppiSqrIntegral/8u32s/err_null_src", nullptr,
                                          srcStep, d.get(), dstStep, q.get(), dstStep, NppiSize{W, H}, f);
  sqr_integral_err<Npp8u, Npp32s, Npp32s>("nppiSqrIntegral_8u32s_C1R", "nppiSqrIntegral/8u32s/err_null_dst", s.get(),
                                          srcStep, nullptr, dstStep, q.get(), dstStep, NppiSize{W, H}, f);
  sqr_integral_err<Npp8u, Npp32s, Npp32s>("nppiSqrIntegral_8u32s_C1R", "nppiSqrIntegral/8u32s/err_null_sqr", s.get(),
                                          srcStep, d.get(), dstStep, nullptr, dstStep, NppiSize{W, H}, f);
  sqr_integral_err<Npp8u, Npp32s, Npp32s>("nppiSqrIntegral_8u32s_C1R", "nppiSqrIntegral/8u32s/err_neg_roi", s.get(),
                                          srcStep, d.get(), dstStep, q.get(), dstStep, NppiSize{-3, H}, f);
  sqr_integral_err<Npp8u, Npp32s, Npp32s>("nppiSqrIntegral_8u32s_C1R", "nppiSqrIntegral/8u32s/err_zero_w", s.get(),
                                          srcStep, d.get(), dstStep, q.get(), dstStep, NppiSize{0, H}, f);
  sqr_integral_err<Npp8u, Npp32s, Npp32s>("nppiSqrIntegral_8u32s_C1R", "nppiSqrIntegral/8u32s/err_src_step", s.get(),
                                          W - 1, d.get(), dstStep, q.get(), dstStep, NppiSize{W, H}, f);
  sqr_integral_err<Npp8u, Npp32s, Npp32s>("nppiSqrIntegral_8u32s_C1R", "nppiSqrIntegral/8u32s/err_dst_step", s.get(),
                                          srcStep, d.get(), W * 4, q.get(), dstStep, NppiSize{W, H}, f);
  sqr_integral_err<Npp8u, Npp32s, Npp32s>("nppiSqrIntegral_8u32s_C1R", "nppiSqrIntegral/8u32s/err_sqr_step", s.get(),
                                          srcStep, d.get(), dstStep, q.get(), W * 4, NppiSize{W, H}, f);
}

TEST(NppiIntegralDiff, SqrIntegral8u32s64f) {
  const auto f = nppiSqrIntegral_8u32s64f_C1R_Ctx;
  sqr_integral_case<Npp8u, Npp32s, Npp64f>("nppiSqrIntegral_8u32s64f_C1R", "nppiSqrIntegral/8u32s64f/17x13", 17, 13, 4,
                                           0, 0.0, f, 31);
  sqr_integral_case<Npp8u, Npp32s, Npp64f>("nppiSqrIntegral_8u32s64f_C1R", "nppiSqrIntegral/8u32s64f/17x13_v", 17, 13,
                                           4, 3, 9.5, f, 32);
  sqr_integral_case<Npp8u, Npp32s, Npp64f>("nppiSqrIntegral_8u32s64f_C1R", "nppiSqrIntegral/8u32s64f/1x1", 1, 1, 0, 1,
                                           2.0, f, 33);
  sqr_integral_case<Npp8u, Npp32s, Npp64f>("nppiSqrIntegral_8u32s64f_C1R", "nppiSqrIntegral/8u32s64f/64x32", 64, 32, 0,
                                           0, 0.0, f, 34);
}

TEST(NppiIntegralDiff, SqrIntegral8u32f64f) {
  const auto f = nppiSqrIntegral_8u32f64f_C1R_Ctx;
  sqr_integral_case<Npp8u, Npp32f, Npp64f>("nppiSqrIntegral_8u32f64f_C1R", "nppiSqrIntegral/8u32f64f/17x13", 17, 13, 4,
                                           0.0f, 0.0, f, 41);
  sqr_integral_case<Npp8u, Npp32f, Npp64f>("nppiSqrIntegral_8u32f64f_C1R", "nppiSqrIntegral/8u32f64f/17x13_v", 17, 13,
                                           4, 2.5f, 4.5, f, 42);
  sqr_integral_case<Npp8u, Npp32f, Npp64f>("nppiSqrIntegral_8u32f64f_C1R", "nppiSqrIntegral/8u32f64f/1x1", 1, 1, 0,
                                           1.0f, 2.0, f, 43);
  sqr_integral_case<Npp8u, Npp32f, Npp64f>("nppiSqrIntegral_8u32f64f_C1R", "nppiSqrIntegral/8u32f64f/64x32", 64, 32, 0,
                                           0.0f, 0.0, f, 44);
}

TEST(NppiIntegralDiff, SqrIntegral8u32f64fErrors) {
  const auto f = nppiSqrIntegral_8u32f64f_C1R_Ctx;
  const int W = 17, H = 13;
  const int srcStep = align_step(W, 1);
  const int dstStep = align_step((W + 1) * 4, 4);
  const int sqrStep = align_step((W + 1) * 8, 8);
  DeviceMemory<Npp8u> &s = arena<Npp8u>((size_t)H * srcStep);
  DeviceMemory<Npp32f> &d = arena<Npp32f>((size_t)(H + 1) * dstStep / 4);
  DeviceMemory<Npp64f> &q = arena<Npp64f>((size_t)(H + 1) * sqrStep / 8);
  s.copyFromHost(make_uniform<Npp8u>((size_t)H * srcStep, 72));
  d.copyFromHost(make_uniform<Npp32f>((size_t)(H + 1) * dstStep / 4, 73));
  q.copyFromHost(make_uniform<Npp64f>((size_t)(H + 1) * sqrStep / 8, 74));
  sqr_integral_err<Npp8u, Npp32f, Npp64f>("nppiSqrIntegral_8u32f64f_C1R", "nppiSqrIntegral/8u32f64f/err_null_src",
                                          nullptr, srcStep, d.get(), dstStep, q.get(), sqrStep, NppiSize{W, H}, f);
  sqr_integral_err<Npp8u, Npp32f, Npp64f>("nppiSqrIntegral_8u32f64f_C1R", "nppiSqrIntegral/8u32f64f/err_null_sqr",
                                          s.get(), srcStep, d.get(), dstStep, nullptr, sqrStep, NppiSize{W, H}, f);
  sqr_integral_err<Npp8u, Npp32f, Npp64f>("nppiSqrIntegral_8u32f64f_C1R", "nppiSqrIntegral/8u32f64f/err_neg_roi",
                                          s.get(), srcStep, d.get(), dstStep, q.get(), sqrStep, NppiSize{W, -2}, f);
  sqr_integral_err<Npp8u, Npp32f, Npp64f>("nppiSqrIntegral_8u32f64f_C1R", "nppiSqrIntegral/8u32f64f/err_zero_h",
                                          s.get(), srcStep, d.get(), dstStep, q.get(), sqrStep, NppiSize{W, 0}, f);
}

#if NPP_DIFF_HAS_NONCTX
// Non-Ctx adapters: same matrix case, same golden id, no stream context.
template <typename SrcT, typename DstT>
void integral_case_nc(const char *fname, const std::string &id, int width, int height, int pad, DstT nVal,
                      NppStatus (*f)(const SrcT *, int, DstT *, int, NppiSize, DstT), uint64_t seed) {
  const int srcStep = align_step(width * (int)sizeof(SrcT) + pad, (int)sizeof(SrcT));
  const int dstStep = align_step((width + 1) * (int)sizeof(DstT) + pad, (int)sizeof(DstT));
  const size_t srcElems = (size_t)height * srcStep / sizeof(SrcT);
  const size_t dstElems = (size_t)(height + 1) * dstStep / sizeof(DstT);
  std::vector<SrcT> hs = make_uniform<SrcT>(srcElems, seed);
  std::vector<DstT> hd = make_uniform<DstT>(dstElems, seed + 101);
  DeviceMemory<SrcT> &s = arena<SrcT>(srcElems);
  DeviceMemory<DstT> &d = arena<DstT>(dstElems);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  Case c(id, fname);
  clear_cuda_error();
  NppStatus st = f(s.get(), srcStep, d.get(), dstStep, NppiSize{width, height}, nVal);
  c.input(hs.data(), hs.size() * sizeof(SrcT));
  c.status(st);
  c.device_output<DstT>(d.get(), dstElems, out_kind<DstT>(), 1e-5, 1e-6);
  c.finish();
}

template <typename SrcT, typename DstT, typename SqrT>
void sqr_integral_case_nc(const char *fname, const std::string &id, int width, int height, int pad, DstT nVal,
                          SqrT nValSqr,
                          NppStatus (*f)(const SrcT *, int, DstT *, int, SqrT *, int, NppiSize, DstT, SqrT),
                          uint64_t seed) {
  const int srcStep = align_step(width * (int)sizeof(SrcT) + pad, (int)sizeof(SrcT));
  const int dstStep = align_step((width + 1) * (int)sizeof(DstT) + pad, (int)sizeof(DstT));
  const int sqrStep = align_step((width + 1) * (int)sizeof(SqrT) + pad, (int)sizeof(SqrT));
  const size_t srcElems = (size_t)height * srcStep / sizeof(SrcT);
  const size_t dstElems = (size_t)(height + 1) * dstStep / sizeof(DstT);
  const size_t sqrElems = (size_t)(height + 1) * sqrStep / sizeof(SqrT);
  std::vector<SrcT> hs = make_uniform<SrcT>(srcElems, seed);
  std::vector<DstT> hd = make_uniform<DstT>(dstElems, seed + 101);
  std::vector<SqrT> hq = make_uniform<SqrT>(sqrElems, seed + 211);
  DeviceMemory<SrcT> &s = arena<SrcT>(srcElems);
  DeviceMemory<DstT> &d = arena<DstT>(dstElems);
  DeviceMemory<SqrT> &q = arena<SqrT>(sqrElems);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  q.copyFromHost(hq);
  Case c(id, fname);
  clear_cuda_error();
  NppStatus st = f(s.get(), srcStep, d.get(), dstStep, q.get(), sqrStep, NppiSize{width, height}, nVal, nValSqr);
  c.input(hs.data(), hs.size() * sizeof(SrcT));
  c.status(st);
  c.device_output<DstT>(d.get(), dstElems, out_kind<DstT>(), 1e-5, 1e-6);
  c.device_output<SqrT>(q.get(), sqrElems, out_kind<SqrT>(), 1e-5, 1e-6);
  c.finish();
}

// Non-Ctx adapters reproduce an existing _Ctx matrix case with identical
// inputs; the forwarder must produce the same result (same golden id).
TEST(NppiIntegralDiff, NonCtxAdapters) {
  {
    const auto f = nppiIntegral_8u32s_C1R;
    integral_case_nc<Npp8u, Npp32s>("nppiIntegral_8u32s_C1R", "nppiIntegral/8u32s/17x13_v7", 17, 13, 4, 7, f, 2);
    integral_case_nc<Npp8u, Npp32s>("nppiIntegral_8u32s_C1R", "nppiIntegral/8u32s/1x1", 1, 1, 0, 3, f, 3);
  }
  {
    const auto f = nppiIntegral_8u32f_C1R;
    integral_case_nc<Npp8u, Npp32f>("nppiIntegral_8u32f_C1R", "nppiIntegral/8u32f/17x13_v5", 17, 13, 4, 5.5f, f, 12);
  }
  {
    const auto f = nppiSqrIntegral_8u32s_C1R;
    sqr_integral_case_nc<Npp8u, Npp32s, Npp32s>("nppiSqrIntegral_8u32s_C1R", "nppiSqrIntegral/8u32s/17x13_v", 17, 13,
                                                4, 3, 7, f, 22);
  }
  {
    const auto f = nppiSqrIntegral_8u32f64f_C1R;
    sqr_integral_case_nc<Npp8u, Npp32f, Npp64f>("nppiSqrIntegral_8u32f64f_C1R", "nppiSqrIntegral/8u32f64f/17x13_v",
                                                17, 13, 4, 2.5f, 4.5, f, 42);
  }
}
#endif
