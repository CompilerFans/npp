// Differential tests for the nppiColorTwist family (package c2, wave 1:
// Npp32f images; the 8u/16u/16s entry points of this family were already real
// and are covered by test_nppi_color_twist.cpp / the 32f-family suite).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiColorTwistDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=... ./diff_tests --gtest_filter='NppiColorTwistDiff.*'
//
// The goldens also pin: rounding/saturation of the twist, the values consumed
// for missing channels (C1/C2), alpha handling (C4 copy vs AC4 untouched),
// ROI/step/pointer validation order and status codes, empty-ROI behaviour and
// the non-Ctx adapters (guarded smoke test).
//
// Measured on NPP 13.4 (RTX 5090): a null aTwist (packed and planar) or a null
// planar pointer array crashes the reference (no status code), so those probes
// are not part of the suite; mcMPP answers NPP_NULL_POINTER_ERROR instead.

#include "npp_color_twist_diff.h"

#include <vector>

namespace {

using namespace npp_color_twist_diff;

// Status-only error probe: single violation per case, no output capture (the
// harness only compares values when the golden status is a success code).
template <typename T>
void packedR_err(const char *fname, const std::string &id, const T *src, int srcStep, T *dst, int dstStep, int w, int h,
                 const Npp32f (*twist)[4],
                 NppStatus (*f)(const T *, int, T *, int, NppiSize, const Npp32f[3][4], NppStreamContext)) {
  NppiSize roi{w, h};
  clear_cuda_error();
  const NppStatus st = f(src, srcStep, dst, dstStep, roi, twist, current_context());
  Case c(id, fname);
  c.status(st);
  c.finish();
}

template <typename T>
void packedIR_err(const char *fname, const std::string &id, T *buf, int step, int w, int h,
                  const Npp32f (*twist)[4],
                  NppStatus (*f)(T *, int, NppiSize, const Npp32f[3][4], NppStreamContext)) {
  NppiSize roi{w, h};
  clear_cuda_error();
  const NppStatus st = f(buf, step, roi, twist, current_context());
  Case c(id, fname);
  c.status(st);
  c.finish();
}

template <typename T>
void planarP3_err(const char *fname, const std::string &id, const T *const *srcs, int srcStep, T *const *dsts,
                  int dstStep, int w, int h,
                  NppStatus (*f)(const T *const *, int, T *const *, int, NppiSize, const Npp32f[3][4], NppStreamContext)) {
  NppiSize roi{w, h};
  clear_cuda_error();
  const NppStatus st = f(srcs, srcStep, dsts, dstStep, roi, twist3(), current_context());
  Case c(id, fname);
  c.status(st);
  c.finish();
}

}  // namespace

class NppiColorTwistDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ---- happy paths, packed out-of-place -----------------------------------------
TEST_F(NppiColorTwistDiff, PackedR_C1R) {
  packedR_case<Npp32f>("nppiColorTwist_32f_C1R", "nppiColorTwist/C1R_17x13", 17, 13, 1, 16, twist3(),
                       [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize r) {
                         return nppiColorTwist_32f_C1R_Ctx(s, ss, d, ds, r, twist3(), current_context());
                       }, 101);
  packedR_case<Npp32f>("nppiColorTwist_32f_C1R", "nppiColorTwist/C1R_5x3", 5, 3, 1, 4, twist3(),
                       [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize r) {
                         return nppiColorTwist_32f_C1R_Ctx(s, ss, d, ds, r, twist3(), current_context());
                       }, 102);
}

TEST_F(NppiColorTwistDiff, PackedR_C2R) {
  packedR_case<Npp32f>("nppiColorTwist_32f_C2R", "nppiColorTwist/C2R_17x13", 17, 13, 2, 16, twist3(),
                       [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize r) {
                         return nppiColorTwist_32f_C2R_Ctx(s, ss, d, ds, r, twist3(), current_context());
                       }, 110);
}

TEST_F(NppiColorTwistDiff, PackedR_C3R) {
  packedR_case<Npp32f>("nppiColorTwist_32f_C3R", "nppiColorTwist/C3R_17x13", 17, 13, 3, 16, twist3(),
                       [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize r) {
                         return nppiColorTwist_32f_C3R_Ctx(s, ss, d, ds, r, twist3(), current_context());
                       }, 120);
  packedR_case<Npp32f>("nppiColorTwist_32f_C3R", "nppiColorTwist/C3R_1x1", 1, 1, 3, 16, twist3(),
                       [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize r) {
                         return nppiColorTwist_32f_C3R_Ctx(s, ss, d, ds, r, twist3(), current_context());
                       }, 121);
  packedR_case<Npp32f>("nppiColorTwist_32f_C3R", "nppiColorTwist/C3R_5x3", 5, 3, 3, 4, twist3(),
                       [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize r) {
                         return nppiColorTwist_32f_C3R_Ctx(s, ss, d, ds, r, twist3(), current_context());
                       }, 122);
}

TEST_F(NppiColorTwistDiff, PackedR_C4R) {
  packedR_case<Npp32f>("nppiColorTwist_32f_C4R", "nppiColorTwist/C4R_17x13", 17, 13, 4, 16, twist3(),
                       [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize r) {
                         return nppiColorTwist_32f_C4R_Ctx(s, ss, d, ds, r, twist3(), current_context());
                       }, 130);
}

TEST_F(NppiColorTwistDiff, PackedR_AC4R) {
  packedR_case<Npp32f>("nppiColorTwist_32f_AC4R", "nppiColorTwist/AC4R_17x13", 17, 13, 4, 16, twist3(),
                       [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize r) {
                         return nppiColorTwist_32f_AC4R_Ctx(s, ss, d, ds, r, twist3(), current_context());
                       }, 140);
}

// Boundary pixel values + big-coefficient twist: NaN/inf propagation and
// saturation behaviour of the 32f variant.
TEST_F(NppiColorTwistDiff, PackedR_C3R_Boundary) {
  packedR_case<Npp32f>("nppiColorTwist_32f_C3R", "nppiColorTwist/C3R_17x13_boundary", 17, 13, 3, 16, twist3(),
                       [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize r) {
                         return nppiColorTwist_32f_C3R_Ctx(s, ss, d, ds, r, twist3(), current_context());
                       }, 150, /*boundary=*/true);
  packedR_case<Npp32f>("nppiColorTwist_32f_C3R", "nppiColorTwist/C3R_17x13_big", 17, 13, 3, 16, twist3b(),
                       [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize r) {
                         return nppiColorTwist_32f_C3R_Ctx(s, ss, d, ds, r, twist3b(), current_context());
                       }, 151, /*boundary=*/true);
}

// ---- happy paths, packed in-place ----------------------------------------------
TEST_F(NppiColorTwistDiff, PackedIR_Variants) {
  packedIR_case<Npp32f>("nppiColorTwist_32f_C1IR", "nppiColorTwist/C1IR_9x7", 9, 7, 1, 8, twist3(),
                        [](Npp32f *b, int st, NppiSize r) { return nppiColorTwist_32f_C1IR_Ctx(b, st, r, twist3(), current_context()); },
                        200);
  packedIR_case<Npp32f>("nppiColorTwist_32f_C2IR", "nppiColorTwist/C2IR_9x7", 9, 7, 2, 8, twist3(),
                        [](Npp32f *b, int st, NppiSize r) { return nppiColorTwist_32f_C2IR_Ctx(b, st, r, twist3(), current_context()); },
                        201);
  packedIR_case<Npp32f>("nppiColorTwist_32f_C3IR", "nppiColorTwist/C3IR_9x7", 9, 7, 3, 8, twist3(),
                        [](Npp32f *b, int st, NppiSize r) { return nppiColorTwist_32f_C3IR_Ctx(b, st, r, twist3(), current_context()); },
                        202);
  packedIR_case<Npp32f>("nppiColorTwist_32f_C4IR", "nppiColorTwist/C4IR_9x7", 9, 7, 4, 8, twist3(),
                        [](Npp32f *b, int st, NppiSize r) { return nppiColorTwist_32f_C4IR_Ctx(b, st, r, twist3(), current_context()); },
                        203);
  packedIR_case<Npp32f>("nppiColorTwist_32f_AC4IR", "nppiColorTwist/AC4IR_9x7", 9, 7, 4, 8, twist3(),
                        [](Npp32f *b, int st, NppiSize r) { return nppiColorTwist_32f_AC4IR_Ctx(b, st, r, twist3(), current_context()); },
                        204);
  packedIR_case<Npp32f>("nppiColorTwist_32f_C3IR", "nppiColorTwist/C3IR_9x7_big", 9, 7, 3, 8, twist3b(),
                        [](Npp32f *b, int st, NppiSize r) { return nppiColorTwist_32f_C3IR_Ctx(b, st, r, twist3b(), current_context()); },
                        205, /*boundary=*/true);
}

// ---- happy paths, planar ---------------------------------------------------------
TEST_F(NppiColorTwistDiff, Planar_P3R) {
  planarP3_case<Npp32f>("nppiColorTwist_32f_P3R", "nppiColorTwist/P3R_17x13", 17, 13, 16, twist3(),
                        [](const Npp32f *const *s, int ss, Npp32f *const *d, int ds, NppiSize r) {
                          return nppiColorTwist_32f_P3R_Ctx(s, ss, d, ds, r, twist3(), current_context());
                        }, 300);
  planarP3_case<Npp32f>("nppiColorTwist_32f_P3R", "nppiColorTwist/P3R_1x1", 1, 1, 16, twist3(),
                        [](const Npp32f *const *s, int ss, Npp32f *const *d, int ds, NppiSize r) {
                          return nppiColorTwist_32f_P3R_Ctx(s, ss, d, ds, r, twist3(), current_context());
                        }, 301);
  planarP3_case<Npp32f>("nppiColorTwist_32f_P3R", "nppiColorTwist/P3R_17x13_big", 17, 13, 16, twist3b(),
                        [](const Npp32f *const *s, int ss, Npp32f *const *d, int ds, NppiSize r) {
                          return nppiColorTwist_32f_P3R_Ctx(s, ss, d, ds, r, twist3b(), current_context());
                        }, 302, /*boundary=*/true);
}

TEST_F(NppiColorTwistDiff, Planar_IP3R) {
  planarIP3_case<Npp32f>("nppiColorTwist_32f_IP3R", "nppiColorTwist/IP3R_9x7", 9, 7, 8, twist3(),
                         [](Npp32f *const *b, int st, NppiSize r) {
                           return nppiColorTwist_32f_IP3R_Ctx(b, st, r, twist3(), current_context());
                         }, 310);
}

// ---- error paths (one probe per TEST so a crashing reference call can be
// identified and dropped without blocking the remaining probes) ----------------

TEST_F(NppiColorTwistDiff, Err_C3R_NullSrc) {
  auto &s = arena<Npp32f>(64);
  auto &d = arena<Npp32f>(64);
  s.copyFromHost(make_uniform<Npp32f>(64, 400));
  d.copyFromHost(make_uniform<Npp32f>(64, 401));
  packedR_err<Npp32f>("nppiColorTwist_32f_C3R", "nppiColorTwist/err_null_src", nullptr, 64, d.get(), 64, 4, 4, twist3(), &nppiColorTwist_32f_C3R_Ctx);
}

TEST_F(NppiColorTwistDiff, Err_C3R_NullDst) {
  auto &s = arena<Npp32f>(64);
  auto &d = arena<Npp32f>(64);
  s.copyFromHost(make_uniform<Npp32f>(64, 400));
  d.copyFromHost(make_uniform<Npp32f>(64, 401));
  packedR_err<Npp32f>("nppiColorTwist_32f_C3R", "nppiColorTwist/err_null_dst", s.get(), 64, nullptr, 64, 4, 4, twist3(), &nppiColorTwist_32f_C3R_Ctx);
}

TEST_F(NppiColorTwistDiff, Err_C3R_SrcStepZero) {
  auto &s = arena<Npp32f>(64);
  auto &d = arena<Npp32f>(64);
  s.copyFromHost(make_uniform<Npp32f>(64, 400));
  d.copyFromHost(make_uniform<Npp32f>(64, 401));
  packedR_err<Npp32f>("nppiColorTwist_32f_C3R", "nppiColorTwist/err_src_step_zero", s.get(), 0, d.get(), 64, 4, 4, twist3(), &nppiColorTwist_32f_C3R_Ctx);
}

TEST_F(NppiColorTwistDiff, Err_C3R_SrcStepNegative) {
  auto &s = arena<Npp32f>(64);
  auto &d = arena<Npp32f>(64);
  s.copyFromHost(make_uniform<Npp32f>(64, 400));
  d.copyFromHost(make_uniform<Npp32f>(64, 401));
  packedR_err<Npp32f>("nppiColorTwist_32f_C3R", "nppiColorTwist/err_src_step_negative", s.get(), -48, d.get(), 64, 4, 4, twist3(), &nppiColorTwist_32f_C3R_Ctx);
}

TEST_F(NppiColorTwistDiff, Err_C3R_DstStepZero) {
  auto &s = arena<Npp32f>(64);
  auto &d = arena<Npp32f>(64);
  s.copyFromHost(make_uniform<Npp32f>(64, 400));
  d.copyFromHost(make_uniform<Npp32f>(64, 401));
  packedR_err<Npp32f>("nppiColorTwist_32f_C3R", "nppiColorTwist/err_dst_step_zero", s.get(), 64, d.get(), 0, 4, 4, twist3(), &nppiColorTwist_32f_C3R_Ctx);
}

TEST_F(NppiColorTwistDiff, Err_C3R_RoiZeroWidth) {
  auto &s = arena<Npp32f>(64);
  auto &d = arena<Npp32f>(64);
  s.copyFromHost(make_uniform<Npp32f>(64, 400));
  d.copyFromHost(make_uniform<Npp32f>(64, 401));
  packedR_err<Npp32f>("nppiColorTwist_32f_C3R", "nppiColorTwist/err_roi_zero_width", s.get(), 64, d.get(), 64, 0, 4, twist3(), &nppiColorTwist_32f_C3R_Ctx);
}

TEST_F(NppiColorTwistDiff, Err_C3R_RoiZeroHeight) {
  auto &s = arena<Npp32f>(64);
  auto &d = arena<Npp32f>(64);
  s.copyFromHost(make_uniform<Npp32f>(64, 400));
  d.copyFromHost(make_uniform<Npp32f>(64, 401));
  packedR_err<Npp32f>("nppiColorTwist_32f_C3R", "nppiColorTwist/err_roi_zero_height", s.get(), 64, d.get(), 64, 4, 0, twist3(), &nppiColorTwist_32f_C3R_Ctx);
}

TEST_F(NppiColorTwistDiff, Err_C3R_RoiNegativeWidth) {
  auto &s = arena<Npp32f>(64);
  auto &d = arena<Npp32f>(64);
  s.copyFromHost(make_uniform<Npp32f>(64, 400));
  d.copyFromHost(make_uniform<Npp32f>(64, 401));
  packedR_err<Npp32f>("nppiColorTwist_32f_C3R", "nppiColorTwist/err_roi_negative_width", s.get(), 64, d.get(), 64, -1, 4, twist3(), &nppiColorTwist_32f_C3R_Ctx);
}

TEST_F(NppiColorTwistDiff, Err_C3R_RoiNegativeHeight) {
  auto &s = arena<Npp32f>(64);
  auto &d = arena<Npp32f>(64);
  s.copyFromHost(make_uniform<Npp32f>(64, 400));
  d.copyFromHost(make_uniform<Npp32f>(64, 401));
  packedR_err<Npp32f>("nppiColorTwist_32f_C3R", "nppiColorTwist/err_roi_negative_height", s.get(), 64, d.get(), 64, 4, -1, twist3(), &nppiColorTwist_32f_C3R_Ctx);
}

TEST_F(NppiColorTwistDiff, Err_C3R_SrcStepTooSmall) {
  auto &s = arena<Npp32f>(64);
  auto &d = arena<Npp32f>(64);
  s.copyFromHost(make_uniform<Npp32f>(64, 400));
  d.copyFromHost(make_uniform<Npp32f>(64, 401));
  packedR_err<Npp32f>("nppiColorTwist_32f_C3R", "nppiColorTwist/err_src_step_too_small", s.get(), 4, d.get(), 64, 17, 4, twist3(), &nppiColorTwist_32f_C3R_Ctx);
}

TEST_F(NppiColorTwistDiff, Err_C3IR_Null) {
  auto &b = arena<Npp32f>(64);
  b.copyFromHost(make_uniform<Npp32f>(64, 410));
  packedIR_err<Npp32f>("nppiColorTwist_32f_C3IR", "nppiColorTwist/err_ir_null", nullptr, 64, 4, 4, twist3(), &nppiColorTwist_32f_C3IR_Ctx);
}

TEST_F(NppiColorTwistDiff, Err_C3IR_StepZero) {
  auto &b = arena<Npp32f>(64);
  b.copyFromHost(make_uniform<Npp32f>(64, 410));
  packedIR_err<Npp32f>("nppiColorTwist_32f_C3IR", "nppiColorTwist/err_ir_step_zero", b.get(), 0, 4, 4, twist3(), &nppiColorTwist_32f_C3IR_Ctx);
}

TEST_F(NppiColorTwistDiff, Err_C3IR_RoiZeroWidth) {
  auto &b = arena<Npp32f>(64);
  b.copyFromHost(make_uniform<Npp32f>(64, 410));
  packedIR_err<Npp32f>("nppiColorTwist_32f_C3IR", "nppiColorTwist/err_ir_roi_zero_width", b.get(), 64, 0, 4, twist3(), &nppiColorTwist_32f_C3IR_Ctx);
}

TEST_F(NppiColorTwistDiff, Err_C3IR_RoiNegative) {
  auto &b = arena<Npp32f>(64);
  b.copyFromHost(make_uniform<Npp32f>(64, 410));
  packedIR_err<Npp32f>("nppiColorTwist_32f_C3IR", "nppiColorTwist/err_ir_roi_negative", b.get(), 64, -1, 4, twist3(), &nppiColorTwist_32f_C3IR_Ctx);
}

TEST_F(NppiColorTwistDiff, Err_P3R_NullPlane) {
  auto &p0 = arena<Npp32f>(64);
  auto &p1 = arena<Npp32f>(64);
  auto &p2 = arena<Npp32f>(64);
  p0.copyFromHost(make_uniform<Npp32f>(64, 420));
  p1.copyFromHost(make_uniform<Npp32f>(64, 421));
  p2.copyFromHost(make_uniform<Npp32f>(64, 422));
  const Npp32f *srcs[3] = {p0.get(), p1.get(), p2.get()};
  Npp32f *dsts[3] = {p0.get(), p1.get(), p2.get()};
  const Npp32f *nulls[3] = {p0.get(), nullptr, p2.get()};
  planarP3_err<Npp32f>("nppiColorTwist_32f_P3R", "nppiColorTwist/err_p3_null_plane", nulls, 16, dsts, 16, 4, 4, &nppiColorTwist_32f_P3R_Ctx);
}

TEST_F(NppiColorTwistDiff, Err_P3R_StepZero) {
  auto &p0 = arena<Npp32f>(64);
  auto &p1 = arena<Npp32f>(64);
  auto &p2 = arena<Npp32f>(64);
  p0.copyFromHost(make_uniform<Npp32f>(64, 420));
  p1.copyFromHost(make_uniform<Npp32f>(64, 421));
  p2.copyFromHost(make_uniform<Npp32f>(64, 422));
  const Npp32f *srcs[3] = {p0.get(), p1.get(), p2.get()};
  Npp32f *dsts[3] = {p0.get(), p1.get(), p2.get()};
  const Npp32f *nulls[3] = {p0.get(), nullptr, p2.get()};
  planarP3_err<Npp32f>("nppiColorTwist_32f_P3R", "nppiColorTwist/err_p3_step_zero", srcs, 0, dsts, 16, 4, 4, &nppiColorTwist_32f_P3R_Ctx);
}

TEST_F(NppiColorTwistDiff, Err_P3R_RoiNegative) {
  auto &p0 = arena<Npp32f>(64);
  auto &p1 = arena<Npp32f>(64);
  auto &p2 = arena<Npp32f>(64);
  p0.copyFromHost(make_uniform<Npp32f>(64, 420));
  p1.copyFromHost(make_uniform<Npp32f>(64, 421));
  p2.copyFromHost(make_uniform<Npp32f>(64, 422));
  const Npp32f *srcs[3] = {p0.get(), p1.get(), p2.get()};
  Npp32f *dsts[3] = {p0.get(), p1.get(), p2.get()};
  const Npp32f *nulls[3] = {p0.get(), nullptr, p2.get()};
  planarP3_err<Npp32f>("nppiColorTwist_32f_P3R", "nppiColorTwist/err_p3_roi_negative", srcs, 16, dsts, 16, -1, 4, &nppiColorTwist_32f_P3R_Ctx);
}

TEST_F(NppiColorTwistDiff, Err_P3R_RoiZeroWidth) {
  auto &p0 = arena<Npp32f>(64);
  auto &p1 = arena<Npp32f>(64);
  auto &p2 = arena<Npp32f>(64);
  p0.copyFromHost(make_uniform<Npp32f>(64, 420));
  p1.copyFromHost(make_uniform<Npp32f>(64, 421));
  p2.copyFromHost(make_uniform<Npp32f>(64, 422));
  const Npp32f *srcs[3] = {p0.get(), p1.get(), p2.get()};
  Npp32f *dsts[3] = {p0.get(), p1.get(), p2.get()};
  const Npp32f *nulls[3] = {p0.get(), nullptr, p2.get()};
  planarP3_err<Npp32f>("nppiColorTwist_32f_P3R", "nppiColorTwist/err_p3_roi_zero_width", srcs, 16, dsts, 16, 0, 4, &nppiColorTwist_32f_P3R_Ctx);
}
// ---- non-Ctx adapter smoke test ----------------------------------------------------
#if NPP_DIFF_HAS_NONCTX
// Reproduces the exact host inputs of PackedR_C3R's 17x13 case (same id, same
// seed), so it shares that golden and exercises the adapter's stream handling.
TEST_F(NppiColorTwistDiff, NonCtx_C3R) {
  packedR_case<Npp32f>("nppiColorTwist_32f_C3R", "nppiColorTwist/C3R_17x13", 17, 13, 3, 16, twist3(),
                       [](const Npp32f *s, int ss, Npp32f *d, int ds, NppiSize r) {
                         return nppiColorTwist_32f_C3R(s, ss, d, ds, r, twist3());
                       }, 120);
}
#endif

// ---- round-2 probes: exact step-validation rules --------------------------------
// Status-only cases with explicit steps, to pin the minimum-step checks.
namespace {
void rawR_case(const std::string &id, const Npp32f *src, int srcStep, Npp32f *dst, int dstStep, int w, int h,
               bool capture) {
  NppiSize roi{w, h};
  clear_cuda_error();
  const NppStatus st = nppiColorTwist_32f_C3R_Ctx(src, srcStep, dst, dstStep, roi, twist3(), current_context());
  Case c(id, "nppiColorTwist_32f_C3R");
  c.status(st);
  if (capture) {
    c.device_output<Npp32f>(dst, 16 * 12, out_kind<Npp32f>(), 1e-5, 1e-6);
  }
  c.finish();
}
}  // namespace

TEST_F(NppiColorTwistDiff, Err2_C3R_StepRules) {
  auto &s = arena<Npp32f>(16 * 12);
  auto &d = arena<Npp32f>(16 * 12);
  s.copyFromHost(make_uniform<Npp32f>(16 * 12, 500));
  d.copyFromHost(make_uniform<Npp32f>(16 * 12, 501));
  // row bytes = 17*3*4 = 204
  rawR_case("nppiColorTwist/err2_dststep_exact", s.get(), 204, d.get(), 204, 17, 4, false);
  rawR_case("nppiColorTwist/err2_dststep_short", s.get(), 204, d.get(), 203, 17, 4, false);
  rawR_case("nppiColorTwist/err2_dststep_negative", s.get(), 204, d.get(), -204, 17, 4, false);
  rawR_case("nppiColorTwist/err2_srcstep_small_dst_valid", s.get(), 4, d.get(), 204, 17, 4, false);
  // srcStep = 0 with a valid dst: does the reference launch (dst written) or return early?
  d.copyFromHost(make_uniform<Npp32f>(16 * 12, 502));
  rawR_case("nppiColorTwist/err2_srcstep_zero_capture", s.get(), 0, d.get(), 204, 17, 4, true);
  // srcStep = 0 with a valid dst and a 4-pixel-wide ROI (row bytes = 48)
  rawR_case("nppiColorTwist/err2_srcstep_zero_capture_small", s.get(), 0, d.get(), 64, 4, 4, true);
}

namespace {
void rawIR_case(const std::string &id, Npp32f *buf, int step, int w, int h) {
  NppiSize roi{w, h};
  clear_cuda_error();
  const NppStatus st = nppiColorTwist_32f_C3IR_Ctx(buf, step, roi, twist3(), current_context());
  Case c(id, "nppiColorTwist_32f_C3IR");
  c.status(st);
  c.finish();
}
}  // namespace

TEST_F(NppiColorTwistDiff, Err2_C3IR_StepRules) {
  auto &b = arena<Npp32f>(16 * 12);
  b.copyFromHost(make_uniform<Npp32f>(16 * 12, 510));
  rawIR_case("nppiColorTwist/err2_ir_step_exact", b.get(), 48, 4, 4);
  rawIR_case("nppiColorTwist/err2_ir_step_small", b.get(), 4, 4, 4);
  rawIR_case("nppiColorTwist/err2_ir_step_negative", b.get(), -48, 4, 4);
}

namespace {
void rawP3_case(const std::string &id, const Npp32f *const *srcs, int srcStep, Npp32f *const *dsts, int dstStep, int w,
                int h) {
  NppiSize roi{w, h};
  clear_cuda_error();
  const NppStatus st =
      nppiColorTwist_32f_P3R_Ctx(srcs, srcStep, dsts, dstStep, roi, twist3(), current_context());
  Case c(id, "nppiColorTwist_32f_P3R");
  c.status(st);
  c.finish();
}
}  // namespace

TEST_F(NppiColorTwistDiff, Err2_P3R_StepRules) {
  auto &p0 = arena<Npp32f>(64);
  auto &p1 = arena<Npp32f>(64);
  auto &p2 = arena<Npp32f>(64);
  p0.copyFromHost(make_uniform<Npp32f>(64, 520));
  p1.copyFromHost(make_uniform<Npp32f>(64, 521));
  p2.copyFromHost(make_uniform<Npp32f>(64, 522));
  const Npp32f *srcs[3] = {p0.get(), p1.get(), p2.get()};
  Npp32f *dsts[3] = {p0.get(), p1.get(), p2.get()};
  // planar row bytes = 4*4 = 16
  rawP3_case("nppiColorTwist/err2_p3_dststep_zero", srcs, 16, dsts, 0, 4, 4);
  rawP3_case("nppiColorTwist/err2_p3_srcstep_small", srcs, 4, dsts, 16, 4, 4);
  rawP3_case("nppiColorTwist/err2_p3_dststep_small", srcs, 16, dsts, 4, 4, 4);
  rawP3_case("nppiColorTwist/err2_p3_steps_exact", srcs, 16, dsts, 16, 4, 4);
  rawP3_case("nppiColorTwist/err2_p3_srcstep_negative", srcs, -16, dsts, 16, 4, 4);
}

namespace {
void rawIP3_case(const std::string &id, Npp32f *const *bufs, int step, int w, int h) {
  NppiSize roi{w, h};
  clear_cuda_error();
  const NppStatus st = nppiColorTwist_32f_IP3R_Ctx(bufs, step, roi, twist3(), current_context());
  Case c(id, "nppiColorTwist_32f_IP3R");
  c.status(st);
  c.finish();
}
}  // namespace

TEST_F(NppiColorTwistDiff, Err2_IP3R_StepRules) {
  auto &p0 = arena<Npp32f>(64);
  auto &p1 = arena<Npp32f>(64);
  auto &p2 = arena<Npp32f>(64);
  p0.copyFromHost(make_uniform<Npp32f>(64, 530));
  p1.copyFromHost(make_uniform<Npp32f>(64, 531));
  p2.copyFromHost(make_uniform<Npp32f>(64, 532));
  Npp32f *bufs[3] = {p0.get(), p1.get(), p2.get()};
  rawIP3_case("nppiColorTwist/err2_ip3_step_zero", bufs, 0, 4, 4);
  rawIP3_case("nppiColorTwist/err2_ip3_step_small", bufs, 4, 4, 4);
  rawIP3_case("nppiColorTwist/err2_ip3_step_negative", bufs, -16, 4, 4);
  rawIP3_case("nppiColorTwist/err2_ip3_step_exact", bufs, 16, 4, 4);
}

namespace {
void rawC1R_case(const std::string &id, const Npp32f *src, int srcStep, Npp32f *dst, int dstStep, int w, int h) {
  NppiSize roi{w, h};
  clear_cuda_error();
  const NppStatus st = nppiColorTwist_32f_C1R_Ctx(src, srcStep, dst, dstStep, roi, twist3(), current_context());
  Case c(id, "nppiColorTwist_32f_C1R");
  c.status(st);
  c.finish();
}
}  // namespace

TEST_F(NppiColorTwistDiff, Err2_C1R_StepRules) {
  auto &s = arena<Npp32f>(64);
  auto &d = arena<Npp32f>(64);
  s.copyFromHost(make_uniform<Npp32f>(64, 540));
  d.copyFromHost(make_uniform<Npp32f>(64, 541));
  // C1 row bytes = 17*1*4 = 68
  rawC1R_case("nppiColorTwist/err2_c1_dststep_short", s.get(), 68, d.get(), 67, 17, 4);
  rawC1R_case("nppiColorTwist/err2_c1_dststep_negative", s.get(), 68, d.get(), -68, 17, 4);
  rawC1R_case("nppiColorTwist/err2_c1_srcstep_zero", s.get(), 0, d.get(), 68, 17, 4);
}

namespace {
// Even-step residue probes (status only).  The 32f packed C4/AC4 kernels are
// vectorized on full pixels, so a step that is not a multiple of the pixel
// size is rejected with NPP_NOT_EVEN_STEP_ERROR before any write; these
// probes pin the modulus per layout and the check order.
using Twist3 = const Npp32f (*)[4];
NppStatus raw3(const char *fname, const std::string &id, const Npp32f *src, int srcStep, Npp32f *dst, int dstStep,
               int w, int h, NppStatus (*f)(const Npp32f *, int, Npp32f *, int, NppiSize, Twist3, NppStreamContext)) {
  clear_cuda_error();
  const NppStatus st = f(src, srcStep, dst, dstStep, NppiSize{w, h}, twist3(), current_context());
  Case c(id, fname);
  c.status(st);
  c.finish();
  return st;
}
NppStatus raw3ir(const char *fname, const std::string &id, Npp32f *buf, int step, int w, int h,
                 NppStatus (*f)(Npp32f *, int, NppiSize, Twist3, NppStreamContext)) {
  clear_cuda_error();
  const NppStatus st = f(buf, step, NppiSize{w, h}, twist3(), current_context());
  Case c(id, fname);
  c.status(st);
  c.finish();
  return st;
}
NppStatus raw3p3(const char *fname, const std::string &id, Npp32f *const *srcs, int srcStep, Npp32f *const *dsts,
                 int dstStep, int w, int h,
                 NppStatus (*f)(const Npp32f *const *, int, Npp32f *const *, int, NppiSize, Twist3, NppStreamContext)) {
  clear_cuda_error();
  const NppStatus st = f(srcs, srcStep, dsts, dstStep, NppiSize{w, h}, twist3(), current_context());
  Case c(id, fname);
  c.status(st);
  c.finish();
  return st;
}
}  // namespace

TEST_F(NppiColorTwistDiff, Err3_EvenStep) {
  auto &s = arena<Npp32f>(4096);
  auto &d = arena<Npp32f>(4096);
  s.copyFromHost(make_uniform<Npp32f>(4096, 560));
  d.copyFromHost(make_uniform<Npp32f>(4096, 561));
  auto &b = arena<Npp32f>(4096);
  b.copyFromHost(make_uniform<Npp32f>(4096, 562));
  // w=17: C1 rowBytes=68, C2=136, C3=204, C4/AC4=272.
  raw3("nppiColorTwist_32f_C1R", "nppiColorTwist/err3_c1_dststep_70", s.get(), 68, d.get(), 70, 17, 4,
       &nppiColorTwist_32f_C1R_Ctx);
  raw3("nppiColorTwist_32f_C2R", "nppiColorTwist/err3_c2_dststep_140", s.get(), 136, d.get(), 140, 17, 4,
       &nppiColorTwist_32f_C2R_Ctx);
  raw3("nppiColorTwist_32f_C3R", "nppiColorTwist/err3_c3_dststep_206", s.get(), 204, d.get(), 206, 17, 4,
       &nppiColorTwist_32f_C3R_Ctx);
  raw3("nppiColorTwist_32f_C4R", "nppiColorTwist/err3_c4_dststep_280", s.get(), 272, d.get(), 280, 17, 4,
       &nppiColorTwist_32f_C4R_Ctx);
  raw3("nppiColorTwist_32f_C4R", "nppiColorTwist/err3_c4_dststep_276", s.get(), 272, d.get(), 276, 17, 4,
       &nppiColorTwist_32f_C4R_Ctx);
  raw3("nppiColorTwist_32f_AC4R", "nppiColorTwist/err3_ac4_dststep_280", s.get(), 272, d.get(), 280, 17, 4,
       &nppiColorTwist_32f_AC4R_Ctx);
  // src step odd (>= rowBytes), dst step even: NPP validates only the dst step
  // and launches (NPP_NO_ERROR, recorded at 17x4 in
  // artifacts/5090/npp_records/nppiColorTwistEvenStep).  The reference kernel
  // then does misaligned vectorized loads on rows >= 1 and the async
  // cudaErrorMisalignedAddress poisons every later test in the process, so
  // this probe uses height 1: the kernel stays on the aligned row 0 while the
  // host still proves the odd src step is accepted.
  raw3("nppiColorTwist_32f_C4R", "nppiColorTwist/err3_c4_srcstep_odd_dst_even", s.get(), 280, d.get(), 272, 17, 1,
       &nppiColorTwist_32f_C4R_Ctx);
  raw3("nppiColorTwist_32f_C4R", "nppiColorTwist/err3_c4_srcstep_even_dst_odd", s.get(), 272, d.get(), 280, 17, 4,
       &nppiColorTwist_32f_C4R_Ctx);
  raw3("nppiColorTwist_32f_C4R", "nppiColorTwist/err3_c4_dststep_negative_odd", s.get(), 272, d.get(), -280, 17, 4,
       &nppiColorTwist_32f_C4R_Ctx);
  raw3("nppiColorTwist_32f_C4R", "nppiColorTwist/err3_c4_nullsrc_oddstep", nullptr, 272, d.get(), 280, 17, 4,
       &nppiColorTwist_32f_C4R_Ctx);
  // w=5: C4 rowBytes=80 (multiple of 16 but not of 32).
  raw3("nppiColorTwist_32f_C4R", "nppiColorTwist/err3_c4_step80", s.get(), 80, d.get(), 80, 5, 4,
       &nppiColorTwist_32f_C4R_Ctx);
  // IR: step 136 < rowBytes 144 (w=9) and odd.
  raw3ir("nppiColorTwist_32f_C4IR", "nppiColorTwist/err3_c4ir_step_small_odd", b.get(), 136, 9, 4,
         &nppiColorTwist_32f_C4IR_Ctx);
  // Planar P3R: rowBytes=68 (w=17), step=70.
  Npp32f *srcs[3] = {s.get(), s.get(), s.get()};
  Npp32f *dsts[3] = {d.get(), d.get(), d.get()};
  raw3p3("nppiColorTwist_32f_P3R", "nppiColorTwist/err3_p3_dststep_70", srcs, 68, dsts, 70, 17, 4,
         &nppiColorTwist_32f_P3R_Ctx);
}

TEST_F(NppiColorTwistDiff, Err4_EvenStepOrder) {
  auto &s = arena<Npp32f>(4096);
  auto &d = arena<Npp32f>(4096);
  s.copyFromHost(make_uniform<Npp32f>(4096, 570));
  d.copyFromHost(make_uniform<Npp32f>(4096, 571));
  // SIZE vs EVEN: negative ROI with an odd dst step.
  raw3("nppiColorTwist_32f_C4R", "nppiColorTwist/err4_c4_roi_negative_oddstep", s.get(), 272, d.get(), 280, -17, 4,
       &nppiColorTwist_32f_C4R_Ctx);
  // empty ROI vs EVEN: zero width with an odd dst step.
  raw3("nppiColorTwist_32f_C4R", "nppiColorTwist/err4_c4_roi_zero_oddstep", s.get(), 272, d.get(), 280, 0, 4,
       &nppiColorTwist_32f_C4R_Ctx);
  // planar: src step odd, dst step even.
  Npp32f *srcs[3] = {s.get(), s.get(), s.get()};
  Npp32f *dsts[3] = {d.get(), d.get(), d.get()};
  raw3p3("nppiColorTwist_32f_P3R", "nppiColorTwist/err4_p3_srcstep_odd", srcs, 70, dsts, 68, 17, 4,
         &nppiColorTwist_32f_P3R_Ctx);
  // IR negative odd step.
  raw3ir("nppiColorTwist_32f_C4IR", "nppiColorTwist/err4_c4ir_step_negative_odd", s.get(), -280, 17, 4,
         &nppiColorTwist_32f_C4IR_Ctx);
}
