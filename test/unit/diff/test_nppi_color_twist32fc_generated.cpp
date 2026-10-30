// Differential suite for the "C" color-twist families (module
// nppi_color_conversion.h): nppiColorTwist32fC (8u/16f C4R/C4IR) and
// nppiColorTwist_32fC_C4R / _C4IR (Npp32f).  These apply a full 4x4 twist
// plus a 4-element additive constant vector to all four packed channels.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='TwistCDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=... ./diff_tests --gtest_filter='TwistCDiff.*'
//
// Probes:
//   * identity4c (identity 4x4 + fractional constants) pins the rounding
//     bias and float->int conversion of the "C" path per dtype
//   * twist4c (dyadic 4x4) pins the rounding of fractional accumulations and
//     that alpha is computed from the 4th matrix row (not copied)
//   * twist4cb (big coefficients) pins saturation (8u) / inf handling
//   * boundary fills include the type limits and NaN/inf (floats)
//
// NVIDIA NPP 13.4 crashes on a null aTwist (measured for the 3x4 families);
// a null aConstants is not probed for the same reason.
#include "npp_color_twist_diff.h"

#include "nppi_color_conversion.h"

#include <gtest/gtest.h>

class TwistCDiff : public ::testing::Test {};

namespace {

using namespace npp_color_twist_diff;

#define CTC_R(dtype, fn)                                                                        \
  [](const dtype *s, int ss, dtype *d, int ds, NppiSize r, const Npp32f(*m)[4], const Npp32f *c) { \
    return fn##_Ctx(s, ss, d, ds, r, m, c, current_context());                                  \
  }
#define CTC_IR(dtype, fn)                                                                       \
  [](dtype *b, int st, NppiSize r, const Npp32f(*m)[4], const Npp32f *c) {                      \
    return fn##_Ctx(b, st, r, m, c, current_context());                                         \
  }

auto c8u_C4 = CTC_R(Npp8u, nppiColorTwist32fC_8u_C4R);
auto c8u_C4IR = CTC_IR(Npp8u, nppiColorTwist32fC_8u_C4IR);
auto c16f_C4 = CTC_R(Npp16f, nppiColorTwist32fC_16f_C4R);
auto c16f_C4IR = CTC_IR(Npp16f, nppiColorTwist32fC_16f_C4IR);
auto c32f_C4 = CTC_R(Npp32f, nppiColorTwist_32fC_C4R);
auto c32f_C4IR = CTC_IR(Npp32f, nppiColorTwist_32fC_C4IR);

void matrices(Npp32f idm[4][4], Npp32f idc[4], Npp32f tm[4][4], Npp32f tc[4], Npp32f bm[4][4], Npp32f bc[4]) {
  identity4c(idm, idc);
  twist4c(tm, tc);
  twist4cb(bm, bc);
}

}  // namespace

TEST_F(TwistCDiff, TwistC_8u_Boundary) {
  Npp32f idm[4][4], idc[4], tm[4][4], tc[4], bm[4][4], bc[4];
  matrices(idm, idc, tm, tc, bm, bc);
  packedC_case<Npp8u>("nppiColorTwist32fC_8u_C4R", "ctc/8u_id_c4r", 8, 5, 4, idm, idc, c8u_C4, 751, true);
  packedC_case<Npp8u>("nppiColorTwist32fC_8u_C4R", "ctc/8u_t4_c4r", 8, 5, 4, tm, tc, c8u_C4, 752, true);
  packedC_case<Npp8u>("nppiColorTwist32fC_8u_C4R", "ctc/8u_sat_c4r", 8, 5, 4, bm, bc, c8u_C4, 753, true);
  packedC_case<Npp8u>("nppiColorTwist32fC_8u_C4R", "ctc/8u_h_c4r", 17, 13, 16, tm, tc, c8u_C4, 754);
  packedCIR_case<Npp8u>("nppiColorTwist32fC_8u_C4IR", "ctc/8u_id_c4ir", 8, 5, 4, idm, idc, c8u_C4IR, 755, true);
  packedCIR_case<Npp8u>("nppiColorTwist32fC_8u_C4IR", "ctc/8u_t4_c4ir", 8, 5, 4, tm, tc, c8u_C4IR, 756, true);
  packedCIR_case<Npp8u>("nppiColorTwist32fC_8u_C4IR", "ctc/8u_h_c4ir", 9, 7, 8, tm, tc, c8u_C4IR, 757);
}

TEST_F(TwistCDiff, TwistC_16f_Boundary) {
  Npp32f idm[4][4], idc[4], tm[4][4], tc[4], bm[4][4], bc[4];
  matrices(idm, idc, tm, tc, bm, bc);
  packedC_case<Npp16f>("nppiColorTwist32fC_16f_C4R", "ctc/16f_id_c4r", 8, 5, 8, idm, idc, c16f_C4, 761, true);
  packedC_case<Npp16f>("nppiColorTwist32fC_16f_C4R", "ctc/16f_t4_c4r", 8, 5, 8, tm, tc, c16f_C4, 762, true);
  packedC_case<Npp16f>("nppiColorTwist32fC_16f_C4R", "ctc/16f_sat_c4r", 8, 5, 8, bm, bc, c16f_C4, 763, true);
  packedC_case<Npp16f>("nppiColorTwist32fC_16f_C4R", "ctc/16f_h_c4r", 17, 13, 16, tm, tc, c16f_C4, 764);
  packedCIR_case<Npp16f>("nppiColorTwist32fC_16f_C4IR", "ctc/16f_id_c4ir", 8, 5, 8, idm, idc, c16f_C4IR, 765, true);
  packedCIR_case<Npp16f>("nppiColorTwist32fC_16f_C4IR", "ctc/16f_t4_c4ir", 8, 5, 8, tm, tc, c16f_C4IR, 766, true);
  packedCIR_case<Npp16f>("nppiColorTwist32fC_16f_C4IR", "ctc/16f_h_c4ir", 9, 7, 8, tm, tc, c16f_C4IR, 767);
}

TEST_F(TwistCDiff, TwistC_32f_Boundary) {
  Npp32f idm[4][4], idc[4], tm[4][4], tc[4], bm[4][4], bc[4];
  matrices(idm, idc, tm, tc, bm, bc);
  packedC_case<Npp32f>("nppiColorTwist_32fC_C4R", "ctc/32f_id_c4r", 8, 5, 8, idm, idc, c32f_C4, 771, true);
  packedC_case<Npp32f>("nppiColorTwist_32fC_C4R", "ctc/32f_t4_c4r", 8, 5, 8, tm, tc, c32f_C4, 772, true);
  packedC_case<Npp32f>("nppiColorTwist_32fC_C4R", "ctc/32f_sat_c4r", 8, 5, 8, bm, bc, c32f_C4, 773, true);
  packedC_case<Npp32f>("nppiColorTwist_32fC_C4R", "ctc/32f_h_c4r", 17, 13, 16, tm, tc, c32f_C4, 774);
  packedCIR_case<Npp32f>("nppiColorTwist_32fC_C4IR", "ctc/32f_id_c4ir", 8, 5, 8, idm, idc, c32f_C4IR, 775, true);
  packedCIR_case<Npp32f>("nppiColorTwist_32fC_C4IR", "ctc/32f_t4_c4ir", 8, 5, 8, tm, tc, c32f_C4IR, 776, true);
  packedCIR_case<Npp32f>("nppiColorTwist_32fC_C4IR", "ctc/32f_h_c4ir", 9, 7, 8, tm, tc, c32f_C4IR, 777);
}

namespace {
void err_case(const std::string &id, const char *fname, NppStatus st) {
  Case c(id, fname);
  c.status(st);
  c.finish();
}
}  // namespace

TEST_F(TwistCDiff, TwistC_Errors) {
  auto &s8 = arena<Npp8u>(512);
  auto &d8 = arena<Npp8u>(512);
  auto &b16 = arena<Npp16f>(512);
  auto &s32 = arena<Npp32f>(512);
  auto &d32 = arena<Npp32f>(512);
  s8.copyFromHost(make_uniform<Npp8u>(512, 791));
  d8.copyFromHost(make_uniform<Npp8u>(512, 792));
  b16.copyFromHost(make_uniform<Npp16f>(512, 793));
  s32.copyFromHost(make_uniform<Npp32f>(512, 794));
  d32.copyFromHost(make_uniform<Npp32f>(512, 795));
  Npp32f m[4][4], c[4];
  twist4c(m, c);
  // 8u C4R: rowBytes = 8*4*1 = 32.
  err_case("ctc/err_8u_c4r_nullsrc", "nppiColorTwist32fC_8u_C4R",
           nppiColorTwist32fC_8u_C4R_Ctx(nullptr, 32, d8.get(), 32, NppiSize{8, 4}, m, c, current_context()));
  err_case("ctc/err_8u_c4r_nulldst", "nppiColorTwist32fC_8u_C4R",
           nppiColorTwist32fC_8u_C4R_Ctx(s8.get(), 32, nullptr, 32, NppiSize{8, 4}, m, c, current_context()));
  err_case("ctc/err_8u_c4r_dststep_zero", "nppiColorTwist32fC_8u_C4R",
           nppiColorTwist32fC_8u_C4R_Ctx(s8.get(), 32, d8.get(), 0, NppiSize{8, 4}, m, c, current_context()));
  err_case("ctc/err_8u_c4r_dststep_short", "nppiColorTwist32fC_8u_C4R",
           nppiColorTwist32fC_8u_C4R_Ctx(s8.get(), 32, d8.get(), 31, NppiSize{8, 4}, m, c, current_context()));
  err_case("ctc/err_8u_c4r_dststep_negative", "nppiColorTwist32fC_8u_C4R",
           nppiColorTwist32fC_8u_C4R_Ctx(s8.get(), 32, d8.get(), -32, NppiSize{8, 4}, m, c, current_context()));
  err_case("ctc/err_8u_c4r_roi_negative", "nppiColorTwist32fC_8u_C4R",
           nppiColorTwist32fC_8u_C4R_Ctx(s8.get(), 32, d8.get(), 32, NppiSize{-8, 4}, m, c, current_context()));
  // 16f C4IR: rowBytes = 8*4*2 = 64.
  err_case("ctc/err_16f_c4ir_step_small", "nppiColorTwist32fC_16f_C4IR",
           nppiColorTwist32fC_16f_C4IR_Ctx(b16.get(), 63, NppiSize{8, 4}, m, c, current_context()));
  err_case("ctc/err_16f_c4ir_step_zero", "nppiColorTwist32fC_16f_C4IR",
           nppiColorTwist32fC_16f_C4IR_Ctx(b16.get(), 0, NppiSize{8, 4}, m, c, current_context()));
  // 32f C4R: rowBytes = 8*4*4 = 128.
  err_case("ctc/err_32f_c4r_dststep_zero", "nppiColorTwist_32fC_C4R",
           nppiColorTwist_32fC_C4R_Ctx(s32.get(), 128, d32.get(), 0, NppiSize{8, 4}, m, c, current_context()));
  // 32f C4R dststep=136: >= rowBytes but not a multiple of 16.
  err_case("ctc/err3_32f_c4r_dststep_odd16", "nppiColorTwist_32fC_C4R",
           nppiColorTwist_32fC_C4R_Ctx(s32.get(), 128, d32.get(), 136, NppiSize{8, 4}, m, c, current_context()));
  // 32f C4R dststep=-280 (negative, |odd| mod 16 = 8).
  err_case("ctc/err3_32f_c4r_dststep_negative_odd", "nppiColorTwist_32fC_C4R",
           nppiColorTwist_32fC_C4R_Ctx(s32.get(), 272, d32.get(), -280, NppiSize{17, 4}, m, c, current_context()));
  // order: null src vs odd dst step.
  err_case("ctc/err3_32f_c4r_nullsrc_oddstep", "nppiColorTwist_32fC_C4R",
           nppiColorTwist_32fC_C4R_Ctx(nullptr, 272, d32.get(), 280, NppiSize{17, 4}, m, c, current_context()));
  // src step odd (>= rowBytes), dst step even.
  err_case("ctc/err3_32f_c4r_srcstep_odd", "nppiColorTwist_32fC_C4R",
           nppiColorTwist_32fC_C4R_Ctx(s32.get(), 280, d32.get(), 272, NppiSize{17, 4}, m, c, current_context()));
  // src step even, dst step odd.
  err_case("ctc/err3_32f_c4r_dststep_odd", "nppiColorTwist_32fC_C4R",
           nppiColorTwist_32fC_C4R_Ctx(s32.get(), 272, d32.get(), 280, NppiSize{17, 4}, m, c, current_context()));
  // 16-vs-32 modulus: w=5 rowBytes=80, step=80 (mult. of 16, not of 32).
  err_case("ctc/err3_32f_c4r_step80", "nppiColorTwist_32fC_C4R",
           nppiColorTwist_32fC_C4R_Ctx(s32.get(), 80, d32.get(), 80, NppiSize{5, 4}, m, c, current_context()));
  // IR: step 136 < rowBytes 144 and odd: STEP_ERROR vs NOT_EVEN order.
  err_case("ctc/err3_32f_c4ir_step_small_odd", "nppiColorTwist_32fC_C4IR",
           nppiColorTwist_32fC_C4IR_Ctx(s32.get(), 136, NppiSize{9, 4}, m, c, current_context()));
  // 8u C4R: rowBytes=32, dststep=34 (not a multiple of 4).
  err_case("ctc/err3_8u_c4r_dststep_odd4", "nppiColorTwist32fC_8u_C4R",
           nppiColorTwist32fC_8u_C4R_Ctx(s8.get(), 32, d8.get(), 34, NppiSize{8, 4}, m, c, current_context()));
  // 16f C4R: rowBytes=64, dststep=68 (not a multiple of 8).
  err_case("ctc/err3_16f_c4r_dststep_odd8", "nppiColorTwist32fC_16f_C4R",
           nppiColorTwist32fC_16f_C4R_Ctx(b16.get(), 64, b16.get(), 68, NppiSize{8, 4}, m, c, current_context()));
}
