// Differential suite for the nppiColorTwist32f family (module
// nppi_color_conversion.h): the newly implemented 8s/16f variants plus
// coverage of the already-implemented 8u/16u/16s variants whose kernels
// were never golden-verified.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='Twist32fDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=... ./diff_tests --gtest_filter='Twist32fDiff.*'
//
// The identity twists discriminate the +0.5f bias / float->int (or
// float->half) conversion per dtype and layout; twist3 (dyadic) probes
// rounding; twist3b (big) probes saturation; boundary fills include the
// type limits (and NaN/inf for 16f).
//
// NVIDIA NPP 13.4 crashes on a null aTwist / null planar pointer array
// (no status code), so those probes are intentionally absent.
#include "npp_color_twist_diff.h"

#include "nppi_color_conversion.h"

#include <gtest/gtest.h>

// Minimal fixture (keeps the suite name in golden metadata stable).
class Twist32fDiff : public ::testing::Test {};

namespace {

using namespace npp_color_twist_diff;

// The case runners hand (src, step(s), dst, roi) to `call`; the twist the
// entry point receives is the same static matrix the runner records.
#define CT32F_R(dtype, fn, tw)                                                                  \
  [](const dtype *s, int ss, dtype *d, int ds, NppiSize r) {                                    \
    return fn##_Ctx(s, ss, d, ds, r, tw(), current_context());                                  \
  }
#define CT32F_IR(dtype, fn, tw)                                                                 \
  [](dtype *b, int st, NppiSize r) { return fn##_Ctx(b, st, r, tw(), current_context()); }
#define CT32F_P3(dtype, fn, tw)                                                                 \
  [](const dtype *const *s, int ss, dtype *const *d, int ds, NppiSize r) {                      \
    return fn##_Ctx(s, ss, d, ds, r, tw(), current_context());                                  \
  }
#define CT32F_IP3(dtype, fn, tw)                                                                \
  [](dtype *const *b, int st, NppiSize r) { return fn##_Ctx(b, st, r, tw(), current_context()); }

auto r8s_C1_id = CT32F_R(Npp8s, nppiColorTwist32f_8s_C1R, identity3);
auto r8s_C2_t3 = CT32F_R(Npp8s, nppiColorTwist32f_8s_C2R, twist3);
auto r8s_C3_id = CT32F_R(Npp8s, nppiColorTwist32f_8s_C3R, identity3);
auto r8s_C3_t3 = CT32F_R(Npp8s, nppiColorTwist32f_8s_C3R, twist3);
auto r8s_C3_t3b = CT32F_R(Npp8s, nppiColorTwist32f_8s_C3R, twist3b);
auto r8s_C4_id = CT32F_R(Npp8s, nppiColorTwist32f_8s_C4R, identity3);
auto r8s_C4_t3 = CT32F_R(Npp8s, nppiColorTwist32f_8s_C4R, twist3);
auto r8s_AC4_id = CT32F_R(Npp8s, nppiColorTwist32f_8s_AC4R, identity3);
auto r8s_AC4_t3 = CT32F_R(Npp8s, nppiColorTwist32f_8s_AC4R, twist3);
auto ir8s_C1_id = CT32F_IR(Npp8s, nppiColorTwist32f_8s_C1IR, identity3);
auto ir8s_C1_t3 = CT32F_IR(Npp8s, nppiColorTwist32f_8s_C1IR, twist3);
auto ir8s_C3_id = CT32F_IR(Npp8s, nppiColorTwist32f_8s_C3IR, identity3);
auto ir8s_C3_t3 = CT32F_IR(Npp8s, nppiColorTwist32f_8s_C3IR, twist3);
auto ir8s_AC4_t3b = CT32F_IR(Npp8s, nppiColorTwist32f_8s_AC4IR, twist3b);
auto ir8s_AC4_t3 = CT32F_IR(Npp8s, nppiColorTwist32f_8s_AC4IR, twist3);
auto p8s_P3_id = CT32F_P3(Npp8s, nppiColorTwist32f_8s_P3R, identity3);
auto p8s_P3_t3 = CT32F_P3(Npp8s, nppiColorTwist32f_8s_P3R, twist3);
auto p8s_P3_t3b = CT32F_P3(Npp8s, nppiColorTwist32f_8s_P3R, twist3b);
auto r8s_C1_t3 = CT32F_R(Npp8s, nppiColorTwist32f_8s_C1R, twist3);
auto ip8s_P3_id = CT32F_IP3(Npp8s, nppiColorTwist32f_8s_IP3R, identity3);
auto ip8s_P3_t3 = CT32F_IP3(Npp8s, nppiColorTwist32f_8s_IP3R, twist3);

auto r16f_C1_id = CT32F_R(Npp16f, nppiColorTwist32f_16f_C1R, identity3);
auto r16f_C1_t3 = CT32F_R(Npp16f, nppiColorTwist32f_16f_C1R, twist3);
auto r16f_C2_t3 = CT32F_R(Npp16f, nppiColorTwist32f_16f_C2R, twist3);
auto r16f_C3_id = CT32F_R(Npp16f, nppiColorTwist32f_16f_C3R, identity3);
auto r16f_C3_t3 = CT32F_R(Npp16f, nppiColorTwist32f_16f_C3R, twist3);
auto r16f_C4_id = CT32F_R(Npp16f, nppiColorTwist32f_16f_C4R, identity3);
auto r16f_C4_t3 = CT32F_R(Npp16f, nppiColorTwist32f_16f_C4R, twist3);
auto ir16f_C1_t3 = CT32F_IR(Npp16f, nppiColorTwist32f_16f_C1IR, twist3);
auto ir16f_C3_id = CT32F_IR(Npp16f, nppiColorTwist32f_16f_C3IR, identity3);
auto ir16f_C3_t3 = CT32F_IR(Npp16f, nppiColorTwist32f_16f_C3IR, twist3);
auto ir16f_C4_t3 = CT32F_IR(Npp16f, nppiColorTwist32f_16f_C4IR, twist3);

auto r8u_C1_id = CT32F_R(Npp8u, nppiColorTwist32f_8u_C1R, identity3);
auto r8u_C1_t3 = CT32F_R(Npp8u, nppiColorTwist32f_8u_C1R, twist3);
auto r8u_C3_id = CT32F_R(Npp8u, nppiColorTwist32f_8u_C3R, identity3);
auto r8u_C3_t3 = CT32F_R(Npp8u, nppiColorTwist32f_8u_C3R, twist3);
auto r8u_C3_t3b = CT32F_R(Npp8u, nppiColorTwist32f_8u_C3R, twist3b);
auto r8u_AC4_id = CT32F_R(Npp8u, nppiColorTwist32f_8u_AC4R, identity3);
auto r8u_AC4_t3 = CT32F_R(Npp8u, nppiColorTwist32f_8u_AC4R, twist3);
auto ir8u_C3_t3 = CT32F_IR(Npp8u, nppiColorTwist32f_8u_C3IR, twist3);
auto p8u_P3_t3 = CT32F_P3(Npp8u, nppiColorTwist32f_8u_P3R, twist3);
auto r16u_C1_t3 = CT32F_R(Npp16u, nppiColorTwist32f_16u_C1R, twist3);
auto r16u_C3_t3 = CT32F_R(Npp16u, nppiColorTwist32f_16u_C3R, twist3);
auto r16u_C3_t3b = CT32F_R(Npp16u, nppiColorTwist32f_16u_C3R, twist3b);
auto r16s_C1_id = CT32F_R(Npp16s, nppiColorTwist32f_16s_C1R, identity3);
auto r16s_C1_t3 = CT32F_R(Npp16s, nppiColorTwist32f_16s_C1R, twist3);
auto r16s_C3_id = CT32F_R(Npp16s, nppiColorTwist32f_16s_C3R, identity3);
auto r16s_C3_t3 = CT32F_R(Npp16s, nppiColorTwist32f_16s_C3R, twist3);
auto r16s_C3_t3b = CT32F_R(Npp16s, nppiColorTwist32f_16s_C3R, twist3b);
auto ir16s_C1_t3 = CT32F_IR(Npp16s, nppiColorTwist32f_16s_C1IR, twist3);
auto p16s_P3_t3 = CT32F_P3(Npp16s, nppiColorTwist32f_16s_P3R, twist3);

}  // namespace

TEST_F(Twist32fDiff, Twist32f_8s_Identity) {
  packedR_case<Npp8s>("nppiColorTwist32f_8s_C1R", "ct32f/8s_id_c1r", 8, 5, 1, 4, identity3(), r8s_C1_id, 601, true);
  packedR_case<Npp8s>("nppiColorTwist32f_8s_C3R", "ct32f/8s_id_c3r", 8, 5, 3, 4, identity3(), r8s_C3_id, 602, true);
  packedR_case<Npp8s>("nppiColorTwist32f_8s_C4R", "ct32f/8s_id_c4r", 8, 5, 4, 4, identity3(), r8s_C4_id, 603, true);
  packedR_case<Npp8s>("nppiColorTwist32f_8s_AC4R", "ct32f/8s_id_ac4r", 8, 5, 4, 4, identity3(), r8s_AC4_id, 604, true);
  packedIR_case<Npp8s>("nppiColorTwist32f_8s_C1IR", "ct32f/8s_id_c1ir", 8, 5, 1, 4, identity3(), ir8s_C1_id, 605, true);
  packedIR_case<Npp8s>("nppiColorTwist32f_8s_C3IR", "ct32f/8s_id_c3ir", 8, 5, 3, 4, identity3(), ir8s_C3_id, 606, true);
  planarP3_case<Npp8s>("nppiColorTwist32f_8s_P3R", "ct32f/8s_id_p3r", 8, 5, 4, identity3(), p8s_P3_id, 607, true);
  planarIP3_case<Npp8s>("nppiColorTwist32f_8s_IP3R", "ct32f/8s_id_ip3r", 8, 5, 4, identity3(), ip8s_P3_id, 608, true);
}

TEST_F(Twist32fDiff, Twist32f_8s_RoundingSaturation) {
  packedR_case<Npp8s>("nppiColorTwist32f_8s_C1R", "ct32f/8s_round_c1r", 8, 5, 1, 4, twist3(), r8s_C1_t3, 611, true);
  packedR_case<Npp8s>("nppiColorTwist32f_8s_C2R", "ct32f/8s_round_c2r", 8, 5, 2, 4, twist3(), r8s_C2_t3, 612, true);
  packedR_case<Npp8s>("nppiColorTwist32f_8s_C3R", "ct32f/8s_round_c3r", 8, 5, 3, 4, twist3(), r8s_C3_t3, 613, true);
  packedR_case<Npp8s>("nppiColorTwist32f_8s_C3R", "ct32f/8s_sat_c3r", 8, 5, 3, 4, twist3b(), r8s_C3_t3b, 614, true);
  packedR_case<Npp8s>("nppiColorTwist32f_8s_C4R", "ct32f/8s_round_c4r", 8, 5, 4, 4, twist3(), r8s_C4_t3, 615, true);
  packedIR_case<Npp8s>("nppiColorTwist32f_8s_AC4IR", "ct32f/8s_sat_ac4ir", 8, 5, 4, 4, twist3b(), ir8s_AC4_t3b, 616, true);
  planarP3_case<Npp8s>("nppiColorTwist32f_8s_P3R", "ct32f/8s_sat_p3r", 8, 5, 4, twist3b(), p8s_P3_t3b, 617, true);
}

TEST_F(Twist32fDiff, Twist32f_8s_Happy) {
  packedR_case<Npp8s>("nppiColorTwist32f_8s_C1R", "ct32f/8s_h_c1r", 17, 13, 1, 16, twist3(), r8s_C1_t3, 621);
  packedR_case<Npp8s>("nppiColorTwist32f_8s_C2R", "ct32f/8s_h_c2r", 17, 13, 2, 16, twist3(), r8s_C2_t3, 622);
  packedR_case<Npp8s>("nppiColorTwist32f_8s_C3R", "ct32f/8s_h_c3r", 17, 13, 3, 16, twist3(), r8s_C3_t3, 623);
  packedR_case<Npp8s>("nppiColorTwist32f_8s_C3R", "ct32f/8s_h_c3r_big", 5, 3, 3, 8, twist3b(), r8s_C3_t3b, 624);
  packedR_case<Npp8s>("nppiColorTwist32f_8s_C4R", "ct32f/8s_h_c4r", 17, 13, 4, 16, twist3(), r8s_C4_t3, 625);
  packedR_case<Npp8s>("nppiColorTwist32f_8s_AC4R", "ct32f/8s_h_ac4r", 17, 13, 4, 16, twist3(), r8s_AC4_t3, 626);
  packedIR_case<Npp8s>("nppiColorTwist32f_8s_C1IR", "ct32f/8s_h_c1ir", 9, 7, 1, 8, twist3(), ir8s_C1_t3, 627);
  packedIR_case<Npp8s>("nppiColorTwist32f_8s_C3IR", "ct32f/8s_h_c3ir", 9, 7, 3, 8, twist3(), ir8s_C3_t3, 628);
  packedIR_case<Npp8s>("nppiColorTwist32f_8s_AC4IR", "ct32f/8s_h_ac4ir", 9, 7, 4, 8, twist3(), ir8s_AC4_t3, 629);
  planarP3_case<Npp8s>("nppiColorTwist32f_8s_P3R", "ct32f/8s_h_p3r", 17, 13, 16, twist3(), p8s_P3_t3, 630);
  planarP3_case<Npp8s>("nppiColorTwist32f_8s_P3R", "ct32f/8s_h_p3r_1x1", 1, 1, 4, twist3(), p8s_P3_t3, 631);
  planarIP3_case<Npp8s>("nppiColorTwist32f_8s_IP3R", "ct32f/8s_h_ip3r", 9, 7, 8, twist3(), ip8s_P3_t3, 632);
}

TEST_F(Twist32fDiff, Twist32f_16f_Identity) {
  packedR_case<Npp16f>("nppiColorTwist32f_16f_C1R", "ct32f/16f_id_c1r", 8, 5, 1, 8, identity3(), r16f_C1_id, 651, true);
  packedR_case<Npp16f>("nppiColorTwist32f_16f_C3R", "ct32f/16f_id_c3r", 8, 5, 3, 8, identity3(), r16f_C3_id, 652, true);
  packedR_case<Npp16f>("nppiColorTwist32f_16f_C4R", "ct32f/16f_id_c4r", 8, 5, 4, 8, identity3(), r16f_C4_id, 653, true);
  packedIR_case<Npp16f>("nppiColorTwist32f_16f_C3IR", "ct32f/16f_id_c3ir", 8, 5, 3, 8, identity3(), ir16f_C3_id, 654, true);
}

TEST_F(Twist32fDiff, Twist32f_16f_RoundingBoundary) {
  packedR_case<Npp16f>("nppiColorTwist32f_16f_C1R", "ct32f/16f_round_c1r", 8, 5, 1, 8, twist3(), r16f_C1_t3, 661, true);
  packedR_case<Npp16f>("nppiColorTwist32f_16f_C2R", "ct32f/16f_round_c2r", 8, 5, 2, 8, twist3(), r16f_C2_t3, 662, true);
  packedR_case<Npp16f>("nppiColorTwist32f_16f_C3R", "ct32f/16f_round_c3r", 8, 5, 3, 8, twist3(), r16f_C3_t3, 663, true);
  packedR_case<Npp16f>("nppiColorTwist32f_16f_C4R", "ct32f/16f_round_c4r", 8, 5, 4, 8, twist3(), r16f_C4_t3, 664, true);
  packedIR_case<Npp16f>("nppiColorTwist32f_16f_C4IR", "ct32f/16f_round_c4ir", 8, 5, 4, 8, twist3(), ir16f_C4_t3, 665, true);
}

TEST_F(Twist32fDiff, Twist32f_16f_Happy) {
  packedR_case<Npp16f>("nppiColorTwist32f_16f_C1R", "ct32f/16f_h_c1r", 17, 13, 1, 16, twist3(), r16f_C1_t3, 671);
  packedR_case<Npp16f>("nppiColorTwist32f_16f_C2R", "ct32f/16f_h_c2r", 17, 13, 2, 16, twist3(), r16f_C2_t3, 672);
  packedR_case<Npp16f>("nppiColorTwist32f_16f_C3R", "ct32f/16f_h_c3r", 17, 13, 3, 16, twist3(), r16f_C3_t3, 673);
  packedR_case<Npp16f>("nppiColorTwist32f_16f_C4R", "ct32f/16f_h_c4r", 17, 13, 4, 16, twist3(), r16f_C4_t3, 674);
  packedIR_case<Npp16f>("nppiColorTwist32f_16f_C1IR", "ct32f/16f_h_c1ir", 9, 7, 1, 8, twist3(), ir16f_C1_t3, 675);
  packedIR_case<Npp16f>("nppiColorTwist32f_16f_C3IR", "ct32f/16f_h_c3ir", 9, 7, 3, 8, twist3(), ir16f_C3_t3, 676);
  packedIR_case<Npp16f>("nppiColorTwist32f_16f_C4IR", "ct32f/16f_h_c4ir", 9, 7, 4, 8, twist3(), ir16f_C4_t3, 677);
}

TEST_F(Twist32fDiff, Twist32f_Existing_Int_Happy) {
  packedR_case<Npp8u>("nppiColorTwist32f_8u_C1R", "ct32f/8u_h_c1r", 17, 13, 1, 16, twist3(), r8u_C1_t3, 701);
  packedR_case<Npp8u>("nppiColorTwist32f_8u_C3R", "ct32f/8u_h_c3r", 17, 13, 3, 16, twist3(), r8u_C3_t3, 702);
  packedR_case<Npp8u>("nppiColorTwist32f_8u_AC4R", "ct32f/8u_h_ac4r", 17, 13, 4, 16, twist3(), r8u_AC4_t3, 703);
  packedIR_case<Npp8u>("nppiColorTwist32f_8u_C3IR", "ct32f/8u_h_c3ir", 9, 7, 3, 8, twist3(), ir8u_C3_t3, 704);
  planarP3_case<Npp8u>("nppiColorTwist32f_8u_P3R", "ct32f/8u_h_p3r", 17, 13, 16, twist3(), p8u_P3_t3, 705);
  packedR_case<Npp16u>("nppiColorTwist32f_16u_C1R", "ct32f/16u_h_c1r", 17, 13, 1, 16, twist3(), r16u_C1_t3, 706);
  packedR_case<Npp16u>("nppiColorTwist32f_16u_C3R", "ct32f/16u_h_c3r", 17, 13, 3, 16, twist3(), r16u_C3_t3, 707);
  packedR_case<Npp16s>("nppiColorTwist32f_16s_C1R", "ct32f/16s_h_c1r", 17, 13, 1, 16, twist3(), r16s_C1_t3, 708);
  packedR_case<Npp16s>("nppiColorTwist32f_16s_C3R", "ct32f/16s_h_c3r", 17, 13, 3, 16, twist3(), r16s_C3_t3, 709);
  packedIR_case<Npp16s>("nppiColorTwist32f_16s_C1IR", "ct32f/16s_h_c1ir", 9, 7, 1, 8, twist3(), ir16s_C1_t3, 710);
  planarP3_case<Npp16s>("nppiColorTwist32f_16s_P3R", "ct32f/16s_h_p3r", 17, 13, 16, twist3(), p16s_P3_t3, 711);
}

TEST_F(Twist32fDiff, Twist32f_Existing_Int_Boundary) {
  packedR_case<Npp8u>("nppiColorTwist32f_8u_C1R", "ct32f/8u_id_c1r", 8, 5, 1, 4, identity3(), r8u_C1_id, 721, true);
  packedR_case<Npp8u>("nppiColorTwist32f_8u_C3R", "ct32f/8u_id_c3r", 8, 5, 3, 4, identity3(), r8u_C3_id, 722, true);
  packedR_case<Npp8u>("nppiColorTwist32f_8u_AC4R", "ct32f/8u_id_ac4r", 8, 5, 4, 4, identity3(), r8u_AC4_id, 723, true);
  packedR_case<Npp8u>("nppiColorTwist32f_8u_C3R", "ct32f/8u_sat_c3r", 8, 5, 3, 4, twist3b(), r8u_C3_t3b, 724, true);
  packedR_case<Npp16u>("nppiColorTwist32f_16u_C3R", "ct32f/16u_sat_c3r", 8, 5, 3, 8, twist3b(), r16u_C3_t3b, 725, true);
  packedR_case<Npp16u>("nppiColorTwist32f_16u_C1R", "ct32f/16u_round_c1r", 8, 5, 1, 8, twist3(), r16u_C1_t3, 726, true);
  packedR_case<Npp16s>("nppiColorTwist32f_16s_C3R", "ct32f/16s_sat_c3r", 8, 5, 3, 8, twist3b(), r16s_C3_t3b, 727, true);
  packedR_case<Npp16s>("nppiColorTwist32f_16s_C1R", "ct32f/16s_id_c1r", 8, 5, 1, 8, identity3(), r16s_C1_id, 728, true);
  packedR_case<Npp16s>("nppiColorTwist32f_16s_C3R", "ct32f/16s_id_c3r", 8, 5, 3, 8, identity3(), r16s_C3_id, 729, true);
}

namespace {
void err_case(const std::string &id, const char *fname, NppStatus st) {
  Case c(id, fname);
  c.status(st);
  c.finish();
}
}  // namespace

TEST_F(Twist32fDiff, Twist32f_Errors) {
  auto &s8 = arena<Npp8s>(256);
  auto &d8 = arena<Npp8s>(256);
  auto &b16 = arena<Npp16f>(256);
  s8.copyFromHost(make_uniform<Npp8s>(256, 801));
  d8.copyFromHost(make_uniform<Npp8s>(256, 802));
  b16.copyFromHost(make_uniform<Npp16f>(256, 803));
  const Npp32f(&tw)[3][4] = twist3();
  // 8s C3R: rowBytes = 8*3*1 = 24.
  err_case("ct32f/err_8s_c3r_nullsrc", "nppiColorTwist32f_8s_C3R",
           nppiColorTwist32f_8s_C3R_Ctx(nullptr, 24, d8.get(), 24, NppiSize{8, 4}, tw, current_context()));
  err_case("ct32f/err_8s_c3r_nulldst", "nppiColorTwist32f_8s_C3R",
           nppiColorTwist32f_8s_C3R_Ctx(s8.get(), 24, nullptr, 24, NppiSize{8, 4}, tw, current_context()));
  err_case("ct32f/err_8s_c3r_dststep_zero", "nppiColorTwist32f_8s_C3R",
           nppiColorTwist32f_8s_C3R_Ctx(s8.get(), 24, d8.get(), 0, NppiSize{8, 4}, tw, current_context()));
  err_case("ct32f/err_8s_c3r_dststep_short", "nppiColorTwist32f_8s_C3R",
           nppiColorTwist32f_8s_C3R_Ctx(s8.get(), 24, d8.get(), 23, NppiSize{8, 4}, tw, current_context()));
  err_case("ct32f/err_8s_c3r_roi_negative", "nppiColorTwist32f_8s_C3R",
           nppiColorTwist32f_8s_C3R_Ctx(s8.get(), 24, d8.get(), 24, NppiSize{-8, 4}, tw, current_context()));
  // 16f C3IR: rowBytes = 8*3*2 = 48.
  err_case("ct32f/err_16f_c3ir_step_small", "nppiColorTwist32f_16f_C3IR",
           nppiColorTwist32f_16f_C3IR_Ctx(b16.get(), 47, NppiSize{8, 4}, tw, current_context()));
  err_case("ct32f/err_16f_c3ir_step_zero", "nppiColorTwist32f_16f_C3IR",
           nppiColorTwist32f_16f_C3IR_Ctx(b16.get(), 0, NppiSize{8, 4}, tw, current_context()));
  // 8s P3R: planar rowBytes = 8.
  Npp8s *srcs[3] = {s8.get(), s8.get(), s8.get()};
  Npp8s *dsts[3] = {d8.get(), d8.get(), d8.get()};
  err_case("ct32f/err_8s_p3r_step_zero", "nppiColorTwist32f_8s_P3R",
           nppiColorTwist32f_8s_P3R_Ctx((const Npp8s *const *)srcs, 0, (Npp8s *const *)dsts, 8, NppiSize{8, 4}, tw,
                                        current_context()));
}
