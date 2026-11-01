// Differential suite for the batch color-twist families (module
// nppi_color_conversion.h): nppiColorTwistBatch (32f images), nppiColorTwist-
// Batch32f (8u / 16f images), nppiColorTwistBatch32fC and nppiColorTwistBatch_
// 32fC_C4R / _C4IR ("C" forms: 4x4 twist + constant column).  All of them take
// a device NppiColorTwistBatchCXR list (per-image buffers, steps and twist
// matrix), a shared ROI and the nMin/nMax clamp.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='TwistBatchDiff.*:TwistBatchCDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=... ./diff_tests --gtest_filter='TwistBatchDiff.*:TwistBatchCDiff.*'
//
// Measured NVIDIA 13.4 batch semantics pinned by these cases:
//   * pixel math: acc_c = sum_j m[c][j]*v_j + m[c][3] (3x4, the 4th column is
//     the bias term); the "C" forms use acc_c = sum_j m4[c][j]*v_j + cst_c on
//     all four channels including alpha.
//   * the +0.5f bias is added BEFORE the clamp: Npp32f on every layout (even
//     C1), Npp8u only for CH >= 3 (incl. the "C" forms), Npp16f never.
//   * the [nMin, nMax] clamp is comparison-based, so NaN survives it; a NaN
//     or +-inf input pixel poisons every output channel through the 0*inf
//     terms of the dot product (16f/32f boundary probes).
//   * Npp8u conversion is (Npp8u)__float2uint_rz(x): the LOW BYTE of the u32
//     result, with no [0,255] saturation (64*v + 8192 wraps); Npp16f is a
//     plain round-to-nearest fp16 conversion.
//   * the non-"C" C4/AC4 kernels never twist alpha: Npp8u/Npp32f leave the
//     destination alpha untouched (the prefill survives), Npp16f COPIES the
//     source alpha; the AC4 kernels also skip the [nMin, nMax] clamp on
//     output channel 2 (min side via 32f identity, max side via 8u twist3b)
//     and convert it with a saturating [0,255] u8 conversion instead.
//   * identity + wide clamp exposes the bias / dtype conversion, twist3 with
//     a fractional clamp exposes the bias-then-clamp-then-truncate order,
//     twist3b / twist4cb with a wide clamp expose the low-byte conversion.
//   * validation: null list / nBatchSize < 2 / negative ROI / empty ROI
//     (the latter returns NPP_CUDA_KERNEL_EXECUTION_ERROR - the degenerate
//     launch - without touching the entries).
// NVIDIA validates NOTHING about the per-entry steps/pointers: the steps used
// here are 16-byte aligned so the reference kernel never faults, and the null
// entry pointers / misaligned steps are NOT probed (they would poison the
// recorder with a sticky CUDA error).
#include "npp_color_twist_diff.h"

#include "nppi_color_conversion.h"

#include <gtest/gtest.h>

class TwistBatchDiff : public ::testing::Test {};
class TwistBatchCDiff : public ::testing::Test {};

namespace {

using namespace npp_color_twist_diff;

#define BATCH_FN(fn)                                                                                                    \
  [](Npp32f nMin, Npp32f nMax, NppiSize roi, NppiColorTwistBatchCXR *l, int n) {                                       \
    return fn##_Ctx(nMin, nMax, roi, l, n, current_context());                                                         \
  }

void matrices3(Npp32f id[3][4]) {
  const Npp32f(&i)[3][4] = identity3();
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c) {
      id[r][c] = i[r][c];
    }
  }
}

}  // namespace

// ---- nppiColorTwistBatch (Npp32f images, 3x4 twist) ----------------------------

TEST_F(TwistBatchDiff, Batch_32f_C1_C3) {
  Npp32f id[3][4];
  matrices3(id);
  const auto cid = coef3(id);
  const auto ct3 = coef3(twist3());
  const auto ctb = coef3(twist3b());
  batchR_case<Npp32f>("nppiColorTwistBatch_32f_C1R", "ctb/32f_c1r_id", 7, 3, 1, cid, ct3, -1e30f, 1e30f,
                      BATCH_FN(nppiColorTwistBatch_32f_C1R), 801);
  batchR_case<Npp32f>("nppiColorTwistBatch_32f_C1R", "ctb/32f_c1r_t3", 7, 3, 1, ct3, cid, 0.5f, 100.5f,
                      BATCH_FN(nppiColorTwistBatch_32f_C1R), 802);
  batchIR_case<Npp32f>("nppiColorTwistBatch_32f_C1IR", "ctb/32f_c1ir_id", 7, 3, 1, cid, ct3, 0.0f, 10.0f,
                       BATCH_FN(nppiColorTwistBatch_32f_C1IR), 803);
  batchR_case<Npp32f>("nppiColorTwistBatch_32f_C3R", "ctb/32f_c3r_id", 5, 3, 3, cid, ct3, -1e30f, 1e30f,
                      BATCH_FN(nppiColorTwistBatch_32f_C3R), 804);
  batchR_case<Npp32f>("nppiColorTwistBatch_32f_C3R", "ctb/32f_c3r_t3", 5, 3, 3, ct3, ctb, 0.5f, 100.5f,
                      BATCH_FN(nppiColorTwistBatch_32f_C3R), 805, true);
  batchIR_case<Npp32f>("nppiColorTwistBatch_32f_C3IR", "ctb/32f_c3ir_id", 5, 3, 3, cid, ct3, -1e30f, 1e30f,
                       BATCH_FN(nppiColorTwistBatch_32f_C3IR), 806);
  batchIR_case<Npp32f>("nppiColorTwistBatch_32f_C3IR", "ctb/32f_c3ir_t3", 5, 3, 3, ct3, cid, 0.5f, 100.5f,
                       BATCH_FN(nppiColorTwistBatch_32f_C3IR), 807);
}

TEST_F(TwistBatchDiff, Batch_32f_C4_AC4) {
  Npp32f id[3][4];
  matrices3(id);
  const auto cid = coef3(id);
  const auto ct3 = coef3(twist3());
  const auto ctb = coef3(twist3b());
  batchR_case<Npp32f>("nppiColorTwistBatch_32f_C4R", "ctb/32f_c4r_id", 5, 3, 4, cid, ct3, -1e30f, 1e30f,
                      BATCH_FN(nppiColorTwistBatch_32f_C4R), 811);
  batchR_case<Npp32f>("nppiColorTwistBatch_32f_C4R", "ctb/32f_c4r_t3", 5, 3, 4, ct3, ctb, 0.5f, 100.5f,
                      BATCH_FN(nppiColorTwistBatch_32f_C4R), 812, true);
  batchR_case<Npp32f>("nppiColorTwistBatch_32f_AC4R", "ctb/32f_ac4r_id", 5, 3, 4, cid, ct3, -1e30f, 1e30f,
                      BATCH_FN(nppiColorTwistBatch_32f_AC4R), 813);
  batchR_case<Npp32f>("nppiColorTwistBatch_32f_AC4R", "ctb/32f_ac4r_t3", 5, 3, 4, ct3, cid, 0.5f, 100.5f,
                      BATCH_FN(nppiColorTwistBatch_32f_AC4R), 814);
  batchIR_case<Npp32f>("nppiColorTwistBatch_32f_C4IR", "ctb/32f_c4ir_id", 5, 3, 4, cid, ct3, -1e30f, 1e30f,
                       BATCH_FN(nppiColorTwistBatch_32f_C4IR), 815);
  batchIR_case<Npp32f>("nppiColorTwistBatch_32f_C4IR", "ctb/32f_c4ir_t3", 5, 3, 4, ct3, cid, 0.5f, 100.5f,
                       BATCH_FN(nppiColorTwistBatch_32f_C4IR), 816);
  batchIR_case<Npp32f>("nppiColorTwistBatch_32f_AC4IR", "ctb/32f_ac4ir_t3", 5, 3, 4, ct3, ctb, 0.5f, 100.5f,
                       BATCH_FN(nppiColorTwistBatch_32f_AC4IR), 817);
}

// ---- nppiColorTwistBatch32f (Npp8u images) -------------------------------------

TEST_F(TwistBatchDiff, Batch_8u_C1_C3) {
  Npp32f id[3][4];
  matrices3(id);
  const auto cid = coef3(id);
  const auto ct3 = coef3(twist3());
  const auto ctb = coef3(twist3b());
  batchR_case<Npp8u>("nppiColorTwistBatch32f_8u_C1R", "ctb/8u_c1r_t3", 9, 3, 1, ct3, cid, 0.5f, 100.5f,
                     BATCH_FN(nppiColorTwistBatch32f_8u_C1R), 821);
  batchR_case<Npp8u>("nppiColorTwistBatch32f_8u_C1R", "ctb/8u_c1r_tb", 9, 3, 1, ctb, ct3, -1e30f, 1e30f,
                     BATCH_FN(nppiColorTwistBatch32f_8u_C1R), 822);
  batchIR_case<Npp8u>("nppiColorTwistBatch32f_8u_C1IR", "ctb/8u_c1ir_t3", 9, 3, 1, ct3, ctb, 0.5f, 100.5f,
                      BATCH_FN(nppiColorTwistBatch32f_8u_C1IR), 823);
  batchR_case<Npp8u>("nppiColorTwistBatch32f_8u_C3R", "ctb/8u_c3r_id", 5, 3, 3, cid, ct3, -1e30f, 1e30f,
                     BATCH_FN(nppiColorTwistBatch32f_8u_C3R), 824);
  batchR_case<Npp8u>("nppiColorTwistBatch32f_8u_C3R", "ctb/8u_c3r_t3", 5, 3, 3, ct3, cid, 0.5f, 100.5f,
                     BATCH_FN(nppiColorTwistBatch32f_8u_C3R), 825, true);
  batchR_case<Npp8u>("nppiColorTwistBatch32f_8u_C3R", "ctb/8u_c3r_tb", 5, 3, 3, ctb, ct3, -1e30f, 1e30f,
                     BATCH_FN(nppiColorTwistBatch32f_8u_C3R), 826);
  batchIR_case<Npp8u>("nppiColorTwistBatch32f_8u_C3IR", "ctb/8u_c3ir_id", 5, 3, 3, cid, ct3, -1e30f, 1e30f,
                      BATCH_FN(nppiColorTwistBatch32f_8u_C3IR), 827);
  batchIR_case<Npp8u>("nppiColorTwistBatch32f_8u_C3IR", "ctb/8u_c3ir_tb", 5, 3, 3, ctb, ct3, -1e30f, 1e30f,
                      BATCH_FN(nppiColorTwistBatch32f_8u_C3IR), 828);
}

TEST_F(TwistBatchDiff, Batch_8u_C4_AC4) {
  Npp32f id[3][4];
  matrices3(id);
  const auto cid = coef3(id);
  const auto ct3 = coef3(twist3());
  const auto ctb = coef3(twist3b());
  batchR_case<Npp8u>("nppiColorTwistBatch32f_8u_C4R", "ctb/8u_c4r_id", 5, 3, 4, cid, ct3, -1e30f, 1e30f,
                     BATCH_FN(nppiColorTwistBatch32f_8u_C4R), 831);
  batchR_case<Npp8u>("nppiColorTwistBatch32f_8u_C4R", "ctb/8u_c4r_t3", 5, 3, 4, ct3, cid, 0.5f, 100.5f,
                     BATCH_FN(nppiColorTwistBatch32f_8u_C4R), 832, true);
  batchR_case<Npp8u>("nppiColorTwistBatch32f_8u_AC4R", "ctb/8u_ac4r_t3", 5, 3, 4, ct3, ctb, 0.5f, 100.5f,
                     BATCH_FN(nppiColorTwistBatch32f_8u_AC4R), 833);
  batchIR_case<Npp8u>("nppiColorTwistBatch32f_8u_C4IR", "ctb/8u_c4ir_id", 5, 3, 4, cid, ct3, -1e30f, 1e30f,
                      BATCH_FN(nppiColorTwistBatch32f_8u_C4IR), 834);
  batchIR_case<Npp8u>("nppiColorTwistBatch32f_8u_C4IR", "ctb/8u_c4ir_t3", 5, 3, 4, ct3, cid, 0.5f, 100.5f,
                      BATCH_FN(nppiColorTwistBatch32f_8u_C4IR), 835);
  batchIR_case<Npp8u>("nppiColorTwistBatch32f_8u_AC4IR", "ctb/8u_ac4ir_t3", 5, 3, 4, ct3, ctb, 0.5f, 100.5f,
                      BATCH_FN(nppiColorTwistBatch32f_8u_AC4IR), 836);
}

// ---- nppiColorTwistBatch32f (Npp16f images) ------------------------------------

TEST_F(TwistBatchDiff, Batch_16f) {
  Npp32f id[3][4];
  matrices3(id);
  const auto cid = coef3(id);
  const auto ct3 = coef3(twist3());
  const auto ctb = coef3(twist3b());
  batchR_case<Npp16f>("nppiColorTwistBatch32f_16f_C1R", "ctb/16f_c1r_id", 7, 3, 1, cid, ct3, -1e30f, 1e30f,
                      BATCH_FN(nppiColorTwistBatch32f_16f_C1R), 841);
  batchR_case<Npp16f>("nppiColorTwistBatch32f_16f_C1R", "ctb/16f_c1r_t3", 7, 3, 1, ct3, ctb, 0.5f, 100.5f,
                      BATCH_FN(nppiColorTwistBatch32f_16f_C1R), 842);
  batchIR_case<Npp16f>("nppiColorTwistBatch32f_16f_C1IR", "ctb/16f_c1ir_t3", 7, 3, 1, ct3, cid, 0.5f, 100.5f,
                       BATCH_FN(nppiColorTwistBatch32f_16f_C1IR), 843);
  batchR_case<Npp16f>("nppiColorTwistBatch32f_16f_C3R", "ctb/16f_c3r_id", 5, 3, 3, cid, ct3, -1e30f, 1e30f,
                      BATCH_FN(nppiColorTwistBatch32f_16f_C3R), 844, true);
  batchR_case<Npp16f>("nppiColorTwistBatch32f_16f_C3R", "ctb/16f_c3r_t3", 5, 3, 3, ct3, cid, 0.5f, 100.5f,
                      BATCH_FN(nppiColorTwistBatch32f_16f_C3R), 845);
  batchIR_case<Npp16f>("nppiColorTwistBatch32f_16f_C3IR", "ctb/16f_c3ir_id", 5, 3, 3, cid, ct3, -1e30f, 1e30f,
                       BATCH_FN(nppiColorTwistBatch32f_16f_C3IR), 846);
  batchR_case<Npp16f>("nppiColorTwistBatch32f_16f_C4R", "ctb/16f_c4r_id", 5, 3, 4, cid, ct3, -1e30f, 1e30f,
                      BATCH_FN(nppiColorTwistBatch32f_16f_C4R), 847);
  batchR_case<Npp16f>("nppiColorTwistBatch32f_16f_C4R", "ctb/16f_c4r_t3", 5, 3, 4, ct3, ctb, 0.5f, 100.5f,
                      BATCH_FN(nppiColorTwistBatch32f_16f_C4R), 848);
  batchIR_case<Npp16f>("nppiColorTwistBatch32f_16f_C4IR", "ctb/16f_c4ir_t3", 5, 3, 4, ct3, cid, 0.5f, 100.5f,
                       BATCH_FN(nppiColorTwistBatch32f_16f_C4IR), 849);
}

// ---- batch error probes ---------------------------------------------------------
// One probe per TEST so a crashing reference call can be identified and dropped
// without blocking the remaining probes.  All steps stay 16-byte aligned: the
// reference kernel validates nothing about the per-entry steps/pointers, so a
// misaligned step would poison the recorder with a sticky CUDA error.

namespace {
NppiColorTwistBatchCXR *err_list() {
  auto &s0 = arena<Npp8u>(512);
  auto &d0 = arena<Npp8u>(512);
  auto &s1 = arena<Npp8u>(512);
  auto &d1 = arena<Npp8u>(512);
  auto &t = arena<Npp32f>(12);
  s0.copyFromHost(make_uniform<Npp8u>(512, 861));
  d0.copyFromHost(make_uniform<Npp8u>(512, 862));
  s1.copyFromHost(make_uniform<Npp8u>(512, 863));
  d1.copyFromHost(make_uniform<Npp8u>(512, 864));
  t.copyFromHost(coef3(twist3()));
  return batch_list(s0, d0, s1, d1, t, 32);
}

void err_case(const std::string &id, const char *fname, NppStatus st) {
  Case c(id, fname);
  c.status(st);
  c.finish();
}
}  // namespace

TEST_F(TwistBatchDiff, Err_NullList) {
  err_case("ctb/err_null_list", "nppiColorTwistBatch32f_8u_C1R",
           nppiColorTwistBatch32f_8u_C1R_Ctx(0.5f, 100.5f, NppiSize{2, 2}, nullptr, 2, current_context()));
}

TEST_F(TwistBatchDiff, Err_Size0) {
  err_case("ctb/err_size0", "nppiColorTwistBatch32f_8u_C1R",
           nppiColorTwistBatch32f_8u_C1R_Ctx(0.5f, 100.5f, NppiSize{2, 2}, err_list(), 0, current_context()));
}

TEST_F(TwistBatchDiff, Err_Size1) {
  err_case("ctb/err_size1", "nppiColorTwistBatch32f_8u_C1R",
           nppiColorTwistBatch32f_8u_C1R_Ctx(0.5f, 100.5f, NppiSize{2, 2}, err_list(), 1, current_context()));
}

TEST_F(TwistBatchDiff, Err_SizeNeg) {
  err_case("ctb/err_size_neg", "nppiColorTwistBatch32f_8u_C1R",
           nppiColorTwistBatch32f_8u_C1R_Ctx(0.5f, 100.5f, NppiSize{2, 2}, err_list(), -3, current_context()));
}

TEST_F(TwistBatchDiff, Err_NullAndSize0) {
  err_case("ctb/err_null_and_size0", "nppiColorTwistBatch32f_8u_C1R",
           nppiColorTwistBatch32f_8u_C1R_Ctx(0.5f, 100.5f, NppiSize{2, 2}, nullptr, 0, current_context()));
}

TEST_F(TwistBatchDiff, Err_Size0AndNegRoi) {
  err_case("ctb/err_size0_and_negroi", "nppiColorTwistBatch32f_8u_C1R",
           nppiColorTwistBatch32f_8u_C1R_Ctx(0.5f, 100.5f, NppiSize{-1, 2}, err_list(), 0, current_context()));
}

TEST_F(TwistBatchDiff, Err_NegRoi) {
  err_case("ctb/err_negroi", "nppiColorTwistBatch32f_8u_C1R",
           nppiColorTwistBatch32f_8u_C1R_Ctx(0.5f, 100.5f, NppiSize{-1, 2}, err_list(), 2, current_context()));
}

TEST_F(TwistBatchDiff, Err_EmptyWidth) {
  err_case("ctb/err_empty_w", "nppiColorTwistBatch32f_8u_C1R",
           nppiColorTwistBatch32f_8u_C1R_Ctx(0.5f, 100.5f, NppiSize{0, 2}, err_list(), 2, current_context()));
}

TEST_F(TwistBatchDiff, Err_EmptyHeight) {
  err_case("ctb/err_empty_h", "nppiColorTwistBatch32f_8u_C1R",
           nppiColorTwistBatch32f_8u_C1R_Ctx(0.5f, 100.5f, NppiSize{2, 0}, err_list(), 2, current_context()));
}

TEST_F(TwistBatchDiff, Err_EmptyRoi32f) {
  err_case("ctb/err_empty_32f", "nppiColorTwistBatch_32f_C1R",
           nppiColorTwistBatch_32f_C1R_Ctx(0.5f, 100.5f, NppiSize{0, 2}, err_list(), 2, current_context()));
}

TEST_F(TwistBatchDiff, Err_Size1C) {
  err_case("ctb/err_size1_c", "nppiColorTwistBatch32fC_8u_C4R",
           nppiColorTwistBatch32fC_8u_C4R_Ctx(0.5f, 100.5f, NppiSize{2, 2}, err_list(), 1, current_context()));
}

TEST_F(TwistBatchDiff, Err_EmptyRoiC) {
  err_case("ctb/err_empty_c", "nppiColorTwistBatch32fC_8u_C4R",
           nppiColorTwistBatch32fC_8u_C4R_Ctx(0.5f, 100.5f, NppiSize{0, 2}, err_list(), 2, current_context()));
}

// ---- extra clamp / dtype-range discriminators -----------------------------------
// These separate semantics the identity/twist3/twist3b cases above cannot:
//   * a tight clamp on C1 (clamped output vs unclamped low byte / fp16 value)
//   * big coefficients on C4 / the "C" forms with a wide clamp (low byte of
//     the u32 conversion on every channel, alpha handling)
//   * the AC4 ch2 clamp exemption on the max side (there is no 16f AC4 batch
//     variant in the API, so 8u / 32f carry the whole measurement).

TEST_F(TwistBatchDiff, Batch_8u_Clamps) {
  const auto ct3 = coef3(twist3());
  const auto ctb = coef3(twist3b());
  batchR_case<Npp8u>("nppiColorTwistBatch32f_8u_C1R", "ctb/8u_c1r_tbc", 9, 3, 1, ctb, ct3, 0.5f, 100.5f,
                     BATCH_FN(nppiColorTwistBatch32f_8u_C1R), 881);
  batchR_case<Npp8u>("nppiColorTwistBatch32f_8u_C4R", "ctb/8u_c4r_tb", 5, 3, 4, ctb, ct3, -1e30f, 1e30f,
                     BATCH_FN(nppiColorTwistBatch32f_8u_C4R), 883);
}

TEST_F(TwistBatchDiff, Batch_16f_Clamps) {
  const auto ct3 = coef3(twist3());
  const auto ctb = coef3(twist3b());
  batchR_case<Npp16f>("nppiColorTwistBatch32f_16f_C1R", "ctb/16f_c1r_tb", 7, 3, 1, ctb, ct3, 0.5f, 100.5f,
                      BATCH_FN(nppiColorTwistBatch32f_16f_C1R), 882);
  batchR_case<Npp16f>("nppiColorTwistBatch32f_16f_C4R", "ctb/16f_c4r_t3c", 5, 3, 4, ct3, ctb, 0.5f, 100.5f,
                      BATCH_FN(nppiColorTwistBatch32f_16f_C4R), 884);
}

TEST_F(TwistBatchDiff, Batch_32f_AC4_Clamp) {
  const auto ct3 = coef3(twist3());
  const auto ctb = coef3(twist3b());
  batchR_case<Npp32f>("nppiColorTwistBatch_32f_AC4R", "ctb/32f_ac4r_tb", 5, 3, 4, ctb, ct3, 0.5f, 100.5f,
                      BATCH_FN(nppiColorTwistBatch_32f_AC4R), 885);
}

// ---- nppiColorTwistBatch32fC / nppiColorTwistBatch_32fC ("C" forms) ------------

TEST_F(TwistBatchCDiff, BatchC_Values) {
  Npp32f idm[4][4], idc[4], tm[4][4], tc[4], bm[4][4], bc[4];
  identity4c(idm, idc);
  twist4c(tm, tc);
  twist4cb(bm, bc);
  const auto cid = coef45(idm, idc);
  const auto ct4 = coef45(tm, tc);
  const auto ctb = coef45(bm, bc);
  batchR_case<Npp8u>("nppiColorTwistBatch32fC_8u_C4R", "ctbc/8u_id", 5, 3, 4, cid, ct4, -1e30f, 1e30f,
                     BATCH_FN(nppiColorTwistBatch32fC_8u_C4R), 871, true);
  batchR_case<Npp8u>("nppiColorTwistBatch32fC_8u_C4R", "ctbc/8u_t4", 5, 3, 4, ct4, ctb, 0.5f, 100.5f,
                     BATCH_FN(nppiColorTwistBatch32fC_8u_C4R), 872);
  batchIR_case<Npp8u>("nppiColorTwistBatch32fC_8u_C4IR", "ctbc/8u_t4ir", 5, 3, 4, ct4, ctb, 0.5f, 100.5f,
                      BATCH_FN(nppiColorTwistBatch32fC_8u_C4IR), 873);
  batchR_case<Npp16f>("nppiColorTwistBatch32fC_16f_C4R", "ctbc/16f_id", 5, 3, 4, cid, ct4, -1e30f, 1e30f,
                      BATCH_FN(nppiColorTwistBatch32fC_16f_C4R), 874, true);
  batchR_case<Npp16f>("nppiColorTwistBatch32fC_16f_C4R", "ctbc/16f_t4", 5, 3, 4, ct4, ctb, 0.5f, 100.5f,
                      BATCH_FN(nppiColorTwistBatch32fC_16f_C4R), 875);
  batchIR_case<Npp16f>("nppiColorTwistBatch32fC_16f_C4IR", "ctbc/16f_idir", 5, 3, 4, cid, ct4, -1e30f, 1e30f,
                      BATCH_FN(nppiColorTwistBatch32fC_16f_C4IR), 876);
  batchR_case<Npp32f>("nppiColorTwistBatch_32fC_C4R", "ctbc/32f_id", 5, 3, 4, cid, ct4, -1e30f, 1e30f,
                      BATCH_FN(nppiColorTwistBatch_32fC_C4R), 877);
  batchR_case<Npp32f>("nppiColorTwistBatch_32fC_C4R", "ctbc/32f_t4", 5, 3, 4, ct4, ctb, 0.5f, 100.5f,
                      BATCH_FN(nppiColorTwistBatch_32fC_C4R), 878);
  batchIR_case<Npp32f>("nppiColorTwistBatch_32fC_C4IR", "ctbc/32f_idir", 5, 3, 4, cid, ct4, 0.0f, 10.0f,
                       BATCH_FN(nppiColorTwistBatch_32fC_C4IR), 879);
  // Big 4x4 + big constants with a wide clamp: every channel (including the
  // twisted alpha) must take the low byte of the u32 conversion.
  batchR_case<Npp8u>("nppiColorTwistBatch32fC_8u_C4R", "ctbc/8u_tcb", 5, 3, 4, ctb, ct4, -1e30f, 1e30f,
                     BATCH_FN(nppiColorTwistBatch32fC_8u_C4R), 889);
}
