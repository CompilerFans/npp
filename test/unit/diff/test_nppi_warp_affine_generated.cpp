// Differential tests for nppiWarpAffine (package g1).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiWarpAffineDiff.*'

#include "npp_warp_affine_diff_helpers.h"

namespace {

using warp_affine_diff::arena;
using warp_affine_diff::align_step;
using warp_affine_diff::current_ctx;
using warp_affine_diff::out_kind;
using warp_affine_diff::warp_packed;
using warp_affine_diff::warp_packed_err;
using warp_affine_diff::warp_planar;

// Forward src -> dst maps (nppiWarpAffine inverts them internally).
const double kScale[2][3] = {{1.25, 0.0, 0.5}, {0.0, 1.1, -0.75}};
const double kHalfX[2][3] = {{1.0, 0.0, 0.5}, {0.0, 1.0, 0.0}};  // NN tie-break probe
const double kOutOfBound[2][3] = {{2.0, 0.0, -3.0}, {0.0, 2.0, -3.0}};
const double kRotate[2][3] = {{0.8660254, -0.5, 4.0717968}, {0.5, 0.8660254, -3.1961524}};
const double kRoiOffset[2][3] = {{1.0, 0.0, 1.5}, {0.0, 1.0, -0.5}};

const NppiRect kFull = {0, 0, 17, 13};
const NppiRect kSrcInner = {2, 3, 13, 8};
const NppiRect kDstInner = {1, 2, 12, 7};
const NppiRect kTiny = {1, 1, 3, 2};

}  // namespace

class NppiWarpAffineDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ===== dtype x layout matrix ================================================
TEST_F(NppiWarpAffineDiff, Packed_8u) {
  warp_packed<Npp8u>("nppiWarpAffine_8u_C1R", "nppiWarpAffine/8u_C1/nn", 17, 13, 1, 16, kScale, NPPI_INTER_NN, kFull,
              kFull, &nppiWarpAffine_8u_C1R_Ctx, 100);
  warp_packed<Npp8u>("nppiWarpAffine_8u_C1R", "nppiWarpAffine/8u_C1/linear", 17, 13, 1, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_8u_C1R_Ctx, 101);
  warp_packed<Npp8u>("nppiWarpAffine_8u_C1R", "nppiWarpAffine/8u_C1/cubic", 17, 13, 1, 16, kScale, NPPI_INTER_CUBIC,
              kFull, kFull, &nppiWarpAffine_8u_C1R_Ctx, 102);
  warp_packed<Npp8u>("nppiWarpAffine_8u_C1R", "nppiWarpAffine/8u_C1/halfx", 17, 13, 1, 16, kHalfX, NPPI_INTER_NN, kFull,
              kFull, &nppiWarpAffine_8u_C1R_Ctx, 103);
  warp_packed<Npp8u>("nppiWarpAffine_8u_C1R", "nppiWarpAffine/8u_C1/oob", 17, 13, 1, 16, kOutOfBound, NPPI_INTER_NN,
              kFull, kFull, &nppiWarpAffine_8u_C1R_Ctx, 104);
  warp_packed<Npp8u>("nppiWarpAffine_8u_C1R", "nppiWarpAffine/8u_C1/rot_tiny", 5, 4, 1, 8, kRotate, NPPI_INTER_LINEAR,
              kTiny, kTiny, &nppiWarpAffine_8u_C1R_Ctx, 105);
  warp_packed<Npp8u>("nppiWarpAffine_8u_C1R", "nppiWarpAffine/8u_C1/roi_offset", 17, 13, 1, 16, kRoiOffset,
              NPPI_INTER_LINEAR, kSrcInner, kDstInner, &nppiWarpAffine_8u_C1R_Ctx, 106);
  warp_packed<Npp8u>("nppiWarpAffine_8u_C3R", "nppiWarpAffine/8u_C3/linear", 17, 13, 3, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_8u_C3R_Ctx, 110);
  warp_packed<Npp8u>("nppiWarpAffine_8u_C4R", "nppiWarpAffine/8u_C4/linear", 17, 13, 4, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_8u_C4R_Ctx, 111);
  warp_packed<Npp8u>("nppiWarpAffine_8u_AC4R", "nppiWarpAffine/8u_AC4/linear", 17, 13, 4, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_8u_AC4R_Ctx, 112);
}

TEST_F(NppiWarpAffineDiff, Packed_16u) {
  warp_packed<Npp16u>("nppiWarpAffine_16u_C1R", "nppiWarpAffine/16u_C1/linear", 17, 13, 1, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_16u_C1R_Ctx, 120);
  warp_packed<Npp16u>("nppiWarpAffine_16u_C3R", "nppiWarpAffine/16u_C3/nn", 17, 13, 3, 16, kScale, NPPI_INTER_NN, kFull,
              kFull, &nppiWarpAffine_16u_C3R_Ctx, 121);
  warp_packed<Npp16u>("nppiWarpAffine_16u_C4R", "nppiWarpAffine/16u_C4/linear", 17, 13, 4, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_16u_C4R_Ctx, 122);
  warp_packed<Npp16u>("nppiWarpAffine_16u_AC4R", "nppiWarpAffine/16u_AC4/linear", 17, 13, 4, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_16u_AC4R_Ctx, 123);
}

TEST_F(NppiWarpAffineDiff, Packed_32s) {
  warp_packed<Npp32s>("nppiWarpAffine_32s_C1R", "nppiWarpAffine/32s_C1/nn", 17, 13, 1, 16, kScale, NPPI_INTER_NN, kFull,
              kFull, &nppiWarpAffine_32s_C1R_Ctx, 130, 1000);
  warp_packed<Npp32s>("nppiWarpAffine_32s_C3R", "nppiWarpAffine/32s_C3/linear", 17, 13, 3, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_32s_C3R_Ctx, 131, 1000);
  warp_packed<Npp32s>("nppiWarpAffine_32s_C4R", "nppiWarpAffine/32s_C4/nn", 17, 13, 4, 16, kScale, NPPI_INTER_NN, kFull,
              kFull, &nppiWarpAffine_32s_C4R_Ctx, 132, 1000);
  warp_packed<Npp32s>("nppiWarpAffine_32s_AC4R", "nppiWarpAffine/32s_AC4/linear", 17, 13, 4, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_32s_AC4R_Ctx, 133, 1000);
}

TEST_F(NppiWarpAffineDiff, Packed_16f) {
  warp_packed<Npp16f>("nppiWarpAffine_16f_C1R", "nppiWarpAffine/16f_C1/linear", 17, 13, 1, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_16f_C1R_Ctx, 140, 0, 2e-3, 2e-3);
  warp_packed<Npp16f>("nppiWarpAffine_16f_C3R", "nppiWarpAffine/16f_C3/linear", 17, 13, 3, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_16f_C3R_Ctx, 141, 0, 2e-3, 2e-3);
  warp_packed<Npp16f>("nppiWarpAffine_16f_C4R", "nppiWarpAffine/16f_C4/linear", 17, 13, 4, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_16f_C4R_Ctx, 142, 0, 2e-3, 2e-3);
}

TEST_F(NppiWarpAffineDiff, Packed_32f) {
  warp_packed<Npp32f>("nppiWarpAffine_32f_C1R", "nppiWarpAffine/32f_C1/nn", 17, 13, 1, 16, kScale, NPPI_INTER_NN, kFull,
              kFull, &nppiWarpAffine_32f_C1R_Ctx, 150);
  warp_packed<Npp32f>("nppiWarpAffine_32f_C1R", "nppiWarpAffine/32f_C1/linear", 17, 13, 1, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_32f_C1R_Ctx, 151);
  warp_packed<Npp32f>("nppiWarpAffine_32f_C1R", "nppiWarpAffine/32f_C1/cubic", 17, 13, 1, 16, kScale, NPPI_INTER_CUBIC,
              kFull, kFull, &nppiWarpAffine_32f_C1R_Ctx, 152, 0, 1e-4, 1e-4);
  warp_packed<Npp32f>("nppiWarpAffine_32f_C3R", "nppiWarpAffine/32f_C3/linear", 17, 13, 3, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_32f_C3R_Ctx, 153);
  warp_packed<Npp32f>("nppiWarpAffine_32f_C4R", "nppiWarpAffine/32f_C4/linear", 17, 13, 4, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_32f_C4R_Ctx, 154);
  warp_packed<Npp32f>("nppiWarpAffine_32f_AC4R", "nppiWarpAffine/32f_AC4/linear", 17, 13, 4, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_32f_AC4R_Ctx, 155);
}

TEST_F(NppiWarpAffineDiff, Packed_64f) {
  warp_packed<Npp64f>("nppiWarpAffine_64f_C1R", "nppiWarpAffine/64f_C1/linear", 17, 13, 1, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_64f_C1R_Ctx, 160);
  warp_packed<Npp64f>("nppiWarpAffine_64f_C3R", "nppiWarpAffine/64f_C3/linear", 17, 13, 3, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_64f_C3R_Ctx, 161);
  warp_packed<Npp64f>("nppiWarpAffine_64f_C4R", "nppiWarpAffine/64f_C4/linear", 17, 13, 4, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_64f_C4R_Ctx, 162);
  warp_packed<Npp64f>("nppiWarpAffine_64f_AC4R", "nppiWarpAffine/64f_AC4/linear", 17, 13, 4, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_64f_AC4R_Ctx, 163);
}

TEST_F(NppiWarpAffineDiff, Planar) {
  warp_planar<Npp8u>("nppiWarpAffine_8u_P3R", "nppiWarpAffine/8u_P3/linear", 17, 13, 3, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_8u_P3R_Ctx, 170);
  warp_planar<Npp8u>("nppiWarpAffine_8u_P4R", "nppiWarpAffine/8u_P4/nn", 17, 13, 4, 16, kScale, NPPI_INTER_NN, kFull,
              kFull, &nppiWarpAffine_8u_P4R_Ctx, 171);
  warp_planar<Npp16u>("nppiWarpAffine_16u_P3R", "nppiWarpAffine/16u_P3/nn", 17, 13, 3, 16, kScale, NPPI_INTER_NN, kFull,
              kFull, &nppiWarpAffine_16u_P3R_Ctx, 172);
  warp_planar<Npp16u>("nppiWarpAffine_16u_P4R", "nppiWarpAffine/16u_P4/linear", 17, 13, 4, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_16u_P4R_Ctx, 173);
  warp_planar<Npp32s>("nppiWarpAffine_32s_P3R", "nppiWarpAffine/32s_P3/nn", 17, 13, 3, 16, kScale, NPPI_INTER_NN, kFull,
              kFull, &nppiWarpAffine_32s_P3R_Ctx, 174, 1000);
  warp_planar<Npp32s>("nppiWarpAffine_32s_P4R", "nppiWarpAffine/32s_P4/nn", 17, 13, 4, 16, kScale, NPPI_INTER_NN, kFull,
              kFull, &nppiWarpAffine_32s_P4R_Ctx, 175, 1000);
  warp_planar<Npp32f>("nppiWarpAffine_32f_P3R", "nppiWarpAffine/32f_P3/linear", 17, 13, 3, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_32f_P3R_Ctx, 176);
  warp_planar<Npp32f>("nppiWarpAffine_32f_P4R", "nppiWarpAffine/32f_P4/linear", 17, 13, 4, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_32f_P4R_Ctx, 177);
  warp_planar<Npp64f>("nppiWarpAffine_64f_P3R", "nppiWarpAffine/64f_P3/linear", 17, 13, 3, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_64f_P3R_Ctx, 178);
  warp_planar<Npp64f>("nppiWarpAffine_64f_P4R", "nppiWarpAffine/64f_P4/linear", 17, 13, 4, 16, kScale, NPPI_INTER_LINEAR,
              kFull, kFull, &nppiWarpAffine_64f_P4R_Ctx, 179);
}

// ===== error paths ==========================================================
TEST_F(NppiWarpAffineDiff, Errors) {
  const double singular[2][3] = {{1.0, 2.0, 0.0}, {2.0, 4.0, 0.0}};
  warp_packed_err<Npp8u>("nppiWarpAffine_8u_C1R", "nppiWarpAffine/err/nullsrc", 17, 13, 1, kScale, NPPI_INTER_NN, kFull,
                  kFull, true, false, &nppiWarpAffine_8u_C1R_Ctx, 190);
  warp_packed_err<Npp8u>("nppiWarpAffine_8u_C1R", "nppiWarpAffine/err/nulldst", 17, 13, 1, kScale, NPPI_INTER_NN, kFull,
                  kFull, false, true, &nppiWarpAffine_8u_C1R_Ctx, 191);
  warp_packed_err<Npp8u>("nppiWarpAffine_8u_C1R", "nppiWarpAffine/err/interp_super", 17, 13, 1, kScale, NPPI_INTER_SUPER,
                  kFull, kFull, false, false, &nppiWarpAffine_8u_C1R_Ctx, 192);
  warp_packed_err<Npp8u>("nppiWarpAffine_8u_C1R", "nppiWarpAffine/err/interp_undef", 17, 13, 1, kScale,
                  NPPI_INTER_UNDEFINED, kFull, kFull, false, false, &nppiWarpAffine_8u_C1R_Ctx, 193);
  warp_packed_err<Npp8u>("nppiWarpAffine_8u_C1R", "nppiWarpAffine/err/singular", 17, 13, 1, singular, NPPI_INTER_NN,
                  kFull, kFull, false, false, &nppiWarpAffine_8u_C1R_Ctx, 194);
  warp_packed_err<Npp8u>("nppiWarpAffine_8u_C1R", "nppiWarpAffine/err/srcsize_zero", 0, 0, 1, kScale, NPPI_INTER_NN,
                  kFull, kFull, false, false, &nppiWarpAffine_8u_C1R_Ctx, 195);
  warp_packed_err<Npp8u>("nppiWarpAffine_8u_C1R", "nppiWarpAffine/err/srcsize_1x1", 1, 1, 1, kScale, NPPI_INTER_NN,
                  NppiRect{0, 0, 1, 1}, NppiRect{0, 0, 1, 1}, false, false, &nppiWarpAffine_8u_C1R_Ctx, 196);
  warp_packed_err<Npp8u>("nppiWarpAffine_8u_C1R", "nppiWarpAffine/err/srcroi_outside", 17, 13, 1, kScale, NPPI_INTER_NN,
                  NppiRect{100, 100, 8, 8}, kFull, false, false, &nppiWarpAffine_8u_C1R_Ctx, 197);
  warp_packed_err<Npp8u>("nppiWarpAffine_8u_C1R", "nppiWarpAffine/err/srcroi_thin", 17, 13, 1, kScale, NPPI_INTER_NN,
                  NppiRect{0, 0, 17, 1}, kFull, false, false, &nppiWarpAffine_8u_C1R_Ctx, 198);
  warp_packed_err<Npp8u>("nppiWarpAffine_8u_C1R", "nppiWarpAffine/err/dstroi_zero_w", 17, 13, 1, kScale, NPPI_INTER_NN,
                  kFull, NppiRect{0, 0, 0, 13}, false, false, &nppiWarpAffine_8u_C1R_Ctx, 199);
  warp_packed_err<Npp8u>("nppiWarpAffine_8u_C1R", "nppiWarpAffine/err/dstroi_neg_h", 17, 13, 1, kScale, NPPI_INTER_NN,
                  kFull, NppiRect{0, 0, 17, -5}, false, false, &nppiWarpAffine_8u_C1R_Ctx, 200);
  warp_packed_err<Npp32f>("nppiWarpAffine_32f_C1R", "nppiWarpAffine/err/singular_32f", 17, 13, 1, singular,
                  NPPI_INTER_LINEAR, kFull, kFull, false, false, &nppiWarpAffine_32f_C1R_Ctx, 201);
}

// ===== probes: step alignment and ROI/size validation rules ================
TEST_F(NppiWarpAffineDiff, Probes_Steps) {
  // Packed 4-channel steps that are not a multiple of 4*sizeof(T).
  warp_packed<Npp8u>("nppiWarpAffine_8u_C4R", "nppiWarpAffine/step/8u_C4_odd", 17, 13, 4, 18, kScale,
                     NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffine_8u_C4R_Ctx, 210);
  warp_packed<Npp16u>("nppiWarpAffine_16u_C4R", "nppiWarpAffine/step/16u_C4_odd", 17, 13, 4, 20, kScale,
                      NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffine_16u_C4R_Ctx, 211);
  warp_packed<Npp32s>("nppiWarpAffine_32s_C4R", "nppiWarpAffine/step/32s_C4_odd", 17, 13, 4, 24, kScale,
                      NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffine_32s_C4R_Ctx, 212, 1000);
  warp_packed<Npp32f>("nppiWarpAffine_32f_C4R", "nppiWarpAffine/step/32f_C4_odd", 17, 13, 4, 24, kScale,
                      NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffine_32f_C4R_Ctx, 213);
  warp_packed<Npp16f>("nppiWarpAffine_16f_C4R", "nppiWarpAffine/step/16f_C4_odd", 17, 13, 4, 20, kScale,
                      NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffine_16f_C4R_Ctx, 214, 0, 2e-3, 2e-3);
  // 64f C4 with a 32-byte multiple step (the 560-byte step variant above
  // yields NPP_NOT_EVEN_STEP_ERROR on NVIDIA NPP 13.4).
  warp_packed<Npp64f>("nppiWarpAffine_64f_C4R", "nppiWarpAffine/step/64f_C4_even", 17, 13, 4, 32, kScale,
                      NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffine_64f_C4R_Ctx, 215);
  // 3-channel non-pixel-multiple step (67 already passed; 68 keeps %3 != 0).
  warp_packed<Npp8u>("nppiWarpAffine_8u_C3R", "nppiWarpAffine/step/8u_C3_odd", 17, 13, 3, 17, kScale,
                     NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffine_8u_C3R_Ctx, 216);
}

TEST_F(NppiWarpAffineDiff, Probes_Roi) {
  warp_packed_err<Npp8u>("nppiWarpAffine_8u_C1R", "nppiWarpAffine/roi/srcroi_neg_outside", 17, 13, 1, kScale,
                         NPPI_INTER_NN, NppiRect{-40, -40, 8, 8}, kFull, false, false,
                         &nppiWarpAffine_8u_C1R_Ctx, 220);
  warp_packed_err<Npp8u>("nppiWarpAffine_8u_C1R", "nppiWarpAffine/roi/srcsize_thin", 17, 1, 1, kScale,
                         NPPI_INTER_NN, NppiRect{0, 0, 17, 1}, NppiRect{0, 0, 17, 1}, false, false,
                         &nppiWarpAffine_8u_C1R_Ctx, 221);
  warp_packed_err<Npp8u>("nppiWarpAffine_8u_C1R", "nppiWarpAffine/roi/dstroi_zero_h", 17, 13, 1, kScale,
                         NPPI_INTER_NN, kFull, NppiRect{0, 0, 17, 0}, false, false, &nppiWarpAffine_8u_C1R_Ctx,
                         222);
  warp_packed_err<Npp8u>("nppiWarpAffine_8u_C1R", "nppiWarpAffine/roi/dstroi_zero_at_offset", 17, 13, 1, kScale,
                         NPPI_INTER_NN, kFull, NppiRect{3, 4, 0, 5}, false, false, &nppiWarpAffine_8u_C1R_Ctx,
                         223);
}

// ===== Non-Ctx adapters (reproduce _Ctx cases -> same golden) ================
#if NPP_DIFF_HAS_NONCTX
TEST_F(NppiWarpAffineDiff, NonCtx_8u_C1_NN) {
  // Reproduces warp_packed(".../8u_C1/nn", 17,13,1,16, kScale, NN, kFull, kFull, seed=100).
  const int es = 1, chans = 1, pad = 16, width = 17, height = 13;
  const int step = align_step(width * chans * es + pad, es);
  const size_t ne = (size_t)height * step / es;
  std::vector<Npp8u> hs = make_uniform<Npp8u>(ne, 100 + 11);
  std::vector<Npp8u> hd = make_uniform<Npp8u>(ne, 100 + 29);
  DeviceMemory<Npp8u> &s = arena<Npp8u>(ne);
  DeviceMemory<Npp8u> &d = arena<Npp8u>(ne);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = nppiWarpAffine_8u_C1R(s.get(), NppiSize{width, height}, step, kFull, d.get(), step, kFull, kScale,
                                       NPPI_INTER_NN);
  Case c("nppiWarpAffine/8u_C1/nn", "nppiWarpAffine_8u_C1R");
  c.input(hs.data(), hs.size() * sizeof(Npp8u));
  c.status(st);
  c.device_output<Npp8u>(d.get(), ne, out_kind<Npp8u>(), 1e-5, 1e-6);
  c.finish();
}

TEST_F(NppiWarpAffineDiff, NonCtx_32f_AC4_Linear) {
  // Reproduces warp_packed(".../32f_AC4/linear", 17,13,4,16, kScale, LINEAR, kFull, kFull, seed=155).
  const int es = 4, chans = 4, pad = 16, width = 17, height = 13;
  const int step = align_step(width * chans * es + pad, es);
  const size_t ne = (size_t)height * step / es;
  std::vector<Npp32f> hs = make_uniform<Npp32f>(ne, 155 + 11);
  std::vector<Npp32f> hd = make_uniform<Npp32f>(ne, 155 + 29);
  DeviceMemory<Npp32f> &s = arena<Npp32f>(ne);
  DeviceMemory<Npp32f> &d = arena<Npp32f>(ne);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  clear_cuda_error();
  NppStatus st = nppiWarpAffine_32f_AC4R(s.get(), NppiSize{width, height}, step, kFull, d.get(), step, kFull,
                                         kScale, NPPI_INTER_LINEAR);
  Case c("nppiWarpAffine/32f_AC4/linear", "nppiWarpAffine_32f_AC4R");
  c.input(hs.data(), hs.size() * sizeof(Npp32f));
  c.status(st);
  c.device_output<Npp32f>(d.get(), ne, out_kind<Npp32f>(), 1e-5, 1e-6);
  c.finish();
}
#endif
