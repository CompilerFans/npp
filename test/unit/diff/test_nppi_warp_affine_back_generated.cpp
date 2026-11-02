// Differential tests for nppiWarpAffineBack (package g1).
//
// nppiWarpAffineBack consumes the dst -> src map directly (no inversion).

#include "npp_warp_affine_diff_helpers.h"

namespace {

using warp_affine_diff::warp_packed;
using warp_affine_diff::warp_packed_err;
using warp_affine_diff::warp_planar;

// dst -> src maps, used as-is.
const double kScale[2][3] = {{0.8, 0.0, 0.5}, {0.0, 0.9, -0.75}};
const double kShear[2][3] = {{1.0, 0.1, 0.0}, {-0.05, 1.0, 0.25}};
const double kRank1[2][3] = {{1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}};  // singular but a legal direct map

const NppiRect kFull = {0, 0, 17, 13};

}  // namespace

class NppiWarpAffineBackDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(NppiWarpAffineBackDiff, Packed_8u) {
  warp_packed<Npp8u>("nppiWarpAffineBack_8u_C1R", "nppiWarpAffineBack/8u_C1/nn", 17, 13, 1, 16, kScale, NPPI_INTER_NN,
              kFull, kFull, &nppiWarpAffineBack_8u_C1R_Ctx, 300);
  warp_packed<Npp8u>("nppiWarpAffineBack_8u_C1R", "nppiWarpAffineBack/8u_C1/linear", 17, 13, 1, 16, kScale,
              NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineBack_8u_C1R_Ctx, 301);
  warp_packed<Npp8u>("nppiWarpAffineBack_8u_C1R", "nppiWarpAffineBack/8u_C1/cubic_shear", 17, 13, 1, 16, kShear,
              NPPI_INTER_CUBIC, kFull, kFull, &nppiWarpAffineBack_8u_C1R_Ctx, 302);
  warp_packed<Npp8u>("nppiWarpAffineBack_8u_C1R", "nppiWarpAffineBack/8u_C1/rank1", 17, 13, 1, 16, kRank1, NPPI_INTER_NN,
              kFull, kFull, &nppiWarpAffineBack_8u_C1R_Ctx, 303);
  warp_packed<Npp8u>("nppiWarpAffineBack_8u_C3R", "nppiWarpAffineBack/8u_C3/linear", 17, 13, 3, 16, kScale,
              NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineBack_8u_C3R_Ctx, 304);
  warp_packed<Npp8u>("nppiWarpAffineBack_8u_C4R", "nppiWarpAffineBack/8u_C4/nn", 17, 13, 4, 16, kScale, NPPI_INTER_NN,
              kFull, kFull, &nppiWarpAffineBack_8u_C4R_Ctx, 305);
  warp_packed<Npp8u>("nppiWarpAffineBack_8u_AC4R", "nppiWarpAffineBack/8u_AC4/linear", 17, 13, 4, 16, kScale,
              NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineBack_8u_AC4R_Ctx, 306);
}

TEST_F(NppiWarpAffineBackDiff, Packed_16u) {
  warp_packed<Npp16u>("nppiWarpAffineBack_16u_C1R", "nppiWarpAffineBack/16u_C1/nn", 17, 13, 1, 16, kScale, NPPI_INTER_NN,
              kFull, kFull, &nppiWarpAffineBack_16u_C1R_Ctx, 310);
  warp_packed<Npp16u>("nppiWarpAffineBack_16u_C3R", "nppiWarpAffineBack/16u_C3/linear", 17, 13, 3, 16, kScale,
              NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineBack_16u_C3R_Ctx, 311);
  warp_packed<Npp16u>("nppiWarpAffineBack_16u_C4R", "nppiWarpAffineBack/16u_C4/nn", 17, 13, 4, 16, kScale, NPPI_INTER_NN,
              kFull, kFull, &nppiWarpAffineBack_16u_C4R_Ctx, 312);
  warp_packed<Npp16u>("nppiWarpAffineBack_16u_AC4R", "nppiWarpAffineBack/16u_AC4/linear", 17, 13, 4, 16, kScale,
              NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineBack_16u_AC4R_Ctx, 313);
}

TEST_F(NppiWarpAffineBackDiff, Packed_32s) {
  warp_packed<Npp32s>("nppiWarpAffineBack_32s_C1R", "nppiWarpAffineBack/32s_C1/nn", 17, 13, 1, 16, kScale, NPPI_INTER_NN,
              kFull, kFull, &nppiWarpAffineBack_32s_C1R_Ctx, 320, 1000);
  warp_packed<Npp32s>("nppiWarpAffineBack_32s_C3R", "nppiWarpAffineBack/32s_C3/linear", 17, 13, 3, 16, kScale,
              NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineBack_32s_C3R_Ctx, 321, 1000);
  warp_packed<Npp32s>("nppiWarpAffineBack_32s_C4R", "nppiWarpAffineBack/32s_C4/nn", 17, 13, 4, 16, kScale, NPPI_INTER_NN,
              kFull, kFull, &nppiWarpAffineBack_32s_C4R_Ctx, 322, 1000);
  warp_packed<Npp32s>("nppiWarpAffineBack_32s_AC4R", "nppiWarpAffineBack/32s_AC4/linear", 17, 13, 4, 16, kScale,
              NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineBack_32s_AC4R_Ctx, 323, 1000);
}

TEST_F(NppiWarpAffineBackDiff, Packed_32f) {
  warp_packed<Npp32f>("nppiWarpAffineBack_32f_C1R", "nppiWarpAffineBack/32f_C1/linear", 17, 13, 1, 16, kScale,
              NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineBack_32f_C1R_Ctx, 330);
  warp_packed<Npp32f>("nppiWarpAffineBack_32f_C3R", "nppiWarpAffineBack/32f_C3/nn", 17, 13, 3, 16, kScale, NPPI_INTER_NN,
              kFull, kFull, &nppiWarpAffineBack_32f_C3R_Ctx, 331);
  warp_packed<Npp32f>("nppiWarpAffineBack_32f_C4R", "nppiWarpAffineBack/32f_C4/linear", 17, 13, 4, 16, kScale,
              NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineBack_32f_C4R_Ctx, 332);
  warp_packed<Npp32f>("nppiWarpAffineBack_32f_AC4R", "nppiWarpAffineBack/32f_AC4/nn", 17, 13, 4, 16, kScale, NPPI_INTER_NN,
              kFull, kFull, &nppiWarpAffineBack_32f_AC4R_Ctx, 333);
}

TEST_F(NppiWarpAffineBackDiff, Planar) {
  warp_planar<Npp8u>("nppiWarpAffineBack_8u_P3R", "nppiWarpAffineBack/8u_P3/nn", 17, 13, 3, 16, kScale, NPPI_INTER_NN,
              kFull, kFull, &nppiWarpAffineBack_8u_P3R_Ctx, 340);
  warp_planar<Npp8u>("nppiWarpAffineBack_8u_P4R", "nppiWarpAffineBack/8u_P4/linear", 17, 13, 4, 16, kScale,
              NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineBack_8u_P4R_Ctx, 341);
  warp_planar<Npp16u>("nppiWarpAffineBack_16u_P3R", "nppiWarpAffineBack/16u_P3/linear", 17, 13, 3, 16, kScale,
              NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineBack_16u_P3R_Ctx, 342);
  warp_planar<Npp16u>("nppiWarpAffineBack_16u_P4R", "nppiWarpAffineBack/16u_P4/nn", 17, 13, 4, 16, kScale, NPPI_INTER_NN,
              kFull, kFull, &nppiWarpAffineBack_16u_P4R_Ctx, 343);
  warp_planar<Npp32s>("nppiWarpAffineBack_32s_P3R", "nppiWarpAffineBack/32s_P3/nn", 17, 13, 3, 16, kScale, NPPI_INTER_NN,
              kFull, kFull, &nppiWarpAffineBack_32s_P3R_Ctx, 344, 1000);
  warp_planar<Npp32s>("nppiWarpAffineBack_32s_P4R", "nppiWarpAffineBack/32s_P4/nn", 17, 13, 4, 16, kScale, NPPI_INTER_NN,
              kFull, kFull, &nppiWarpAffineBack_32s_P4R_Ctx, 345, 1000);
  warp_planar<Npp32f>("nppiWarpAffineBack_32f_P3R", "nppiWarpAffineBack/32f_P3/linear", 17, 13, 3, 16, kScale,
              NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineBack_32f_P3R_Ctx, 346);
  warp_planar<Npp32f>("nppiWarpAffineBack_32f_P4R", "nppiWarpAffineBack/32f_P4/nn", 17, 13, 4, 16, kScale, NPPI_INTER_NN,
              kFull, kFull, &nppiWarpAffineBack_32f_P4R_Ctx, 347);
}

TEST_F(NppiWarpAffineBackDiff, Errors) {
  warp_packed_err<Npp8u>("nppiWarpAffineBack_8u_C1R", "nppiWarpAffineBack/err/nulldst", 17, 13, 1, kScale, NPPI_INTER_NN,
                  kFull, kFull, false, true, &nppiWarpAffineBack_8u_C1R_Ctx, 350);
  warp_packed_err<Npp8u>("nppiWarpAffineBack_8u_C1R", "nppiWarpAffineBack/err/interp_lanczos", 17, 13, 1, kScale,
                  NPPI_INTER_LANCZOS, kFull, kFull, false, false, &nppiWarpAffineBack_8u_C1R_Ctx, 351);
  warp_packed_err<Npp8u>("nppiWarpAffineBack_8u_C1R", "nppiWarpAffineBack/err/srcroi_outside", 17, 13, 1, kScale,
                  NPPI_INTER_NN, NppiRect{-40, -40, 8, 8}, kFull, false, false, &nppiWarpAffineBack_8u_C1R_Ctx,
                  352);
  warp_packed_err<Npp8u>("nppiWarpAffineBack_8u_C1R", "nppiWarpAffineBack/err/dstroi_zero", 17, 13, 1, kScale,
                  NPPI_INTER_NN, kFull, NppiRect{3, 4, 10, 0}, false, false, &nppiWarpAffineBack_8u_C1R_Ctx, 353);
  warp_packed_err<Npp32f>("nppiWarpAffineBack_32f_C1R", "nppiWarpAffineBack/err/srcsize_zero", 0, 0, 1, kScale,
                  NPPI_INTER_NN, kFull, kFull, false, false, &nppiWarpAffineBack_32f_C1R_Ctx, 354);
}

TEST_F(NppiWarpAffineBackDiff, Probes_Roi) {
  warp_packed_err<Npp8u>("nppiWarpAffineBack_8u_C1R", "nppiWarpAffineBack/roi/srcroi_pos_outside", 17, 13, 1,
                         kScale, NPPI_INTER_NN, NppiRect{100, 100, 8, 8}, kFull, false, false,
                         &nppiWarpAffineBack_8u_C1R_Ctx, 360);
  warp_packed_err<Npp8u>("nppiWarpAffineBack_8u_C1R", "nppiWarpAffineBack/roi/srcroi_thin", 17, 13, 1, kScale,
                         NPPI_INTER_NN, NppiRect{0, 0, 17, 1}, kFull, false, false,
                         &nppiWarpAffineBack_8u_C1R_Ctx, 361);
  warp_packed_err<Npp8u>("nppiWarpAffineBack_8u_C1R", "nppiWarpAffineBack/roi/srcroi_partial", 17, 13, 1, kScale,
                         NPPI_INTER_NN, NppiRect{-5, 3, 20, 10}, kFull, false, false,
                         &nppiWarpAffineBack_8u_C1R_Ctx, 362);
  warp_packed_err<Npp8u>("nppiWarpAffineBack_8u_C1R", "nppiWarpAffineBack/roi/srcsize_thin", 17, 1, 1, kScale,
                         NPPI_INTER_NN, NppiRect{0, 0, 17, 1}, NppiRect{0, 0, 17, 1}, false, false,
                         &nppiWarpAffineBack_8u_C1R_Ctx, 363);
}
