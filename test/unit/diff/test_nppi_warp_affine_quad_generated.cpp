// Differential tests for nppiWarpAffineQuad (package g1).
//
// The affine transform is derived from the first three src/dst quad vertex
// pairs; a mismatching fourth vertex yields NPP_AFFINE_QUAD_INCORRECT_WARNING
// (the warp still runs).

#include "npp_warp_affine_diff_helpers.h"

namespace {

using warp_affine_diff::quad_packed;
using warp_affine_diff::quad_planar;

// Exact affine quads (no warning expected).
const double kSrcRect[4][2] = {{0, 0}, {16, 0}, {16, 12}, {0, 12}};
const double kDstRect[4][2] = {{1, 1}, {15, 1}, {15, 11}, {1, 11}};
const double kDstPara[4][2] = {{0, 0}, {14, 2}, {10, 12}, {-4, 10}};
// Fourth dst vertex off the transform -> warning.
const double kDstBad[4][2] = {{0, 0}, {14, 2}, {10, 12}, {-4, 9}};
// Quads inside shifted ROIs (absolute-coordinate probe).
const double kSrcInner[4][2] = {{2, 3}, {14, 3}, {14, 10}, {2, 10}};
const double kDstInner[4][2] = {{1, 2}, {12, 2}, {12, 8}, {1, 8}};

const NppiRect kFull = {0, 0, 17, 13};
const NppiRect kSrcRoi = {2, 3, 13, 8};
const NppiRect kDstRoi = {1, 2, 12, 7};

}  // namespace

class NppiWarpAffineQuadDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(NppiWarpAffineQuadDiff, Packed_8u) {
  quad_packed<Npp8u>("nppiWarpAffineQuad_8u_C1R", "nppiWarpAffineQuad/8u_C1/rect_nn", 17, 13, 1, 16, kSrcRect, kDstRect,
              NPPI_INTER_NN, kFull, kFull, &nppiWarpAffineQuad_8u_C1R_Ctx, 400);
  quad_packed<Npp8u>("nppiWarpAffineQuad_8u_C1R", "nppiWarpAffineQuad/8u_C1/rect_linear", 17, 13, 1, 16, kSrcRect,
              kDstRect, NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineQuad_8u_C1R_Ctx, 401);
  quad_packed<Npp8u>("nppiWarpAffineQuad_8u_C1R", "nppiWarpAffineQuad/8u_C1/para_cubic", 17, 13, 1, 16, kSrcRect, kDstPara,
              NPPI_INTER_CUBIC, kFull, kFull, &nppiWarpAffineQuad_8u_C1R_Ctx, 402);
  quad_packed<Npp8u>("nppiWarpAffineQuad_8u_C1R", "nppiWarpAffineQuad/8u_C1/bad_quad", 17, 13, 1, 16, kSrcRect, kDstBad,
              NPPI_INTER_NN, kFull, kFull, &nppiWarpAffineQuad_8u_C1R_Ctx, 403);
  quad_packed<Npp8u>("nppiWarpAffineQuad_8u_C1R", "nppiWarpAffineQuad/8u_C1/roi_offset", 17, 13, 1, 16, kSrcInner,
              kDstInner, NPPI_INTER_LINEAR, kSrcRoi, kDstRoi, &nppiWarpAffineQuad_8u_C1R_Ctx, 404);
  quad_packed<Npp8u>("nppiWarpAffineQuad_8u_C3R", "nppiWarpAffineQuad/8u_C3/rect_linear", 17, 13, 3, 16, kSrcRect,
              kDstRect, NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineQuad_8u_C3R_Ctx, 405);
  quad_packed<Npp8u>("nppiWarpAffineQuad_8u_C4R", "nppiWarpAffineQuad/8u_C4/rect_nn", 17, 13, 4, 16, kSrcRect, kDstRect,
              NPPI_INTER_NN, kFull, kFull, &nppiWarpAffineQuad_8u_C4R_Ctx, 406);
  quad_packed<Npp8u>("nppiWarpAffineQuad_8u_AC4R", "nppiWarpAffineQuad/8u_AC4/rect_linear", 17, 13, 4, 16, kSrcRect,
              kDstRect, NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineQuad_8u_AC4R_Ctx, 407);
}

TEST_F(NppiWarpAffineQuadDiff, Packed_16u) {
  quad_packed<Npp16u>("nppiWarpAffineQuad_16u_C1R", "nppiWarpAffineQuad/16u_C1/rect_linear", 17, 13, 1, 16, kSrcRect,
              kDstRect, NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineQuad_16u_C1R_Ctx, 410);
  quad_packed<Npp16u>("nppiWarpAffineQuad_16u_C3R", "nppiWarpAffineQuad/16u_C3/rect_nn", 17, 13, 3, 16, kSrcRect, kDstRect,
              NPPI_INTER_NN, kFull, kFull, &nppiWarpAffineQuad_16u_C3R_Ctx, 411);
  quad_packed<Npp16u>("nppiWarpAffineQuad_16u_C4R", "nppiWarpAffineQuad/16u_C4/rect_linear", 17, 13, 4, 16, kSrcRect,
              kDstRect, NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineQuad_16u_C4R_Ctx, 412);
  quad_packed<Npp16u>("nppiWarpAffineQuad_16u_AC4R", "nppiWarpAffineQuad/16u_AC4/para_nn", 17, 13, 4, 16, kSrcRect, kDstPara,
              NPPI_INTER_NN, kFull, kFull, &nppiWarpAffineQuad_16u_AC4R_Ctx, 413);
}

TEST_F(NppiWarpAffineQuadDiff, Packed_32s) {
  quad_packed<Npp32s>("nppiWarpAffineQuad_32s_C1R", "nppiWarpAffineQuad/32s_C1/rect_nn", 17, 13, 1, 16, kSrcRect, kDstRect,
              NPPI_INTER_NN, kFull, kFull, &nppiWarpAffineQuad_32s_C1R_Ctx, 420, 1000);
  quad_packed<Npp32s>("nppiWarpAffineQuad_32s_C3R", "nppiWarpAffineQuad/32s_C3/rect_linear", 17, 13, 3, 16, kSrcRect,
              kDstRect, NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineQuad_32s_C3R_Ctx, 421, 1000);
  quad_packed<Npp32s>("nppiWarpAffineQuad_32s_C4R", "nppiWarpAffineQuad/32s_C4/rect_nn", 17, 13, 4, 16, kSrcRect, kDstRect,
              NPPI_INTER_NN, kFull, kFull, &nppiWarpAffineQuad_32s_C4R_Ctx, 422, 1000);
  quad_packed<Npp32s>("nppiWarpAffineQuad_32s_AC4R", "nppiWarpAffineQuad/32s_AC4/rect_linear", 17, 13, 4, 16, kSrcRect,
              kDstRect, NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineQuad_32s_AC4R_Ctx, 423, 1000);
}

TEST_F(NppiWarpAffineQuadDiff, Packed_32f) {
  quad_packed<Npp32f>("nppiWarpAffineQuad_32f_C1R", "nppiWarpAffineQuad/32f_C1/rect_nn", 17, 13, 1, 16, kSrcRect, kDstRect,
              NPPI_INTER_NN, kFull, kFull, &nppiWarpAffineQuad_32f_C1R_Ctx, 430);
  quad_packed<Npp32f>("nppiWarpAffineQuad_32f_C1R", "nppiWarpAffineQuad/32f_C1/para_linear", 17, 13, 1, 16, kSrcRect,
              kDstPara, NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineQuad_32f_C1R_Ctx, 431);
  quad_packed<Npp32f>("nppiWarpAffineQuad_32f_C3R", "nppiWarpAffineQuad/32f_C3/rect_linear", 17, 13, 3, 16, kSrcRect,
              kDstRect, NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineQuad_32f_C3R_Ctx, 432);
  quad_packed<Npp32f>("nppiWarpAffineQuad_32f_C4R", "nppiWarpAffineQuad/32f_C4/rect_nn", 17, 13, 4, 16, kSrcRect, kDstRect,
              NPPI_INTER_NN, kFull, kFull, &nppiWarpAffineQuad_32f_C4R_Ctx, 433);
  quad_packed<Npp32f>("nppiWarpAffineQuad_32f_AC4R", "nppiWarpAffineQuad/32f_AC4/bad_quad", 17, 13, 4, 16, kSrcRect, kDstBad,
              NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineQuad_32f_AC4R_Ctx, 434);
}

TEST_F(NppiWarpAffineQuadDiff, Planar) {
  quad_planar<Npp8u>("nppiWarpAffineQuad_8u_P3R", "nppiWarpAffineQuad/8u_P3/rect_nn", 17, 13, 3, 16, kSrcRect, kDstRect,
              NPPI_INTER_NN, kFull, kFull, &nppiWarpAffineQuad_8u_P3R_Ctx, 440);
  quad_planar<Npp8u>("nppiWarpAffineQuad_8u_P4R", "nppiWarpAffineQuad/8u_P4/rect_linear", 17, 13, 4, 16, kSrcRect, kDstRect,
              NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineQuad_8u_P4R_Ctx, 441);
  quad_planar<Npp16u>("nppiWarpAffineQuad_16u_P3R", "nppiWarpAffineQuad/16u_P3/rect_linear", 17, 13, 3, 16, kSrcRect,
              kDstRect, NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineQuad_16u_P3R_Ctx, 442);
  quad_planar<Npp16u>("nppiWarpAffineQuad_16u_P4R", "nppiWarpAffineQuad/16u_P4/rect_nn", 17, 13, 4, 16, kSrcRect, kDstRect,
              NPPI_INTER_NN, kFull, kFull, &nppiWarpAffineQuad_16u_P4R_Ctx, 443);
  quad_planar<Npp32s>("nppiWarpAffineQuad_32s_P3R", "nppiWarpAffineQuad/32s_P3/rect_nn", 17, 13, 3, 16, kSrcRect, kDstRect,
              NPPI_INTER_NN, kFull, kFull, &nppiWarpAffineQuad_32s_P3R_Ctx, 444, 1000);
  quad_planar<Npp32s>("nppiWarpAffineQuad_32s_P4R", "nppiWarpAffineQuad/32s_P4/rect_nn", 17, 13, 4, 16, kSrcRect, kDstRect,
              NPPI_INTER_NN, kFull, kFull, &nppiWarpAffineQuad_32s_P4R_Ctx, 445, 1000);
  quad_planar<Npp32f>("nppiWarpAffineQuad_32f_P3R", "nppiWarpAffineQuad/32f_P3/rect_linear", 17, 13, 3, 16, kSrcRect,
              kDstRect, NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineQuad_32f_P3R_Ctx, 446);
  quad_planar<Npp32f>("nppiWarpAffineQuad_32f_P4R", "nppiWarpAffineQuad/32f_P4/rect_nn", 17, 13, 4, 16, kSrcRect, kDstRect,
              NPPI_INTER_NN, kFull, kFull, &nppiWarpAffineQuad_32f_P4R_Ctx, 447);
}
