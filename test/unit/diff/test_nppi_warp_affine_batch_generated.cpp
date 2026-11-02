// Differential tests for nppiWarpAffineBatch (package g1).
//
// The batch list must be prepared by nppiWarpAffineBatchInit (also exercised
// here); pCoeffs holds the forward src -> dst map per image.

#include "npp_warp_affine_diff_helpers.h"

#include <array>
#include <vector>

namespace {

using warp_affine_diff::arena;
using warp_affine_diff::batch_entry;
using warp_affine_diff::current_ctx;
using warp_affine_diff::warp_batch;

const NppiRect kFull = {0, 0, 7, 5};

// Two images with different forward transforms.
const std::vector<std::array<double, 6>> kTwo = {
    {1.0, 0.0, 1.0, 0.0, 1.0, 0.5},                     // translation
    {0.75, 0.0, 0.0, 0.0, 1.25, 0.25},                  // anisotropic scale
};
const std::vector<std::array<double, 6>> kThree = {
    {1.0, 0.0, -1.0, 0.0, 1.0, 0.0},                    // translation
    {0.8660254, -0.5, 2.0, 0.5, 0.8660254, 0.0},        // rotation
    {1.1, 0.15, 0.3, -0.1, 0.9, 0.2},                   // shear
};

}  // namespace

class NppiWarpAffineBatchDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(NppiWarpAffineBatchDiff, Batch_8u) {
  warp_batch<Npp8u>("nppiWarpAffineBatch_8u_C1R", "nppiWarpAffineBatch/8u_C1/nn", 7, 5, 1, 8, kTwo, NPPI_INTER_NN,
                    kFull, kFull, &nppiWarpAffineBatch_8u_C1R_Ctx, 500);
  warp_batch<Npp8u>("nppiWarpAffineBatch_8u_C1R", "nppiWarpAffineBatch/8u_C1/linear", 7, 5, 1, 8, kThree,
                    NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineBatch_8u_C1R_Ctx, 501);
  warp_batch<Npp8u>("nppiWarpAffineBatch_8u_C3R", "nppiWarpAffineBatch/8u_C3/linear", 7, 5, 3, 8, kTwo,
                    NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineBatch_8u_C3R_Ctx, 502);
  warp_batch<Npp8u>("nppiWarpAffineBatch_8u_C4R", "nppiWarpAffineBatch/8u_C4/nn", 7, 5, 4, 8, kThree, NPPI_INTER_NN,
                    kFull, kFull, &nppiWarpAffineBatch_8u_C4R_Ctx, 503);
  warp_batch<Npp8u>("nppiWarpAffineBatch_8u_AC4R", "nppiWarpAffineBatch/8u_AC4/linear", 7, 5, 4, 8, kTwo,
                    NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineBatch_8u_AC4R_Ctx, 504);
}

TEST_F(NppiWarpAffineBatchDiff, Batch_16f) {
  warp_batch<Npp16f>("nppiWarpAffineBatch_16f_C1R", "nppiWarpAffineBatch/16f_C1/nn", 7, 5, 1, 8, kTwo,
                     NPPI_INTER_NN, kFull, kFull, &nppiWarpAffineBatch_16f_C1R_Ctx, 510, 0, 2e-3, 2e-3);
  warp_batch<Npp16f>("nppiWarpAffineBatch_16f_C3R", "nppiWarpAffineBatch/16f_C3/linear", 7, 5, 3, 8, kThree,
                     NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineBatch_16f_C3R_Ctx, 511, 0, 2e-3, 2e-3);
  warp_batch<Npp16f>("nppiWarpAffineBatch_16f_C4R", "nppiWarpAffineBatch/16f_C4/nn", 7, 5, 4, 8, kTwo,
                     NPPI_INTER_NN, kFull, kFull, &nppiWarpAffineBatch_16f_C4R_Ctx, 512, 0, 2e-3, 2e-3);
}

TEST_F(NppiWarpAffineBatchDiff, Batch_32f) {
  warp_batch<Npp32f>("nppiWarpAffineBatch_32f_C1R", "nppiWarpAffineBatch/32f_C1/linear", 7, 5, 1, 8, kTwo,
                     NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineBatch_32f_C1R_Ctx, 520);
  warp_batch<Npp32f>("nppiWarpAffineBatch_32f_C3R", "nppiWarpAffineBatch/32f_C3/nn", 7, 5, 3, 8, kThree,
                     NPPI_INTER_NN, kFull, kFull, &nppiWarpAffineBatch_32f_C3R_Ctx, 521);
  // NB: NVIDIA NPP 13.4 crashes (misaligned address) on the batch CUBIC path
  // unless every row start is 16-byte aligned; pad 16 keeps step = 128.
  warp_batch<Npp32f>("nppiWarpAffineBatch_32f_C4R", "nppiWarpAffineBatch/32f_C4/cubic", 7, 5, 4, 16, kTwo,
                     NPPI_INTER_CUBIC, kFull, kFull, &nppiWarpAffineBatch_32f_C4R_Ctx, 522, 0, 1e-4, 1e-4);
  // 16-byte aligned rows: NVIDIA NPP 13.4 vectorizes the 32f 4-channel batch
  // path and faults on unaligned rows (see the comment on the C4 case above).
  warp_batch<Npp32f>("nppiWarpAffineBatch_32f_AC4R", "nppiWarpAffineBatch/32f_AC4/linear", 7, 5, 4, 16, kThree,
                     NPPI_INTER_LINEAR, kFull, kFull, &nppiWarpAffineBatch_32f_AC4R_Ctx, 523);
}

TEST_F(NppiWarpAffineBatchDiff, Errors) {
  warp_batch<Npp8u>("nppiWarpAffineBatch_8u_C1R", "nppiWarpAffineBatch/err/srcsize_1", 1, 5, 1, 8, kTwo,
                    NPPI_INTER_NN, NppiRect{0, 0, 1, 5}, NppiRect{0, 0, 1, 5}, &nppiWarpAffineBatch_8u_C1R_Ctx,
                    530);
  warp_batch<Npp8u>("nppiWarpAffineBatch_8u_C1R", "nppiWarpAffineBatch/err/interp_super", 7, 5, 1, 8, kTwo,
                    NPPI_INTER_SUPER, kFull, kFull, &nppiWarpAffineBatch_8u_C1R_Ctx, 531);
  warp_batch<Npp8u>("nppiWarpAffineBatch_8u_C1R", "nppiWarpAffineBatch/err/srcroi_outside", 7, 5, 1, 8, kTwo,
                    NPPI_INTER_NN, NppiRect{20, 20, 4, 4}, kFull, &nppiWarpAffineBatch_8u_C1R_Ctx, 532);
  // Valid list, batch size 0 (no kernel launch expected).
  {
    std::vector<NppiWarpAffineBatchCXR> hl(1);
    batch_entry(hl[0], nullptr, 0, nullptr, 0, nullptr);
    DeviceMemory<NppiWarpAffineBatchCXR> &bl = arena<NppiWarpAffineBatchCXR>(1);
    bl.copyFromHost(hl);
    clear_cuda_error();
    NppStatus st = nppiWarpAffineBatch_8u_C1R_Ctx(NppiSize{7, 5}, kFull, kFull, NPPI_INTER_NN, bl.get(), 0,
                                                  current_ctx());
    Case c("nppiWarpAffineBatch/err/batchsize0", "nppiWarpAffineBatch_8u_C1R");
    c.status(st);
    c.finish();
  }
  // Null batch list.
  {
    clear_cuda_error();
    NppStatus st = nppiWarpAffineBatch_8u_C1R_Ctx(NppiSize{7, 5}, kFull, kFull, NPPI_INTER_NN, nullptr, 2,
                                                  current_ctx());
    Case c("nppiWarpAffineBatch/err/nulllist", "nppiWarpAffineBatch_8u_C1R");
    c.status(st);
    c.finish();
  }
}

TEST_F(NppiWarpAffineBatchDiff, Probes_Roi) {
  // nBatchSize must be > 1 (NPP_BAD_ARGUMENT_ERROR otherwise), so the identity
  // probes carry two entries.
  const std::vector<std::array<double, 6>> kId = {{1.0, 0.0, 0.0, 0.0, 1.0, 0.0},
                                                  {1.0, 0.0, 0.0, 0.0, 1.0, 0.0}};
  // Partial src-ROI overlap: intersection with the 7x5 image is 2x2.
  warp_batch<Npp8u>("nppiWarpAffineBatch_8u_C1R", "nppiWarpAffineBatch/roi/srcroi_partial", 7, 5, 1, 8, kTwo,
                    NPPI_INTER_NN, NppiRect{5, 3, 4, 4}, kFull, &nppiWarpAffineBatch_8u_C1R_Ctx, 540);
  // Mapping semantics: src ROI smaller than dst ROI, identity transform.
  warp_batch<Npp8u>("nppiWarpAffineBatch_8u_C1R", "nppiWarpAffineBatch/roi/srcroi_mapping", 7, 5, 1, 8, kId,
                    NPPI_INTER_NN, NppiRect{2, 1, 5, 4}, NppiRect{0, 0, 7, 5}, &nppiWarpAffineBatch_8u_C1R_Ctx,
                    541);
  warp_batch<Npp8u>("nppiWarpAffineBatch_8u_C1R", "nppiWarpAffineBatch/roi/srcroi_mapping_eq", 7, 5, 1, 8, kId,
                    NPPI_INTER_NN, NppiRect{2, 1, 5, 4}, NppiRect{0, 0, 5, 4}, &nppiWarpAffineBatch_8u_C1R_Ctx,
                    542);
  // Non-zero destination ROI offset: dst writes land at absolute coordinates
  // and only pixels mapped inside the source ROI are written.
  warp_batch<Npp8u>("nppiWarpAffineBatch_8u_C1R", "nppiWarpAffineBatch/roi/dstroi_offset", 7, 5, 1, 8, kId,
                    NPPI_INTER_NN, NppiRect{2, 1, 5, 4}, NppiRect{1, 1, 4, 3}, &nppiWarpAffineBatch_8u_C1R_Ctx, 545);
  // Linear taps that cross the source ROI still read the full image.
  warp_batch<Npp8u>("nppiWarpAffineBatch_8u_C1R", "nppiWarpAffineBatch/roi/srcroi_partial_linear", 7, 5, 1, 8, kTwo,
                    NPPI_INTER_LINEAR, NppiRect{5, 3, 4, 4}, kFull, &nppiWarpAffineBatch_8u_C1R_Ctx, 546);
  warp_batch<Npp8u>("nppiWarpAffineBatch_8u_C1R", "nppiWarpAffineBatch/roi/dstroi_zero", 7, 5, 1, 8, kTwo,
                    NPPI_INTER_NN, kFull, NppiRect{0, 0, 7, 0}, &nppiWarpAffineBatch_8u_C1R_Ctx, 543);
  warp_batch<Npp8u>("nppiWarpAffineBatch_8u_C1R", "nppiWarpAffineBatch/roi/dstroi_neg", 7, 5, 1, 8, kTwo,
                    NPPI_INTER_NN, kFull, NppiRect{0, 0, 7, -3}, &nppiWarpAffineBatch_8u_C1R_Ctx, 544);
  // NB: a batch C4 step that is not a multiple of 4*sizeof(T) is NOT
  // recorded: NVIDIA NPP 13.4 faults (async misaligned address) on the
  // vectorized 4-channel batch path for odd 4-byte steps, for every dtype.
  // mcMPP returns NPP_NOT_EVEN_STEP_ERROR there instead, which is strictly
  // safer.
}
