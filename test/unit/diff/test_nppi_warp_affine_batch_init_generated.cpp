// Differential tests for nppiWarpAffineBatchInit (package g1).
//
// Records the aTransformedCoeffs written into the (device-resident) batch
// list: identity / translation / scale / rotation maps plus a singular map
// (pre-filled sentinel distinguishes "zeroed" from "left untouched").

#include "npp_warp_affine_diff_helpers.h"

#include <array>
#include <vector>

namespace {

using warp_affine_diff::batch_init_case;

}  // namespace

class NppiWarpAffineBatchInitDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(NppiWarpAffineBatchInitDiff, Invert) {
  batch_init_case("nppiWarpAffineBatchInit", "nppiWarpAffineBatchInit/regular", {{1, 0, 0, 0, 1, 0},
                                                                                  {1, 0, 2.5, 0, 1, -1.5},
                                                                                  {1.25, 0, 0.5, 0, 0.8, -0.75}},
                  600);
  batch_init_case("nppiWarpAffineBatchInit", "nppiWarpAffineBatchInit/rotate",
                  {{0.8660254, -0.5, 3.0, 0.5, 0.8660254, -2.0}}, 601);
  batch_init_case("nppiWarpAffineBatchInit", "nppiWarpAffineBatchInit/singular", {{1, 2, 0, 2, 4, 0}}, 602);
}

TEST_F(NppiWarpAffineBatchInitDiff, Errors) {
  // NB: a null pBatchList is NOT recorded: NVIDIA NPP 13.4 dereferences it on
  // the device (async illegal-address fault).  mcMPP returns
  // NPP_NULL_POINTER_ERROR there instead, which is strictly safer.
  // Batch size 0 with a valid list pointer.
  {
    std::vector<NppiWarpAffineBatchCXR> hl(1);
    warp_affine_diff::batch_entry(hl[0], nullptr, 0, nullptr, 0, nullptr);
    DeviceMemory<NppiWarpAffineBatchCXR> &bl = warp_affine_diff::arena<NppiWarpAffineBatchCXR>(1);
    bl.copyFromHost(hl);
    clear_cuda_error();
    NppStatus st = nppiWarpAffineBatchInit_Ctx(bl.get(), 0, warp_affine_diff::current_ctx());
    Case c("nppiWarpAffineBatchInit/err/batchsize0", "nppiWarpAffineBatchInit");
    c.status(st);
    c.finish();
  }
}

TEST_F(NppiWarpAffineBatchInitDiff, Probes) {
  // Negative determinant (mirror): inverse of [[-1,0],[0,1]] is itself.
  batch_init_case("nppiWarpAffineBatchInit", "nppiWarpAffineBatchInit/mirror", {{-1.0, 0.0, 3.0, 0.0, 1.0, 2.0}},
                  610);
  // Extra singular matrices, to pin down the implementation-defined output.
  batch_init_case("nppiWarpAffineBatchInit", "nppiWarpAffineBatchInit/singular2", {{2, 1, 3, 4, 2, 5}}, 611);
  batch_init_case("nppiWarpAffineBatchInit", "nppiWarpAffineBatchInit/singular3", {{0, 0, 1, 0, 0, 1}}, 612);
}
