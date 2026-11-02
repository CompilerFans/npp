// Status-contract differential tests for nppiSegmentWatershed_16u.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   tools/remote5090.sh record 'NppiWatershedDiff.*'
//
// The reference pSrcDst output is an internal cell-state image and its exact
// flooding order is not reproduced (see nppi_watershed.cu), so these cases
// check only the measured host contract: null src/buffer -8, zero or negative
// ROI -6, unsupported norm -5, unsupported boundary type -2, optional labels.
#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;

namespace {

NppStreamContext current_context() {
  NppStreamContext ctx{};
  NppStatus st = nppGetStreamContext(&ctx);
  EXPECT_EQ(st, NPP_NO_ERROR) << "nppGetStreamContext failed";
  return ctx;
}

void status_case(const std::string &id, NppStatus st) {
  Case c(id, "nppiSegmentWatershed_16u_C1IR_Ctx");
  c.status(st);
  c.finish();
}

}  // namespace

TEST(NppiWatershedDiff, Contract) {
  const int w = 16, h = 16;
  std::vector<Npp16u> img((size_t)w * h, 5);
  DeviceMemory<Npp16u> src(img.size());
  src.copyFromHost(img);
  DeviceMemory<Npp32u> labels(img.size());
  size_t bufferSize = 0;
  nppiSegmentWatershedGetBufferSize_16u_C1R({w, h}, &bufferSize);
  DeviceMemory<Npp8u> buffer(bufferSize);
  NppStreamContext ctx = current_context();

  status_case("s2/Watershed/valid_none", nppiSegmentWatershed_16u_C1IR_Ctx(
                                              src.get(), w * 2, labels.get(), w * 4, nppiNormL1,
                                              NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE, {w, h}, buffer.get(), ctx));
  status_case("s2/Watershed/valid_black", nppiSegmentWatershed_16u_C1IR_Ctx(
                                               src.get(), w * 2, labels.get(), w * 4, nppiNormL1,
                                               NPP_WATERSHED_SEGMENT_BOUNDARIES_BLACK, {w, h}, buffer.get(), ctx));
  status_case("s2/Watershed/no_labels", nppiSegmentWatershed_16u_C1IR_Ctx(
                                             src.get(), w * 2, nullptr, 0, nppiNormL1,
                                             NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE, {w, h}, buffer.get(), ctx));
  status_case("s2/Watershed/null_src", nppiSegmentWatershed_16u_C1IR_Ctx(
                                            nullptr, w * 2, labels.get(), w * 4, nppiNormL1,
                                            NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE, {w, h}, buffer.get(), ctx));
  status_case("s2/Watershed/null_buffer", nppiSegmentWatershed_16u_C1IR_Ctx(
                                               src.get(), w * 2, labels.get(), w * 4, nppiNormL1,
                                               NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE, {w, h}, nullptr, ctx));
  status_case("s2/Watershed/roi_zero_w", nppiSegmentWatershed_16u_C1IR_Ctx(
                                              src.get(), w * 2, labels.get(), w * 4, nppiNormL1,
                                              NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE, {0, h}, buffer.get(), ctx));
  status_case("s2/Watershed/roi_zero_h", nppiSegmentWatershed_16u_C1IR_Ctx(
                                              src.get(), w * 2, labels.get(), w * 4, nppiNormL1,
                                              NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE, {w, 0}, buffer.get(), ctx));
  status_case("s2/Watershed/roi_negative", nppiSegmentWatershed_16u_C1IR_Ctx(
                                                src.get(), w * 2, labels.get(), w * 4, nppiNormL1,
                                                NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE, {-1, h}, buffer.get(), ctx));
  status_case("s2/Watershed/norm_bad", nppiSegmentWatershed_16u_C1IR_Ctx(
                                            src.get(), w * 2, labels.get(), w * 4, (NppiNorm)7,
                                            NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE, {w, h}, buffer.get(), ctx));
  status_case("s2/Watershed/boundary_bad", nppiSegmentWatershed_16u_C1IR_Ctx(
                                                src.get(), w * 2, labels.get(), w * 4, nppiNormL1,
                                                (NppiWatershedSegmentBoundaryType)9, {w, h}, buffer.get(), ctx));
  clear_cuda_error();
}
