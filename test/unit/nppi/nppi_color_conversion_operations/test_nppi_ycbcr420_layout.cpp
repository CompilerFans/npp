#include "npp.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

namespace {

class YCbCr420LayoutTest : public ::testing::Test {
protected:
  void SetUp() override { ASSERT_EQ(cudaSetDevice(0), cudaSuccess); }
  void TearDown() override { EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess); }
};

TEST_F(YCbCr420LayoutTest, P2P3RAndP3P2RRoundTrip) {
  const int width = 8;
  const int height = 6;
  const NppiSize roi{width, height};
  std::vector<Npp8u> y(static_cast<size_t>(width) * height);
  std::vector<Npp8u> cbcr(static_cast<size_t>(width) * height / 2);
  for (size_t i = 0; i < y.size(); ++i) {
    y[i] = static_cast<Npp8u>((17 + i * 3) % 251);
  }
  for (int row = 0; row < height / 2; ++row) {
    for (int x = 0; x < width; x += 2) {
      cbcr[static_cast<size_t>(row) * width + x] = static_cast<Npp8u>(61 + row * 7 + x);
      cbcr[static_cast<size_t>(row) * width + x + 1] = static_cast<Npp8u>(113 + row * 5 + x);
    }
  }

  int srcYStep = 0;
  int srcCbCrStep = 0;
  int dstYStep = 0;
  int dstCbStep = 0;
  int dstCrStep = 0;
  int mergedYStep = 0;
  int mergedCbCrStep = 0;
  Npp8u *dSrcY = nppiMalloc_8u_C1(width, height, &srcYStep);
  Npp8u *dSrcCbCr = nppiMalloc_8u_C1(width, height / 2, &srcCbCrStep);
  Npp8u *dDstY = nppiMalloc_8u_C1(width, height, &dstYStep);
  Npp8u *dDstCb = nppiMalloc_8u_C1(width / 2, height / 2, &dstCbStep);
  Npp8u *dDstCr = nppiMalloc_8u_C1(width / 2, height / 2, &dstCrStep);
  Npp8u *dMergedY = nppiMalloc_8u_C1(width, height, &mergedYStep);
  Npp8u *dMergedCbCr = nppiMalloc_8u_C1(width, height / 2, &mergedCbCrStep);
  ASSERT_NE(dSrcY, nullptr);
  ASSERT_NE(dSrcCbCr, nullptr);
  ASSERT_NE(dDstY, nullptr);
  ASSERT_NE(dDstCb, nullptr);
  ASSERT_NE(dDstCr, nullptr);
  ASSERT_NE(dMergedY, nullptr);
  ASSERT_NE(dMergedCbCr, nullptr);
  ASSERT_EQ(cudaMemcpy2D(dSrcY, srcYStep, y.data(), width, width, height, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(dSrcCbCr, srcCbCrStep, cbcr.data(), width, width, height / 2, cudaMemcpyHostToDevice),
            cudaSuccess);

  Npp8u *dstPlanes[3] = {dDstY, dDstCb, dDstCr};
  int dstSteps[3] = {dstYStep, dstCbStep, dstCrStep};
  ASSERT_EQ(nppiYCbCr420_8u_P2P3R(dSrcY, srcYStep, dSrcCbCr, srcCbCrStep, dstPlanes, dstSteps, roi), NPP_SUCCESS);
  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);
  ASSERT_EQ(nppiYCbCr420_8u_P2P3R_Ctx(dSrcY, srcYStep, dSrcCbCr, srcCbCrStep, dstPlanes, dstSteps, roi, context),
            NPP_SUCCESS);

  std::vector<Npp8u> outY(static_cast<size_t>(dstYStep) * height);
  std::vector<Npp8u> outCb(static_cast<size_t>(dstCbStep) * height / 2);
  std::vector<Npp8u> outCr(static_cast<size_t>(dstCrStep) * height / 2);
  ASSERT_EQ(cudaMemcpy(outY.data(), dDstY, outY.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(outCb.data(), dDstCb, outCb.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(outCr.data(), dDstCr, outCr.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  for (int row = 0; row < height; ++row) {
    for (int x = 0; x < width; ++x) {
      EXPECT_EQ(outY[static_cast<size_t>(row) * dstYStep + x], y[static_cast<size_t>(row) * width + x]);
    }
  }
  for (int row = 0; row < height / 2; ++row) {
    for (int x = 0; x < width / 2; ++x) {
      EXPECT_EQ(outCb[static_cast<size_t>(row) * dstCbStep + x], cbcr[static_cast<size_t>(row) * width + x * 2]);
      EXPECT_EQ(outCr[static_cast<size_t>(row) * dstCrStep + x],
                cbcr[static_cast<size_t>(row) * width + x * 2 + 1]);
    }
  }

  const Npp8u *srcPlanes[3] = {dDstY, dDstCb, dDstCr};
  int srcSteps[3] = {dstYStep, dstCbStep, dstCrStep};
  ASSERT_EQ(nppiYCbCr420_8u_P3P2R_Ctx(srcPlanes, srcSteps, dMergedY, mergedYStep, dMergedCbCr, mergedCbCrStep, roi,
                                      context),
            NPP_SUCCESS);
  ASSERT_EQ(nppiYCbCr420_8u_P3P2R(srcPlanes, srcSteps, dMergedY, mergedYStep, dMergedCbCr, mergedCbCrStep, roi),
            NPP_SUCCESS);

  std::vector<Npp8u> roundTripY(static_cast<size_t>(mergedYStep) * height);
  std::vector<Npp8u> roundTripCbCr(static_cast<size_t>(mergedCbCrStep) * height / 2);
  ASSERT_EQ(cudaMemcpy(roundTripY.data(), dMergedY, roundTripY.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(roundTripCbCr.data(), dMergedCbCr, roundTripCbCr.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  for (int row = 0; row < height; ++row) {
    for (int x = 0; x < width; ++x) {
      EXPECT_EQ(roundTripY[static_cast<size_t>(row) * mergedYStep + x], y[static_cast<size_t>(row) * width + x]);
    }
  }
  for (int row = 0; row < height / 2; ++row) {
    for (int x = 0; x < width; ++x) {
      EXPECT_EQ(roundTripCbCr[static_cast<size_t>(row) * mergedCbCrStep + x],
                cbcr[static_cast<size_t>(row) * width + x]);
    }
  }

  nppiFree(dSrcY);
  nppiFree(dSrcCbCr);
  nppiFree(dDstY);
  nppiFree(dDstCb);
  nppiFree(dDstCr);
  nppiFree(dMergedY);
  nppiFree(dMergedCbCr);
}

TEST_F(YCbCr420LayoutTest, YCbCr420ToBGRMatchesKnownValues) {
  const int width = 4;
  const int height = 4;
  const NppiSize roi{width, height};
  const Npp8u y[16] = {32, 46, 61, 75, 41, 55, 69, 83, 49, 64, 78, 92, 58, 72, 86, 101};
  const Npp8u cb[4] = {135, 124, 138, 127};
  const Npp8u cr[4] = {125, 141, 121, 138};
  const Npp8u expected[48] = {32, 18, 13, 49, 34, 30, 44, 43, 73, 60, 59, 89, 43, 28, 24, 59,
                              45, 40, 53, 52, 82, 69, 68, 98, 58, 40, 27, 76, 57, 44, 70, 64,
                              88, 86, 80, 104, 69, 50, 37, 85, 66, 54, 79, 73, 97, 96, 91, 114};

  int yStep = 0;
  int cbStep = 0;
  int crStep = 0;
  int dstStep = 0;
  Npp8u *dY = nppiMalloc_8u_C1(width, height, &yStep);
  Npp8u *dCb = nppiMalloc_8u_C1(width / 2, height / 2, &cbStep);
  Npp8u *dCr = nppiMalloc_8u_C1(width / 2, height / 2, &crStep);
  Npp8u *dDst = nppiMalloc_8u_C3(width, height, &dstStep);
  ASSERT_NE(dY, nullptr);
  ASSERT_NE(dCb, nullptr);
  ASSERT_NE(dCr, nullptr);
  ASSERT_NE(dDst, nullptr);
  ASSERT_EQ(cudaMemcpy2D(dY, yStep, y, width, width, height, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(dCb, cbStep, cb, width / 2, width / 2, height / 2, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(dCr, crStep, cr, width / 2, width / 2, height / 2, cudaMemcpyHostToDevice), cudaSuccess);
  const Npp8u *planes[3] = {dY, dCb, dCr};
  int steps[3] = {yStep, cbStep, crStep};

  ASSERT_EQ(nppiYCbCr420ToBGR_8u_P3C3R(planes, steps, dDst, dstStep, roi), NPP_SUCCESS);
  std::vector<Npp8u> output(static_cast<size_t>(dstStep) * height);
  ASSERT_EQ(cudaMemcpy(output.data(), dDst, output.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  for (int row = 0; row < height; ++row) {
    for (int x = 0; x < width * 3; ++x) {
      EXPECT_EQ(output[static_cast<size_t>(row) * dstStep + x], expected[static_cast<size_t>(row) * width * 3 + x]);
    }
  }

  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);
  EXPECT_EQ(nppiYCbCr420ToBGR_8u_P3C3R_Ctx(planes, steps, dDst, dstStep, roi, context), NPP_SUCCESS);

  nppiFree(dY);
  nppiFree(dCb);
  nppiFree(dCr);
  nppiFree(dDst);
}

TEST_F(YCbCr420LayoutTest, ValidatesArguments) {
  Npp8u *planes[3] = {nullptr, nullptr, nullptr};
  int steps[3] = {8, 4, 4};
  EXPECT_EQ(nppiYCbCr420_8u_P2P3R(nullptr, 8, nullptr, 8, planes, steps, {8, 4}), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiYCbCr420_8u_P3P2R(nullptr, steps, nullptr, 8, nullptr, 8, {8, 4}), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiYCbCr420ToBGR_8u_P3C3R(nullptr, steps, nullptr, 24, {8, 4}), NPP_NULL_POINTER_ERROR);
}

} // namespace
