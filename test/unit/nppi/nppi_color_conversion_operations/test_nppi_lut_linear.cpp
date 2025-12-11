#include "npp.h"
#include "framework/npp_test_base.h"
#include <gtest/gtest.h>
#include <vector>

using namespace npp_functional_test;

/**
 * @brief NPP LUT (Look-Up Table) Linear Interpolation Tests
 *
 * Tests for nppiLUT_Linear_8u_C1R family of functions.
 * These tests verify linear interpolation lookup table operations
 * on 8-bit unsigned single-channel images.
 */
class NPPILUTTest : public NppTestBase {
protected:
  void SetUp() override {
    NppTestBase::SetUp();
    width = 16;
    height = 12;
    roi.width = width;
    roi.height = height;
  }

  int width, height;
  NppiSize roi;
};

/**
 * @brief Test nppiLUT_Linear_8u_C1R basic functionality
 *
 * Verifies linear interpolation LUT with inversion mapping:
 * - Input 0 -> Output 255
 * - Input 128 -> Output 127
 * - Input 255 -> Output 0
 */
TEST_F(NPPILUTTest, LUT_Linear_8u_C1R) {
  size_t dataSize = width * height;

  std::vector<Npp8u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i++) {
    srcData[i] = (Npp8u)(i % 256);
  }

  std::vector<Npp32s> pLevels = {0, 128, 255};
  std::vector<Npp32s> pValues = {255, 127, 0};
  int nLevels = pLevels.size();

  NppImageMemory<Npp8u> src(width, height, 1);
  NppImageMemory<Npp8u> dst(width, height, 1);

  Npp32s *d_pValues, *d_pLevels;
  cudaMalloc(&d_pValues, nLevels * sizeof(Npp32s));
  cudaMalloc(&d_pLevels, nLevels * sizeof(Npp32s));
  cudaMemcpy(d_pValues, pValues.data(), nLevels * sizeof(Npp32s), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pLevels, pLevels.data(), nLevels * sizeof(Npp32s), cudaMemcpyHostToDevice);

  src.copyFromHost(srcData);

  NppStatus status = nppiLUT_Linear_8u_C1R(
      src.get(), src.step(),
      dst.get(), dst.step(),
      roi,
      d_pValues, d_pLevels, nLevels
  );

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> dstData;
  dst.copyToHost(dstData);

  EXPECT_EQ(dstData[0], 255);
  EXPECT_EQ(dstData[128], 127);
  if (dataSize > 255) {
    EXPECT_EQ(dstData[255], 0);
  }

  cudaFree(d_pValues);
  cudaFree(d_pLevels);
}

/**
 * @brief Test nppiLUT_Linear_8u_C1R_Ctx with stream context
 *
 * Verifies the context-aware version that supports custom CUDA streams.
 */
TEST_F(NPPILUTTest, LUT_Linear_8u_C1R_Ctx) {
  size_t dataSize = width * height;

  std::vector<Npp8u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i++) {
    srcData[i] = (Npp8u)(i % 256);
  }

  std::vector<Npp32s> pLevels = {0, 128, 255};
  std::vector<Npp32s> pValues = {255, 127, 0};
  int nLevels = pLevels.size();

  NppImageMemory<Npp8u> src(width, height, 1);
  NppImageMemory<Npp8u> dst(width, height, 1);

  Npp32s *d_pValues, *d_pLevels;
  cudaMalloc(&d_pValues, nLevels * sizeof(Npp32s));
  cudaMalloc(&d_pLevels, nLevels * sizeof(Npp32s));
  cudaMemcpy(d_pValues, pValues.data(), nLevels * sizeof(Npp32s), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pLevels, pLevels.data(), nLevels * sizeof(Npp32s), cudaMemcpyHostToDevice);

  src.copyFromHost(srcData);

  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status = nppiLUT_Linear_8u_C1R_Ctx(
      src.get(), src.step(),
      dst.get(), dst.step(),
      roi,
      d_pValues, d_pLevels, nLevels,
      nppStreamCtx
  );

  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> dstData;
  dst.copyToHost(dstData);

  EXPECT_EQ(dstData[0], 255);
  EXPECT_EQ(dstData[128], 127);
  if (dataSize > 255) {
    EXPECT_EQ(dstData[255], 0);
  }

  cudaFree(d_pValues);
  cudaFree(d_pLevels);
}
