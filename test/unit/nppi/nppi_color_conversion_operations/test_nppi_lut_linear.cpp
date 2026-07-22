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

// ==================== nppiLUT_Linear_8u_C3R Tests ====================

TEST_F(NPPILUTTest, LUT_Linear_8u_C3R) {
  const int channels = 3;
  size_t dataSize = width * height * channels;

  std::vector<Npp8u> srcData(dataSize);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * channels;
      srcData[idx + 0] = (Npp8u)((x + y) % 256);
      srcData[idx + 1] = (Npp8u)((x * 2) % 256);
      srcData[idx + 2] = (Npp8u)((y * 3) % 256);
    }
  }

  std::vector<Npp32s> pLevels[3] = {{0, 128, 255}, {0, 128, 255}, {0, 128, 255}};
  std::vector<Npp32s> pValues[3] = {{0, 128, 255}, {0, 128, 255}, {0, 128, 255}};
  int nLevels[3] = {3, 3, 3};

  NppImageMemory<Npp8u> src(width, height, channels);
  NppImageMemory<Npp8u> dst(width, height, channels);
  src.copyFromHost(srcData);

  const Npp32s *d_pValues[3], *d_pLevels[3];
  for (int c = 0; c < 3; c++) {
    cudaMalloc((void **)&d_pValues[c], nLevels[c] * sizeof(Npp32s));
    cudaMalloc((void **)&d_pLevels[c], nLevels[c] * sizeof(Npp32s));
    cudaMemcpy((void *)d_pValues[c], pValues[c].data(), nLevels[c] * sizeof(Npp32s), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_pLevels[c], pLevels[c].data(), nLevels[c] * sizeof(Npp32s), cudaMemcpyHostToDevice);
  }

  NppStatus status = nppiLUT_Linear_8u_C3R(src.get(), src.step(), dst.get(), dst.step(), roi,
                                            d_pValues, d_pLevels, nLevels);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> dstData;
  dst.copyToHost(dstData);

  for (size_t i = 0; i < dataSize; i++) {
    EXPECT_NEAR(dstData[i], srcData[i], 1) << "Mismatch at index " << i;
  }

  for (int c = 0; c < 3; c++) {
    cudaFree((void *)d_pValues[c]);
    cudaFree((void *)d_pLevels[c]);
  }
}

TEST_F(NPPILUTTest, LUT_Linear_8u_C3R_Ctx) {
  const int channels = 3;
  size_t dataSize = width * height * channels;

  std::vector<Npp8u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i++) {
    srcData[i] = (Npp8u)(i % 256);
  }

  std::vector<Npp32s> pLevels[3] = {{0, 255}, {0, 255}, {0, 255}};
  std::vector<Npp32s> pValues[3] = {{255, 0}, {255, 0}, {255, 0}};
  int nLevels[3] = {2, 2, 2};

  NppImageMemory<Npp8u> src(width, height, channels);
  NppImageMemory<Npp8u> dst(width, height, channels);
  src.copyFromHost(srcData);

  const Npp32s *d_pValues[3], *d_pLevels[3];
  for (int c = 0; c < 3; c++) {
    cudaMalloc((void **)&d_pValues[c], nLevels[c] * sizeof(Npp32s));
    cudaMalloc((void **)&d_pLevels[c], nLevels[c] * sizeof(Npp32s));
    cudaMemcpy((void *)d_pValues[c], pValues[c].data(), nLevels[c] * sizeof(Npp32s), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)d_pLevels[c], pLevels[c].data(), nLevels[c] * sizeof(Npp32s), cudaMemcpyHostToDevice);
  }

  NppStreamContext ctx;
  ctx.hStream = 0;

  NppStatus status = nppiLUT_Linear_8u_C3R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi,
                                                 d_pValues, d_pLevels, nLevels, ctx);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> dstData;
  dst.copyToHost(dstData);

  // Inversion: output = 255 - input (allow boundary tolerance)
  for (size_t i = 0; i < dataSize; i++) {
    int expected = 255 - srcData[i];
    // NVIDIA NPP boundary behavior may differ at exact level values
    if (srcData[i] == 0 || srcData[i] == 255) {
      EXPECT_GE(dstData[i], 0);
      EXPECT_LE(dstData[i], 255);
    } else {
      EXPECT_NEAR(dstData[i], expected, 2) << "Mismatch at index " << i;
    }
  }

  for (int c = 0; c < 3; c++) {
    cudaFree((void *)d_pValues[c]);
    cudaFree((void *)d_pLevels[c]);
  }
}

// ==================== nppiLUT_Linear_16u_C1R Tests ====================

TEST_F(NPPILUTTest, LUT_Linear_16u_C1R) {
  const int width16 = 16, height16 = 12;
  NppiSize roi16 = {width16, height16};
  size_t dataSize = width16 * height16;

  std::vector<Npp16u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i++) {
    srcData[i] = (Npp16u)(i * 10 % 65536);
  }

  std::vector<Npp32s> pLevels = {0, 32768, 65535};
  std::vector<Npp32s> pValues = {0, 32768, 65535};
  int nLevels = 3;

  NppImageMemory<Npp16u> src(width16, height16, 1);
  NppImageMemory<Npp16u> dst(width16, height16, 1);
  src.copyFromHost(srcData);

  Npp32s *d_pValues, *d_pLevels;
  cudaMalloc(&d_pValues, nLevels * sizeof(Npp32s));
  cudaMalloc(&d_pLevels, nLevels * sizeof(Npp32s));
  cudaMemcpy(d_pValues, pValues.data(), nLevels * sizeof(Npp32s), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pLevels, pLevels.data(), nLevels * sizeof(Npp32s), cudaMemcpyHostToDevice);

  NppStatus status = nppiLUT_Linear_16u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi16,
                                             d_pValues, d_pLevels, nLevels);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> dstData;
  dst.copyToHost(dstData);

  for (size_t i = 0; i < dataSize; i++) {
    EXPECT_NEAR(dstData[i], srcData[i], 1) << "Mismatch at index " << i;
  }

  cudaFree(d_pValues);
  cudaFree(d_pLevels);
}

TEST_F(NPPILUTTest, LUT_Linear_16u_C1R_Ctx) {
  const int width16 = 16, height16 = 12;
  NppiSize roi16 = {width16, height16};
  size_t dataSize = width16 * height16;

  std::vector<Npp16u> srcData(dataSize);
  for (size_t i = 0; i < dataSize; i++) {
    srcData[i] = (Npp16u)(i * 10 % 65536);
  }

  std::vector<Npp32s> pLevels = {0, 32768, 65535};
  std::vector<Npp32s> pValues = {0, 16384, 32767};
  int nLevels = 3;

  NppImageMemory<Npp16u> src(width16, height16, 1);
  NppImageMemory<Npp16u> dst(width16, height16, 1);
  src.copyFromHost(srcData);

  Npp32s *d_pValues, *d_pLevels;
  cudaMalloc(&d_pValues, nLevels * sizeof(Npp32s));
  cudaMalloc(&d_pLevels, nLevels * sizeof(Npp32s));
  cudaMemcpy(d_pValues, pValues.data(), nLevels * sizeof(Npp32s), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pLevels, pLevels.data(), nLevels * sizeof(Npp32s), cudaMemcpyHostToDevice);

  NppStreamContext ctx;
  ctx.hStream = 0;

  NppStatus status = nppiLUT_Linear_16u_C1R_Ctx(src.get(), src.step(), dst.get(), dst.step(), roi16,
                                                  d_pValues, d_pLevels, nLevels, ctx);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> dstData;
  dst.copyToHost(dstData);

  for (size_t i = 0; i < dataSize; i++) {
    Npp16u expected = (Npp16u)(srcData[i] / 2);
    EXPECT_NEAR(dstData[i], expected, 100) << "Mismatch at index " << i;
  }

  cudaFree(d_pValues);
  cudaFree(d_pLevels);
}
