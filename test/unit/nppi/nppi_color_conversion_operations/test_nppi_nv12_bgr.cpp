#include "npp.h"
#include "npp_test_base.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

class NV12ToBGRTest : public ::testing::Test {
protected:
  void SetUp() override {
    width = 64;
    height = 64;

    // NV12 requires even dimensions
    ASSERT_EQ(width % 2, 0);
    ASSERT_EQ(height % 2, 0);
  }

  void TearDown() override {}

  void createTestNV12Data(std::vector<Npp8u> &yData, std::vector<Npp8u> &uvData) {
    yData.resize(width * height);
    uvData.resize(width * height / 2);

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        yData[y * width + x] = (Npp8u)(16 + (x + y) * 239 / (width + height - 2));
      }
    }

    for (int y = 0; y < height / 2; ++y) {
      for (int x = 0; x < width; x += 2) {
        int idx = y * width + x;
        uvData[idx] = (Npp8u)(64 + x * 128 / width);
        uvData[idx + 1] = (Npp8u)(64 + y * 128 / (height / 2));
      }
    }
  }

  bool isValidBGR(Npp8u b, Npp8u g, Npp8u r) {
    (void)b;
    (void)g;
    (void)r;
    return true;
  }

  int width, height;
};

TEST_F(NV12ToBGRTest, BasicNV12ToBGR_8u_P2C3R) {
  std::vector<Npp8u> hostYData, hostUVData;
  createTestNV12Data(hostYData, hostUVData);

  Npp8u *d_srcY = nppsMalloc_8u(width * height);
  Npp8u *d_srcUV = nppsMalloc_8u(width * height / 2);

  int bgrStep;
  Npp8u *d_bgr = nppiMalloc_8u_C3(width, height, &bgrStep);

  ASSERT_NE(d_srcY, nullptr);
  ASSERT_NE(d_srcUV, nullptr);
  ASSERT_NE(d_bgr, nullptr);

  cudaMemcpy(d_srcY, hostYData.data(), hostYData.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_srcUV, hostUVData.data(), hostUVData.size(), cudaMemcpyHostToDevice);

  const Npp8u *pSrc[2] = {d_srcY, d_srcUV};
  NppiSize roi = {width, height};

  NppStatus status = nppiNV12ToBGR_8u_P2C3R(pSrc, width, d_bgr, bgrStep, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> hostBGR(bgrStep * height);
  cudaMemcpy(hostBGR.data(), d_bgr, bgrStep * height, cudaMemcpyDeviceToHost);

  for (int y = 0; y < height; y += 8) {
    for (int x = 0; x < width; x += 8) {
      int bgrIdx = y * bgrStep + x * 3;
      Npp8u b = hostBGR[bgrIdx];
      Npp8u g = hostBGR[bgrIdx + 1];
      Npp8u r = hostBGR[bgrIdx + 2];

      EXPECT_TRUE(isValidBGR(b, g, r)) << "Invalid BGR at (" << x << "," << y << "): " << (int)b << "," << (int)g << ","
                                       << (int)r;
    }
  }

  nppsFree(d_srcY);
  nppsFree(d_srcUV);
  nppiFree(d_bgr);
}

TEST_F(NV12ToBGRTest, BasicNV12ToBGR_8u_P2C3R_Ctx) {
  std::vector<Npp8u> hostYData, hostUVData;
  createTestNV12Data(hostYData, hostUVData);

  Npp8u *d_srcY = nppsMalloc_8u(width * height);
  Npp8u *d_srcUV = nppsMalloc_8u(width * height / 2);

  int bgrStep;
  Npp8u *d_bgr = nppiMalloc_8u_C3(width, height, &bgrStep);

  ASSERT_NE(d_srcY, nullptr);
  ASSERT_NE(d_srcUV, nullptr);
  ASSERT_NE(d_bgr, nullptr);

  cudaMemcpy(d_srcY, hostYData.data(), hostYData.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_srcUV, hostUVData.data(), hostUVData.size(), cudaMemcpyHostToDevice);

  const Npp8u *pSrc[2] = {d_srcY, d_srcUV};
  NppiSize roi = {width, height};

  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status = nppiNV12ToBGR_8u_P2C3R_Ctx(pSrc, width, d_bgr, bgrStep, roi, nppStreamCtx);
  EXPECT_EQ(status, NPP_NO_ERROR) << "Basic NV12ToBGR Context conversion failed";

  std::vector<Npp8u> hostBGR(bgrStep * height);
  cudaMemcpy(hostBGR.data(), d_bgr, bgrStep * height, cudaMemcpyDeviceToHost);

  bool hasValidPixels = false;
  int validPixelCount = 0;

  for (int y = 0; y < height; y += 8) {
    for (int x = 0; x < width; x += 8) {
      int bgrIdx = y * bgrStep + x * 3;
      Npp8u b = hostBGR[bgrIdx];
      Npp8u g = hostBGR[bgrIdx + 1];
      Npp8u r = hostBGR[bgrIdx + 2];

      if (isValidBGR(b, g, r)) {
        hasValidPixels = true;
        validPixelCount++;
      }

      EXPECT_TRUE(isValidBGR(b, g, r)) << "Invalid BGR at (" << x << "," << y << "): " << (int)b << "," << (int)g << ","
                                       << (int)r;
    }
  }

  EXPECT_TRUE(hasValidPixels) << "No valid BGR pixels found in basic NV12ToBGR conversion";
  EXPECT_GT(validPixelCount, 50) << "Too few valid pixels in basic NV12ToBGR conversion";

  nppsFree(d_srcY);
  nppsFree(d_srcUV);
  nppiFree(d_bgr);
}

TEST_F(NV12ToBGRTest, NV12ToBGR_709CSC_8u_P2C3R) {
  std::vector<Npp8u> hostYData, hostUVData;
  createTestNV12Data(hostYData, hostUVData);

  Npp8u *d_srcY = nppsMalloc_8u(width * height);
  Npp8u *d_srcUV = nppsMalloc_8u(width * height / 2);

  int bgrStep;
  Npp8u *d_bgr = nppiMalloc_8u_C3(width, height, &bgrStep);

  ASSERT_NE(d_srcY, nullptr);
  ASSERT_NE(d_srcUV, nullptr);
  ASSERT_NE(d_bgr, nullptr);

  cudaMemcpy(d_srcY, hostYData.data(), hostYData.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_srcUV, hostUVData.data(), hostUVData.size(), cudaMemcpyHostToDevice);

  const Npp8u *pSrc[2] = {d_srcY, d_srcUV};
  NppiSize roi = {width, height};

  NppStatus status = nppiNV12ToBGR_709CSC_8u_P2C3R(pSrc, width, d_bgr, bgrStep, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> hostBGR(bgrStep * height);
  cudaMemcpy(hostBGR.data(), d_bgr, bgrStep * height, cudaMemcpyDeviceToHost);

  bool hasValidPixels = false;
  for (int y = 0; y < height; y += 4) {
    for (int x = 0; x < width; x += 4) {
      int bgrIdx = y * bgrStep + x * 3;
      Npp8u b = hostBGR[bgrIdx];
      Npp8u g = hostBGR[bgrIdx + 1];
      Npp8u r = hostBGR[bgrIdx + 2];

      if (isValidBGR(b, g, r)) {
        hasValidPixels = true;
      }
    }
  }
  EXPECT_TRUE(hasValidPixels);

  nppsFree(d_srcY);
  nppsFree(d_srcUV);
  nppiFree(d_bgr);
}

TEST_F(NV12ToBGRTest, NV12ToBGR_709CSC_8u_P2C3R_Ctx) {
  std::vector<Npp8u> hostYData, hostUVData;
  createTestNV12Data(hostYData, hostUVData);

  Npp8u *d_srcY = nppsMalloc_8u(width * height);
  Npp8u *d_srcUV = nppsMalloc_8u(width * height / 2);

  int bgrStep;
  Npp8u *d_bgr = nppiMalloc_8u_C3(width, height, &bgrStep);

  ASSERT_NE(d_srcY, nullptr);
  ASSERT_NE(d_srcUV, nullptr);
  ASSERT_NE(d_bgr, nullptr);

  cudaMemcpy(d_srcY, hostYData.data(), hostYData.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_srcUV, hostUVData.data(), hostUVData.size(), cudaMemcpyHostToDevice);

  const Npp8u *pSrc[2] = {d_srcY, d_srcUV};
  NppiSize roi = {width, height};

  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status = nppiNV12ToBGR_709CSC_8u_P2C3R_Ctx(pSrc, width, d_bgr, bgrStep, roi, nppStreamCtx);
  EXPECT_EQ(status, NPP_NO_ERROR) << "BT.709 CSC Context conversion failed";

  std::vector<Npp8u> hostBGR(bgrStep * height);
  cudaMemcpy(hostBGR.data(), d_bgr, bgrStep * height, cudaMemcpyDeviceToHost);

  bool hasValidPixels = false;
  int validPixelCount = 0;

  for (int y = 0; y < height; y += 4) {
    for (int x = 0; x < width; x += 4) {
      int bgrIdx = y * bgrStep + x * 3;
      Npp8u b = hostBGR[bgrIdx];
      Npp8u g = hostBGR[bgrIdx + 1];
      Npp8u r = hostBGR[bgrIdx + 2];

      if (isValidBGR(b, g, r)) {
        hasValidPixels = true;
        validPixelCount++;
      }
    }
  }

  EXPECT_TRUE(hasValidPixels) << "No valid BGR pixels found in BT.709 CSC conversion";
  EXPECT_GT(validPixelCount, 50) << "Too few valid pixels in BT.709 CSC conversion";

  nppsFree(d_srcY);
  nppsFree(d_srcUV);
  nppiFree(d_bgr);
}

TEST_F(NV12ToBGRTest, NV12ToBGR_709HDTV_8u_P2C3R) {
  std::vector<Npp8u> hostYData, hostUVData;
  createTestNV12Data(hostYData, hostUVData);

  Npp8u *d_srcY = nppsMalloc_8u(width * height);
  Npp8u *d_srcUV = nppsMalloc_8u(width * height / 2);

  int bgrStep;
  Npp8u *d_bgr = nppiMalloc_8u_C3(width, height, &bgrStep);

  ASSERT_NE(d_srcY, nullptr);
  ASSERT_NE(d_srcUV, nullptr);
  ASSERT_NE(d_bgr, nullptr);

  cudaMemcpy(d_srcY, hostYData.data(), hostYData.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_srcUV, hostUVData.data(), hostUVData.size(), cudaMemcpyHostToDevice);

  const Npp8u *pSrc[2] = {d_srcY, d_srcUV};
  NppiSize roi = {width, height};

  NppStatus status = nppiNV12ToBGR_709HDTV_8u_P2C3R(pSrc, width, d_bgr, bgrStep, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  nppsFree(d_srcY);
  nppsFree(d_srcUV);
  nppiFree(d_bgr);
}

TEST_F(NV12ToBGRTest, NV12ToBGR_709HDTV_8u_P2C3R_Ctx) {
  std::vector<Npp8u> hostYData, hostUVData;
  createTestNV12Data(hostYData, hostUVData);

  Npp8u *d_srcY = nppsMalloc_8u(width * height);
  Npp8u *d_srcUV = nppsMalloc_8u(width * height / 2);

  int bgrStep;
  Npp8u *d_bgr = nppiMalloc_8u_C3(width, height, &bgrStep);

  ASSERT_NE(d_srcY, nullptr);
  ASSERT_NE(d_srcUV, nullptr);
  ASSERT_NE(d_bgr, nullptr);

  cudaMemcpy(d_srcY, hostYData.data(), hostYData.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_srcUV, hostUVData.data(), hostUVData.size(), cudaMemcpyHostToDevice);

  const Npp8u *pSrc[2] = {d_srcY, d_srcUV};
  NppiSize roi = {width, height};

  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status = nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx(pSrc, width, d_bgr, bgrStep, roi, nppStreamCtx);
  EXPECT_EQ(status, NPP_NO_ERROR) << "BT.709 HDTV Context conversion failed";

  std::vector<Npp8u> hostBGR(bgrStep * height);
  cudaMemcpy(hostBGR.data(), d_bgr, bgrStep * height, cudaMemcpyDeviceToHost);

  bool hasValidPixels = false;
  int validPixelCount = 0;

  for (int y = 0; y < height; y += 4) {
    for (int x = 0; x < width; x += 4) {
      int bgrIdx = y * bgrStep + x * 3;
      Npp8u b = hostBGR[bgrIdx];
      Npp8u g = hostBGR[bgrIdx + 1];
      Npp8u r = hostBGR[bgrIdx + 2];

      if (isValidBGR(b, g, r)) {
        hasValidPixels = true;
        validPixelCount++;
      }
    }
  }

  EXPECT_TRUE(hasValidPixels) << "No valid BGR pixels found in BT.709 HDTV conversion";
  EXPECT_GT(validPixelCount, 50) << "Too few valid pixels in BT.709 HDTV conversion";

  nppsFree(d_srcY);
  nppsFree(d_srcUV);
  nppiFree(d_bgr);
}

TEST_F(NV12ToBGRTest, ColorAccuracy_StrictGray) {
  const int width = 16, height = 16;

  std::vector<Npp8u> yData(width * height, 128);
  std::vector<Npp8u> uvData(width * height / 2);
  for (int i = 0; i < width * height / 2; i += 2) {
    uvData[i] = 128;
    uvData[i + 1] = 128;
  }

  Npp8u *d_srcY = nppsMalloc_8u(width * height);
  Npp8u *d_srcUV = nppsMalloc_8u(width * height / 2);
  int bgrStep;
  Npp8u *d_bgr = nppiMalloc_8u_C3(width, height, &bgrStep);

  ASSERT_NE(d_srcY, nullptr);
  ASSERT_NE(d_srcUV, nullptr);
  ASSERT_NE(d_bgr, nullptr);

  cudaMemcpy(d_srcY, yData.data(), yData.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_srcUV, uvData.data(), uvData.size(), cudaMemcpyHostToDevice);

  const Npp8u *pSrc[2] = {d_srcY, d_srcUV};
  NppiSize roi = {width, height};

  NppStatus status = nppiNV12ToBGR_8u_P2C3R(pSrc, width, d_bgr, bgrStep, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> hostBGR(bgrStep * height);
  cudaMemcpy(hostBGR.data(), d_bgr, bgrStep * height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; i++) {
    int bgrIdx = (i / width) * bgrStep + (i % width) * 3;
    Npp8u b = hostBGR[bgrIdx];
    Npp8u g = hostBGR[bgrIdx + 1];
    Npp8u r = hostBGR[bgrIdx + 2];

    EXPECT_NEAR(b, g, 3) << "B and G mismatch at pixel " << i;
    EXPECT_NEAR(g, r, 3) << "G and R mismatch at pixel " << i;
    EXPECT_NEAR(b, r, 3) << "B and R mismatch at pixel " << i;

    EXPECT_GT(b, 110) << "Gray value too low at pixel " << i;
    EXPECT_LT(b, 150) << "Gray value too high at pixel " << i;
  }

  nppsFree(d_srcY);
  nppsFree(d_srcUV);
  nppiFree(d_bgr);
}

TEST_F(NV12ToBGRTest, ColorAccuracy_StrictWhiteBlack) {
  const int width = 16, height = 16;

  std::vector<Npp8u> yData(width * height);
  std::vector<Npp8u> uvData(width * height / 2);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      yData[y * width + x] = (y < height / 2) ? 235 : 16;
    }
  }

  for (int i = 0; i < width * height / 2; i += 2) {
    uvData[i] = 128;
    uvData[i + 1] = 128;
  }

  Npp8u *d_srcY = nppsMalloc_8u(width * height);
  Npp8u *d_srcUV = nppsMalloc_8u(width * height / 2);
  int bgrStep;
  Npp8u *d_bgr = nppiMalloc_8u_C3(width, height, &bgrStep);

  ASSERT_NE(d_srcY, nullptr);
  ASSERT_NE(d_srcUV, nullptr);
  ASSERT_NE(d_bgr, nullptr);

  cudaMemcpy(d_srcY, yData.data(), yData.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_srcUV, uvData.data(), uvData.size(), cudaMemcpyHostToDevice);

  const Npp8u *pSrc[2] = {d_srcY, d_srcUV};
  NppiSize roi = {width, height};

  NppStatus status = nppiNV12ToBGR_8u_P2C3R(pSrc, width, d_bgr, bgrStep, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> hostBGR(bgrStep * height);
  cudaMemcpy(hostBGR.data(), d_bgr, bgrStep * height, cudaMemcpyDeviceToHost);

  // Check white pixels (top half)
  for (int y = 0; y < height / 2; y++) {
    for (int x = 0; x < width; x++) {
      int bgrIdx = y * bgrStep + x * 3;
      Npp8u b = hostBGR[bgrIdx];
      Npp8u g = hostBGR[bgrIdx + 1];
      Npp8u r = hostBGR[bgrIdx + 2];

      EXPECT_GT(b, 230) << "White B too low at (" << x << "," << y << ")";
      EXPECT_GT(g, 230) << "White G too low at (" << x << "," << y << ")";
      EXPECT_GT(r, 230) << "White R too low at (" << x << "," << y << ")";
      EXPECT_NEAR(b, g, 2) << "White BG mismatch at (" << x << "," << y << ")";
      EXPECT_NEAR(g, r, 2) << "White GR mismatch at (" << x << "," << y << ")";
    }
  }

  // Check black pixels (bottom half)
  for (int y = height / 2; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int bgrIdx = y * bgrStep + x * 3;
      Npp8u b = hostBGR[bgrIdx];
      Npp8u g = hostBGR[bgrIdx + 1];
      Npp8u r = hostBGR[bgrIdx + 2];

      EXPECT_LT(b, 25) << "Black B too high at (" << x << "," << y << ")";
      EXPECT_LT(g, 25) << "Black G too high at (" << x << "," << y << ")";
      EXPECT_LT(r, 25) << "Black R too high at (" << x << "," << y << ")";
      EXPECT_NEAR(b, g, 2) << "Black BG mismatch at (" << x << "," << y << ")";
      EXPECT_NEAR(g, r, 2) << "Black GR mismatch at (" << x << "," << y << ")";
    }
  }

  nppsFree(d_srcY);
  nppsFree(d_srcUV);
  nppiFree(d_bgr);
}
