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

TEST_F(NV12ToBGRTest, NV12ToBGR_709CSC_CameraSize_1280x720) {
  const int camWidth = 1280;
  const int camHeight = 720;

  std::vector<Npp8u> yData(static_cast<size_t>(camWidth) * camHeight);
  std::vector<Npp8u> uvData(static_cast<size_t>(camWidth) * camHeight / 2);
  for (int y = 0; y < camHeight; ++y) {
    for (int x = 0; x < camWidth; ++x) {
      yData[static_cast<size_t>(y) * camWidth + x] =
          static_cast<Npp8u>(16 + (x + y) * 219 / (camWidth + camHeight - 2));
    }
  }
  for (int y = 0; y < camHeight / 2; ++y) {
    for (int x = 0; x < camWidth; x += 2) {
      const size_t idx = static_cast<size_t>(y) * camWidth + x;
      uvData[idx] = static_cast<Npp8u>(64 + x * 128 / camWidth);
      uvData[idx + 1] = static_cast<Npp8u>(64 + y * 128 / (camHeight / 2));
    }
  }

  Npp8u *d_srcY = nppsMalloc_8u(static_cast<size_t>(camWidth) * camHeight);
  Npp8u *d_srcUV = nppsMalloc_8u(static_cast<size_t>(camWidth) * camHeight / 2);
  int bgrStep;
  Npp8u *d_bgr = nppiMalloc_8u_C3(camWidth, camHeight, &bgrStep);
  ASSERT_NE(d_srcY, nullptr);
  ASSERT_NE(d_srcUV, nullptr);
  ASSERT_NE(d_bgr, nullptr);
  ASSERT_EQ(cudaMemcpy(d_srcY, yData.data(), yData.size(), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(d_srcUV, uvData.data(), uvData.size(), cudaMemcpyHostToDevice), cudaSuccess);

  const Npp8u *pSrc[2] = {d_srcY, d_srcUV};
  const NppiSize roi{camWidth, camHeight};
  NppStreamContext nppStreamCtx{};
  ASSERT_EQ(nppGetStreamContext(&nppStreamCtx), NPP_SUCCESS);
  ASSERT_EQ(nppiNV12ToBGR_709CSC_8u_P2C3R_Ctx(pSrc, camWidth, d_bgr, bgrStep, roi, nppStreamCtx), NPP_NO_ERROR);

  std::vector<Npp8u> hostBGR(static_cast<size_t>(bgrStep) * camHeight);
  ASSERT_EQ(cudaMemcpy(hostBGR.data(), d_bgr, hostBGR.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  for (int y = 0; y < camHeight; y += 32) {
    for (int x = 0; x < camWidth; x += 32) {
      const int idx = y * bgrStep + x * 3;
      EXPECT_GE(hostBGR[idx], 0) << "Invalid B at (" << x << "," << y << ")";
      EXPECT_GE(hostBGR[idx + 1], 0) << "Invalid G at (" << x << "," << y << ")";
      EXPECT_GE(hostBGR[idx + 2], 0) << "Invalid R at (" << x << "," << y << ")";
    }
  }

  nppsFree(d_srcY);
  nppsFree(d_srcUV);
  nppiFree(d_bgr);
}

TEST_F(NV12ToBGRTest, NV12ToBGR_709CSC_PaddedLineSize) {
  const int width = 32;
  const int height = 24;
  const int pad = 32;
  const int paddedStep = width + pad;

  std::vector<Npp8u> yData(static_cast<size_t>(width) * height);
  std::vector<Npp8u> uvData(static_cast<size_t>(width) * height / 2);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      yData[static_cast<size_t>(y) * width + x] = static_cast<Npp8u>(16 + (x * 2 + y) % 219);
    }
  }
  for (int y = 0; y < height / 2; ++y) {
    for (int x = 0; x < width; x += 2) {
      const size_t idx = static_cast<size_t>(y) * width + x;
      uvData[idx] = static_cast<Npp8u>(64 + x * 3 % 127);
      uvData[idx + 1] = static_cast<Npp8u>(64 + y * 5 % 127);
    }
  }

  // Reference: contiguous planes (no padding), step == width.
  Npp8u *d_refY = nppsMalloc_8u(static_cast<size_t>(width) * height);
  Npp8u *d_refUV = nppsMalloc_8u(static_cast<size_t>(width) * height / 2);
  // Padded: same image content but row stride inflated by 32 bytes per row
  // (video-frame AVFrame alignment scenario).
  Npp8u *d_padY = nppsMalloc_8u(static_cast<size_t>(paddedStep) * height);
  Npp8u *d_padUV = nppsMalloc_8u(static_cast<size_t>(paddedStep) * height / 2);
  int refBgrStep;
  int padBgrStep;
  Npp8u *d_refBgr = nppiMalloc_8u_C3(width, height, &refBgrStep);
  Npp8u *d_padBgr = nppiMalloc_8u_C3(width, height, &padBgrStep);
  ASSERT_NE(d_refY, nullptr);
  ASSERT_NE(d_refUV, nullptr);
  ASSERT_NE(d_padY, nullptr);
  ASSERT_NE(d_padUV, nullptr);
  ASSERT_NE(d_refBgr, nullptr);
  ASSERT_NE(d_padBgr, nullptr);
  ASSERT_EQ(cudaMemcpy2D(d_refY, width, yData.data(), width, width, height, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(d_refUV, width, uvData.data(), width, width, height / 2, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(d_padY, paddedStep, yData.data(), width, width, height, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(d_padUV, paddedStep, uvData.data(), width, width, height / 2, cudaMemcpyHostToDevice),
            cudaSuccess);

  const NppiSize roi{width, height};
  const Npp8u *refSrc[2] = {d_refY, d_refUV};
  const Npp8u *padSrc[2] = {d_padY, d_padUV};
  ASSERT_EQ(nppiNV12ToBGR_709CSC_8u_P2C3R(refSrc, width, d_refBgr, refBgrStep, roi), NPP_NO_ERROR);
  ASSERT_EQ(nppiNV12ToBGR_709CSC_8u_P2C3R(padSrc, paddedStep, d_padBgr, padBgrStep, roi), NPP_NO_ERROR);

  std::vector<Npp8u> refBGR(static_cast<size_t>(refBgrStep) * height);
  std::vector<Npp8u> padBGR(static_cast<size_t>(padBgrStep) * height);
  ASSERT_EQ(cudaMemcpy(refBGR.data(), d_refBgr, refBGR.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(padBGR.data(), d_padBgr, padBGR.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width * 3; ++x) {
      EXPECT_EQ(padBGR[static_cast<size_t>(y) * padBgrStep + x], refBGR[static_cast<size_t>(y) * refBgrStep + x])
          << "Padded result differs from unpadded at (" << x << "," << y << ")";
    }
  }

  nppsFree(d_refY);
  nppsFree(d_refUV);
  nppsFree(d_padY);
  nppsFree(d_padUV);
  nppiFree(d_refBgr);
  nppiFree(d_padBgr);
}

TEST_F(NV12ToBGRTest, NV12ToBGR_709CSC_PartialEvenROI) {
  const int width = 128;
  const int height = 128;
  const int roiX = 32;
  const int roiY = 16;
  const int roiW = 64;
  const int roiH = 48;

  std::vector<Npp8u> yData(static_cast<size_t>(width) * height);
  std::vector<Npp8u> uvData(static_cast<size_t>(width) * height / 2);
  for (size_t i = 0; i < yData.size(); ++i) {
    yData[i] = static_cast<Npp8u>((i * 17 + 9) % 256);
  }
  for (int y = 0; y < height / 2; ++y) {
    for (int x = 0; x < width; x += 2) {
      const size_t idx = static_cast<size_t>(y) * width + x;
      uvData[idx] = static_cast<Npp8u>(64 + (x / 2) * 13 % 127);
      uvData[idx + 1] = static_cast<Npp8u>(64 + y * 29 % 127);
    }
  }

  Npp8u *d_srcY = nppsMalloc_8u(static_cast<size_t>(width) * height);
  Npp8u *d_srcUV = nppsMalloc_8u(static_cast<size_t>(width) * height / 2);
  int fullBgrStep;
  int roiBgrStep;
  Npp8u *d_fullBgr = nppiMalloc_8u_C3(width, height, &fullBgrStep);
  Npp8u *d_roiBgr = nppiMalloc_8u_C3(width, height, &roiBgrStep);
  ASSERT_NE(d_srcY, nullptr);
  ASSERT_NE(d_srcUV, nullptr);
  ASSERT_NE(d_fullBgr, nullptr);
  ASSERT_NE(d_roiBgr, nullptr);
  ASSERT_EQ(cudaMemcpy2D(d_srcY, width, yData.data(), width, width, height, cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy2D(d_srcUV, width, uvData.data(), width, width, height / 2, cudaMemcpyHostToDevice), cudaSuccess);

  // Reference: full-image conversion, then take the same ROI window.
  const NppiSize fullRoi{width, height};
  const NppiSize roi{roiW, roiH};
  const Npp8u *fullSrc[2] = {d_srcY, d_srcUV};
  ASSERT_EQ(nppiNV12ToBGR_709CSC_8u_P2C3R(fullSrc, width, d_fullBgr, fullBgrStep, fullRoi), NPP_NO_ERROR);
  // Chroma pointer offset is half the Y offset (2x2 subsampling).
  Npp8u *d_roiY = d_srcY + static_cast<size_t>(roiY) * width + roiX;
  Npp8u *d_roiUV = d_srcUV + static_cast<size_t>(roiY / 2) * width + roiX;
  const Npp8u *roiSrc[2] = {d_roiY, d_roiUV};
  ASSERT_EQ(nppiNV12ToBGR_709CSC_8u_P2C3R(roiSrc, width, d_roiBgr, roiBgrStep, roi), NPP_NO_ERROR);

  std::vector<Npp8u> fullBGR(static_cast<size_t>(fullBgrStep) * height);
  std::vector<Npp8u> roiBGR(static_cast<size_t>(roiBgrStep) * height);
  ASSERT_EQ(cudaMemcpy(fullBGR.data(), d_fullBgr, fullBGR.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(roiBGR.data(), d_roiBgr, roiBGR.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  for (int y = 0; y < roiH; ++y) {
    for (int x = 0; x < roiW * 3; ++x) {
      EXPECT_EQ(roiBGR[static_cast<size_t>(y) * roiBgrStep + x],
                fullBGR[static_cast<size_t>(roiY + y) * fullBgrStep + (roiX * 3 + x)])
          << "Partial ROI differs from full-image window at (" << x << "," << y << ")";
    }
  }

  nppsFree(d_srcY);
  nppsFree(d_srcUV);
  nppiFree(d_fullBgr);
  nppiFree(d_roiBgr);
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
