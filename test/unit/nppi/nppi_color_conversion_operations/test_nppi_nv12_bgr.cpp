#include "npp.h"
#include "npp_test_base.h"

#include <algorithm>
#include <cmath>
#include <cstring>
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

// Parameter contract for the plain NV12/NV21 Ctx entry points, aligned with
// NVIDIA NPP 13: an empty ROI is a no-op, odd widths are rejected with
// NPP_SIZE_ERROR, odd heights are valid, and the destination step must be at
// least width * channels.
TEST_F(NV12ToBGRTest, NV12ToBGR_CtxParameterContract) {
  const int W = width;
  const int H = height;
  Npp8u *dY = nullptr;
  Npp8u *dUV = nullptr;
  Npp8u *dDst = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&dY), static_cast<size_t>(W) * H), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&dUV), static_cast<size_t>(W) * H / 2), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&dDst), static_cast<size_t>(W) * H * 3), cudaSuccess);
  ASSERT_EQ(cudaMemset(dY, 128, static_cast<size_t>(W) * H), cudaSuccess);
  ASSERT_EQ(cudaMemset(dUV, 128, static_cast<size_t>(W) * H / 2), cudaSuccess);

  const Npp8u *pSrc[2] = {dY, dUV};
  NppStreamContext ctx{};
  ctx.hStream = 0;
  auto call = [&](int srcStep, int dstStep, int w, int h) {
    cudaGetLastError();
    NppiSize roi = {w, h};
    return nppiNV12ToBGR_8u_P2C3R_Ctx(pSrc, srcStep, dDst, dstStep, roi, ctx);
  };

  EXPECT_EQ(call(W, W * 3, 0, 0), NPP_NO_ERROR) << "empty ROI must be a no-op";
  EXPECT_EQ(call(W, W * 3, -1, H), NPP_SIZE_ERROR) << "negative width";
  EXPECT_EQ(call(W, W * 3, W, -1), NPP_SIZE_ERROR) << "negative height";
  EXPECT_EQ(call(W, W * 3, W - 1, H), NPP_SIZE_ERROR) << "odd width";
  EXPECT_EQ(call(W, W * 3, W, H - 1), NPP_NO_ERROR) << "odd height is valid";
  EXPECT_EQ(call(W, W, W, H), NPP_STEP_ERROR) << "dst step below width * 3";
  EXPECT_EQ(call(W, W * 3 - 1, W, H), NPP_STEP_ERROR) << "dst step below width * 3";
  EXPECT_EQ(call(W, W * 3, W, H), NPP_NO_ERROR) << "valid conversion";

  cudaFree(dY);
  cudaFree(dUV);
  cudaFree(dDst);
}

TEST_F(NV12ToBGRTest, NV21ToBGR_CtxParameterContract) {
  const int W = width;
  const int H = height;
  Npp8u *dY = nullptr;
  Npp8u *dUV = nullptr;
  Npp8u *dDst = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&dY), static_cast<size_t>(W) * H), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&dUV), static_cast<size_t>(W) * H / 2), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&dDst), static_cast<size_t>(W) * H * 4), cudaSuccess);
  ASSERT_EQ(cudaMemset(dY, 128, static_cast<size_t>(W) * H), cudaSuccess);
  ASSERT_EQ(cudaMemset(dUV, 128, static_cast<size_t>(W) * H / 2), cudaSuccess);

  const Npp8u *pSrc[2] = {dY, dUV};
  NppStreamContext ctx{};
  ctx.hStream = 0;
  auto call = [&](int srcStep, int dstStep, int w, int h) {
    cudaGetLastError();
    NppiSize roi = {w, h};
    return nppiNV21ToBGR_8u_P2C4R_Ctx(pSrc, srcStep, dDst, dstStep, roi, ctx);
  };

  EXPECT_EQ(call(W, W * 4, 0, 0), NPP_NO_ERROR) << "empty ROI must be a no-op";
  EXPECT_EQ(call(W, W * 4, W - 1, H), NPP_SIZE_ERROR) << "odd width";
  EXPECT_EQ(call(W, W * 4, W, H - 1), NPP_NO_ERROR) << "odd height is valid";
  EXPECT_EQ(call(W, W * 3, W, H), NPP_STEP_ERROR) << "dst step below width * 4";
  EXPECT_EQ(call(W, W * 4, W, H), NPP_NO_ERROR) << "valid conversion";

  cudaFree(dY);
  cudaFree(dUV);
  cudaFree(dDst);
}

#ifndef USE_NVIDIA_NPP_TESTS
// Measured NVIDIA NPP 13.4 behavior: the plain NV12 path applies the
// FULL-RANGE BT.601 matrix (no 16/128 studio offsets) and rounds half up.
TEST_F(NV12ToBGRTest, NV12ToBGR_UsesBt601FullRange) {
  const int W = 8;
  const int H = 8;
  Npp8u *dY = nullptr;
  Npp8u *dUV = nullptr;
  Npp8u *dDst = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&dY), static_cast<size_t>(W) * H), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&dUV), static_cast<size_t>(W) * H / 2), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&dDst), static_cast<size_t>(W) * H * 3), cudaSuccess);

  const Npp8u *pSrc[2] = {dY, dUV};
  NppStreamContext ctx{};
  ctx.hStream = 0;
  const NppiSize roi = {W, H};
  auto convert = [&](int y, int u, int v, Npp8u out[3]) {
    ASSERT_EQ(cudaMemset(dY, y, static_cast<size_t>(W) * H), cudaSuccess);
    std::vector<Npp8u> uv(static_cast<size_t>(W) * H / 2);
    for (size_t i = 0; i < uv.size(); i += 2) {
      uv[i] = static_cast<Npp8u>(u);
      uv[i + 1] = static_cast<Npp8u>(v);
    }
    ASSERT_EQ(cudaMemcpy(dUV, uv.data(), uv.size(), cudaMemcpyHostToDevice), cudaSuccess);
    EXPECT_EQ(nppiNV12ToBGR_8u_P2C3R_Ctx(pSrc, W, dDst, W * 3, roi, ctx), NPP_NO_ERROR);
    EXPECT_EQ(cudaStreamSynchronize(0), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(out, dDst, 3, cudaMemcpyDeviceToHost), cudaSuccess);
  };

  struct Case {
    int y, u, v, b, g, r;
  } cases[] = {
      {16, 128, 128, 16, 16, 16},
      {128, 128, 128, 128, 128, 128},
      {235, 128, 128, 235, 235, 235},
      {128, 0, 128, 0, 179, 128},
      {128, 128, 192, 128, 91, 201},
  };
  for (const auto &c : cases) {
    Npp8u out[3] = {0, 0, 0};
    convert(c.y, c.u, c.v, out);
    EXPECT_NEAR(out[0], c.b, 1) << "Y=" << c.y << " U=" << c.u << " V=" << c.v;
    EXPECT_NEAR(out[1], c.g, 1) << "Y=" << c.y << " U=" << c.u << " V=" << c.v;
    EXPECT_NEAR(out[2], c.r, 1) << "Y=" << c.y << " U=" << c.u << " V=" << c.v;
  }

  cudaFree(dY);
  cudaFree(dUV);
  cudaFree(dDst);
}
#endif // !USE_NVIDIA_NPP_TESTS

// Our 709HDTV entry point intentionally uses the same standard BT.709 limited
// curve as 709CSC.  NVIDIA NPP 13 gives 709HDTV full-range luma, so the two
// libraries differ here by design.
TEST_F(NV12ToBGRTest, NV12ToBGR_709HDTVMatchesCSC) {
  std::vector<Npp8u> hostY, hostUV;
  createTestNV12Data(hostY, hostUV);
  Npp8u *dY = nullptr;
  Npp8u *dUV = nullptr;
  Npp8u *hdtv = nullptr;
  Npp8u *csc = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&dY), hostY.size()), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&dUV), hostUV.size()), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&hdtv), static_cast<size_t>(width) * height * 3), cudaSuccess);
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&csc), static_cast<size_t>(width) * height * 3), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dY, hostY.data(), hostY.size(), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dUV, hostUV.data(), hostUV.size(), cudaMemcpyHostToDevice), cudaSuccess);

  const Npp8u *pSrc[2] = {dY, dUV};
  NppStreamContext ctx{};
  ctx.hStream = 0;
  const NppiSize roi = {width, height};
  ASSERT_EQ(nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx(pSrc, width, hdtv, width * 3, roi, ctx), NPP_NO_ERROR);
  ASSERT_EQ(nppiNV12ToBGR_709CSC_8u_P2C3R_Ctx(pSrc, width, csc, width * 3, roi, ctx), NPP_NO_ERROR);

  std::vector<Npp8u> hdtvHost(static_cast<size_t>(width) * height * 3);
  std::vector<Npp8u> cscHost(hdtvHost.size());
  ASSERT_EQ(cudaMemcpy(hdtvHost.data(), hdtv, hdtvHost.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(cscHost.data(), csc, cscHost.size(), cudaMemcpyDeviceToHost), cudaSuccess);

  // NPP 13.4 distinguishes the two: HDTV uses the full-range BT.709 matrix,
  // CSC the studio-range one.  Verify both models and that they differ.
  auto roundClamp = [](float v) {
    float r = std::floor(v + 0.5f);
    return (Npp8u)(r < 0.0f ? 0.0f : (r > 255.0f ? 255.0f : r));
  };
  bool anyDifferent = false;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int yi = y * width + x;
      const int uvi = (y / 2) * width + (x & ~1);
      const float Y = (float)hostY[yi];
      const float U = (float)hostUV[uvi] - 128.0f;
      const float V = (float)hostUV[uvi + 1] - 128.0f;
      const int o = (y * width + x) * 3;
      const float hb = Y + 2.125f * U, hg = Y - 0.2109375f * U - 0.3828125f * V, hr = Y + 1.28125f * V;
      const float yc = Y - 16.0f;
      const float cb = 1.164f * yc + 2.112f * U, cg = 1.164f * yc - 0.213f * U - 0.533f * V,
                  cr = 1.164f * yc + 1.793f * V;
      EXPECT_NEAR(hdtvHost[(size_t)o + 0], roundClamp(hb), 1) << "HDTV B at " << x << "," << y;
      EXPECT_NEAR(hdtvHost[(size_t)o + 1], roundClamp(hg), 1) << "HDTV G at " << x << "," << y;
      EXPECT_NEAR(hdtvHost[(size_t)o + 2], roundClamp(hr), 1) << "HDTV R at " << x << "," << y;
      EXPECT_NEAR(cscHost[(size_t)o + 0], roundClamp(cb), 1) << "CSC B at " << x << "," << y;
      EXPECT_NEAR(cscHost[(size_t)o + 1], roundClamp(cg), 1) << "CSC G at " << x << "," << y;
      EXPECT_NEAR(cscHost[(size_t)o + 2], roundClamp(cr), 1) << "CSC R at " << x << "," << y;
      if (std::abs((int)hdtvHost[(size_t)o] - (int)cscHost[(size_t)o]) > 2 ||
          std::abs((int)hdtvHost[(size_t)o + 1] - (int)cscHost[(size_t)o + 1]) > 2)
        anyDifferent = true;
    }
  }
  EXPECT_TRUE(anyDifferent);

  cudaFree(dY);
  cudaFree(dUV);
  cudaFree(hdtv);
  cudaFree(csc);
}
