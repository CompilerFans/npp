#include "npp_test_base.h"
#include <algorithm>
#include <cmath>

using namespace npp_functional_test;

class NV12ToBGRFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }

  // Helper to create standard NV12 test data
  void createNV12TestData(int width, int height, std::vector<Npp8u>& yData, std::vector<Npp8u>& uvData) {
    yData.resize(width * height);
    uvData.resize(width * height / 2);

    for (int i = 0; i < width * height; i++) {
      yData[i] = static_cast<Npp8u>(128 + (i % 128));
    }

    for (int i = 0; i < width * height / 2; i += 2) {
      uvData[i] = 128;     // U
      uvData[i + 1] = 128; // V
    }
  }
};

// Test standard NV12 to BGR conversion
TEST_F(NV12ToBGRFunctionalTest, NV12ToBGR_8u_P2C3R_Ctx_Basic) {
  const int width = 64, height = 64;
  const int chromaHeight = height / 2;

  std::vector<Npp8u> yData, uvData;
  createNV12TestData(width, height, yData, uvData);

  NppImageMemory<Npp8u> yPlane(width, height);
  NppImageMemory<Npp8u> uvPlane(width, chromaHeight);
  NppImageMemory<Npp8u> bgrImage(width, height, 3);

  yPlane.copyFromHost(yData);
  uvPlane.copyFromHost(uvData);

  const Npp8u *pSrc[2] = {yPlane.get(), uvPlane.get()};
  int srcStep = yPlane.step();
  NppiSize oSizeROI = {width, height};

  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status = nppiNV12ToBGR_8u_P2C3R_Ctx(pSrc, srcStep, bgrImage.get(),
                                                  bgrImage.step(), oSizeROI, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> bgrData(width * height * 3);
  bgrImage.copyToHost(bgrData);

  // Neutral UV should produce grayscale
  bool hasValidData = false;
  for (int i = 0; i < width * height; i++) {
    Npp8u b = bgrData[i * 3 + 0];
    Npp8u g = bgrData[i * 3 + 1];
    Npp8u r = bgrData[i * 3 + 2];

    if (b > 0 || g > 0 || r > 0) {
      hasValidData = true;
    }

    int diff = std::abs(r - g) + std::abs(g - b) + std::abs(r - b);
    ASSERT_LT(diff, 30);
  }

  ASSERT_TRUE(hasValidData);
}

// Test without stream context
TEST_F(NV12ToBGRFunctionalTest, NV12ToBGR_8u_P2C3R_NoContext) {
  const int width = 32, height = 32;
  const int chromaHeight = height / 2;

  std::vector<Npp8u> yData, uvData;
  createNV12TestData(width, height, yData, uvData);

  NppImageMemory<Npp8u> yPlane(width, height);
  NppImageMemory<Npp8u> uvPlane(width, chromaHeight);
  NppImageMemory<Npp8u> bgrImage(width, height, 3);

  yPlane.copyFromHost(yData);
  uvPlane.copyFromHost(uvData);

  const Npp8u *pSrc[2] = {yPlane.get(), uvPlane.get()};
  int srcStep = yPlane.step();
  NppiSize oSizeROI = {width, height};

  NppStatus status = nppiNV12ToBGR_8u_P2C3R(pSrc, srcStep, bgrImage.get(),
                                            bgrImage.step(), oSizeROI);

  ASSERT_EQ(status, NPP_SUCCESS);

  cudaDeviceSynchronize();
  std::vector<Npp8u> bgrData(width * height * 3);
  bgrImage.copyToHost(bgrData);

  bool hasValidData = false;
  for (int i = 0; i < width * height && i < 100; i++) {
    if (bgrData[i * 3] > 0 || bgrData[i * 3 + 1] > 0 || bgrData[i * 3 + 2] > 0) {
      hasValidData = true;
      break;
    }
  }
  ASSERT_TRUE(hasValidData);
}

// Test BT.709 HDTV color space
TEST_F(NV12ToBGRFunctionalTest, NV12ToBGR_709HDTV_8u_P2C3R_Ctx) {
  const int width = 64, height = 64;
  const int chromaHeight = height / 2;

  std::vector<Npp8u> yData, uvData;
  createNV12TestData(width, height, yData, uvData);

  NppImageMemory<Npp8u> yPlane(width, height);
  NppImageMemory<Npp8u> uvPlane(width, chromaHeight);
  NppImageMemory<Npp8u> bgrImage(width, height, 3);

  yPlane.copyFromHost(yData);
  uvPlane.copyFromHost(uvData);

  const Npp8u *pSrc[2] = {yPlane.get(), uvPlane.get()};
  int srcStep = yPlane.step();
  NppiSize oSizeROI = {width, height};

  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status = nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx(pSrc, srcStep, bgrImage.get(),
                                                         bgrImage.step(), oSizeROI, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> bgrData(width * height * 3);
  bgrImage.copyToHost(bgrData);

  bool hasValidData = false;
  for (int i = 0; i < 100; i++) {
    if (bgrData[i * 3] > 0 || bgrData[i * 3 + 1] > 0 || bgrData[i * 3 + 2] > 0) {
      hasValidData = true;
      break;
    }
  }
  ASSERT_TRUE(hasValidData);
}

// Test BT.709 CSC color space
TEST_F(NV12ToBGRFunctionalTest, NV12ToBGR_709CSC_8u_P2C3R_Ctx) {
  const int width = 64, height = 64;
  const int chromaHeight = height / 2;

  std::vector<Npp8u> yData, uvData;
  createNV12TestData(width, height, yData, uvData);

  NppImageMemory<Npp8u> yPlane(width, height);
  NppImageMemory<Npp8u> uvPlane(width, chromaHeight);
  NppImageMemory<Npp8u> bgrImage(width, height, 3);

  yPlane.copyFromHost(yData);
  uvPlane.copyFromHost(uvData);

  const Npp8u *pSrc[2] = {yPlane.get(), uvPlane.get()};
  int srcStep = yPlane.step();
  NppiSize oSizeROI = {width, height};

  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status = nppiNV12ToBGR_709CSC_8u_P2C3R_Ctx(pSrc, srcStep, bgrImage.get(),
                                                        bgrImage.step(), oSizeROI, nppStreamCtx);

  ASSERT_EQ(status, NPP_SUCCESS);

  cudaStreamSynchronize(nppStreamCtx.hStream);
  std::vector<Npp8u> bgrData(width * height * 3);
  bgrImage.copyToHost(bgrData);

  bool hasValidData = false;
  for (int i = 0; i < 100; i++) {
    if (bgrData[i * 3] > 0 || bgrData[i * 3 + 1] > 0 || bgrData[i * 3 + 2] > 0) {
      hasValidData = true;
      break;
    }
  }
  ASSERT_TRUE(hasValidData);
}

// Test with known Y=128, U=128, V=128 (neutral gray)
TEST_F(NV12ToBGRFunctionalTest, NV12ToBGR_ColorAccuracy_Gray) {
  const int width = 16, height = 16;
  const int chromaHeight = height / 2;

  // Pure gray: Y=128, U=128, V=128
  std::vector<Npp8u> yData(width * height, 128);
  std::vector<Npp8u> uvData(width * chromaHeight);
  for (int i = 0; i < width * chromaHeight; i += 2) {
    uvData[i] = 128;     // U neutral
    uvData[i + 1] = 128; // V neutral
  }

  NppImageMemory<Npp8u> yPlane(width, height);
  NppImageMemory<Npp8u> uvPlane(width, chromaHeight);
  NppImageMemory<Npp8u> bgrImage(width, height, 3);

  yPlane.copyFromHost(yData);
  uvPlane.copyFromHost(uvData);

  const Npp8u *pSrc[2] = {yPlane.get(), uvPlane.get()};
  int srcStep = yPlane.step();
  NppiSize oSizeROI = {width, height};

  NppStatus status = nppiNV12ToBGR_8u_P2C3R(pSrc, srcStep, bgrImage.get(),
                                            bgrImage.step(), oSizeROI);

  ASSERT_EQ(status, NPP_SUCCESS);

  cudaDeviceSynchronize();
  std::vector<Npp8u> bgrData(width * height * 3);
  bgrImage.copyToHost(bgrData);

  // For neutral gray, all channels should be equal within tight tolerance
  for (int i = 0; i < width * height; i++) {
    Npp8u b = bgrData[i * 3 + 0];
    Npp8u g = bgrData[i * 3 + 1];
    Npp8u r = bgrData[i * 3 + 2];

    // All channels should be approximately equal for gray
    EXPECT_NEAR(b, g, 3) << "B and G should match at pixel " << i;
    EXPECT_NEAR(g, r, 3) << "G and R should match at pixel " << i;
    EXPECT_NEAR(r, b, 3) << "R and B should match at pixel " << i;

    // Expected gray value around 130 for Y=128 using BT.601
    EXPECT_GT(b, 110) << "Gray value too low at pixel " << i;
    EXPECT_LT(b, 150) << "Gray value too high at pixel " << i;
  }
}

// Test pure white and black values
TEST_F(NV12ToBGRFunctionalTest, NV12ToBGR_ColorAccuracy_WhiteBlack) {
  const int width = 16, height = 16;
  const int chromaHeight = height / 2;

  std::vector<Npp8u> yData(width * height);
  std::vector<Npp8u> uvData(width * chromaHeight);

  // Top half: white (Y=235, U=128, V=128)
  // Bottom half: black (Y=16, U=128, V=128)
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      yData[y * width + x] = (y < height / 2) ? 235 : 16;
    }
  }

  for (int i = 0; i < width * chromaHeight; i += 2) {
    uvData[i] = 128;
    uvData[i + 1] = 128;
  }

  NppImageMemory<Npp8u> yPlane(width, height);
  NppImageMemory<Npp8u> uvPlane(width, chromaHeight);
  NppImageMemory<Npp8u> bgrImage(width, height, 3);

  yPlane.copyFromHost(yData);
  uvPlane.copyFromHost(uvData);

  const Npp8u *pSrc[2] = {yPlane.get(), uvPlane.get()};
  int srcStep = yPlane.step();
  NppiSize oSizeROI = {width, height};

  NppStatus status = nppiNV12ToBGR_8u_P2C3R(pSrc, srcStep, bgrImage.get(),
                                            bgrImage.step(), oSizeROI);

  ASSERT_EQ(status, NPP_SUCCESS);

  cudaDeviceSynchronize();
  std::vector<Npp8u> bgrData(width * height * 3);
  bgrImage.copyToHost(bgrData);

  // Check white pixels (top half)
  for (int i = 0; i < width * (height / 2); i++) {
    Npp8u b = bgrData[i * 3 + 0];
    Npp8u g = bgrData[i * 3 + 1];
    Npp8u r = bgrData[i * 3 + 2];

    EXPECT_GT(b, 230) << "White B too low at pixel " << i;
    EXPECT_GT(g, 230) << "White G too low at pixel " << i;
    EXPECT_GT(r, 230) << "White R too low at pixel " << i;
    EXPECT_NEAR(b, g, 2) << "White BGR mismatch at pixel " << i;
    EXPECT_NEAR(g, r, 2) << "White GR mismatch at pixel " << i;
  }

  // Check black pixels (bottom half)
  for (int i = width * (height / 2); i < width * height; i++) {
    Npp8u b = bgrData[i * 3 + 0];
    Npp8u g = bgrData[i * 3 + 1];
    Npp8u r = bgrData[i * 3 + 2];

    EXPECT_LT(b, 25) << "Black B too high at pixel " << i;
    EXPECT_LT(g, 25) << "Black G too high at pixel " << i;
    EXPECT_LT(r, 25) << "Black R too high at pixel " << i;
    EXPECT_NEAR(b, g, 2) << "Black BG mismatch at pixel " << i;
    EXPECT_NEAR(g, r, 2) << "Black GR mismatch at pixel " << i;
  }
}
