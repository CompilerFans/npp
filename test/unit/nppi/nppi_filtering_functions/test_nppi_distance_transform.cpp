#include "npp.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

class NPPIDistanceTransformTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Distance transform requires minimum ROI size of 64x64
    width = 64;
    height = 64;
    oSizeROI.width = width;
    oSizeROI.height = height;
  }

  void TearDown() override {
    // Cleanup handled by RAII guards in tests
  }

  int width, height;
  NppiSize oSizeROI;

  // RAII guard for CUDA memory
  struct CudaMemGuard {
    void *ptr;
    CudaMemGuard(void *p) : ptr(p) {}
    ~CudaMemGuard() {
      if (ptr) cudaFree(ptr);
    }
  };
};

// Test buffer size calculation for distance transform
TEST_F(NPPIDistanceTransformTest, GetBufferSize_Basic) {
  size_t bufferSize = 0;

  NppStatus status = nppiDistanceTransformPBAGetBufferSize(oSizeROI, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);
}

// Test antialiasing buffer size
TEST_F(NPPIDistanceTransformTest, GetAntialiasingBufferSize_Basic) {
  size_t bufferSize = 0;

  NppStatus status = nppiDistanceTransformPBAGetAntialiasingBufferSize(oSizeROI, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);
}

// Test signed distance transform buffer size
TEST_F(NPPIDistanceTransformTest, SignedDistanceTransformGetBufferSize_Basic) {
  size_t bufferSize = 0;

  NppStatus status = nppiSignedDistanceTransformPBAGetBufferSize(oSizeROI, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);
}

// Test signed distance transform 64f buffer size
TEST_F(NPPIDistanceTransformTest, SignedDistanceTransformGet64fBufferSize_Basic) {
  size_t bufferSize = 0;

  NppStatus status = nppiSignedDistanceTransformPBAGet64fBufferSize(oSizeROI, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);
  EXPECT_GT(bufferSize, 0);
}

// Test distance transform 8u to 16u with absolute Manhattan distances
TEST_F(NPPIDistanceTransformTest, DistanceTransformAbsPBA_8u16u_C1R_Basic) {
  size_t dataSize = width * height;
  std::vector<Npp8u> srcData(dataSize);

  // Create test pattern: rectangle of sites in the center
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (x >= width / 4 && x < 3 * width / 4 && y >= height / 4 && y < 3 * height / 4) {
        srcData[y * width + x] = 0;  // Site (foreground)
      } else {
        srcData[y * width + x] = 255;  // Background
      }
    }
  }

  // Get buffer size
  size_t bufferSize = 0;
  NppStatus status = nppiDistanceTransformPBAGetBufferSize(oSizeROI, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Allocate GPU memory
  Npp8u *d_src = nullptr;
  Npp16u *d_transform = nullptr;
  Npp8u *d_buffer = nullptr;

  int srcStep = width * sizeof(Npp8u);
  int transformStep = width * sizeof(Npp16u);

  cudaMalloc(&d_src, dataSize * sizeof(Npp8u));
  cudaMalloc(&d_transform, dataSize * sizeof(Npp16u));
  cudaMalloc(&d_buffer, bufferSize);

  CudaMemGuard srcGuard(d_src);
  CudaMemGuard transformGuard(d_transform);
  CudaMemGuard bufferGuard(d_buffer);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_transform, nullptr);
  ASSERT_NE(d_buffer, nullptr);

  cudaMemcpy(d_src, srcData.data(), dataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // Create stream context
  NppStreamContext nppStreamCtx;
  memset(&nppStreamCtx, 0, sizeof(nppStreamCtx));
  nppStreamCtx.hStream = 0;

  // Call distance transform
  status = nppiDistanceTransformAbsPBA_8u16u_C1R_Ctx(
      d_src, srcStep, 0, 0,  // Sites are pixels with value 0
      nullptr, 0,  // No Voronoi output
      nullptr, 0,  // No Voronoi indices
      nullptr, 0,  // No Manhattan distances
      d_transform, transformStep, oSizeROI,
      d_buffer, nppStreamCtx);

  EXPECT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp16u> resultData(dataSize);
  cudaMemcpy(resultData.data(), d_transform, dataSize * sizeof(Npp16u), cudaMemcpyDeviceToHost);

  // Verify: pixels inside the rectangle should have distance 0
  // Corner pixels should have larger distances
  Npp16u centerDist = resultData[(height / 2) * width + (width / 2)];
  Npp16u cornerDist = resultData[0];

  EXPECT_EQ(centerDist, 0);  // Center of rectangle is a site
  EXPECT_GT(cornerDist, 0);   // Corner is not a site
}

// Test distance transform 8u to 32f
TEST_F(NPPIDistanceTransformTest, DistanceTransformPBA_8u32f_C1R_Basic) {
  size_t dataSize = width * height;
  std::vector<Npp8u> srcData(dataSize);

  // Create test pattern: single site in the center
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (x == width / 2 && y == height / 2) {
        srcData[y * width + x] = 0;  // Site
      } else {
        srcData[y * width + x] = 255;  // Background
      }
    }
  }

  // Get buffer size
  size_t bufferSize = 0;
  NppStatus status = nppiDistanceTransformPBAGetBufferSize(oSizeROI, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Allocate GPU memory
  Npp8u *d_src = nullptr;
  Npp32f *d_transform = nullptr;
  Npp8u *d_buffer = nullptr;

  int srcStep = width * sizeof(Npp8u);
  int transformStep = width * sizeof(Npp32f);

  cudaMalloc(&d_src, dataSize * sizeof(Npp8u));
  cudaMalloc(&d_transform, dataSize * sizeof(Npp32f));
  cudaMalloc(&d_buffer, bufferSize);

  CudaMemGuard srcGuard(d_src);
  CudaMemGuard transformGuard(d_transform);
  CudaMemGuard bufferGuard(d_buffer);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_transform, nullptr);
  ASSERT_NE(d_buffer, nullptr);

  cudaMemcpy(d_src, srcData.data(), dataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // Create stream context
  NppStreamContext nppStreamCtx;
  memset(&nppStreamCtx, 0, sizeof(nppStreamCtx));
  nppStreamCtx.hStream = 0;

  // Call distance transform
  status = nppiDistanceTransformPBA_8u32f_C1R_Ctx(
      d_src, srcStep, 0, 0,  // Sites are pixels with value 0
      nullptr, 0,  // No Voronoi output
      nullptr, 0,  // No Voronoi indices
      nullptr, 0,  // No Manhattan distances
      d_transform, transformStep, oSizeROI,
      d_buffer, nppStreamCtx);

  EXPECT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32f> resultData(dataSize);
  cudaMemcpy(resultData.data(), d_transform, dataSize * sizeof(Npp32f), cudaMemcpyDeviceToHost);

  // Verify: center pixel should have distance 0
  Npp32f centerDist = resultData[(height / 2) * width + (width / 2)];
  EXPECT_FLOAT_EQ(centerDist, 0.0f);

  // Adjacent pixels should have distance 1
  Npp32f adjDist = resultData[(height / 2) * width + (width / 2 + 1)];
  EXPECT_FLOAT_EQ(adjDist, 1.0f);
}

// Test distance transform 8u to 64f
TEST_F(NPPIDistanceTransformTest, DistanceTransformPBA_8u64f_C1R_Basic) {
  size_t dataSize = width * height;
  std::vector<Npp8u> srcData(dataSize);

  // Create test pattern: diagonal line of sites
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (x == y) {
        srcData[y * width + x] = 0;  // Site
      } else {
        srcData[y * width + x] = 255;  // Background
      }
    }
  }

  // Get buffer size
  size_t bufferSize = 0;
  NppStatus status = nppiDistanceTransformPBAGetBufferSize(oSizeROI, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Allocate GPU memory
  Npp8u *d_src = nullptr;
  Npp64f *d_transform = nullptr;
  Npp8u *d_buffer = nullptr;

  int srcStep = width * sizeof(Npp8u);
  int transformStep = width * sizeof(Npp64f);

  cudaMalloc(&d_src, dataSize * sizeof(Npp8u));
  cudaMalloc(&d_transform, dataSize * sizeof(Npp64f));
  cudaMalloc(&d_buffer, bufferSize);

  CudaMemGuard srcGuard(d_src);
  CudaMemGuard transformGuard(d_transform);
  CudaMemGuard bufferGuard(d_buffer);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_transform, nullptr);
  ASSERT_NE(d_buffer, nullptr);

  cudaMemcpy(d_src, srcData.data(), dataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // Create stream context
  NppStreamContext nppStreamCtx;
  memset(&nppStreamCtx, 0, sizeof(nppStreamCtx));
  nppStreamCtx.hStream = 0;

  // Call distance transform
  status = nppiDistanceTransformPBA_8u64f_C1R_Ctx(
      d_src, srcStep, 0, 0,  // Sites are pixels with value 0
      nullptr, 0,  // No Voronoi output
      nullptr, 0,  // No Voronoi indices
      nullptr, 0,  // No Manhattan distances
      d_transform, transformStep, oSizeROI,
      d_buffer, nullptr, nppStreamCtx);

  EXPECT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp64f> resultData(dataSize);
  cudaMemcpy(resultData.data(), d_transform, dataSize * sizeof(Npp64f), cudaMemcpyDeviceToHost);

  // Verify: diagonal pixels should have distance 0
  for (int i = 0; i < std::min(width, height); i++) {
    Npp64f diagDist = resultData[i * width + i];
    EXPECT_DOUBLE_EQ(diagDist, 0.0);
  }
}

// Test signed distance transform 32f to 64f
TEST_F(NPPIDistanceTransformTest, SignedDistanceTransformPBA_32f64f_C1R_Basic) {
  size_t dataSize = width * height;
  std::vector<Npp32f> srcData(dataSize);

  // Create test pattern: filled circle in center
  int centerX = width / 2;
  int centerY = height / 2;
  int radius = width / 4;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      float dx = (float)(x - centerX);
      float dy = (float)(y - centerY);
      float dist = sqrtf(dx * dx + dy * dy);

      if (dist < radius) {
        srcData[y * width + x] = 1.0f;  // Inside
      } else {
        srcData[y * width + x] = -1.0f;  // Outside
      }
    }
  }

  // Get buffer size
  size_t bufferSize = 0;
  NppStatus status = nppiSignedDistanceTransformPBAGet64fBufferSize(oSizeROI, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Allocate GPU memory
  Npp32f *d_src = nullptr;
  Npp64f *d_transform = nullptr;
  Npp8u *d_buffer = nullptr;

  int srcStep = width * sizeof(Npp32f);
  int transformStep = width * sizeof(Npp64f);

  cudaMalloc(&d_src, dataSize * sizeof(Npp32f));
  cudaMalloc(&d_transform, dataSize * sizeof(Npp64f));
  cudaMalloc(&d_buffer, bufferSize);

  CudaMemGuard srcGuard(d_src);
  CudaMemGuard transformGuard(d_transform);
  CudaMemGuard bufferGuard(d_buffer);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_transform, nullptr);
  ASSERT_NE(d_buffer, nullptr);

  cudaMemcpy(d_src, srcData.data(), dataSize * sizeof(Npp32f), cudaMemcpyHostToDevice);

  // Create stream context
  NppStreamContext nppStreamCtx;
  memset(&nppStreamCtx, 0, sizeof(nppStreamCtx));
  nppStreamCtx.hStream = 0;

  // Call signed distance transform
  status = nppiSignedDistanceTransformPBA_32f64f_C1R_Ctx(
      d_src, srcStep, 0.0f,  // Cutoff value
      0.0, 0.0,  // No sub-pixel shift
      nullptr, 0,  // No Voronoi output
      nullptr, 0,  // No Voronoi indices
      nullptr, 0,  // No Manhattan distances
      d_transform, transformStep, oSizeROI,
      d_buffer, nullptr, nppStreamCtx);

  EXPECT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp64f> resultData(dataSize);
  cudaMemcpy(resultData.data(), d_transform, dataSize * sizeof(Npp64f), cudaMemcpyDeviceToHost);

  // Verify: center pixel should have positive distance (inside)
  Npp64f centerDist = resultData[centerY * width + centerX];
  EXPECT_GT(centerDist, 0.0);  // Inside region

  // Corner pixel should have negative distance (outside)
  Npp64f cornerDist = resultData[0];
  EXPECT_LT(cornerDist, 0.0);  // Outside region
}

// Test signed distance transform 64f to 64f
TEST_F(NPPIDistanceTransformTest, SignedDistanceTransformPBA_64f_C1R_Basic) {
  size_t dataSize = width * height;
  std::vector<Npp64f> srcData(dataSize);

  // Create test pattern: horizontal stripe in center
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (y >= height / 3 && y < 2 * height / 3) {
        srcData[y * width + x] = 1.0;  // Inside
      } else {
        srcData[y * width + x] = -1.0;  // Outside
      }
    }
  }

  // Get buffer size
  size_t bufferSize = 0;
  NppStatus status = nppiSignedDistanceTransformPBAGet64fBufferSize(oSizeROI, &bufferSize);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Allocate GPU memory
  Npp64f *d_src = nullptr;
  Npp64f *d_transform = nullptr;
  Npp8u *d_buffer = nullptr;

  int srcStep = width * sizeof(Npp64f);
  int transformStep = width * sizeof(Npp64f);

  cudaMalloc(&d_src, dataSize * sizeof(Npp64f));
  cudaMalloc(&d_transform, dataSize * sizeof(Npp64f));
  cudaMalloc(&d_buffer, bufferSize);

  CudaMemGuard srcGuard(d_src);
  CudaMemGuard transformGuard(d_transform);
  CudaMemGuard bufferGuard(d_buffer);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_transform, nullptr);
  ASSERT_NE(d_buffer, nullptr);

  cudaMemcpy(d_src, srcData.data(), dataSize * sizeof(Npp64f), cudaMemcpyHostToDevice);

  // Create stream context
  NppStreamContext nppStreamCtx;
  memset(&nppStreamCtx, 0, sizeof(nppStreamCtx));
  nppStreamCtx.hStream = 0;

  // Call signed distance transform
  status = nppiSignedDistanceTransformPBA_64f_C1R_Ctx(
      d_src, srcStep, 0.0,  // Cutoff value
      0.0, 0.0,  // No sub-pixel shift
      nullptr, 0,  // No Voronoi output
      nullptr, 0,  // No Voronoi indices
      nullptr, 0,  // No Manhattan distances
      d_transform, transformStep, oSizeROI,
      d_buffer, nullptr, nppStreamCtx);

  EXPECT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp64f> resultData(dataSize);
  cudaMemcpy(resultData.data(), d_transform, dataSize * sizeof(Npp64f), cudaMemcpyDeviceToHost);

  // Verify: center pixel should have positive distance (inside stripe)
  Npp64f centerDist = resultData[(height / 2) * width + (width / 2)];
  EXPECT_GT(centerDist, 0.0);  // Inside stripe

  // Top pixel should have negative distance (outside stripe)
  Npp64f topDist = resultData[0];
  EXPECT_LT(topDist, 0.0);  // Outside stripe
}
