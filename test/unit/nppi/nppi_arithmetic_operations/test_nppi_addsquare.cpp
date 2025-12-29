#include "npp.h"
#include "npp_test_base.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

class NppiAddSquareTest : public ::testing::Test {
protected:
  void SetUp() override {
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess);
  }

  void TearDown() override {
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
  }
};

// ============================================================================
// AddSquare Tests
// ============================================================================

TEST_F(NppiAddSquareTest, AddSquare_8u32f_C1IR_BasicOperation) {
  const int width = 4;
  const int height = 3;
  const int totalPixels = width * height;

  std::vector<Npp8u> hostSrc = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<Npp32f> hostDst = {100.0f, 200.0f, 300.0f, 400.0f,  500.0f,  600.0f,
                                 700.0f, 800.0f, 900.0f, 1000.0f, 1100.0f, 1200.0f};

  // Expected result: dst + (src * src)
  std::vector<Npp32f> expected = {101.0f, 204.0f, 309.0f, 416.0f,  525.0f,  636.0f,
                                  749.0f, 864.0f, 981.0f, 1100.0f, 1221.0f, 1344.0f};

  // Allocate GPU memory
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostSrcStep = width * sizeof(Npp8u);
  int hostDstStep = width * sizeof(Npp32f);

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostSrcStep, hostSrcStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), hostDstStep, hostDstStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAddSquare_8u32f_C1IR(d_src, srcStep, d_dst, dstStep, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostDstStep, d_dst, dstStep, hostDstStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-5f) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiAddSquareTest, AddSquare_32f_C1IR_FloatingPoint) {
  const int width = 3;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp32f> hostSrc = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f};
  std::vector<Npp32f> hostDst = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

  // Expected: dst + (src * src)
  std::vector<Npp32f> expected = {12.25f, 26.25f, 42.25f, 60.25f, 80.25f, 102.25f};

  // Allocate GPU memory
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C1(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * sizeof(Npp32f);
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAddSquare_32f_C1IR(d_src, srcStep, d_dst, dstStep, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-5f) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

// Scale factor versions will be implemented later
// TEST_F(NppiAddSquareTest, AddSquare_8u_C1IRSfs_ScaleFactor) - TODO: implement

TEST_F(NppiAddSquareTest, AddSquare_16u32f_C1IR_MixedType) {
  const int width = 3;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp16u> hostSrc = {10, 20, 30, 40, 50, 60};
  std::vector<Npp32f> hostDst = {100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f};

  // Expected: dst + (src * src)
  std::vector<Npp32f> expected = {200.0f, 600.0f, 1200.0f, 2000.0f, 3000.0f, 4200.0f};

  // Allocate GPU memory
  int srcStep, dstStep;
  Npp16u *d_src = nppiMalloc_16u_C1(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostSrcStep = width * sizeof(Npp16u);
  int hostDstStep = width * sizeof(Npp32f);

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostSrcStep, hostSrcStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), hostDstStep, hostDstStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAddSquare_16u32f_C1IR(d_src, srcStep, d_dst, dstStep, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostDstStep, d_dst, dstStep, hostDstStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-5f) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

// ============================================================================
// Context (_Ctx) version tests
// ============================================================================

TEST_F(NppiAddSquareTest, AddSquare_8u32f_C1IR_Ctx_BasicOperation) {
  const int width = 4;
  const int height = 3;
  const int totalPixels = width * height;

  std::vector<Npp8u> hostSrc = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<Npp32f> hostDst = {100.0f, 200.0f, 300.0f, 400.0f,  500.0f,  600.0f,
                                 700.0f, 800.0f, 900.0f, 1000.0f, 1100.0f, 1200.0f};

  std::vector<Npp32f> expected = {101.0f, 204.0f, 309.0f, 416.0f,  525.0f,  636.0f,
                                  749.0f, 864.0f, 981.0f, 1100.0f, 1221.0f, 1344.0f};

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostSrcStep = width * sizeof(Npp8u);
  int hostDstStep = width * sizeof(Npp32f);

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostSrcStep, hostSrcStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), hostDstStep, hostDstStep, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  NppStatus status = nppiAddSquare_8u32f_C1IR_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostDstStep, d_dst, dstStep, hostDstStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-5f) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiAddSquareTest, AddSquare_16u32f_C1IR_Ctx_MixedType) {
  const int width = 3;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp16u> hostSrc = {10, 20, 30, 40, 50, 60};
  std::vector<Npp32f> hostDst = {100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f};

  std::vector<Npp32f> expected = {200.0f, 600.0f, 1200.0f, 2000.0f, 3000.0f, 4200.0f};

  int srcStep, dstStep;
  Npp16u *d_src = nppiMalloc_16u_C1(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostSrcStep = width * sizeof(Npp16u);
  int hostDstStep = width * sizeof(Npp32f);

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostSrcStep, hostSrcStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), hostDstStep, hostDstStep, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  NppStatus status = nppiAddSquare_16u32f_C1IR_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostDstStep, d_dst, dstStep, hostDstStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-5f) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiAddSquareTest, AddSquare_32f_C1IR_Ctx_FloatingPoint) {
  const int width = 3;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp32f> hostSrc = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f};
  std::vector<Npp32f> hostDst = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

  std::vector<Npp32f> expected = {12.25f, 26.25f, 42.25f, 60.25f, 80.25f, 102.25f};

  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C1(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostStep = width * sizeof(Npp32f);
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  NppStatus status = nppiAddSquare_32f_C1IR_Ctx(d_src, srcStep, d_dst, dstStep, oSizeROI, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-5f) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// ============================================================================
// Masked version tests (_C1IMR)
// ============================================================================

TEST_F(NppiAddSquareTest, AddSquare_8u32f_C1IMR_MaskedOperation) {
  const int width = 4;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp8u> hostSrc = {2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<Npp32f> hostDst = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
  std::vector<Npp8u> hostMask = {255, 0, 255, 0, 0, 255, 0, 255};

  // dst + src*src only where mask != 0
  std::vector<Npp32f> expected = {14.0f, 20.0f, 46.0f, 40.0f, 50.0f, 109.0f, 70.0f, 161.0f};

  int srcStep, dstStep, maskStep;
  Npp8u *d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);
  Npp8u *d_mask = nppiMalloc_8u_C1(width, height, &maskStep);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);
  ASSERT_NE(d_mask, nullptr);

  int hostSrcStep = width * sizeof(Npp8u);
  int hostDstStep = width * sizeof(Npp32f);
  int hostMaskStep = width * sizeof(Npp8u);

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostSrcStep, hostSrcStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), hostDstStep, hostDstStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_mask, maskStep, hostMask.data(), hostMaskStep, hostMaskStep, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAddSquare_8u32f_C1IMR(d_src, srcStep, d_mask, maskStep, d_dst, dstStep, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostDstStep, d_dst, dstStep, hostDstStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-5f) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
  nppiFree(d_mask);
}

TEST_F(NppiAddSquareTest, AddSquare_16u32f_C1IMR_MaskedOperation) {
  const int width = 4;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp16u> hostSrc = {10, 20, 30, 40, 50, 60, 70, 80};
  std::vector<Npp32f> hostDst = {100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f, 700.0f, 800.0f};
  std::vector<Npp8u> hostMask = {255, 0, 255, 0, 0, 255, 0, 255};

  // dst + src*src only where mask != 0
  std::vector<Npp32f> expected = {200.0f, 200.0f, 1200.0f, 400.0f, 500.0f, 4200.0f, 700.0f, 7200.0f};

  int srcStep, dstStep, maskStep;
  Npp16u *d_src = nppiMalloc_16u_C1(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);
  Npp8u *d_mask = nppiMalloc_8u_C1(width, height, &maskStep);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);
  ASSERT_NE(d_mask, nullptr);

  int hostSrcStep = width * sizeof(Npp16u);
  int hostDstStep = width * sizeof(Npp32f);
  int hostMaskStep = width * sizeof(Npp8u);

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostSrcStep, hostSrcStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), hostDstStep, hostDstStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_mask, maskStep, hostMask.data(), hostMaskStep, hostMaskStep, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAddSquare_16u32f_C1IMR(d_src, srcStep, d_mask, maskStep, d_dst, dstStep, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostDstStep, d_dst, dstStep, hostDstStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-5f) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
  nppiFree(d_mask);
}

TEST_F(NppiAddSquareTest, AddSquare_32f_C1IMR_MaskedOperation) {
  const int width = 4;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp32f> hostSrc = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f, 7.5f, 8.5f};
  std::vector<Npp32f> hostDst = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};
  std::vector<Npp8u> hostMask = {255, 0, 255, 0, 0, 255, 0, 255};

  // dst + src*src only where mask != 0
  std::vector<Npp32f> expected = {12.25f, 20.0f, 42.25f, 40.0f, 50.0f, 102.25f, 70.0f, 152.25f};

  int srcStep, dstStep, maskStep;
  Npp32f *d_src = nppiMalloc_32f_C1(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);
  Npp8u *d_mask = nppiMalloc_8u_C1(width, height, &maskStep);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);
  ASSERT_NE(d_mask, nullptr);

  int hostStep = width * sizeof(Npp32f);
  int hostMaskStep = width * sizeof(Npp8u);

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_mask, maskStep, hostMask.data(), hostMaskStep, hostMaskStep, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAddSquare_32f_C1IMR(d_src, srcStep, d_mask, maskStep, d_dst, dstStep, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-5f) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
  nppiFree(d_mask);
}