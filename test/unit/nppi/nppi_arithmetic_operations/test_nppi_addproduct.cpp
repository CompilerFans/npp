#include "npp.h"
#include "npp_test_base.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <vector>

class NppiAddProductTest : public ::testing::Test {
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
// AddProduct Tests
// ============================================================================

TEST_F(NppiAddProductTest, AddProduct_8u32f_C1IR_BasicOperation) {
  const int width = 4;
  const int height = 3;
  const int totalPixels = width * height;

  std::vector<Npp8u> hostSrc1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<Npp8u> hostSrc2 = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  std::vector<Npp32f> hostDst = {100.0f, 200.0f, 300.0f, 400.0f,  500.0f,  600.0f,
                                 700.0f, 800.0f, 900.0f, 1000.0f, 1100.0f, 1200.0f};

  // Expected result: dst + (src1 * src2)
  std::vector<Npp32f> expected = {102.0f, 206.0f, 312.0f, 420.0f,  530.0f,  642.0f,
                                  756.0f, 872.0f, 990.0f, 1110.0f, 1232.0f, 1356.0f};

  // Allocate GPU memory
  int src1Step, src2Step, dstStep;
  Npp8u *d_src1 = nppiMalloc_8u_C1(width, height, &src1Step);
  Npp8u *d_src2 = nppiMalloc_8u_C1(width, height, &src2Step);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostSrc1Step = width * sizeof(Npp8u);
  int hostSrc2Step = width * sizeof(Npp8u);
  int hostDstStep = width * sizeof(Npp32f);

  cudaMemcpy2D(d_src1, src1Step, hostSrc1.data(), hostSrc1Step, hostSrc1Step, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, hostSrc2.data(), hostSrc2Step, hostSrc2Step, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), hostDstStep, hostDstStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAddProduct_8u32f_C1IR(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostDstStep, d_dst, dstStep, hostDstStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-5f) << "Mismatch at index " << i;
  }

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

TEST_F(NppiAddProductTest, AddProduct_32f_C1IR_FloatingPoint) {
  const int width = 3;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp32f> hostSrc1 = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f};
  std::vector<Npp32f> hostSrc2 = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<Npp32f> hostDst = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

  // Expected: dst + (src1 * src2)
  std::vector<Npp32f> expected = {13.0f, 27.5f, 44.0f, 62.5f, 83.0f, 105.5f};

  // Allocate GPU memory
  int src1Step, src2Step, dstStep;
  Npp32f *d_src1 = nppiMalloc_32f_C1(width, height, &src1Step);
  Npp32f *d_src2 = nppiMalloc_32f_C1(width, height, &src2Step);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * sizeof(Npp32f);
  cudaMemcpy2D(d_src1, src1Step, hostSrc1.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, hostSrc2.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAddProduct_32f_C1IR(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-5f) << "Mismatch at index " << i;
  }

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

// ============================================================================
// Context Version Tests
// ============================================================================

TEST_F(NppiAddProductTest, AddProduct_8u32f_C1IR_Ctx_BasicOperation) {
  const int width = 4;
  const int height = 3;
  const int totalPixels = width * height;

  std::vector<Npp8u> hostSrc1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<Npp8u> hostSrc2 = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  std::vector<Npp32f> hostDst = {100.0f, 200.0f, 300.0f, 400.0f,  500.0f,  600.0f,
                                 700.0f, 800.0f, 900.0f, 1000.0f, 1100.0f, 1200.0f};

  std::vector<Npp32f> expected = {102.0f, 206.0f, 312.0f, 420.0f,  530.0f,  642.0f,
                                  756.0f, 872.0f, 990.0f, 1110.0f, 1232.0f, 1356.0f};

  int src1Step, src2Step, dstStep;
  Npp8u *d_src1 = nppiMalloc_8u_C1(width, height, &src1Step);
  Npp8u *d_src2 = nppiMalloc_8u_C1(width, height, &src2Step);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostSrc1Step = width * sizeof(Npp8u);
  int hostSrc2Step = width * sizeof(Npp8u);
  int hostDstStep = width * sizeof(Npp32f);

  cudaMemcpy2D(d_src1, src1Step, hostSrc1.data(), hostSrc1Step, hostSrc1Step, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, hostSrc2.data(), hostSrc2Step, hostSrc2Step, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), hostDstStep, hostDstStep, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status =
      nppiAddProduct_8u32f_C1IR_Ctx(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, oSizeROI, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostDstStep, d_dst, dstStep, hostDstStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-5f) << "Mismatch at index " << i;
  }

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

TEST_F(NppiAddProductTest, AddProduct_16u32f_C1IR_BasicOperation) {
  const int width = 4;
  const int height = 3;
  const int totalPixels = width * height;

  std::vector<Npp16u> hostSrc1 = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200};
  std::vector<Npp16u> hostSrc2 = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  std::vector<Npp32f> hostDst = {1000.0f, 2000.0f, 3000.0f, 4000.0f,  5000.0f,  6000.0f,
                                 7000.0f, 8000.0f, 9000.0f, 10000.0f, 11000.0f, 12000.0f};

  std::vector<Npp32f> expected(totalPixels);
  for (int i = 0; i < totalPixels; ++i) {
    expected[i] = hostDst[i] + (float)(hostSrc1[i] * hostSrc2[i]);
  }

  int src1Step, src2Step, dstStep;
  Npp16u *d_src1 = nppiMalloc_16u_C1(width, height, &src1Step);
  Npp16u *d_src2 = nppiMalloc_16u_C1(width, height, &src2Step);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostSrc1Step = width * sizeof(Npp16u);
  int hostSrc2Step = width * sizeof(Npp16u);
  int hostDstStep = width * sizeof(Npp32f);

  cudaMemcpy2D(d_src1, src1Step, hostSrc1.data(), hostSrc1Step, hostSrc1Step, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, hostSrc2.data(), hostSrc2Step, hostSrc2Step, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), hostDstStep, hostDstStep, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiAddProduct_16u32f_C1IR(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostDstStep, d_dst, dstStep, hostDstStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-3f) << "Mismatch at index " << i;
  }

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

TEST_F(NppiAddProductTest, AddProduct_16u32f_C1IR_Ctx_BasicOperation) {
  const int width = 4;
  const int height = 3;
  const int totalPixels = width * height;

  std::vector<Npp16u> hostSrc1 = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200};
  std::vector<Npp16u> hostSrc2 = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  std::vector<Npp32f> hostDst = {1000.0f, 2000.0f, 3000.0f, 4000.0f,  5000.0f,  6000.0f,
                                 7000.0f, 8000.0f, 9000.0f, 10000.0f, 11000.0f, 12000.0f};

  std::vector<Npp32f> expected(totalPixels);
  for (int i = 0; i < totalPixels; ++i) {
    expected[i] = hostDst[i] + (float)(hostSrc1[i] * hostSrc2[i]);
  }

  int src1Step, src2Step, dstStep;
  Npp16u *d_src1 = nppiMalloc_16u_C1(width, height, &src1Step);
  Npp16u *d_src2 = nppiMalloc_16u_C1(width, height, &src2Step);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostSrc1Step = width * sizeof(Npp16u);
  int hostSrc2Step = width * sizeof(Npp16u);
  int hostDstStep = width * sizeof(Npp32f);

  cudaMemcpy2D(d_src1, src1Step, hostSrc1.data(), hostSrc1Step, hostSrc1Step, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, hostSrc2.data(), hostSrc2Step, hostSrc2Step, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), hostDstStep, hostDstStep, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status =
      nppiAddProduct_16u32f_C1IR_Ctx(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, oSizeROI, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostDstStep, d_dst, dstStep, hostDstStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-3f) << "Mismatch at index " << i;
  }

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

TEST_F(NppiAddProductTest, AddProduct_32f_C1IR_Ctx_FloatingPoint) {
  const int width = 3;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp32f> hostSrc1 = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f};
  std::vector<Npp32f> hostSrc2 = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<Npp32f> hostDst = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

  std::vector<Npp32f> expected = {13.0f, 27.5f, 44.0f, 62.5f, 83.0f, 105.5f};

  int src1Step, src2Step, dstStep;
  Npp32f *d_src1 = nppiMalloc_32f_C1(width, height, &src1Step);
  Npp32f *d_src2 = nppiMalloc_32f_C1(width, height, &src2Step);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostStep = width * sizeof(Npp32f);
  cudaMemcpy2D(d_src1, src1Step, hostSrc1.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, hostSrc2.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status =
      nppiAddProduct_32f_C1IR_Ctx(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, oSizeROI, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-5f) << "Mismatch at index " << i;
  }

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

// ============================================================================
// Masked Version Tests
// ============================================================================

TEST_F(NppiAddProductTest, AddProduct_8u32f_C1IMR_MaskedOperation) {
  const int width = 4;
  const int height = 3;
  const int totalPixels = width * height;

  std::vector<Npp8u> hostSrc1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<Npp8u> hostSrc2 = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  std::vector<Npp8u> hostMask = {255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0};
  std::vector<Npp32f> hostDst = {100.0f, 200.0f, 300.0f, 400.0f,  500.0f,  600.0f,
                                 700.0f, 800.0f, 900.0f, 1000.0f, 1100.0f, 1200.0f};

  // Expected: update only where mask != 0
  std::vector<Npp32f> expected = {102.0f, 200.0f, 312.0f, 400.0f,  530.0f,  600.0f,
                                  756.0f, 800.0f, 990.0f, 1000.0f, 1232.0f, 1200.0f};

  int src1Step, src2Step, maskStep, dstStep;
  Npp8u *d_src1 = nppiMalloc_8u_C1(width, height, &src1Step);
  Npp8u *d_src2 = nppiMalloc_8u_C1(width, height, &src2Step);
  Npp8u *d_mask = nppiMalloc_8u_C1(width, height, &maskStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_mask, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostSrc1Step = width * sizeof(Npp8u);
  int hostDstStep = width * sizeof(Npp32f);

  cudaMemcpy2D(d_src1, src1Step, hostSrc1.data(), hostSrc1Step, hostSrc1Step, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, hostSrc2.data(), hostSrc1Step, hostSrc1Step, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_mask, maskStep, hostMask.data(), hostSrc1Step, hostSrc1Step, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), hostDstStep, hostDstStep, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStatus status =
      nppiAddProduct_8u32f_C1IMR(d_src1, src1Step, d_src2, src2Step, d_mask, maskStep, d_dst, dstStep, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostDstStep, d_dst, dstStep, hostDstStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-5f) << "Mismatch at index " << i;
  }

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_mask);
  nppiFree(d_dst);
}

TEST_F(NppiAddProductTest, AddProduct_16u32f_C1IMR_MaskedOperation) {
  const int width = 4;
  const int height = 3;
  const int totalPixels = width * height;

  std::vector<Npp16u> hostSrc1 = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200};
  std::vector<Npp16u> hostSrc2 = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
  std::vector<Npp8u> hostMask = {255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0};
  std::vector<Npp32f> hostDst = {1000.0f, 2000.0f, 3000.0f, 4000.0f,  5000.0f,  6000.0f,
                                 7000.0f, 8000.0f, 9000.0f, 10000.0f, 11000.0f, 12000.0f};

  // Calculate expected values (only update where mask != 0)
  std::vector<Npp32f> expected(totalPixels);
  for (int i = 0; i < totalPixels; ++i) {
    if (hostMask[i] != 0) {
      expected[i] = hostDst[i] + (float)(hostSrc1[i] * hostSrc2[i]);
    } else {
      expected[i] = hostDst[i];
    }
  }

  int src1Step, src2Step, maskStep, dstStep;
  Npp16u *d_src1 = nppiMalloc_16u_C1(width, height, &src1Step);
  Npp16u *d_src2 = nppiMalloc_16u_C1(width, height, &src2Step);
  Npp8u *d_mask = nppiMalloc_8u_C1(width, height, &maskStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_mask, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostSrc1Step = width * sizeof(Npp16u);
  int hostMaskStep = width * sizeof(Npp8u);
  int hostDstStep = width * sizeof(Npp32f);

  cudaMemcpy2D(d_src1, src1Step, hostSrc1.data(), hostSrc1Step, hostSrc1Step, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, hostSrc2.data(), hostSrc1Step, hostSrc1Step, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_mask, maskStep, hostMask.data(), hostMaskStep, hostMaskStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), hostDstStep, hostDstStep, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStatus status =
      nppiAddProduct_16u32f_C1IMR(d_src1, src1Step, d_src2, src2Step, d_mask, maskStep, d_dst, dstStep, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostDstStep, d_dst, dstStep, hostDstStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-3f) << "Mismatch at index " << i;
  }

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_mask);
  nppiFree(d_dst);
}

TEST_F(NppiAddProductTest, AddProduct_32f_C1IMR_MaskedOperation) {
  const int width = 3;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp32f> hostSrc1 = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f};
  std::vector<Npp32f> hostSrc2 = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  std::vector<Npp8u> hostMask = {255, 0, 255, 0, 255, 0};
  std::vector<Npp32f> hostDst = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

  // Expected: only update where mask != 0
  std::vector<Npp32f> expected = {13.0f, 20.0f, 44.0f, 40.0f, 83.0f, 60.0f};

  int src1Step, src2Step, maskStep, dstStep;
  Npp32f *d_src1 = nppiMalloc_32f_C1(width, height, &src1Step);
  Npp32f *d_src2 = nppiMalloc_32f_C1(width, height, &src2Step);
  Npp8u *d_mask = nppiMalloc_8u_C1(width, height, &maskStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_mask, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostSrcStep = width * sizeof(Npp32f);
  int hostMaskStep = width * sizeof(Npp8u);

  cudaMemcpy2D(d_src1, src1Step, hostSrc1.data(), hostSrcStep, hostSrcStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, hostSrc2.data(), hostSrcStep, hostSrcStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_mask, maskStep, hostMask.data(), hostMaskStep, hostMaskStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), hostSrcStep, hostSrcStep, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStatus status =
      nppiAddProduct_32f_C1IMR(d_src1, src1Step, d_src2, src2Step, d_mask, maskStep, d_dst, dstStep, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostSrcStep, d_dst, dstStep, hostSrcStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_NEAR(hostResult[i], expected[i], 1e-5f) << "Mismatch at index " << i;
  }

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_mask);
  nppiFree(d_dst);
}