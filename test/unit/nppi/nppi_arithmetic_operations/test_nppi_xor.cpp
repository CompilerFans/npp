#include "npp.h"
#include "npp_test_base.h"
#include <gtest/gtest.h>
#include <vector>

class NppiXorTest : public ::testing::Test {
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

TEST_F(NppiXorTest, Xor_8u_C1R_BasicOperation) {
  const int width = 4;
  const int height = 2;
  const int totalPixels = width * height;

  // Create test data
  std::vector<Npp8u> hostSrc1 = {0xFF, 0x00, 0xAA, 0x55, 0x33, 0xCC, 0x0F, 0xF0};
  std::vector<Npp8u> hostSrc2 = {0x0F, 0xF0, 0x55, 0xAA, 0xCC, 0x33, 0xFF, 0x00};
  std::vector<Npp8u> expected = {0xF0, 0xF0, 0xFF, 0xFF, 0xFF, 0xFF, 0xF0, 0xF0};

  // Allocate GPU memory
  int step;
  Npp8u *d_src1 = nppiMalloc_8u_C1(width, height, &step);
  Npp8u *d_src2 = nppiMalloc_8u_C1(width, height, &step);
  Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &step);

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * sizeof(Npp8u);
  cudaMemcpy2D(d_src1, step, hostSrc1.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, step, hostSrc2.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiXor_8u_C1R(d_src1, step, d_src2, step, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp8u> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

TEST_F(NppiXorTest, Xor_8u_C1IR_InPlace) {
  const int width = 3;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp8u> hostSrcDst = {0xFF, 0x00, 0xAA, 0x55, 0x33, 0xCC};
  std::vector<Npp8u> hostSrc = {0x0F, 0xF0, 0x55, 0xAA, 0xCC, 0x33};
  std::vector<Npp8u> expected = {0xF0, 0xF0, 0xFF, 0xFF, 0xFF, 0xFF};

  // Allocate GPU memory
  int step;
  Npp8u *d_srcDst = nppiMalloc_8u_C1(width, height, &step);
  Npp8u *d_src = nppiMalloc_8u_C1(width, height, &step);

  ASSERT_NE(d_srcDst, nullptr);
  ASSERT_NE(d_src, nullptr);

  // Copy data to GPU
  int hostStep = width * sizeof(Npp8u);
  cudaMemcpy2D(d_srcDst, step, hostSrcDst.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute in-place operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiXor_8u_C1IR(d_src, step, d_srcDst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp8u> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_srcDst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_srcDst);
  nppiFree(d_src);
}

TEST_F(NppiXorTest, Xor_16u_C1R_DataTypes) {
  const int width = 2;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp16u> hostSrc1 = {0xFFFF, 0x0000, 0xAAAA, 0x5555};
  std::vector<Npp16u> hostSrc2 = {0x0F0F, 0xF0F0, 0x5555, 0xAAAA};
  std::vector<Npp16u> expected = {0xF0F0, 0xF0F0, 0xFFFF, 0xFFFF};

  // Allocate GPU memory
  int step;
  Npp16u *d_src1 = nppiMalloc_16u_C1(width, height, &step);
  Npp16u *d_src2 = nppiMalloc_16u_C1(width, height, &step);
  Npp16u *d_dst = nppiMalloc_16u_C1(width, height, &step);

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * sizeof(Npp16u);
  cudaMemcpy2D(d_src1, step, hostSrc1.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, step, hostSrc2.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiXor_16u_C1R(d_src1, step, d_src2, step, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp16u> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

TEST_F(NppiXorTest, Xor_32s_C3R_MultiChannel) {
  const int width = 2;
  const int height = 1;
  const int channels = 3;
  const int totalElements = width * height * channels;

  std::vector<Npp32s> hostSrc1 = {(Npp32s)0xFFFFFFFF, 0x00000000, (Npp32s)0xAAAAAAAA,  // pixel 0
                                  0x55555555,         0x33333333, (Npp32s)0xCCCCCCCC}; // pixel 1

  std::vector<Npp32s> hostSrc2 = {0x0F0F0F0F,         (Npp32s)0xF0F0F0F0, 0x55555555,  // pixel 0
                                  (Npp32s)0xAAAAAAAA, (Npp32s)0xCCCCCCCC, 0x33333333}; // pixel 1

  std::vector<Npp32s> expected = {(Npp32s)0xF0F0F0F0, (Npp32s)0xF0F0F0F0, (Npp32s)0xFFFFFFFF,  // pixel 0
                                  (Npp32s)0xFFFFFFFF, (Npp32s)0xFFFFFFFF, (Npp32s)0xFFFFFFFF}; // pixel 1

  // Allocate GPU memory
  int step;
  Npp32s *d_src1 = (Npp32s *)nppiMalloc_32s_C3(width, height, &step);
  Npp32s *d_src2 = (Npp32s *)nppiMalloc_32s_C3(width, height, &step);
  Npp32s *d_dst = (Npp32s *)nppiMalloc_32s_C3(width, height, &step);

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * channels * sizeof(Npp32s);
  cudaMemcpy2D(d_src1, step, hostSrc1.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, step, hostSrc2.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiXor_32s_C3R(d_src1, step, d_src2, step, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32s> hostResult(totalElements);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalElements; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

TEST_F(NppiXorTest, XorC_8u_C1R_ConstantOperation) {
  const int width = 4;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp8u> hostSrc = {0xFF, 0x00, 0xAA, 0x55, 0x33, 0xCC, 0x0F, 0xF0};
  Npp8u constant = 0x0F;
  std::vector<Npp8u> expected = {0xF0, 0x0F, 0xA5, 0x5A, 0x3C, 0xC3, 0x00, 0xFF};

  // Allocate GPU memory
  int step;
  Npp8u *d_src = nppiMalloc_8u_C1(width, height, &step);
  Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * sizeof(Npp8u);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiXorC_8u_C1R(d_src, step, constant, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp8u> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiXorTest, XorC_8u_C3R_MultiChannelConstant) {
  const int width = 2;
  const int height = 1;
  const int channels = 3;
  const int totalElements = width * height * channels;

  std::vector<Npp8u> hostSrc = {0xFF, 0x00, 0xAA, 0x55, 0x33, 0xCC};
  Npp8u constants[3] = {0x0F, 0xF0, 0x55};
  std::vector<Npp8u> expected = {0xF0, 0xF0, 0xFF, 0x5A, 0xC3, 0x99};

  // Allocate GPU memory
  int step;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &step);
  Npp8u *d_dst = nppiMalloc_8u_C3(width, height, &step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * channels * sizeof(Npp8u);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiXorC_8u_C3R(d_src, step, constants, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp8u> hostResult(totalElements);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalElements; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}
