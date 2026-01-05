#include "npp.h"
#include "npp_test_base.h"
#include <gtest/gtest.h>
#include <vector>

class NppiRShiftCTest : public ::testing::Test {
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
// RShiftC Tests - 8u variants
// ============================================================================

TEST_F(NppiRShiftCTest, RShiftC_8u_C1R_BasicOperation) {
  const int width = 4;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp8u> hostSrc = {128, 64, 32, 16, 8, 4, 2, 1};
  Npp32u shiftCount = 2;
  std::vector<Npp8u> expected = {32, 16, 8, 4, 2, 1, 0, 0}; // src >> 2

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
  NppStatus status = nppiRShiftC_8u_C1R(d_src, step, shiftCount, d_dst, step, oSizeROI);
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

TEST_F(NppiRShiftCTest, RShiftC_8u_C1IR_InPlace) {
  const int width = 4;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp8u> hostSrc = {240, 200, 160, 120, 80, 40, 20, 10};
  Npp32u shiftCount = 3;
  std::vector<Npp8u> expected = {30, 25, 20, 15, 10, 5, 2, 1}; // src >> 3

  // Allocate GPU memory
  int step;
  Npp8u *d_srcDst = nppiMalloc_8u_C1(width, height, &step);

  ASSERT_NE(d_srcDst, nullptr);

  // Copy data to GPU
  int hostStep = width * sizeof(Npp8u);
  cudaMemcpy2D(d_srcDst, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute in-place operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiRShiftC_8u_C1IR(shiftCount, d_srcDst, step, oSizeROI);
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
}

TEST_F(NppiRShiftCTest, RShiftC_8u_C3R_MultiChannel) {
  const int width = 2;
  const int height = 2;
  const int channels = 3;
  const int totalPixels = width * height * channels;

  std::vector<Npp8u> hostSrc = {
      128, 64, 32, // pixel 0: R, G, B
      96,  48, 24, // pixel 1: R, G, B
      64,  32, 16, // pixel 2: R, G, B
      32,  16, 8   // pixel 3: R, G, B
  };
  Npp32u shiftConstants[3] = {2, 3, 1};
  std::vector<Npp8u> expected = {
      static_cast<Npp8u>(128 >> shiftConstants[0]), static_cast<Npp8u>(64 >> shiftConstants[1]),
      static_cast<Npp8u>(32 >> shiftConstants[2]),  static_cast<Npp8u>(96 >> shiftConstants[0]),
      static_cast<Npp8u>(48 >> shiftConstants[1]),  static_cast<Npp8u>(24 >> shiftConstants[2]),
      static_cast<Npp8u>(64 >> shiftConstants[0]),  static_cast<Npp8u>(32 >> shiftConstants[1]),
      static_cast<Npp8u>(16 >> shiftConstants[2]),  static_cast<Npp8u>(32 >> shiftConstants[0]),
      static_cast<Npp8u>(16 >> shiftConstants[1]),  static_cast<Npp8u>(8 >> shiftConstants[2])};

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
  NppStatus status = nppiRShiftC_8u_C3R(d_src, step, shiftConstants, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp8u> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    // printf("i%d: src=%d, result=%d, expected=%d\n", i, hostSrc[i], hostResult[i], expected[i]);
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiRShiftCTest, RShiftC_8u_C3R_Shift8ProducesZero) {
  const int width = 2;
  const int height = 1;
  const int channels = 3;
  const int totalPixels = width * height * channels;

  std::vector<Npp8u> hostSrc = {255, 128, 64, 32, 16, 8};
  Npp32u shiftConstants[3] = {8, 8, 8};
  std::vector<Npp8u> expected(totalPixels, 0);

  int step;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &step);
  Npp8u *d_dst = nppiMalloc_8u_C3(width, height, &step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostStep = width * channels * sizeof(Npp8u);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiRShiftC_8u_C3R(d_src, step, shiftConstants, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiRShiftCTest, RShiftC_8u_C4R_FourChannels) {
  const int width = 2;
  const int height = 1;
  const int channels = 4;
  const int totalPixels = width * height * channels;

  std::vector<Npp8u> hostSrc = {
      128, 64, 32, 16, // pixel 0: R, G, B, A
      96,  48, 24, 12  // pixel 1: R, G, B, A
  };
  Npp32u shiftConstants[4] = {2, 3, 1, 0};
  std::vector<Npp8u> expected = {
      static_cast<Npp8u>(128 >> shiftConstants[0]), static_cast<Npp8u>(64 >> shiftConstants[1]),
      static_cast<Npp8u>(32 >> shiftConstants[2]),  static_cast<Npp8u>(16 >> shiftConstants[3]),
      static_cast<Npp8u>(96 >> shiftConstants[0]),  static_cast<Npp8u>(48 >> shiftConstants[1]),
      static_cast<Npp8u>(24 >> shiftConstants[2]),  static_cast<Npp8u>(12 >> shiftConstants[3])};

  // Allocate GPU memory
  int step;
  Npp8u *d_src = nppiMalloc_8u_C4(width, height, &step);
  Npp8u *d_dst = nppiMalloc_8u_C4(width, height, &step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * channels * sizeof(Npp8u);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiRShiftC_8u_C4R(d_src, step, shiftConstants, d_dst, step, oSizeROI);
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

// ============================================================================
// RShiftC Tests - 8s variants
// ============================================================================
TEST_F(NppiRShiftCTest, RShiftC_8s_C3R_BasicOperation) {
  const int width = 2;
  const int height = 1;
  const int channels = 3;
  const int totalPixels = width * height * channels;

  std::vector<Npp8s> hostSrc = {
      -128, 64,  -32,
      96,   -48, 24
  };
  Npp32u shiftConstants[3] = {1, 2, 3};
  std::vector<Npp8s> expected(totalPixels);
  for (int i = 0; i < totalPixels; ++i) {
    const int c = i % channels;
    expected[i] = static_cast<Npp8s>(hostSrc[i] >> shiftConstants[c]);
  }

  int step = width * channels * sizeof(Npp8s);
  Npp8s *d_src = nullptr;
  Npp8s *d_dst = nullptr;
  cudaMalloc(&d_src, height * step);
  cudaMalloc(&d_dst, height * step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, step, hostSrc.data(), step, step, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiRShiftC_8s_C3R(d_src, step, shiftConstants, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8s> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), step, d_dst, step, step, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  cudaFree(d_src);
  cudaFree(d_dst);
}

TEST_F(NppiRShiftCTest, RShiftC_8s_C1R_Shift8Arithmetic) {
  const int width = 4;
  const int height = 1;
  const int totalPixels = width * height;

  std::vector<Npp8s> hostSrc = {127, -64, 32, -1};
  Npp32u shiftCount = 8;
  std::vector<Npp8s> expected(totalPixels);
  for (int i = 0; i < totalPixels; ++i) {
    expected[i] = static_cast<Npp8s>(static_cast<int>(hostSrc[i]) >> 8);
  }

  int step = width * sizeof(Npp8s);
  Npp8s *d_src = nullptr;
  Npp8s *d_dst = nullptr;
  cudaMalloc(&d_src, height * step);
  cudaMalloc(&d_dst, height * step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, step, hostSrc.data(), step, step, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiRShiftC_8s_C1R(d_src, step, shiftCount, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8s> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), step, d_dst, step, step, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  cudaFree(d_src);
  cudaFree(d_dst);
}

TEST_F(NppiRShiftCTest, RShiftC_8s_C1IR_Shift8Arithmetic) {
  const int width = 4;
  const int height = 1;
  const int totalPixels = width * height;

  std::vector<Npp8s> hostSrc = {127, -64, 32, -1};
  Npp32u shiftCount = 8;
  std::vector<Npp8s> expected(totalPixels);
  for (int i = 0; i < totalPixels; ++i) {
    expected[i] = static_cast<Npp8s>(static_cast<int>(hostSrc[i]) >> 8);
  }

  int step = width * sizeof(Npp8s);
  Npp8s *d_srcDst = nullptr;
  cudaMalloc(&d_srcDst, height * step);

  ASSERT_NE(d_srcDst, nullptr);

  cudaMemcpy2D(d_srcDst, step, hostSrc.data(), step, step, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiRShiftC_8s_C1IR(shiftCount, d_srcDst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8s> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), step, d_srcDst, step, step, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  cudaFree(d_srcDst);
}

TEST_F(NppiRShiftCTest, RShiftC_8s_C3R_Shift8Arithmetic) {
  const int width = 2;
  const int height = 1;
  const int channels = 3;
  const int totalPixels = width * height * channels;

  std::vector<Npp8s> hostSrc = {127, -64, 32, -16, 8, -4};
  Npp32u shiftConstants[3] = {8, 8, 8};
  std::vector<Npp8s> expected(totalPixels);
  for (int i = 0; i < totalPixels; ++i) {
    expected[i] = static_cast<Npp8s>(static_cast<int>(hostSrc[i]) >> 8);
  }

  int step = width * channels * sizeof(Npp8s);
  Npp8s *d_src = nullptr;
  Npp8s *d_dst = nullptr;
  cudaMalloc(&d_src, height * step);
  cudaMalloc(&d_dst, height * step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, step, hostSrc.data(), step, step, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiRShiftC_8s_C3R(d_src, step, shiftConstants, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8s> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), step, d_dst, step, step, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  cudaFree(d_src);
  cudaFree(d_dst);
}

TEST_F(NppiRShiftCTest, RShiftC_8s_C4R_BasicOperation) {
  const int width = 2;
  const int height = 1;
  const int channels = 4;
  const int totalPixels = width * height * channels;

  std::vector<Npp8s> hostSrc = {
      -128, 64,  -32, 16,
      96,   -48, 24,  -8
  };
  Npp32u shiftConstants[4] = {1, 2, 3, 0};
  std::vector<Npp8s> expected(totalPixels);
  for (int i = 0; i < totalPixels; ++i) {
    const int c = i % channels;
    expected[i] = static_cast<Npp8s>(hostSrc[i] >> shiftConstants[c]);
  }

  int step = width * channels * sizeof(Npp8s);
  Npp8s *d_src = nullptr;
  Npp8s *d_dst = nullptr;
  cudaMalloc(&d_src, height * step);
  cudaMalloc(&d_dst, height * step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, step, hostSrc.data(), step, step, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiRShiftC_8s_C4R(d_src, step, shiftConstants, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8s> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), step, d_dst, step, step, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  cudaFree(d_src);
  cudaFree(d_dst);
}

TEST_F(NppiRShiftCTest, RShiftC_8s_C4R_Shift8Arithmetic) {
  const int width = 2;
  const int height = 1;
  const int channels = 4;
  const int totalPixels = width * height * channels;

  std::vector<Npp8s> hostSrc = {127, -64, 32, -16, 8, -4, 2, -1};
  Npp32u shiftConstants[4] = {8, 8, 8, 8};
  std::vector<Npp8s> expected(totalPixels);
  for (int i = 0; i < totalPixels; ++i) {
    expected[i] = static_cast<Npp8s>(static_cast<int>(hostSrc[i]) >> 8);
  }

  int step = width * channels * sizeof(Npp8s);
  Npp8s *d_src = nullptr;
  Npp8s *d_dst = nullptr;
  cudaMalloc(&d_src, height * step);
  cudaMalloc(&d_dst, height * step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, step, hostSrc.data(), step, step, height, cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiRShiftC_8s_C4R(d_src, step, shiftConstants, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8s> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), step, d_dst, step, step, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  cudaFree(d_src);
  cudaFree(d_dst);
}

// ============================================================================
// RShiftC Tests - 16u variants
// ============================================================================

TEST_F(NppiRShiftCTest, RShiftC_16u_C1R_DataTypes) {
  const int width = 3;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp16u> hostSrc = {4800, 4000, 3200, 2400, 1600, 800};
  Npp32u shiftCount = 3;
  std::vector<Npp16u> expected = {600, 500, 400, 300, 200, 100}; // src >> 3

  // Allocate GPU memory
  int step;
  Npp16u *d_src = nppiMalloc_16u_C1(width, height, &step);
  Npp16u *d_dst = nppiMalloc_16u_C1(width, height, &step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * sizeof(Npp16u);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiRShiftC_16u_C1R(d_src, step, shiftCount, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp16u> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiRShiftCTest, RShiftC_16u_C3R_MultiChannel) {
  const int width = 2;
  const int height = 1;
  const int channels = 3;
  const int totalPixels = width * height * channels;

  std::vector<Npp16u> hostSrc = {
      1600, 800, 400, // pixel 0: R, G, B
      1200, 600, 300  // pixel 1: R, G, B
  };
  Npp32u shiftConstants[3] = {3, 2, 1}; // Note: Currently only first constant is used
  std::vector<Npp16u> expected = {
      static_cast<Npp16u>(1600 >> shiftConstants[0]), static_cast<Npp16u>(800 >> shiftConstants[1]),
      static_cast<Npp16u>(400 >> shiftConstants[2]),  static_cast<Npp16u>(1200 >> shiftConstants[0]),
      static_cast<Npp16u>(600 >> shiftConstants[1]),  static_cast<Npp16u>(300 >> shiftConstants[2])};

  // Allocate GPU memory
  int step;
  Npp16u *d_src = nppiMalloc_16u_C3(width, height, &step);
  Npp16u *d_dst = nppiMalloc_16u_C3(width, height, &step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * channels * sizeof(Npp16u);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiRShiftC_16u_C3R(d_src, step, shiftConstants, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp16u> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiRShiftCTest, RShiftC_16u_C4R_FourChannels) {
  const int width = 1;
  const int height = 2;
  const int channels = 4;
  const int totalPixels = width * height * channels;

  std::vector<Npp16u> hostSrc = {
      1600, 800, 400, 200, // pixel 0: R, G, B, A
      1200, 600, 300, 150  // pixel 1: R, G, B, A
  };
  Npp32u shiftConstants[4] = {3, 2, 1, 0};
  std::vector<Npp16u> expected = {
      static_cast<Npp16u>(1600 >> shiftConstants[0]), static_cast<Npp16u>(800 >> shiftConstants[1]),
      static_cast<Npp16u>(400 >> shiftConstants[2]),  static_cast<Npp16u>(200 >> shiftConstants[3]),
      static_cast<Npp16u>(1200 >> shiftConstants[0]), static_cast<Npp16u>(600 >> shiftConstants[1]),
      static_cast<Npp16u>(300 >> shiftConstants[2]),  static_cast<Npp16u>(150 >> shiftConstants[3])};

  // Allocate GPU memory
  int step;
  Npp16u *d_src = nppiMalloc_16u_C4(width, height, &step);
  Npp16u *d_dst = nppiMalloc_16u_C4(width, height, &step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * channels * sizeof(Npp16u);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiRShiftC_16u_C4R(d_src, step, shiftConstants, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp16u> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

// ============================================================================
// RShiftC Tests - 16s variants (using manual memory allocation)
// ============================================================================

TEST_F(NppiRShiftCTest, RShiftC_16s_C1R_SignedDataType) {
  const int width = 3;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp16s> hostSrc = {-128, -64, -32, 128, 64, 32};
  Npp32u shiftCount = 2;
  std::vector<Npp16s> expected = {-32, -16, -8, 32, 16, 8}; // src >> 2 (arithmetic shift for signed)

  // Allocate GPU memory using cudaMalloc for 16s types
  int step = width * sizeof(Npp16s);
  Npp16s *d_src, *d_dst;
  cudaMalloc(&d_src, height * step);
  cudaMalloc(&d_dst, height * step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  cudaMemcpy2D(d_src, step, hostSrc.data(), step, step, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiRShiftC_16s_C1R(d_src, step, shiftCount, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp16s> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), step, d_dst, step, step, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  cudaFree(d_src);
  cudaFree(d_dst);
}

TEST_F(NppiRShiftCTest, RShiftC_16s_C3R_MultiChannel) {
  const int width = 2;
  const int height = 1;
  const int channels = 3;
  const int totalPixels = width * height * channels;

  std::vector<Npp16s> hostSrc = {
      -128, 64,  -32, // pixel 0: R, G, B
      96,   -48, 24   // pixel 1: R, G, B
  };
  Npp32u shiftConstants[3] = {2, 3, 1};
  std::vector<Npp16s> expected = {
      static_cast<Npp16s>(-128 >> shiftConstants[0]), static_cast<Npp16s>(64 >> shiftConstants[1]),
      static_cast<Npp16s>(-32 >> shiftConstants[2]),  static_cast<Npp16s>(96 >> shiftConstants[0]),
      static_cast<Npp16s>(-48 >> shiftConstants[1]),  static_cast<Npp16s>(24 >> shiftConstants[2])};

  // Allocate GPU memory using cudaMalloc for 16s C3 data
  int step = width * channels * sizeof(Npp16s);
  Npp16s *d_src, *d_dst;
  cudaMalloc(&d_src, height * step);
  cudaMalloc(&d_dst, height * step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  cudaMemcpy2D(d_src, step, hostSrc.data(), step, step, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiRShiftC_16s_C3R(d_src, step, shiftConstants, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp16s> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), step, d_dst, step, step, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  cudaFree(d_src);
  cudaFree(d_dst);
}

TEST_F(NppiRShiftCTest, RShiftC_16s_C4R_FourChannels) {
  const int width = 1;
  const int height = 2;
  const int channels = 4;
  const int totalPixels = width * height * channels;

  std::vector<Npp16s> hostSrc = {
      -128, 64,  -32, 16, // pixel 0: R, G, B, A
      96,   -48, 24,  -8  // pixel 1: R, G, B, A
  };
  Npp32u shiftConstants[4] = {2, 3, 1, 0};
  std::vector<Npp16s> expected = {
      static_cast<Npp16s>(-128 >> shiftConstants[0]), static_cast<Npp16s>(64 >> shiftConstants[1]),
      static_cast<Npp16s>(-32 >> shiftConstants[2]),  static_cast<Npp16s>(16 >> shiftConstants[3]),
      static_cast<Npp16s>(96 >> shiftConstants[0]),   static_cast<Npp16s>(-48 >> shiftConstants[1]),
      static_cast<Npp16s>(24 >> shiftConstants[2]),   static_cast<Npp16s>(-8 >> shiftConstants[3])};

  // Allocate GPU memory using cudaMalloc for 16s C4 data
  int step = width * channels * sizeof(Npp16s);
  Npp16s *d_src, *d_dst;
  cudaMalloc(&d_src, height * step);
  cudaMalloc(&d_dst, height * step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  cudaMemcpy2D(d_src, step, hostSrc.data(), step, step, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiRShiftC_16s_C4R(d_src, step, shiftConstants, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp16s> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), step, d_dst, step, step, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  cudaFree(d_src);
  cudaFree(d_dst);
}

// ============================================================================
// RShiftC Tests - 32s variants (using manual memory allocation)
// ============================================================================

TEST_F(NppiRShiftCTest, RShiftC_32s_C1R_DataTypes) {
  const int width = 3;
  const int height = 1;
  const int totalPixels = width * height;

  std::vector<Npp32s> hostSrc = {32000, 16000, 8000};
  Npp32u shiftCount = 4;
  std::vector<Npp32s> expected = {2000, 1000, 500}; // src >> 4

  // Allocate GPU memory using cudaMalloc for 32s types
  int step = width * sizeof(Npp32s);
  Npp32s *d_src, *d_dst;
  cudaMalloc(&d_src, height * step);
  cudaMalloc(&d_dst, height * step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  cudaMemcpy2D(d_src, step, hostSrc.data(), step, step, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiRShiftC_32s_C1R(d_src, step, shiftCount, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32s> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), step, d_dst, step, step, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  cudaFree(d_src);
  cudaFree(d_dst);
}

TEST_F(NppiRShiftCTest, RShiftC_32s_C3R_MultiChannel) {
  const int width = 2;
  const int height = 1;
  const int channels = 3;
  const int totalPixels = width * height * channels;

  std::vector<Npp32s> hostSrc = {
      -32000, 16000,  -8000, // pixel 0: R, G, B
      24000,  -12000, 6000   // pixel 1: R, G, B
  };
  Npp32u shiftConstants[3] = {3, 4, 2};
  std::vector<Npp32s> expected = {-32000 >> shiftConstants[0], 16000 >> shiftConstants[1],  -8000 >> shiftConstants[2],
                                  24000 >> shiftConstants[0],  -12000 >> shiftConstants[1], 6000 >> shiftConstants[2]};

  // Allocate GPU memory using cudaMalloc for 32s C3 data
  int step = width * channels * sizeof(Npp32s);
  Npp32s *d_src, *d_dst;
  cudaMalloc(&d_src, height * step);
  cudaMalloc(&d_dst, height * step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  cudaMemcpy2D(d_src, step, hostSrc.data(), step, step, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiRShiftC_32s_C3R(d_src, step, shiftConstants, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32s> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), step, d_dst, step, step, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  cudaFree(d_src);
  cudaFree(d_dst);
}

TEST_F(NppiRShiftCTest, RShiftC_32s_C4R_FourChannels) {
  const int width = 1;
  const int height = 2;
  const int channels = 4;
  const int totalPixels = width * height * channels;

  std::vector<Npp32s> hostSrc = {
      -32000, 16000,  -8000, 4000, // pixel 0: R, G, B, A
      24000,  -12000, 6000,  -3000 // pixel 1: R, G, B, A
  };
  Npp32u shiftConstants[4] = {3, 2, 1, 0};
  std::vector<Npp32s> expected = {-32000 >> shiftConstants[0], 16000 >> shiftConstants[1], -8000 >> shiftConstants[2],
                                  4000 >> shiftConstants[3],   24000 >> shiftConstants[0], -12000 >> shiftConstants[1],
                                  6000 >> shiftConstants[2],   -3000 >> shiftConstants[3]};

  // Allocate GPU memory using cudaMalloc for 32s C4 data
  int step = width * channels * sizeof(Npp32s);
  Npp32s *d_src, *d_dst;
  cudaMalloc(&d_src, height * step);
  cudaMalloc(&d_dst, height * step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  cudaMemcpy2D(d_src, step, hostSrc.data(), step, step, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiRShiftC_32s_C4R(d_src, step, shiftConstants, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32s> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), step, d_dst, step, step, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  cudaFree(d_src);
  cudaFree(d_dst);
}