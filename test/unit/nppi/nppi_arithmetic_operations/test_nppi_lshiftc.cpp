#include "npp.h"
#include "npp_test_base.h"
#include <gtest/gtest.h>
#include <vector>

class NppiLShiftCTest : public ::testing::Test {
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
// LShiftC Tests - 8u variants
// ============================================================================

TEST_F(NppiLShiftCTest, LShiftC_8u_C1R_BasicOperation) {
  const int width = 4;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp8u> hostSrc = {1, 2, 4, 8, 16, 32, 64, 128};
  Npp32u shiftCount = 2;
  std::vector<Npp8u> expected = {4, 8, 16, 32, 64, 128, 0, 0}; // src << 2 (with overflow saturation)

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
  NppStatus status = nppiLShiftC_8u_C1R(d_src, step, shiftCount, d_dst, step, oSizeROI);
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

TEST_F(NppiLShiftCTest, LShiftC_8u_C1IR_InPlace) {
  const int width = 4;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp8u> hostSrc = {3, 5, 7, 9, 11, 13, 15, 17};
  Npp32u shiftCount = 1;
  std::vector<Npp8u> expected = {6, 10, 14, 18, 22, 26, 30, 34}; // src << 1

  // Allocate GPU memory
  int step;
  Npp8u *d_srcDst = nppiMalloc_8u_C1(width, height, &step);

  ASSERT_NE(d_srcDst, nullptr);

  // Copy data to GPU
  int hostStep = width * sizeof(Npp8u);
  cudaMemcpy2D(d_srcDst, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute in-place operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiLShiftC_8u_C1IR(shiftCount, d_srcDst, step, oSizeROI);
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

TEST_F(NppiLShiftCTest, LShiftC_8u_C3R_MultiChannel) {
  const int width = 2;
  const int height = 2;
  const int channels = 3;
  const int totalPixels = width * height * channels;

  std::vector<Npp8u> hostSrc = {
      1,  2,  3, // pixel 0: R, G, B
      4,  5,  6, // pixel 1: R, G, B
      7,  8,  9, // pixel 2: R, G, B
      10, 11, 12 // pixel 3: R, G, B
  };

  // each channel use different shift constant
  Npp32u shiftConstants[3] = {1, 2, 1};

  std::vector<Npp8u> expected = {// pixel 0: R(1<<1)=2, G(2<<2)=8, B(3<<3)=24
                                 static_cast<Npp8u>(1 << shiftConstants[0]), static_cast<Npp8u>(2 << shiftConstants[1]),
                                 static_cast<Npp8u>(3 << shiftConstants[2]),

                                 // pixel 1: R(4<<1)=8, G(5<<2)=20, B(6<<3)=48
                                 static_cast<Npp8u>(4 << shiftConstants[0]), static_cast<Npp8u>(5 << shiftConstants[1]),
                                 static_cast<Npp8u>(6 << shiftConstants[2]),

                                 // pixel 2: R(7<<1)=14, G(8<<2)=32, B(9<<3)=72
                                 static_cast<Npp8u>(7 << shiftConstants[0]), static_cast<Npp8u>(8 << shiftConstants[1]),
                                 static_cast<Npp8u>(9 << shiftConstants[2]),

                                 // pixel 3: R(10<<1)=20, G(11<<2)=44, B(12<<3)=96
                                 static_cast<Npp8u>(10 << shiftConstants[0]),
                                 static_cast<Npp8u>(11 << shiftConstants[1]),
                                 static_cast<Npp8u>(12 << shiftConstants[2])};

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
  NppStatus status = nppiLShiftC_8u_C3R(d_src, step, shiftConstants, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp8u> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    // printf("i%d: src=%d, result=%d, expected=%d\n", i, hostSrc[i], hostResult[i], expected[i]);
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at pixel " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiLShiftCTest, LShiftC_8u_C3R_Shift8ProducesZero) {
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
  NppStatus status = nppiLShiftC_8u_C3R(d_src, step, shiftConstants, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiLShiftCTest, LShiftC_8u_C4R_FourChannels) {
  const int width = 2;
  const int height = 1;
  const int channels = 4;
  const int totalPixels = width * height * channels;

  std::vector<Npp8u> hostSrc = {
      1, 2, 3, 4, // pixel 0: R, G, B, A
      5, 6, 7, 8  // pixel 1: R, G, B, A
  };
  Npp32u shiftConstants[4] = {1, 2, 1, 0};
  std::vector<Npp8u> expected = {
      static_cast<Npp8u>(1 << shiftConstants[0]), static_cast<Npp8u>(2 << shiftConstants[1]),
      static_cast<Npp8u>(3 << shiftConstants[2]), static_cast<Npp8u>(4 << shiftConstants[3]),
      static_cast<Npp8u>(5 << shiftConstants[0]), static_cast<Npp8u>(6 << shiftConstants[1]),
      static_cast<Npp8u>(7 << shiftConstants[2]), static_cast<Npp8u>(8 << shiftConstants[3])};

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
  NppStatus status = nppiLShiftC_8u_C4R(d_src, step, shiftConstants, d_dst, step, oSizeROI);
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
// LShiftC Tests - 16u variants
// ============================================================================

TEST_F(NppiLShiftCTest, LShiftC_16u_C1R_DataTypes) {
  const int width = 3;
  const int height = 2;
  const int totalPixels = width * height;

  std::vector<Npp16u> hostSrc = {100, 200, 300, 400, 500, 600};
  Npp32u shiftCount = 3;
  std::vector<Npp16u> expected = {800, 1600, 2400, 3200, 4000, 4800}; // src << 3

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
  NppStatus status = nppiLShiftC_16u_C1R(d_src, step, shiftCount, d_dst, step, oSizeROI);
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

TEST_F(NppiLShiftCTest, LShiftC_16u_C3R_MultiChannel) {
  const int width = 2;
  const int height = 1;
  const int channels = 3;
  const int totalPixels = width * height * channels;

  std::vector<Npp16u> hostSrc = {
      100, 200, 300, // pixel 0: R, G, B
      400, 500, 600  // pixel 1: R, G, B
  };
  Npp32u shiftConstants[3] = {2, 3, 1};
  std::vector<Npp16u> expected = {
      static_cast<Npp16u>(100 << shiftConstants[0]), static_cast<Npp16u>(200 << shiftConstants[1]),
      static_cast<Npp16u>(300 << shiftConstants[2]), static_cast<Npp16u>(400 << shiftConstants[0]),
      static_cast<Npp16u>(500 << shiftConstants[1]), static_cast<Npp16u>(600 << shiftConstants[2])};

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
  NppStatus status = nppiLShiftC_16u_C3R(d_src, step, shiftConstants, d_dst, step, oSizeROI);
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

TEST_F(NppiLShiftCTest, LShiftC_16u_C4R_FourChannels) {
  const int width = 1;
  const int height = 2;
  const int channels = 4;
  const int totalPixels = width * height * channels;

  std::vector<Npp16u> hostSrc = {
      10, 20, 30, 40, // pixel 0: R, G, B, A
      50, 60, 70, 80  // pixel 1: R, G, B, A
  };
  Npp32u shiftConstants[4] = {2, 1, 3, 0};
  std::vector<Npp16u> expected = {
      static_cast<Npp16u>(10 << shiftConstants[0]), static_cast<Npp16u>(20 << shiftConstants[1]),
      static_cast<Npp16u>(30 << shiftConstants[2]), static_cast<Npp16u>(40 << shiftConstants[3]),
      static_cast<Npp16u>(50 << shiftConstants[0]), static_cast<Npp16u>(60 << shiftConstants[1]),
      static_cast<Npp16u>(70 << shiftConstants[2]), static_cast<Npp16u>(80 << shiftConstants[3])};

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
  NppStatus status = nppiLShiftC_16u_C4R(d_src, step, shiftConstants, d_dst, step, oSizeROI);
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
// LShiftC Tests - 32s variants
// ============================================================================

TEST_F(NppiLShiftCTest, LShiftC_32s_C1R_DataTypes) {
  const int width = 3;
  const int height = 1;
  const int totalPixels = width * height;

  std::vector<Npp32s> hostSrc = {1000, 2000, 3000};
  Npp32u shiftCount = 4;
  std::vector<Npp32s> expected = {16000, 32000, 48000}; // src << 4

  // Allocate GPU memory
  int step;
  Npp32s *d_src = nppiMalloc_32s_C1(width, height, &step);
  Npp32s *d_dst = nppiMalloc_32s_C1(width, height, &step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * sizeof(Npp32s);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiLShiftC_32s_C1R(d_src, step, shiftCount, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32s> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiLShiftCTest, LShiftC_32s_C3R_MultiChannel) {
  const int width = 2;
  const int height = 1;
  const int channels = 3;
  const int totalPixels = width * height * channels;

  std::vector<Npp32s> hostSrc = {
      100, 200, 300, // pixel 0: R, G, B
      400, 500, 600  // pixel 1: R, G, B
  };
  Npp32u shiftConstants[3] = {2, 3, 4};
  std::vector<Npp32s> expected = {100 << shiftConstants[0], 200 << shiftConstants[1], 300 << shiftConstants[2],
                                  400 << shiftConstants[0], 500 << shiftConstants[1], 600 << shiftConstants[2]};

  // Allocate GPU memory
  int step;
  Npp32s *d_src = nppiMalloc_32s_C3(width, height, &step);
  Npp32s *d_dst = nppiMalloc_32s_C3(width, height, &step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * channels * sizeof(Npp32s);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiLShiftC_32s_C3R(d_src, step, shiftConstants, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32s> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiLShiftCTest, LShiftC_32s_C4R_FourChannels) {
  const int width = 1;
  const int height = 2;
  const int channels = 4;
  const int totalPixels = width * height * channels;

  std::vector<Npp32s> hostSrc = {
      100, 200, 300, 400, // pixel 0: R, G, B, A
      500, 600, 700, 800  // pixel 1: R, G, B, A
  };
  Npp32u shiftConstants[4] = {3, 2, 1, 0};
  std::vector<Npp32s> expected = {100 << shiftConstants[0], 200 << shiftConstants[1], 300 << shiftConstants[2],
                                  400 << shiftConstants[3], 500 << shiftConstants[0], 600 << shiftConstants[1],
                                  700 << shiftConstants[2], 800 << shiftConstants[3]};

  // Allocate GPU memory
  int step;
  Npp32s *d_src = nppiMalloc_32s_C4(width, height, &step);
  Npp32s *d_dst = nppiMalloc_32s_C4(width, height, &step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * channels * sizeof(Npp32s);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiLShiftC_32s_C4R(d_src, step, shiftConstants, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32s> hostResult(totalPixels);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalPixels; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}
