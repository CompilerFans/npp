#include "npp.h"
#include "npp_test_base.h"
#include <gtest/gtest.h>
#include <vector>

class NppiOrCTest : public ::testing::Test {
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

TEST_F(NppiOrCTest, OrC_8u_C3R_BasicOperation) {
  const int width = 2;
  const int height = 1;
  const int channels = 3;
  const int totalElements = width * height * channels;

  std::vector<Npp8u> hostSrc = {
    0xFF, 0x00, 0xAA,  // pixel 0
    0x55, 0x33, 0xCC   // pixel 1
  };
  Npp8u constants[3] = {0x0F, 0xF0, 0x55};
  std::vector<Npp8u> expected = {
    0xFF, 0xF0, 0xFF,  // pixel 0
    0x5F, 0xF3, 0xDD   // pixel 1
  };

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
  NppStatus status = nppiOrC_8u_C3R(d_src, step, constants, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp8u> hostResult(totalElements);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalElements; ++i) {
    // printf("i: %d, host: %x, expect: %x", i, hostResult[i], expected[i]);
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiOrCTest, OrC_8u_C3R_Ctx_WithStreamContext) {
  const int width = 2;
  const int height = 1;
  const int channels = 3;
  const int totalElements = width * height * channels;

  std::vector<Npp8u> hostSrc = {
    0xFF, 0x00, 0xAA,  // pixel 0
    0x55, 0x33, 0xCC   // pixel 1
  };
  Npp8u constants[3] = {0x0F, 0xF0, 0x55};
  std::vector<Npp8u> expected = {
    0xFF, 0xF0, 0xFF,  // pixel 0
    0x5F, 0xF3, 0xDD   // pixel 1
  };

  // Allocate GPU memory
  int step;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &step);
  Npp8u *d_dst = nppiMalloc_8u_C3(width, height, &step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * channels * sizeof(Npp8u);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation with context
  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status = nppiOrC_8u_C3R_Ctx(d_src, step, constants, d_dst, step, oSizeROI, nppStreamCtx);
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

TEST_F(NppiOrCTest, OrC_16u_C3R_16BitData) {
  const int width = 2;
  const int height = 1;
  const int channels = 3;
  const int totalElements = width * height * channels;

  std::vector<Npp16u> hostSrc = {
    0xFFFF, 0x0000, 0xAAAA,  // pixel 0
    0x5555, 0x3333, 0xCCCC   // pixel 1
  };
  Npp16u constants[3] = {0x0F0F, 0xF0F0, 0x5555};
  // 修正的期望结果
  std::vector<Npp16u> expected = {
    0xFFFF | 0x0F0F, 0x0000 | 0xF0F0, 0xAAAA | 0x5555,  // pixel 0
    0x5555 | 0x0F0F, 0x3333 | 0xF0F0, 0xCCCC | 0x5555   // pixel 1
  };

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
  NppStatus status = nppiOrC_16u_C3R(d_src, step, constants, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp16u> hostResult(totalElements);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalElements; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiOrCTest, OrC_16u_C3R_Ctx_WithStreamContext) {
  const int width = 2;
  const int height = 1;
  const int channels = 3;
  const int totalElements = width * height * channels;

  std::vector<Npp16u> hostSrc = {
    0xFFFF, 0x0000, 0xAAAA,  // pixel 0
    0x5555, 0x3333, 0xCCCC   // pixel 1
  };
  Npp16u constants[3] = {0x0F0F, 0xF0F0, 0x5555};
  // 修正的期望结果
  std::vector<Npp16u> expected = {
    0xFFFF | 0x0F0F, 0x0000 | 0xF0F0, 0xAAAA | 0x5555,  // pixel 0
    0x5555 | 0x0F0F, 0x3333 | 0xF0F0, 0xCCCC | 0x5555   // pixel 1
  };

  // Allocate GPU memory
  int step;
  Npp16u *d_src = nppiMalloc_16u_C3(width, height, &step);
  Npp16u *d_dst = nppiMalloc_16u_C3(width, height, &step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * channels * sizeof(Npp16u);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation with context
  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status = nppiOrC_16u_C3R_Ctx(d_src, step, constants, d_dst, step, oSizeROI, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp16u> hostResult(totalElements);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalElements; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiOrCTest, OrC_16u_C4R_FourChannel) {
  const int width = 2;
  const int height = 1;
  const int channels = 4;
  const int totalElements = width * height * channels;

  std::vector<Npp16u> hostSrc = {
    0xFFFF, 0x0000, 0xAAAA, 0x5555,  // pixel 0
    0x3333, 0xCCCC, 0x0F0F, 0xF0F0   // pixel 1
  };
  Npp16u constants[4] = {0x0F0F, 0xF0F0, 0x5555, 0xAAAA};
  // 修正的期望结果
  std::vector<Npp16u> expected = {
    0xFFFF | 0x0F0F, 0x0000 | 0xF0F0, 0xAAAA | 0x5555, 0x5555 | 0xAAAA,  // pixel 0
    0x3333 | 0x0F0F, 0xCCCC | 0xF0F0, 0x0F0F | 0x5555, 0xF0F0 | 0xAAAA   // pixel 1
  };

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
  NppStatus status = nppiOrC_16u_C4R(d_src, step, constants, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp16u> hostResult(totalElements);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalElements; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiOrCTest, OrC_16u_C4R_Ctx_WithStreamContext) {
  const int width = 2;
  const int height = 1;
  const int channels = 4;
  const int totalElements = width * height * channels;

  std::vector<Npp16u> hostSrc = {
    0xFFFF, 0x0000, 0xAAAA, 0x5555,  // pixel 0
    0x3333, 0xCCCC, 0x0F0F, 0xF0F0   // pixel 1
  };
  Npp16u constants[4] = {0x0F0F, 0xF0F0, 0x5555, 0xAAAA};
  // 修正的期望结果
  std::vector<Npp16u> expected = {
    0xFFFF | 0x0F0F, 0x0000 | 0xF0F0, 0xAAAA | 0x5555, 0x5555 | 0xAAAA,  // pixel 0
    0x3333 | 0x0F0F, 0xCCCC | 0xF0F0, 0x0F0F | 0x5555, 0xF0F0 | 0xAAAA   // pixel 1
  };

  // Allocate GPU memory
  int step;
  Npp16u *d_src = nppiMalloc_16u_C4(width, height, &step);
  Npp16u *d_dst = nppiMalloc_16u_C4(width, height, &step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * channels * sizeof(Npp16u);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation with context
  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status = nppiOrC_16u_C4R_Ctx(d_src, step, constants, d_dst, step, oSizeROI, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp16u> hostResult(totalElements);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalElements; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiOrCTest, OrC_32s_C3R_32BitData) {
  const int width = 2;
  const int height = 1;
  const int channels = 3;
  const int totalElements = width * height * channels;

  std::vector<Npp32s> hostSrc = {
    (Npp32s)0xFFFFFFFF, 0x00000000, (Npp32s)0xAAAAAAAA,  // pixel 0
    0x55555555,         0x33333333, (Npp32s)0xCCCCCCCC   // pixel 1
  };
  Npp32s constants[3] = {0x0F0F0F0F, (Npp32s)0xF0F0F0F0, 0x55555555};
  std::vector<Npp32s> expected = {
    (Npp32s)0xFFFFFFFF, (Npp32s)0xF0F0F0F0, (Npp32s)0xFFFFFFFF,  // pixel 0
    0x5F5F5F5F,         (Npp32s)0xF3F3F3F3, (Npp32s)0xDDDDDDDD   // pixel 1
  };

  // Allocate GPU memory
  int step;
  Npp32s *d_src = (Npp32s *)nppiMalloc_32s_C3(width, height, &step);
  Npp32s *d_dst = (Npp32s *)nppiMalloc_32s_C3(width, height, &step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * channels * sizeof(Npp32s);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiOrC_32s_C3R(d_src, step, constants, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32s> hostResult(totalElements);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalElements; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiOrCTest, OrC_32s_C3R_Ctx_WithStreamContext) {
  const int width = 2;
  const int height = 1;
  const int channels = 3;
  const int totalElements = width * height * channels;

  std::vector<Npp32s> hostSrc = {
    (Npp32s)0xFFFFFFFF, 0x00000000, (Npp32s)0xAAAAAAAA,  // pixel 0
    0x55555555,         0x33333333, (Npp32s)0xCCCCCCCC   // pixel 1
  };
  Npp32s constants[3] = {0x0F0F0F0F, (Npp32s)0xF0F0F0F0, 0x55555555};
  std::vector<Npp32s> expected = {
    (Npp32s)0xFFFFFFFF, (Npp32s)0xF0F0F0F0, (Npp32s)0xFFFFFFFF,  // pixel 0
    0x5F5F5F5F,         (Npp32s)0xF3F3F3F3, (Npp32s)0xDDDDDDDD   // pixel 1
  };

  // Allocate GPU memory
  int step;
  Npp32s *d_src = (Npp32s *)nppiMalloc_32s_C3(width, height, &step);
  Npp32s *d_dst = (Npp32s *)nppiMalloc_32s_C3(width, height, &step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * channels * sizeof(Npp32s);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation with context
  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status = nppiOrC_32s_C3R_Ctx(d_src, step, constants, d_dst, step, oSizeROI, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32s> hostResult(totalElements);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalElements; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiOrCTest, OrC_32s_C4R_FourChannel32Bit) {
  const int width = 2;
  const int height = 1;
  const int channels = 4;
  const int totalElements = width * height * channels;

  std::vector<Npp32s> hostSrc = {
    (Npp32s)0xFFFFFFFF, 0x00000000, (Npp32s)0xAAAAAAAA, 0x55555555,  // pixel 0
    0x33333333,         (Npp32s)0xCCCCCCCC, 0x0F0F0F0F, (Npp32s)0xF0F0F0F0  // pixel 1
  };
  Npp32s constants[4] = {0x0F0F0F0F, (Npp32s)0xF0F0F0F0, 0x55555555, (Npp32s)0xAAAAAAAA};
  std::vector<Npp32s> expected = {
    (Npp32s)0xFFFFFFFF, (Npp32s)0xF0F0F0F0, (Npp32s)0xFFFFFFFF, (Npp32s)0xFFFFFFFF,  // pixel 0
    0x3F3F3F3F,         (Npp32s)0xFCFCFCFC, 0x5F5F5F5F, (Npp32s)0xFAFAFAFA          // pixel 1
  };

  // Allocate GPU memory
  int step;
  Npp32s *d_src = (Npp32s *)nppiMalloc_32s_C4(width, height, &step);
  Npp32s *d_dst = (Npp32s *)nppiMalloc_32s_C4(width, height, &step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * channels * sizeof(Npp32s);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStatus status = nppiOrC_32s_C4R(d_src, step, constants, d_dst, step, oSizeROI);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32s> hostResult(totalElements);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalElements; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(NppiOrCTest, OrC_32s_C4R_Ctx_WithStreamContext) {
  const int width = 2;
  const int height = 1;
  const int channels = 4;
  const int totalElements = width * height * channels;

  std::vector<Npp32s> hostSrc = {
    (Npp32s)0xFFFFFFFF, 0x00000000, (Npp32s)0xAAAAAAAA, 0x55555555,  // pixel 0
    0x33333333,         (Npp32s)0xCCCCCCCC, 0x0F0F0F0F, (Npp32s)0xF0F0F0F0  // pixel 1
  };
  Npp32s constants[4] = {0x0F0F0F0F, (Npp32s)0xF0F0F0F0, 0x55555555, (Npp32s)0xAAAAAAAA};
  std::vector<Npp32s> expected = {
    (Npp32s)0xFFFFFFFF, (Npp32s)0xF0F0F0F0, (Npp32s)0xFFFFFFFF, (Npp32s)0xFFFFFFFF,  // pixel 0
    0x3F3F3F3F,         (Npp32s)0xFCFCFCFC, 0x5F5F5F5F, (Npp32s)0xFAFAFAFA          // pixel 1
  };

  // Allocate GPU memory
  int step;
  Npp32s *d_src = (Npp32s *)nppiMalloc_32s_C4(width, height, &step);
  Npp32s *d_dst = (Npp32s *)nppiMalloc_32s_C4(width, height, &step);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  int hostStep = width * channels * sizeof(Npp32s);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute operation with context
  NppiSize oSizeROI = {width, height};
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  NppStatus status = nppiOrC_32s_C4R_Ctx(d_src, step, constants, d_dst, step, oSizeROI, nppStreamCtx);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp32s> hostResult(totalElements);
  cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalElements; ++i) {
    EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  nppiFree(d_src);
  nppiFree(d_dst);
}