#include "npp.h"
#include "npp_version_compat.h"
#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

class NPPSIntegralTest : public ::testing::Test {
protected:
  void SetUp() override {}
  void TearDown() override {}

  // Helper function to compute expected integral (exclusive prefix sum)
  // NVIDIA NPP uses exclusive scan: result[i] = sum of all elements before index i
  std::vector<Npp32s> computeExpectedIntegral(const std::vector<Npp32s> &src) {
    std::vector<Npp32s> result(src.size());
    Npp32s sum = 0;
    for (size_t i = 0; i < src.size(); i++) {
      result[i] = sum;
      sum += src[i];
    }
    return result;
  }
};

// ==============================================================================
// nppsIntegralGetBufferSize_32s Tests
// ==============================================================================

TEST_F(NPPSIntegralTest, GetBufferSize_32s_BasicOperation) {
  NppSignalLength bufferSize = 0;

  NppStatus status = nppsIntegralGetBufferSize_32s(1024, &bufferSize);
  ASSERT_EQ(status, NPP_NO_ERROR);
  EXPECT_GT(bufferSize, 0);
}

TEST_F(NPPSIntegralTest, GetBufferSize_32s_VariousLengths) {
  NppSignalLength bufferSize = 0;

  for (NppSignalLength nLength : {1, 10, 100, 256, 512, 1000, 10000}) {
    NppStatus status = nppsIntegralGetBufferSize_32s(nLength, &bufferSize);
    ASSERT_EQ(status, NPP_NO_ERROR) << "Failed for length " << nLength;
    EXPECT_GT(bufferSize, 0) << "Buffer size should be > 0 for length " << nLength;
  }
}

// ==============================================================================
// nppsIntegral_32s Tests
// ==============================================================================

TEST_F(NPPSIntegralTest, Integral_32s_SmallArray) {
  const NppSignalLength nLength = 10;

  // Input: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  std::vector<Npp32s> src(nLength);
  for (NppSignalLength i = 0; i < nLength; i++) {
    src[i] = static_cast<Npp32s>(i + 1);
  }

  // Expected output: [1, 3, 6, 10, 15, 21, 28, 36, 45, 55]
  std::vector<Npp32s> expected = computeExpectedIntegral(src);

  // Allocate GPU memory
  Npp32s *d_src = nullptr;
  Npp32s *d_dst = nullptr;
  Npp8u *d_buffer = nullptr;

  cudaMalloc(&d_src, nLength * sizeof(Npp32s));
  cudaMalloc(&d_dst, nLength * sizeof(Npp32s));

  NppSignalLength bufferSize = 0;
  NppStatus status = nppsIntegralGetBufferSize_32s(nLength, &bufferSize);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaMalloc(&d_buffer, bufferSize);
  ASSERT_TRUE(d_src && d_dst && d_buffer);

  // Copy data to GPU
  cudaMemcpy(d_src, src.data(), nLength * sizeof(Npp32s), cudaMemcpyHostToDevice);

  // Execute integral
  status = nppsIntegral_32s(d_src, d_dst, nLength, d_buffer);
  ASSERT_EQ(status, NPP_NO_ERROR);

  // Copy results back
  std::vector<Npp32s> result(nLength);
  cudaMemcpy(result.data(), d_dst, nLength * sizeof(Npp32s), cudaMemcpyDeviceToHost);

  // Verify results
  for (NppSignalLength i = 0; i < nLength; i++) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
  }

  // Cleanup
  cudaFree(d_src);
  cudaFree(d_dst);
  cudaFree(d_buffer);
}

TEST_F(NPPSIntegralTest, Integral_32s_ConstantValue) {
  const NppSignalLength nLength = 100;
  const Npp32s constValue = 5;

  std::vector<Npp32s> src(nLength, constValue);
  std::vector<Npp32s> expected = computeExpectedIntegral(src);

  Npp32s *d_src = nullptr;
  Npp32s *d_dst = nullptr;
  Npp8u *d_buffer = nullptr;

  cudaMalloc(&d_src, nLength * sizeof(Npp32s));
  cudaMalloc(&d_dst, nLength * sizeof(Npp32s));

  NppSignalLength bufferSize = 0;
  NppStatus status = nppsIntegralGetBufferSize_32s(nLength, &bufferSize);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaMalloc(&d_buffer, bufferSize);
  ASSERT_TRUE(d_src && d_dst && d_buffer);

  cudaMemcpy(d_src, src.data(), nLength * sizeof(Npp32s), cudaMemcpyHostToDevice);

  status = nppsIntegral_32s(d_src, d_dst, nLength, d_buffer);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32s> result(nLength);
  cudaMemcpy(result.data(), d_dst, nLength * sizeof(Npp32s), cudaMemcpyDeviceToHost);

  for (NppSignalLength i = 0; i < nLength; i++) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
  }

  cudaFree(d_src);
  cudaFree(d_dst);
  cudaFree(d_buffer);
}

TEST_F(NPPSIntegralTest, Integral_32s_MediumArray) {
  const NppSignalLength nLength = 512;

  std::vector<Npp32s> src(nLength);
  for (NppSignalLength i = 0; i < nLength; i++) {
    src[i] = static_cast<Npp32s>((i % 10) - 5); // Values from -5 to 4
  }

  std::vector<Npp32s> expected = computeExpectedIntegral(src);

  Npp32s *d_src = nullptr;
  Npp32s *d_dst = nullptr;
  Npp8u *d_buffer = nullptr;

  cudaMalloc(&d_src, nLength * sizeof(Npp32s));
  cudaMalloc(&d_dst, nLength * sizeof(Npp32s));

  NppSignalLength bufferSize = 0;
  NppStatus status = nppsIntegralGetBufferSize_32s(nLength, &bufferSize);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaMalloc(&d_buffer, bufferSize);
  ASSERT_TRUE(d_src && d_dst && d_buffer);

  cudaMemcpy(d_src, src.data(), nLength * sizeof(Npp32s), cudaMemcpyHostToDevice);

  status = nppsIntegral_32s(d_src, d_dst, nLength, d_buffer);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32s> result(nLength);
  cudaMemcpy(result.data(), d_dst, nLength * sizeof(Npp32s), cudaMemcpyDeviceToHost);

  for (NppSignalLength i = 0; i < nLength; i++) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
  }

  cudaFree(d_src);
  cudaFree(d_dst);
  cudaFree(d_buffer);
}

TEST_F(NPPSIntegralTest, Integral_32s_LargeArray) {
  const NppSignalLength nLength = 10000;

  std::vector<Npp32s> src(nLength);
  for (NppSignalLength i = 0; i < nLength; i++) {
    src[i] = static_cast<Npp32s>((i % 100) - 50);
  }

  std::vector<Npp32s> expected = computeExpectedIntegral(src);

  Npp32s *d_src = nullptr;
  Npp32s *d_dst = nullptr;
  Npp8u *d_buffer = nullptr;

  cudaMalloc(&d_src, nLength * sizeof(Npp32s));
  cudaMalloc(&d_dst, nLength * sizeof(Npp32s));

  NppSignalLength bufferSize = 0;
  NppStatus status = nppsIntegralGetBufferSize_32s(nLength, &bufferSize);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaMalloc(&d_buffer, bufferSize);
  ASSERT_TRUE(d_src && d_dst && d_buffer);

  cudaMemcpy(d_src, src.data(), nLength * sizeof(Npp32s), cudaMemcpyHostToDevice);

  status = nppsIntegral_32s(d_src, d_dst, nLength, d_buffer);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32s> result(nLength);
  cudaMemcpy(result.data(), d_dst, nLength * sizeof(Npp32s), cudaMemcpyDeviceToHost);

  for (NppSignalLength i = 0; i < nLength; i++) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
  }

  cudaFree(d_src);
  cudaFree(d_dst);
  cudaFree(d_buffer);
}

TEST_F(NPPSIntegralTest, Integral_32s_SingleElement) {
  const NppSignalLength nLength = 1;
  const Npp32s value = 42;

  std::vector<Npp32s> src = {value};

  Npp32s *d_src = nullptr;
  Npp32s *d_dst = nullptr;
  Npp8u *d_buffer = nullptr;

  cudaMalloc(&d_src, nLength * sizeof(Npp32s));
  cudaMalloc(&d_dst, nLength * sizeof(Npp32s));

  NppSignalLength bufferSize = 0;
  NppStatus status = nppsIntegralGetBufferSize_32s(nLength, &bufferSize);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaMalloc(&d_buffer, bufferSize);
  ASSERT_TRUE(d_src && d_dst && d_buffer);

  cudaMemcpy(d_src, src.data(), nLength * sizeof(Npp32s), cudaMemcpyHostToDevice);

  status = nppsIntegral_32s(d_src, d_dst, nLength, d_buffer);
  ASSERT_EQ(status, NPP_NO_ERROR);

  Npp32s result;
  cudaMemcpy(&result, d_dst, sizeof(Npp32s), cudaMemcpyDeviceToHost);

  // Exclusive prefix sum: first element is always 0
  EXPECT_EQ(result, 0);

  cudaFree(d_src);
  cudaFree(d_dst);
  cudaFree(d_buffer);
}

TEST_F(NPPSIntegralTest, Integral_32s_Ctx_BasicOperation) {
  const NppSignalLength nLength = 256;

  std::vector<Npp32s> src(nLength);
  for (NppSignalLength i = 0; i < nLength; i++) {
    src[i] = static_cast<Npp32s>(i + 1);
  }

  std::vector<Npp32s> expected = computeExpectedIntegral(src);

  Npp32s *d_src = nullptr;
  Npp32s *d_dst = nullptr;
  Npp8u *d_buffer = nullptr;

  cudaMalloc(&d_src, nLength * sizeof(Npp32s));
  cudaMalloc(&d_dst, nLength * sizeof(Npp32s));

  NppSignalLength bufferSize = 0;
  NppStatus status = nppsIntegralGetBufferSize_32s(nLength, &bufferSize);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaMalloc(&d_buffer, bufferSize);
  ASSERT_TRUE(d_src && d_dst && d_buffer);

  cudaMemcpy(d_src, src.data(), nLength * sizeof(Npp32s), cudaMemcpyHostToDevice);

  // Use Ctx version
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);

  status = nppsIntegral_32s_Ctx(d_src, d_dst, nLength, d_buffer, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32s> result(nLength);
  cudaMemcpy(result.data(), d_dst, nLength * sizeof(Npp32s), cudaMemcpyDeviceToHost);

  for (NppSignalLength i = 0; i < nLength; i++) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
  }

  cudaFree(d_src);
  cudaFree(d_dst);
  cudaFree(d_buffer);
}

TEST_F(NPPSIntegralTest, Integral_32s_ZeroValues) {
  const NppSignalLength nLength = 50;

  std::vector<Npp32s> src(nLength, 0);

  Npp32s *d_src = nullptr;
  Npp32s *d_dst = nullptr;
  Npp8u *d_buffer = nullptr;

  cudaMalloc(&d_src, nLength * sizeof(Npp32s));
  cudaMalloc(&d_dst, nLength * sizeof(Npp32s));

  NppSignalLength bufferSize = 0;
  NppStatus status = nppsIntegralGetBufferSize_32s(nLength, &bufferSize);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaMalloc(&d_buffer, bufferSize);
  ASSERT_TRUE(d_src && d_dst && d_buffer);

  cudaMemcpy(d_src, src.data(), nLength * sizeof(Npp32s), cudaMemcpyHostToDevice);

  status = nppsIntegral_32s(d_src, d_dst, nLength, d_buffer);
  ASSERT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp32s> result(nLength);
  cudaMemcpy(result.data(), d_dst, nLength * sizeof(Npp32s), cudaMemcpyDeviceToHost);

  for (NppSignalLength i = 0; i < nLength; i++) {
    EXPECT_EQ(result[i], 0) << "Expected 0 at index " << i;
  }

  cudaFree(d_src);
  cudaFree(d_dst);
  cudaFree(d_buffer);
}
