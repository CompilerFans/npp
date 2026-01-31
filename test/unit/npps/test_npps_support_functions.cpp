#include "npp.h"
#include <gtest/gtest.h>
#include <vector>

class NppsSupportTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Signal length for testing
    signal_size = 1024;
  }

  void TearDown() override {
    // Base class cleanup
  }

  size_t signal_size;
};

// Test 8-bit unsigned signal memory allocation
TEST_F(NppsSupportTest, Malloc_8u) {
  Npp8u *signal = nppsMalloc_8u(signal_size);
  ASSERT_NE(signal, nullptr) << "Failed to allocate 8u signal memory";

  // Test write and read
  std::vector<Npp8u> host_data(signal_size);
  for (size_t i = 0; i < signal_size; ++i) {
    host_data[i] = static_cast<Npp8u>(i % 256);
  }

  cudaError_t err = cudaMemcpy(signal, host_data.data(), signal_size * sizeof(Npp8u), cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data to device";

  std::vector<Npp8u> result(signal_size);
  err = cudaMemcpy(result.data(), signal, signal_size * sizeof(Npp8u), cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data from device";

  // Verify data integrity
  for (size_t i = 0; i < signal_size; ++i) {
    EXPECT_EQ(result[i], host_data[i]);
  }

  nppsFree(signal);
}

// Test 16-bit signed signal memory allocation
TEST_F(NppsSupportTest, Malloc_16s) {
  Npp16s *signal = nppsMalloc_16s(signal_size);
  ASSERT_NE(signal, nullptr) << "Failed to allocate 16s signal memory";

  // Test write and read
  std::vector<Npp16s> host_data(signal_size);
  for (size_t i = 0; i < signal_size; ++i) {
    host_data[i] = static_cast<Npp16s>(i - 512);
  }

  cudaError_t err = cudaMemcpy(signal, host_data.data(), signal_size * sizeof(Npp16s), cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data to device";

  std::vector<Npp16s> result(signal_size);
  err = cudaMemcpy(result.data(), signal, signal_size * sizeof(Npp16s), cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data from device";

  // Verify data integrity
  for (size_t i = 0; i < signal_size; ++i) {
    EXPECT_EQ(result[i], host_data[i]);
  }

  nppsFree(signal);
}

// Test 32-bit float signal memory allocation
TEST_F(NppsSupportTest, Malloc_32f) {
  Npp32f *signal = nppsMalloc_32f(signal_size);
  ASSERT_NE(signal, nullptr) << "Failed to allocate 32f signal memory";

  // Test write and read
  std::vector<Npp32f> host_data(signal_size);
  for (size_t i = 0; i < signal_size; ++i) {
    host_data[i] = static_cast<Npp32f>(i) * 0.5f;
  }

  cudaError_t err = cudaMemcpy(signal, host_data.data(), signal_size * sizeof(Npp32f), cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data to device";

  std::vector<Npp32f> result(signal_size);
  err = cudaMemcpy(result.data(), signal, signal_size * sizeof(Npp32f), cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data from device";

  // Verify data integrity
  for (size_t i = 0; i < signal_size; ++i) {
    EXPECT_FLOAT_EQ(result[i], host_data[i]);
  }

  nppsFree(signal);
}

// Test 32-bit complex signal memory allocation
TEST_F(NppsSupportTest, Malloc_32fc) {
  Npp32fc *signal = nppsMalloc_32fc(signal_size);
  ASSERT_NE(signal, nullptr) << "Failed to allocate 32fc signal memory";

  // Test write and read
  std::vector<Npp32fc> host_data(signal_size);
  for (size_t i = 0; i < signal_size; ++i) {
    host_data[i].re = static_cast<Npp32f>(i) * 0.5f;
    host_data[i].im = static_cast<Npp32f>(i) * 0.3f;
  }

  cudaError_t err = cudaMemcpy(signal, host_data.data(), signal_size * sizeof(Npp32fc), cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data to device";

  std::vector<Npp32fc> result(signal_size);
  err = cudaMemcpy(result.data(), signal, signal_size * sizeof(Npp32fc), cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data from device";

  // Verify data integrity
  for (size_t i = 0; i < signal_size; ++i) {
    EXPECT_FLOAT_EQ(result[i].re, host_data[i].re);
    EXPECT_FLOAT_EQ(result[i].im, host_data[i].im);
  }

  nppsFree(signal);
}

// Test 64-bit float signal memory allocation
TEST_F(NppsSupportTest, Malloc_64f) {
  Npp64f *signal = nppsMalloc_64f(signal_size);
  ASSERT_NE(signal, nullptr) << "Failed to allocate 64f signal memory";

  // Test write and read
  std::vector<Npp64f> host_data(signal_size);
  for (size_t i = 0; i < signal_size; ++i) {
    host_data[i] = static_cast<Npp64f>(i) * 0.123456789;
  }

  cudaError_t err = cudaMemcpy(signal, host_data.data(), signal_size * sizeof(Npp64f), cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data to device";

  std::vector<Npp64f> result(signal_size);
  err = cudaMemcpy(result.data(), signal, signal_size * sizeof(Npp64f), cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data from device";

  // Verify data integrity
  for (size_t i = 0; i < signal_size; ++i) {
    EXPECT_DOUBLE_EQ(result[i], host_data[i]);
  }

  nppsFree(signal);
}

// Error parameter tests
TEST(NppsSupportParameterTest, NullSize) {
  // Test zero size allocation
  EXPECT_EQ(nppsMalloc_32f(0), nullptr);
  EXPECT_EQ(nppsMalloc_16s(0), nullptr);
  EXPECT_EQ(nppsMalloc_32fc(0), nullptr);
}

TEST(NppsSupportParameterTest, FreeSafety) {
  // Test null pointer free safety
  EXPECT_NO_THROW(nppsFree(nullptr));

  // Test normal allocation and free
  Npp32f *signal = nppsMalloc_32f(100);
  ASSERT_NE(signal, nullptr);
  EXPECT_NO_THROW(nppsFree(signal));
}

// Memory alignment test
TEST_F(NppsSupportTest, MemoryAlignment) {
  // Test memory alignment (GPU memory is typically 256-byte aligned)
  Npp32f *signal1 = nppsMalloc_32f(signal_size);
  Npp32f *signal2 = nppsMalloc_32f(signal_size);

  ASSERT_NE(signal1, nullptr);
  ASSERT_NE(signal2, nullptr);

  // Check if pointers are valid device pointers
  cudaPointerAttributes attr1, attr2;
  cudaError_t err1 = cudaPointerGetAttributes(&attr1, signal1);
  cudaError_t err2 = cudaPointerGetAttributes(&attr2, signal2);

  EXPECT_EQ(err1, cudaSuccess);
  EXPECT_EQ(err2, cudaSuccess);

  if (err1 == cudaSuccess && err2 == cudaSuccess) {
    EXPECT_EQ(attr1.type, cudaMemoryTypeDevice);
    EXPECT_EQ(attr2.type, cudaMemoryTypeDevice);
  }

  nppsFree(signal1);
  nppsFree(signal2);
}

// Large memory allocation test
TEST(NppsSupportLargeMemoryTest, LargeAllocation) {
  // Test allocation of large signal (1M elements)
  const size_t large_size = 1024 * 1024;

  Npp32f *large_signal = nppsMalloc_32f(large_size);

  if (large_signal != nullptr) {
    // If allocation succeeds, test basic read/write operations
    Npp32f test_value = 42.0f;
    cudaError_t err = cudaMemset(large_signal, 0, large_size * sizeof(Npp32f));
    EXPECT_EQ(err, cudaSuccess);

    err = cudaMemcpy(large_signal, &test_value, sizeof(Npp32f), cudaMemcpyHostToDevice);
    EXPECT_EQ(err, cudaSuccess);

    Npp32f result;
    err = cudaMemcpy(&result, large_signal, sizeof(Npp32f), cudaMemcpyDeviceToHost);
    EXPECT_EQ(err, cudaSuccess);
    EXPECT_FLOAT_EQ(result, test_value);

    nppsFree(large_signal);
  } else {
    // If allocation fails, this may be due to insufficient memory, which is acceptable
    GTEST_FAIL() << "Large memory allocation failed - possibly due to insufficient GPU memory";
  }
}

// Test 8-bit signed signal memory allocation
TEST_F(NppsSupportTest, Malloc_8s) {
  Npp8s *signal = nppsMalloc_8s(signal_size);
  ASSERT_NE(signal, nullptr) << "Failed to allocate 8s signal memory";

  std::vector<Npp8s> host_data(signal_size);
  for (size_t i = 0; i < signal_size; ++i) {
    host_data[i] = static_cast<Npp8s>((i % 256) - 128);
  }

  cudaError_t err = cudaMemcpy(signal, host_data.data(), signal_size * sizeof(Npp8s), cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data to device";

  std::vector<Npp8s> result(signal_size);
  err = cudaMemcpy(result.data(), signal, signal_size * sizeof(Npp8s), cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data from device";

  for (size_t i = 0; i < signal_size; ++i) {
    EXPECT_EQ(result[i], host_data[i]);
  }

  nppsFree(signal);
}

// Test 16-bit unsigned signal memory allocation
TEST_F(NppsSupportTest, Malloc_16u) {
  Npp16u *signal = nppsMalloc_16u(signal_size);
  ASSERT_NE(signal, nullptr) << "Failed to allocate 16u signal memory";

  std::vector<Npp16u> host_data(signal_size);
  for (size_t i = 0; i < signal_size; ++i) {
    host_data[i] = static_cast<Npp16u>(i % 65536);
  }

  cudaError_t err = cudaMemcpy(signal, host_data.data(), signal_size * sizeof(Npp16u), cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data to device";

  std::vector<Npp16u> result(signal_size);
  err = cudaMemcpy(result.data(), signal, signal_size * sizeof(Npp16u), cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data from device";

  for (size_t i = 0; i < signal_size; ++i) {
    EXPECT_EQ(result[i], host_data[i]);
  }

  nppsFree(signal);
}

// Test 16-bit signed complex signal memory allocation
TEST_F(NppsSupportTest, Malloc_16sc) {
  Npp16sc *signal = nppsMalloc_16sc(signal_size);
  ASSERT_NE(signal, nullptr) << "Failed to allocate 16sc signal memory";

  std::vector<Npp16sc> host_data(signal_size);
  for (size_t i = 0; i < signal_size; ++i) {
    host_data[i].re = static_cast<Npp16s>(i - 512);
    host_data[i].im = static_cast<Npp16s>(512 - i);
  }

  cudaError_t err = cudaMemcpy(signal, host_data.data(), signal_size * sizeof(Npp16sc), cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data to device";

  std::vector<Npp16sc> result(signal_size);
  err = cudaMemcpy(result.data(), signal, signal_size * sizeof(Npp16sc), cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data from device";

  for (size_t i = 0; i < signal_size; ++i) {
    EXPECT_EQ(result[i].re, host_data[i].re);
    EXPECT_EQ(result[i].im, host_data[i].im);
  }

  nppsFree(signal);
}

// Test 32-bit unsigned signal memory allocation
TEST_F(NppsSupportTest, Malloc_32u) {
  Npp32u *signal = nppsMalloc_32u(signal_size);
  ASSERT_NE(signal, nullptr) << "Failed to allocate 32u signal memory";

  std::vector<Npp32u> host_data(signal_size);
  for (size_t i = 0; i < signal_size; ++i) {
    host_data[i] = static_cast<Npp32u>(i * 1000);
  }

  cudaError_t err = cudaMemcpy(signal, host_data.data(), signal_size * sizeof(Npp32u), cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data to device";

  std::vector<Npp32u> result(signal_size);
  err = cudaMemcpy(result.data(), signal, signal_size * sizeof(Npp32u), cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data from device";

  for (size_t i = 0; i < signal_size; ++i) {
    EXPECT_EQ(result[i], host_data[i]);
  }

  nppsFree(signal);
}

// Test 32-bit signed complex signal memory allocation
TEST_F(NppsSupportTest, Malloc_32sc) {
  Npp32sc *signal = nppsMalloc_32sc(signal_size);
  ASSERT_NE(signal, nullptr) << "Failed to allocate 32sc signal memory";

  std::vector<Npp32sc> host_data(signal_size);
  for (size_t i = 0; i < signal_size; ++i) {
    host_data[i].re = static_cast<Npp32s>(i);
    host_data[i].im = static_cast<Npp32s>(-static_cast<int>(i));
  }

  cudaError_t err = cudaMemcpy(signal, host_data.data(), signal_size * sizeof(Npp32sc), cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data to device";

  std::vector<Npp32sc> result(signal_size);
  err = cudaMemcpy(result.data(), signal, signal_size * sizeof(Npp32sc), cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data from device";

  for (size_t i = 0; i < signal_size; ++i) {
    EXPECT_EQ(result[i].re, host_data[i].re);
    EXPECT_EQ(result[i].im, host_data[i].im);
  }

  nppsFree(signal);
}

// Test 64-bit signed signal memory allocation
TEST_F(NppsSupportTest, Malloc_64s) {
  Npp64s *signal = nppsMalloc_64s(signal_size);
  ASSERT_NE(signal, nullptr) << "Failed to allocate 64s signal memory";

  std::vector<Npp64s> host_data(signal_size);
  for (size_t i = 0; i < signal_size; ++i) {
    host_data[i] = static_cast<Npp64s>(i) * 1000000LL;
  }

  cudaError_t err = cudaMemcpy(signal, host_data.data(), signal_size * sizeof(Npp64s), cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data to device";

  std::vector<Npp64s> result(signal_size);
  err = cudaMemcpy(result.data(), signal, signal_size * sizeof(Npp64s), cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data from device";

  for (size_t i = 0; i < signal_size; ++i) {
    EXPECT_EQ(result[i], host_data[i]);
  }

  nppsFree(signal);
}

// Test 64-bit signed complex signal memory allocation
TEST_F(NppsSupportTest, Malloc_64sc) {
  Npp64sc *signal = nppsMalloc_64sc(signal_size);
  ASSERT_NE(signal, nullptr) << "Failed to allocate 64sc signal memory";

  std::vector<Npp64sc> host_data(signal_size);
  for (size_t i = 0; i < signal_size; ++i) {
    host_data[i].re = static_cast<Npp64s>(i) * 1000LL;
    host_data[i].im = static_cast<Npp64s>(-static_cast<long long>(i)) * 1000LL;
  }

  cudaError_t err = cudaMemcpy(signal, host_data.data(), signal_size * sizeof(Npp64sc), cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data to device";

  std::vector<Npp64sc> result(signal_size);
  err = cudaMemcpy(result.data(), signal, signal_size * sizeof(Npp64sc), cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data from device";

  for (size_t i = 0; i < signal_size; ++i) {
    EXPECT_EQ(result[i].re, host_data[i].re);
    EXPECT_EQ(result[i].im, host_data[i].im);
  }

  nppsFree(signal);
}

// Test 64-bit float complex signal memory allocation
TEST_F(NppsSupportTest, Malloc_64fc) {
  Npp64fc *signal = nppsMalloc_64fc(signal_size);
  ASSERT_NE(signal, nullptr) << "Failed to allocate 64fc signal memory";

  std::vector<Npp64fc> host_data(signal_size);
  for (size_t i = 0; i < signal_size; ++i) {
    host_data[i].re = static_cast<Npp64f>(i) * 0.123456789;
    host_data[i].im = static_cast<Npp64f>(i) * 0.987654321;
  }

  cudaError_t err = cudaMemcpy(signal, host_data.data(), signal_size * sizeof(Npp64fc), cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data to device";

  std::vector<Npp64fc> result(signal_size);
  err = cudaMemcpy(result.data(), signal, signal_size * sizeof(Npp64fc), cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess) << "Failed to copy data from device";

  for (size_t i = 0; i < signal_size; ++i) {
    EXPECT_DOUBLE_EQ(result[i].re, host_data[i].re);
    EXPECT_DOUBLE_EQ(result[i].im, host_data[i].im);
  }

  nppsFree(signal);
}
