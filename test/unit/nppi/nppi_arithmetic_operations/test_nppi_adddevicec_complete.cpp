#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <npp.h>
#include <vector>

// Helper unions for 16f conversion
union Fp16Union {
  Npp16f u16;
  __half h;
};

// Helper to convert Npp16f (unsigned short) to float
static inline float npp16f_to_float(Npp16f val) {
  Fp16Union u;
  u.u16 = val;
  return __half2float(u.h);
}

// Helper to convert float to Npp16f
static inline Npp16f float_to_npp16f(float val) {
  Fp16Union u;
  u.h = __float2half(val);
  return u.u16;
}

// Test parameter structure for AddDeviceC operations
struct AddDeviceCParam {
  std::string name;
  int dataType; // 0=8u, 1=16u, 2=16s, 3=32s, 4=16f, 5=32f
  int channels; // 1, 3, 4, or -4 for AC4
  bool inplace;
  bool hasScale; // Integer types have scale factor
};

class AddDeviceCTest : public ::testing::TestWithParam<AddDeviceCParam> {
protected:
  void SetUp() override { cudaSetDevice(0); }

  void TearDown() override { cudaDeviceSynchronize(); }
};

// Helper to get actual channel count
static int getChannelCount(int channels) { return (channels == -4) ? 4 : channels; }

// ============================================================================
// 8u Tests
// ============================================================================

class AddDeviceC_8u_Test : public ::testing::TestWithParam<AddDeviceCParam> {
protected:
  void SetUp() override { cudaSetDevice(0); }
  void TearDown() override { cudaDeviceSynchronize(); }
};

TEST_P(AddDeviceC_8u_Test, Compute) {
  const auto &p = GetParam();
  const int width = 4;
  const int height = 3;
  const int channelCount = getChannelCount(p.channels);
  const int totalElements = width * height * channelCount;

  // Prepare source data
  std::vector<Npp8u> hostSrc(totalElements);
  for (int i = 0; i < totalElements; ++i) {
    hostSrc[i] = static_cast<Npp8u>(10 + (i % 50));
  }

  // Prepare constants (one per channel for multi-channel, one for C1)
  std::vector<Npp8u> hostConstants(channelCount);
  for (int c = 0; c < channelCount; ++c) {
    hostConstants[c] = static_cast<Npp8u>(5 + c * 3);
  }

  // Allocate device memory
  int step;
  Npp8u *d_src = nullptr;
  Npp8u *d_dst = nullptr;
  Npp8u *d_constants = nullptr;

  if (channelCount == 1) {
    d_src = nppiMalloc_8u_C1(width, height, &step);
    if (!p.inplace)
      d_dst = nppiMalloc_8u_C1(width, height, &step);
  } else if (channelCount == 3) {
    d_src = nppiMalloc_8u_C3(width, height, &step);
    if (!p.inplace)
      d_dst = nppiMalloc_8u_C3(width, height, &step);
  } else {
    d_src = nppiMalloc_8u_C4(width, height, &step);
    if (!p.inplace)
      d_dst = nppiMalloc_8u_C4(width, height, &step);
  }

  cudaMalloc(&d_constants, channelCount * sizeof(Npp8u));

  ASSERT_NE(d_src, nullptr);
  if (!p.inplace) {
    ASSERT_NE(d_dst, nullptr);
  }
  ASSERT_NE(d_constants, nullptr);

  // Copy data to device
  int hostStep = width * channelCount * sizeof(Npp8u);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy(d_constants, hostConstants.data(), channelCount * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // For AC4 non-inplace, copy src to dst first to preserve alpha channel
  if (p.channels == -4 && !p.inplace) {
    cudaMemcpy2D(d_dst, step, d_src, step, width * channelCount * sizeof(Npp8u), height, cudaMemcpyDeviceToDevice);
  }

  // Execute operation
  NppiSize oSizeROI = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = NPP_ERROR;
  int scaleFactor = 0;

  if (p.inplace) {
    if (channelCount == 1) {
      status = nppiAddDeviceC_8u_C1IRSfs_Ctx(d_constants, d_src, step, oSizeROI, scaleFactor, ctx);
    } else if (channelCount == 3) {
      status = nppiAddDeviceC_8u_C3IRSfs_Ctx(d_constants, d_src, step, oSizeROI, scaleFactor, ctx);
    } else if (p.channels == -4) {
      status = nppiAddDeviceC_8u_AC4IRSfs_Ctx(d_constants, d_src, step, oSizeROI, scaleFactor, ctx);
    } else {
      status = nppiAddDeviceC_8u_C4IRSfs_Ctx(d_constants, d_src, step, oSizeROI, scaleFactor, ctx);
    }
  } else {
    if (channelCount == 1) {
      status = nppiAddDeviceC_8u_C1RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, ctx);
    } else if (channelCount == 3) {
      status = nppiAddDeviceC_8u_C3RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, ctx);
    } else if (p.channels == -4) {
      status = nppiAddDeviceC_8u_AC4RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, ctx);
    } else {
      status = nppiAddDeviceC_8u_C4RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, ctx);
    }
  }
  ASSERT_EQ(status, NPP_SUCCESS);

  // Copy result back
  std::vector<Npp8u> hostResult(totalElements);
  Npp8u *resultPtr = p.inplace ? d_src : d_dst;
  cudaMemcpy2D(hostResult.data(), hostStep, resultPtr, step, hostStep, height, cudaMemcpyDeviceToHost);

  // Verify results
  for (int i = 0; i < totalElements; ++i) {
    int ch = i % channelCount;
    // For AC4, channel 3 (alpha) is not modified
    if (p.channels == -4 && ch == 3) {
      EXPECT_EQ(hostResult[i], hostSrc[i]) << "Alpha channel should be unchanged at index " << i;
    } else {
      int expected = std::min(255, static_cast<int>(hostSrc[i]) + static_cast<int>(hostConstants[ch]));
      EXPECT_EQ(hostResult[i], static_cast<Npp8u>(expected)) << "Mismatch at index " << i;
    }
  }

  // Cleanup
  nppiFree(d_src);
  if (!p.inplace) {
    nppiFree(d_dst);
  }
  cudaFree(d_constants);
}

INSTANTIATE_TEST_SUITE_P(
    AddDeviceC_8u, AddDeviceC_8u_Test,
    ::testing::Values(AddDeviceCParam{"8u_C1RSfs", 0, 1, false, true}, AddDeviceCParam{"8u_C1IRSfs", 0, 1, true, true},
                      AddDeviceCParam{"8u_C3RSfs", 0, 3, false, true}, AddDeviceCParam{"8u_C3IRSfs", 0, 3, true, true},
                      AddDeviceCParam{"8u_AC4RSfs", 0, -4, false, true},
                      AddDeviceCParam{"8u_AC4IRSfs", 0, -4, true, true},
                      AddDeviceCParam{"8u_C4RSfs", 0, 4, false, true}, AddDeviceCParam{"8u_C4IRSfs", 0, 4, true, true}),
    [](const ::testing::TestParamInfo<AddDeviceCParam> &info) { return info.param.name; });

// ============================================================================
// 16u Tests
// ============================================================================

class AddDeviceC_16u_Test : public ::testing::TestWithParam<AddDeviceCParam> {
protected:
  void SetUp() override { cudaSetDevice(0); }
  void TearDown() override { cudaDeviceSynchronize(); }
};

TEST_P(AddDeviceC_16u_Test, Compute) {
  const auto &p = GetParam();
  const int width = 4;
  const int height = 3;
  const int channelCount = getChannelCount(p.channels);
  const int totalElements = width * height * channelCount;

  std::vector<Npp16u> hostSrc(totalElements);
  for (int i = 0; i < totalElements; ++i) {
    hostSrc[i] = static_cast<Npp16u>(100 + (i % 500));
  }

  std::vector<Npp16u> hostConstants(channelCount);
  for (int c = 0; c < channelCount; ++c) {
    hostConstants[c] = static_cast<Npp16u>(50 + c * 30);
  }

  int step;
  Npp16u *d_src = nullptr;
  Npp16u *d_dst = nullptr;
  Npp16u *d_constants = nullptr;

  if (channelCount == 1) {
    d_src = nppiMalloc_16u_C1(width, height, &step);
    if (!p.inplace)
      d_dst = nppiMalloc_16u_C1(width, height, &step);
  } else if (channelCount == 3) {
    d_src = nppiMalloc_16u_C3(width, height, &step);
    if (!p.inplace)
      d_dst = nppiMalloc_16u_C3(width, height, &step);
  } else {
    d_src = nppiMalloc_16u_C4(width, height, &step);
    if (!p.inplace)
      d_dst = nppiMalloc_16u_C4(width, height, &step);
  }

  cudaMalloc(&d_constants, channelCount * sizeof(Npp16u));

  ASSERT_NE(d_src, nullptr);
  if (!p.inplace) {
    ASSERT_NE(d_dst, nullptr);
  }
  ASSERT_NE(d_constants, nullptr);

  int hostStep = width * channelCount * sizeof(Npp16u);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy(d_constants, hostConstants.data(), channelCount * sizeof(Npp16u), cudaMemcpyHostToDevice);

  // For AC4 non-inplace, copy src to dst first to preserve alpha channel
  if (p.channels == -4 && !p.inplace) {
    cudaMemcpy2D(d_dst, step, d_src, step, width * channelCount * sizeof(Npp16u), height, cudaMemcpyDeviceToDevice);
  }

  NppiSize oSizeROI = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = NPP_ERROR;
  int scaleFactor = 0;

  if (p.inplace) {
    if (channelCount == 1) {
      status = nppiAddDeviceC_16u_C1IRSfs_Ctx(d_constants, d_src, step, oSizeROI, scaleFactor, ctx);
    } else if (channelCount == 3) {
      status = nppiAddDeviceC_16u_C3IRSfs_Ctx(d_constants, d_src, step, oSizeROI, scaleFactor, ctx);
    } else if (p.channels == -4) {
      status = nppiAddDeviceC_16u_AC4IRSfs_Ctx(d_constants, d_src, step, oSizeROI, scaleFactor, ctx);
    } else {
      status = nppiAddDeviceC_16u_C4IRSfs_Ctx(d_constants, d_src, step, oSizeROI, scaleFactor, ctx);
    }
  } else {
    if (channelCount == 1) {
      status = nppiAddDeviceC_16u_C1RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, ctx);
    } else if (channelCount == 3) {
      status = nppiAddDeviceC_16u_C3RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, ctx);
    } else if (p.channels == -4) {
      status = nppiAddDeviceC_16u_AC4RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, ctx);
    } else {
      status = nppiAddDeviceC_16u_C4RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, ctx);
    }
  }
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> hostResult(totalElements);
  Npp16u *resultPtr = p.inplace ? d_src : d_dst;
  cudaMemcpy2D(hostResult.data(), hostStep, resultPtr, step, hostStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalElements; ++i) {
    int ch = i % channelCount;
    if (p.channels == -4 && ch == 3) {
      EXPECT_EQ(hostResult[i], hostSrc[i]) << "Alpha channel should be unchanged at index " << i;
    } else {
      int expected = std::min(65535, static_cast<int>(hostSrc[i]) + static_cast<int>(hostConstants[ch]));
      EXPECT_EQ(hostResult[i], static_cast<Npp16u>(expected)) << "Mismatch at index " << i;
    }
  }

  nppiFree(d_src);
  if (!p.inplace) {
    nppiFree(d_dst);
  }
  cudaFree(d_constants);
}

INSTANTIATE_TEST_SUITE_P(AddDeviceC_16u, AddDeviceC_16u_Test,
                         ::testing::Values(AddDeviceCParam{"16u_C1RSfs", 1, 1, false, true},
                                           AddDeviceCParam{"16u_C1IRSfs", 1, 1, true, true},
                                           AddDeviceCParam{"16u_C3RSfs", 1, 3, false, true},
                                           AddDeviceCParam{"16u_C3IRSfs", 1, 3, true, true},
                                           AddDeviceCParam{"16u_AC4RSfs", 1, -4, false, true},
                                           AddDeviceCParam{"16u_AC4IRSfs", 1, -4, true, true},
                                           AddDeviceCParam{"16u_C4RSfs", 1, 4, false, true},
                                           AddDeviceCParam{"16u_C4IRSfs", 1, 4, true, true}),
                         [](const ::testing::TestParamInfo<AddDeviceCParam> &info) { return info.param.name; });

// ============================================================================
// 16s Tests
// ============================================================================

class AddDeviceC_16s_Test : public ::testing::TestWithParam<AddDeviceCParam> {
protected:
  void SetUp() override { cudaSetDevice(0); }
  void TearDown() override { cudaDeviceSynchronize(); }
};

TEST_P(AddDeviceC_16s_Test, Compute) {
  const auto &p = GetParam();
  const int width = 4;
  const int height = 3;
  const int channelCount = getChannelCount(p.channels);
  const int totalElements = width * height * channelCount;

  std::vector<Npp16s> hostSrc(totalElements);
  for (int i = 0; i < totalElements; ++i) {
    hostSrc[i] = static_cast<Npp16s>(-100 + (i % 200));
  }

  std::vector<Npp16s> hostConstants(channelCount);
  for (int c = 0; c < channelCount; ++c) {
    hostConstants[c] = static_cast<Npp16s>(25 + c * 15);
  }

  int step;
  Npp16s *d_src = nullptr;
  Npp16s *d_dst = nullptr;
  Npp16s *d_constants = nullptr;

  // Only C1 is supported for 16s (nppiMalloc_16s_C3/C4 don't exist)
  d_src = nppiMalloc_16s_C1(width, height, &step);
  if (!p.inplace) {
    d_dst = nppiMalloc_16s_C1(width, height, &step);
  }

  cudaMalloc(&d_constants, channelCount * sizeof(Npp16s));

  ASSERT_NE(d_src, nullptr);
  if (!p.inplace) {
    ASSERT_NE(d_dst, nullptr);
  }
  ASSERT_NE(d_constants, nullptr);

  int hostStep = width * channelCount * sizeof(Npp16s);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy(d_constants, hostConstants.data(), channelCount * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = NPP_ERROR;
  int scaleFactor = 0;

  if (p.inplace) {
    status = nppiAddDeviceC_16s_C1IRSfs_Ctx(d_constants, d_src, step, oSizeROI, scaleFactor, ctx);
  } else {
    status = nppiAddDeviceC_16s_C1RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, ctx);
  }
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16s> hostResult(totalElements);
  Npp16s *resultPtr = p.inplace ? d_src : d_dst;
  cudaMemcpy2D(hostResult.data(), hostStep, resultPtr, step, hostStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalElements; ++i) {
    int expected = std::max(-32768, std::min(32767, static_cast<int>(hostSrc[i]) + static_cast<int>(hostConstants[0])));
    EXPECT_EQ(hostResult[i], static_cast<Npp16s>(expected)) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  if (!p.inplace) {
    nppiFree(d_dst);
  }
  cudaFree(d_constants);
}

INSTANTIATE_TEST_SUITE_P(AddDeviceC_16s, AddDeviceC_16s_Test,
                         ::testing::Values(AddDeviceCParam{"16s_C1RSfs", 2, 1, false, true},
                                           AddDeviceCParam{"16s_C1IRSfs", 2, 1, true, true}),
                         [](const ::testing::TestParamInfo<AddDeviceCParam> &info) { return info.param.name; });

// ============================================================================
// 32s Tests
// ============================================================================

class AddDeviceC_32s_Test : public ::testing::TestWithParam<AddDeviceCParam> {
protected:
  void SetUp() override { cudaSetDevice(0); }
  void TearDown() override { cudaDeviceSynchronize(); }
};

TEST_P(AddDeviceC_32s_Test, Compute) {
  const auto &p = GetParam();
  const int width = 4;
  const int height = 3;
  const int channelCount = getChannelCount(p.channels);
  const int totalElements = width * height * channelCount;

  std::vector<Npp32s> hostSrc(totalElements);
  for (int i = 0; i < totalElements; ++i) {
    hostSrc[i] = static_cast<Npp32s>(-1000 + (i % 2000));
  }

  std::vector<Npp32s> hostConstants(channelCount);
  for (int c = 0; c < channelCount; ++c) {
    hostConstants[c] = static_cast<Npp32s>(250 + c * 150);
  }

  int step;
  Npp32s *d_src = nullptr;
  Npp32s *d_dst = nullptr;
  Npp32s *d_constants = nullptr;

  if (channelCount == 1) {
    d_src = nppiMalloc_32s_C1(width, height, &step);
    if (!p.inplace)
      d_dst = nppiMalloc_32s_C1(width, height, &step);
  } else if (channelCount == 3) {
    d_src = nppiMalloc_32s_C3(width, height, &step);
    if (!p.inplace)
      d_dst = nppiMalloc_32s_C3(width, height, &step);
  } else {
    d_src = nppiMalloc_32s_C4(width, height, &step);
    if (!p.inplace)
      d_dst = nppiMalloc_32s_C4(width, height, &step);
  }

  cudaMalloc(&d_constants, channelCount * sizeof(Npp32s));

  ASSERT_NE(d_src, nullptr);
  if (!p.inplace) {
    ASSERT_NE(d_dst, nullptr);
  }
  ASSERT_NE(d_constants, nullptr);

  int hostStep = width * channelCount * sizeof(Npp32s);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy(d_constants, hostConstants.data(), channelCount * sizeof(Npp32s), cudaMemcpyHostToDevice);

  NppiSize oSizeROI = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = NPP_ERROR;
  int scaleFactor = 0;

  if (p.inplace) {
    if (channelCount == 1) {
      status = nppiAddDeviceC_32s_C1IRSfs_Ctx(d_constants, d_src, step, oSizeROI, scaleFactor, ctx);
    } else {
      status = nppiAddDeviceC_32s_C3IRSfs_Ctx(d_constants, d_src, step, oSizeROI, scaleFactor, ctx);
    }
  } else {
    if (channelCount == 1) {
      status = nppiAddDeviceC_32s_C1RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, ctx);
    } else {
      status = nppiAddDeviceC_32s_C3RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, ctx);
    }
  }
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32s> hostResult(totalElements);
  Npp32s *resultPtr = p.inplace ? d_src : d_dst;
  cudaMemcpy2D(hostResult.data(), hostStep, resultPtr, step, hostStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalElements; ++i) {
    int ch = i % channelCount;
    Npp32s expected = hostSrc[i] + hostConstants[ch];
    EXPECT_EQ(hostResult[i], expected) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  if (!p.inplace) {
    nppiFree(d_dst);
  }
  cudaFree(d_constants);
}

INSTANTIATE_TEST_SUITE_P(AddDeviceC_32s, AddDeviceC_32s_Test,
                         ::testing::Values(AddDeviceCParam{"32s_C1RSfs", 3, 1, false, true},
                                           AddDeviceCParam{"32s_C1IRSfs", 3, 1, true, true},
                                           AddDeviceCParam{"32s_C3RSfs", 3, 3, false, true},
                                           AddDeviceCParam{"32s_C3IRSfs", 3, 3, true, true}),
                         [](const ::testing::TestParamInfo<AddDeviceCParam> &info) { return info.param.name; });

// ============================================================================
// 32f Tests
// ============================================================================

class AddDeviceC_32f_Test : public ::testing::TestWithParam<AddDeviceCParam> {
protected:
  void SetUp() override { cudaSetDevice(0); }
  void TearDown() override { cudaDeviceSynchronize(); }
};

TEST_P(AddDeviceC_32f_Test, Compute) {
  const auto &p = GetParam();
  const int width = 4;
  const int height = 3;
  const int channelCount = getChannelCount(p.channels);
  const int totalElements = width * height * channelCount;

  std::vector<Npp32f> hostSrc(totalElements);
  for (int i = 0; i < totalElements; ++i) {
    hostSrc[i] = 1.0f + (i % 10) * 0.5f;
  }

  std::vector<Npp32f> hostConstants(channelCount);
  for (int c = 0; c < channelCount; ++c) {
    hostConstants[c] = 0.25f + c * 0.1f;
  }

  int step;
  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  Npp32f *d_constants = nullptr;

  if (channelCount == 1) {
    d_src = nppiMalloc_32f_C1(width, height, &step);
    if (!p.inplace)
      d_dst = nppiMalloc_32f_C1(width, height, &step);
  } else if (channelCount == 3) {
    d_src = nppiMalloc_32f_C3(width, height, &step);
    if (!p.inplace)
      d_dst = nppiMalloc_32f_C3(width, height, &step);
  } else {
    d_src = nppiMalloc_32f_C4(width, height, &step);
    if (!p.inplace)
      d_dst = nppiMalloc_32f_C4(width, height, &step);
  }

  cudaMalloc(&d_constants, channelCount * sizeof(Npp32f));

  ASSERT_NE(d_src, nullptr);
  if (!p.inplace) {
    ASSERT_NE(d_dst, nullptr);
  }
  ASSERT_NE(d_constants, nullptr);

  int hostStep = width * channelCount * sizeof(Npp32f);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy(d_constants, hostConstants.data(), channelCount * sizeof(Npp32f), cudaMemcpyHostToDevice);

  // For AC4 non-inplace, copy src to dst first to preserve alpha channel
  if (p.channels == -4 && !p.inplace) {
    cudaMemcpy2D(d_dst, step, d_src, step, width * channelCount * sizeof(Npp32f), height, cudaMemcpyDeviceToDevice);
  }

  NppiSize oSizeROI = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = NPP_ERROR;

  if (p.inplace) {
    if (channelCount == 1) {
      status = nppiAddDeviceC_32f_C1IR_Ctx(d_constants, d_src, step, oSizeROI, ctx);
    } else if (channelCount == 3) {
      status = nppiAddDeviceC_32f_C3IR_Ctx(d_constants, d_src, step, oSizeROI, ctx);
    } else if (p.channels == -4) {
      status = nppiAddDeviceC_32f_AC4IR_Ctx(d_constants, d_src, step, oSizeROI, ctx);
    } else {
      status = nppiAddDeviceC_32f_C4IR_Ctx(d_constants, d_src, step, oSizeROI, ctx);
    }
  } else {
    if (channelCount == 1) {
      status = nppiAddDeviceC_32f_C1R_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, ctx);
    } else if (channelCount == 3) {
      status = nppiAddDeviceC_32f_C3R_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, ctx);
    } else if (p.channels == -4) {
      status = nppiAddDeviceC_32f_AC4R_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, ctx);
    } else {
      status = nppiAddDeviceC_32f_C4R_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, ctx);
    }
  }
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> hostResult(totalElements);
  Npp32f *resultPtr = p.inplace ? d_src : d_dst;
  cudaMemcpy2D(hostResult.data(), hostStep, resultPtr, step, hostStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalElements; ++i) {
    int ch = i % channelCount;
    if (p.channels == -4 && ch == 3) {
      EXPECT_NEAR(hostResult[i], hostSrc[i], 1e-5f) << "Alpha channel should be unchanged at index " << i;
    } else {
      float expected = hostSrc[i] + hostConstants[ch];
      EXPECT_NEAR(hostResult[i], expected, 1e-5f) << "Mismatch at index " << i;
    }
  }

  nppiFree(d_src);
  if (!p.inplace) {
    nppiFree(d_dst);
  }
  cudaFree(d_constants);
}

INSTANTIATE_TEST_SUITE_P(
    AddDeviceC_32f, AddDeviceC_32f_Test,
    ::testing::Values(AddDeviceCParam{"32f_C1R", 5, 1, false, false}, AddDeviceCParam{"32f_C1IR", 5, 1, true, false},
                      AddDeviceCParam{"32f_C3R", 5, 3, false, false}, AddDeviceCParam{"32f_C3IR", 5, 3, true, false},
                      AddDeviceCParam{"32f_AC4R", 5, -4, false, false},
                      AddDeviceCParam{"32f_AC4IR", 5, -4, true, false}, AddDeviceCParam{"32f_C4R", 5, 4, false, false},
                      AddDeviceCParam{"32f_C4IR", 5, 4, true, false}),
    [](const ::testing::TestParamInfo<AddDeviceCParam> &info) { return info.param.name; });
