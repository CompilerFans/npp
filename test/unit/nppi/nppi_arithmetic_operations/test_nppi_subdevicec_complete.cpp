#include <gtest/gtest.h>
#include <npp.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

struct SubDeviceCParam {
  std::string name;
  int dataType;
  int channels;
  bool inplace;
  bool hasScale;
};

static int getChannelCount(int channels) {
  return (channels == -4) ? 4 : channels;
}

// ============================================================================
// 8u Tests
// ============================================================================

class SubDeviceC_8u_Test : public ::testing::TestWithParam<SubDeviceCParam> {
protected:
  void SetUp() override { cudaSetDevice(0); }
  void TearDown() override { cudaDeviceSynchronize(); }
};

TEST_P(SubDeviceC_8u_Test, Compute) {
  const auto& p = GetParam();
  const int width = 4;
  const int height = 3;
  const int channelCount = getChannelCount(p.channels);
  const int totalElements = width * height * channelCount;

  std::vector<Npp8u> hostSrc(totalElements);
  for (int i = 0; i < totalElements; ++i) {
    hostSrc[i] = static_cast<Npp8u>(50 + (i % 100));
  }

  std::vector<Npp8u> hostConstants(channelCount);
  for (int c = 0; c < channelCount; ++c) {
    hostConstants[c] = static_cast<Npp8u>(5 + c * 3);
  }

  int step;
  Npp8u* d_src = nullptr;
  Npp8u* d_dst = nullptr;
  Npp8u* d_constants = nullptr;

  if (channelCount == 1) {
    d_src = nppiMalloc_8u_C1(width, height, &step);
    if (!p.inplace) d_dst = nppiMalloc_8u_C1(width, height, &step);
  } else if (channelCount == 3) {
    d_src = nppiMalloc_8u_C3(width, height, &step);
    if (!p.inplace) d_dst = nppiMalloc_8u_C3(width, height, &step);
  } else {
    d_src = nppiMalloc_8u_C4(width, height, &step);
    if (!p.inplace) d_dst = nppiMalloc_8u_C4(width, height, &step);
  }

  cudaMalloc(&d_constants, channelCount * sizeof(Npp8u));

  ASSERT_NE(d_src, nullptr);
  if (!p.inplace) {
    ASSERT_NE(d_dst, nullptr);
  }
  ASSERT_NE(d_constants, nullptr);

  int hostStep = width * channelCount * sizeof(Npp8u);
  cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy(d_constants, hostConstants.data(), channelCount * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // For AC4 non-inplace, copy src to dst first to preserve alpha channel
  if (p.channels == -4 && !p.inplace) {
    cudaMemcpy2D(d_dst, step, d_src, step, width * channelCount * sizeof(Npp8u), height, cudaMemcpyDeviceToDevice);
  }

  NppiSize oSizeROI = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);
  NppStatus status = NPP_ERROR;
  int scaleFactor = 0;

  if (p.inplace) {
    if (channelCount == 1) {
      status = nppiSubDeviceC_8u_C1IRSfs_Ctx(d_constants, d_src, step, oSizeROI, scaleFactor, ctx);
    } else if (channelCount == 3) {
      status = nppiSubDeviceC_8u_C3IRSfs_Ctx(d_constants, d_src, step, oSizeROI, scaleFactor, ctx);
    } else if (p.channels == -4) {
      status = nppiSubDeviceC_8u_AC4IRSfs_Ctx(d_constants, d_src, step, oSizeROI, scaleFactor, ctx);
    } else {
      status = nppiSubDeviceC_8u_C4IRSfs_Ctx(d_constants, d_src, step, oSizeROI, scaleFactor, ctx);
    }
  } else {
    if (channelCount == 1) {
      status = nppiSubDeviceC_8u_C1RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, ctx);
    } else if (channelCount == 3) {
      status = nppiSubDeviceC_8u_C3RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, ctx);
    } else if (p.channels == -4) {
      status = nppiSubDeviceC_8u_AC4RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, ctx);
    } else {
      status = nppiSubDeviceC_8u_C4RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, ctx);
    }
  }
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> hostResult(totalElements);
  Npp8u* resultPtr = p.inplace ? d_src : d_dst;
  cudaMemcpy2D(hostResult.data(), hostStep, resultPtr, step, hostStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalElements; ++i) {
    int ch = i % channelCount;
    if (p.channels == -4 && ch == 3) {
      EXPECT_EQ(hostResult[i], hostSrc[i]) << "Alpha channel should be unchanged at index " << i;
    } else {
      int expected = std::max(0, static_cast<int>(hostSrc[i]) - static_cast<int>(hostConstants[ch]));
      EXPECT_EQ(hostResult[i], static_cast<Npp8u>(expected)) << "Mismatch at index " << i;
    }
  }

  nppiFree(d_src);
  if (!p.inplace) {
    nppiFree(d_dst);
  }
  cudaFree(d_constants);
}

INSTANTIATE_TEST_SUITE_P(
    SubDeviceC_8u,
    SubDeviceC_8u_Test,
    ::testing::Values(
        SubDeviceCParam{"8u_C1RSfs", 0, 1, false, true},
        SubDeviceCParam{"8u_C1IRSfs", 0, 1, true, true},
        SubDeviceCParam{"8u_C3RSfs", 0, 3, false, true},
        SubDeviceCParam{"8u_C3IRSfs", 0, 3, true, true},
        SubDeviceCParam{"8u_AC4RSfs", 0, -4, false, true},
        SubDeviceCParam{"8u_AC4IRSfs", 0, -4, true, true},
        SubDeviceCParam{"8u_C4RSfs", 0, 4, false, true},
        SubDeviceCParam{"8u_C4IRSfs", 0, 4, true, true}
    ),
    [](const ::testing::TestParamInfo<SubDeviceCParam>& info) {
      return info.param.name;
    }
);

// ============================================================================
// 16u Tests
// ============================================================================

class SubDeviceC_16u_Test : public ::testing::TestWithParam<SubDeviceCParam> {
protected:
  void SetUp() override { cudaSetDevice(0); }
  void TearDown() override { cudaDeviceSynchronize(); }
};

TEST_P(SubDeviceC_16u_Test, Compute) {
  const auto& p = GetParam();
  const int width = 4;
  const int height = 3;
  const int channelCount = getChannelCount(p.channels);
  const int totalElements = width * height * channelCount;

  std::vector<Npp16u> hostSrc(totalElements);
  for (int i = 0; i < totalElements; ++i) {
    hostSrc[i] = static_cast<Npp16u>(500 + (i % 1000));
  }

  std::vector<Npp16u> hostConstants(channelCount);
  for (int c = 0; c < channelCount; ++c) {
    hostConstants[c] = static_cast<Npp16u>(50 + c * 30);
  }

  int step;
  Npp16u* d_src = nullptr;
  Npp16u* d_dst = nullptr;
  Npp16u* d_constants = nullptr;

  if (channelCount == 1) {
    d_src = nppiMalloc_16u_C1(width, height, &step);
    if (!p.inplace) d_dst = nppiMalloc_16u_C1(width, height, &step);
  } else if (channelCount == 3) {
    d_src = nppiMalloc_16u_C3(width, height, &step);
    if (!p.inplace) d_dst = nppiMalloc_16u_C3(width, height, &step);
  } else {
    d_src = nppiMalloc_16u_C4(width, height, &step);
    if (!p.inplace) d_dst = nppiMalloc_16u_C4(width, height, &step);
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
      status = nppiSubDeviceC_16u_C1IRSfs_Ctx(d_constants, d_src, step, oSizeROI, scaleFactor, ctx);
    } else if (channelCount == 3) {
      status = nppiSubDeviceC_16u_C3IRSfs_Ctx(d_constants, d_src, step, oSizeROI, scaleFactor, ctx);
    } else if (p.channels == -4) {
      status = nppiSubDeviceC_16u_AC4IRSfs_Ctx(d_constants, d_src, step, oSizeROI, scaleFactor, ctx);
    } else {
      status = nppiSubDeviceC_16u_C4IRSfs_Ctx(d_constants, d_src, step, oSizeROI, scaleFactor, ctx);
    }
  } else {
    if (channelCount == 1) {
      status = nppiSubDeviceC_16u_C1RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, ctx);
    } else if (channelCount == 3) {
      status = nppiSubDeviceC_16u_C3RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, ctx);
    } else if (p.channels == -4) {
      status = nppiSubDeviceC_16u_AC4RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, ctx);
    } else {
      status = nppiSubDeviceC_16u_C4RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, ctx);
    }
  }
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> hostResult(totalElements);
  Npp16u* resultPtr = p.inplace ? d_src : d_dst;
  cudaMemcpy2D(hostResult.data(), hostStep, resultPtr, step, hostStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalElements; ++i) {
    int ch = i % channelCount;
    if (p.channels == -4 && ch == 3) {
      EXPECT_EQ(hostResult[i], hostSrc[i]) << "Alpha channel should be unchanged at index " << i;
    } else {
      int expected = std::max(0, static_cast<int>(hostSrc[i]) - static_cast<int>(hostConstants[ch]));
      EXPECT_EQ(hostResult[i], static_cast<Npp16u>(expected)) << "Mismatch at index " << i;
    }
  }

  nppiFree(d_src);
  if (!p.inplace) {
    nppiFree(d_dst);
  }
  cudaFree(d_constants);
}

INSTANTIATE_TEST_SUITE_P(
    SubDeviceC_16u,
    SubDeviceC_16u_Test,
    ::testing::Values(
        SubDeviceCParam{"16u_C1RSfs", 1, 1, false, true},
        SubDeviceCParam{"16u_C1IRSfs", 1, 1, true, true},
        SubDeviceCParam{"16u_C3RSfs", 1, 3, false, true},
        SubDeviceCParam{"16u_C3IRSfs", 1, 3, true, true},
        SubDeviceCParam{"16u_AC4RSfs", 1, -4, false, true},
        SubDeviceCParam{"16u_AC4IRSfs", 1, -4, true, true},
        SubDeviceCParam{"16u_C4RSfs", 1, 4, false, true},
        SubDeviceCParam{"16u_C4IRSfs", 1, 4, true, true}
    ),
    [](const ::testing::TestParamInfo<SubDeviceCParam>& info) {
      return info.param.name;
    }
);

// ============================================================================
// 16s Tests
// ============================================================================

class SubDeviceC_16s_Test : public ::testing::TestWithParam<SubDeviceCParam> {
protected:
  void SetUp() override { cudaSetDevice(0); }
  void TearDown() override { cudaDeviceSynchronize(); }
};

TEST_P(SubDeviceC_16s_Test, Compute) {
  const auto& p = GetParam();
  const int width = 4;
  const int height = 3;
  const int channelCount = getChannelCount(p.channels);
  const int totalElements = width * height * channelCount;

  std::vector<Npp16s> hostSrc(totalElements);
  for (int i = 0; i < totalElements; ++i) {
    hostSrc[i] = static_cast<Npp16s>(100 + (i % 200));
  }

  std::vector<Npp16s> hostConstants(channelCount);
  for (int c = 0; c < channelCount; ++c) {
    hostConstants[c] = static_cast<Npp16s>(25 + c * 15);
  }

  int step;
  Npp16s* d_src = nullptr;
  Npp16s* d_dst = nullptr;
  Npp16s* d_constants = nullptr;

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
    status = nppiSubDeviceC_16s_C1IRSfs_Ctx(d_constants, d_src, step, oSizeROI, scaleFactor, ctx);
  } else {
    status = nppiSubDeviceC_16s_C1RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, ctx);
  }
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16s> hostResult(totalElements);
  Npp16s* resultPtr = p.inplace ? d_src : d_dst;
  cudaMemcpy2D(hostResult.data(), hostStep, resultPtr, step, hostStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalElements; ++i) {
    Npp16s expected = static_cast<Npp16s>(hostSrc[i] - hostConstants[0]);
    EXPECT_EQ(hostResult[i], expected) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  if (!p.inplace) {
    nppiFree(d_dst);
  }
  cudaFree(d_constants);
}

INSTANTIATE_TEST_SUITE_P(
    SubDeviceC_16s,
    SubDeviceC_16s_Test,
    ::testing::Values(
        SubDeviceCParam{"16s_C1RSfs", 2, 1, false, true},
        SubDeviceCParam{"16s_C1IRSfs", 2, 1, true, true}
    ),
    [](const ::testing::TestParamInfo<SubDeviceCParam>& info) {
      return info.param.name;
    }
);

// ============================================================================
// 32s Tests
// ============================================================================

class SubDeviceC_32s_Test : public ::testing::TestWithParam<SubDeviceCParam> {
protected:
  void SetUp() override { cudaSetDevice(0); }
  void TearDown() override { cudaDeviceSynchronize(); }
};

TEST_P(SubDeviceC_32s_Test, Compute) {
  const auto& p = GetParam();
  const int width = 4;
  const int height = 3;
  const int channelCount = getChannelCount(p.channels);
  const int totalElements = width * height * channelCount;

  std::vector<Npp32s> hostSrc(totalElements);
  for (int i = 0; i < totalElements; ++i) {
    hostSrc[i] = static_cast<Npp32s>(1000 + (i % 2000));
  }

  std::vector<Npp32s> hostConstants(channelCount);
  for (int c = 0; c < channelCount; ++c) {
    hostConstants[c] = static_cast<Npp32s>(250 + c * 150);
  }

  int step;
  Npp32s* d_src = nullptr;
  Npp32s* d_dst = nullptr;
  Npp32s* d_constants = nullptr;

  if (channelCount == 1) {
    d_src = nppiMalloc_32s_C1(width, height, &step);
    if (!p.inplace) d_dst = nppiMalloc_32s_C1(width, height, &step);
  } else if (channelCount == 3) {
    d_src = nppiMalloc_32s_C3(width, height, &step);
    if (!p.inplace) d_dst = nppiMalloc_32s_C3(width, height, &step);
  } else {
    d_src = nppiMalloc_32s_C4(width, height, &step);
    if (!p.inplace) d_dst = nppiMalloc_32s_C4(width, height, &step);
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
      status = nppiSubDeviceC_32s_C1IRSfs_Ctx(d_constants, d_src, step, oSizeROI, scaleFactor, ctx);
    } else {
      status = nppiSubDeviceC_32s_C3IRSfs_Ctx(d_constants, d_src, step, oSizeROI, scaleFactor, ctx);
    }
  } else {
    if (channelCount == 1) {
      status = nppiSubDeviceC_32s_C1RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, ctx);
    } else {
      status = nppiSubDeviceC_32s_C3RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, ctx);
    }
  }
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32s> hostResult(totalElements);
  Npp32s* resultPtr = p.inplace ? d_src : d_dst;
  cudaMemcpy2D(hostResult.data(), hostStep, resultPtr, step, hostStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalElements; ++i) {
    int ch = i % channelCount;
    Npp32s expected = hostSrc[i] - hostConstants[ch];
    EXPECT_EQ(hostResult[i], expected) << "Mismatch at index " << i;
  }

  nppiFree(d_src);
  if (!p.inplace) {
    nppiFree(d_dst);
  }
  cudaFree(d_constants);
}

INSTANTIATE_TEST_SUITE_P(
    SubDeviceC_32s,
    SubDeviceC_32s_Test,
    ::testing::Values(
        SubDeviceCParam{"32s_C1RSfs", 3, 1, false, true},
        SubDeviceCParam{"32s_C1IRSfs", 3, 1, true, true},
        SubDeviceCParam{"32s_C3RSfs", 3, 3, false, true},
        SubDeviceCParam{"32s_C3IRSfs", 3, 3, true, true}
    ),
    [](const ::testing::TestParamInfo<SubDeviceCParam>& info) {
      return info.param.name;
    }
);

// ============================================================================
// 32f Tests
// ============================================================================

class SubDeviceC_32f_Test : public ::testing::TestWithParam<SubDeviceCParam> {
protected:
  void SetUp() override { cudaSetDevice(0); }
  void TearDown() override { cudaDeviceSynchronize(); }
};

TEST_P(SubDeviceC_32f_Test, Compute) {
  const auto& p = GetParam();
  const int width = 4;
  const int height = 3;
  const int channelCount = getChannelCount(p.channels);
  const int totalElements = width * height * channelCount;

  std::vector<Npp32f> hostSrc(totalElements);
  for (int i = 0; i < totalElements; ++i) {
    hostSrc[i] = 5.0f + (i % 10) * 0.5f;
  }

  std::vector<Npp32f> hostConstants(channelCount);
  for (int c = 0; c < channelCount; ++c) {
    hostConstants[c] = 0.25f + c * 0.1f;
  }

  int step;
  Npp32f* d_src = nullptr;
  Npp32f* d_dst = nullptr;
  Npp32f* d_constants = nullptr;

  if (channelCount == 1) {
    d_src = nppiMalloc_32f_C1(width, height, &step);
    if (!p.inplace) d_dst = nppiMalloc_32f_C1(width, height, &step);
  } else if (channelCount == 3) {
    d_src = nppiMalloc_32f_C3(width, height, &step);
    if (!p.inplace) d_dst = nppiMalloc_32f_C3(width, height, &step);
  } else {
    d_src = nppiMalloc_32f_C4(width, height, &step);
    if (!p.inplace) d_dst = nppiMalloc_32f_C4(width, height, &step);
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
      status = nppiSubDeviceC_32f_C1IR_Ctx(d_constants, d_src, step, oSizeROI, ctx);
    } else if (channelCount == 3) {
      status = nppiSubDeviceC_32f_C3IR_Ctx(d_constants, d_src, step, oSizeROI, ctx);
    } else if (p.channels == -4) {
      status = nppiSubDeviceC_32f_AC4IR_Ctx(d_constants, d_src, step, oSizeROI, ctx);
    } else {
      status = nppiSubDeviceC_32f_C4IR_Ctx(d_constants, d_src, step, oSizeROI, ctx);
    }
  } else {
    if (channelCount == 1) {
      status = nppiSubDeviceC_32f_C1R_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, ctx);
    } else if (channelCount == 3) {
      status = nppiSubDeviceC_32f_C3R_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, ctx);
    } else if (p.channels == -4) {
      status = nppiSubDeviceC_32f_AC4R_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, ctx);
    } else {
      status = nppiSubDeviceC_32f_C4R_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, ctx);
    }
  }
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> hostResult(totalElements);
  Npp32f* resultPtr = p.inplace ? d_src : d_dst;
  cudaMemcpy2D(hostResult.data(), hostStep, resultPtr, step, hostStep, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalElements; ++i) {
    int ch = i % channelCount;
    if (p.channels == -4 && ch == 3) {
      EXPECT_NEAR(hostResult[i], hostSrc[i], 1e-5f) << "Alpha channel should be unchanged at index " << i;
    } else {
      float expected = hostSrc[i] - hostConstants[ch];
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
    SubDeviceC_32f,
    SubDeviceC_32f_Test,
    ::testing::Values(
        SubDeviceCParam{"32f_C1R", 5, 1, false, false},
        SubDeviceCParam{"32f_C1IR", 5, 1, true, false},
        SubDeviceCParam{"32f_C3R", 5, 3, false, false},
        SubDeviceCParam{"32f_C3IR", 5, 3, true, false},
        SubDeviceCParam{"32f_AC4R", 5, -4, false, false},
        SubDeviceCParam{"32f_AC4IR", 5, -4, true, false},
        SubDeviceCParam{"32f_C4R", 5, 4, false, false},
        SubDeviceCParam{"32f_C4IR", 5, 4, true, false}
    ),
    [](const ::testing::TestParamInfo<SubDeviceCParam>& info) {
      return info.param.name;
    }
);
