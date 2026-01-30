#include "npp.h"
#include "npp_test_base.h"
#include <algorithm>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <vector>

namespace {

struct LutCase {
  int width;
  int height;
  int seed;
};

class LutMissingTestBase : public npp_functional_test::NppTestBase {
protected:
  void SetUp() override {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      has_cuda_ = false;
      GTEST_SKIP() << "CUDA device unavailable for LUT missing tests";
    }
    has_cuda_ = true;
    npp_functional_test::NppTestBase::SetUp();
  }

  void TearDown() override {
    if (has_cuda_) {
      npp_functional_test::NppTestBase::TearDown();
    }
  }

  void fillU8(std::vector<Npp8u> &data, int seed) {
    std::mt19937 rng(static_cast<unsigned int>(seed));
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto &v : data) {
      v = static_cast<Npp8u>(dist(rng));
    }
  }

  void fillU16(std::vector<Npp16u> &data, int seed) {
    std::mt19937 rng(static_cast<unsigned int>(seed));
    std::uniform_int_distribution<int> dist(0, 65535);
    for (auto &v : data) {
      v = static_cast<Npp16u>(dist(rng));
    }
  }

  bool has_cuda_ = false;
};

class Lut8uC1RParamTest : public LutMissingTestBase, public ::testing::WithParamInterface<LutCase> {};
class Lut8uC3RParamTest : public LutMissingTestBase, public ::testing::WithParamInterface<LutCase> {};
class Lut8uC4RParamTest : public LutMissingTestBase, public ::testing::WithParamInterface<LutCase> {};
class Lut16uC1RParamTest : public LutMissingTestBase, public ::testing::WithParamInterface<LutCase> {};

#ifdef USE_NVIDIA_NPP_TESTS
template <typename T> T clampToType(Npp32s value);
template <> Npp8u clampToType<Npp8u>(Npp32s value) {
  return static_cast<Npp8u>(std::min(std::max(value, 0), 255));
}
template <> Npp16u clampToType<Npp16u>(Npp32s value) {
  return static_cast<Npp16u>(std::min(std::max(value, 0), 65535));
}

template <typename T>
T applyLutNoInterpolation(T input, const std::vector<Npp32s> &values, const std::vector<Npp32s> &levels) {
  int index = 0;
  const int inputValue = static_cast<int>(input);
  for (size_t i = 1; i < levels.size(); ++i) {
    if (inputValue >= levels[i]) {
      index = static_cast<int>(i);
    } else {
      break;
    }
  }
  return clampToType<T>(values[static_cast<size_t>(index)]);
}
#endif

template <typename T>
bool allocAndCopyToDevice(const std::vector<T> &host, T **devicePtr) {
  *devicePtr = nullptr;
  if (host.empty()) {
    return false;
  }
  cudaError_t err = cudaMalloc(reinterpret_cast<void **>(devicePtr), host.size() * sizeof(T));
  if (err != cudaSuccess) {
    return false;
  }
  err = cudaMemcpy(*devicePtr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(*devicePtr);
    *devicePtr = nullptr;
    return false;
  }
  return true;
}

} // namespace

TEST_P(Lut8uC1RParamTest, ReturnNotImplementedAndNoWrite) {
  const auto param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const NppiSize roi = {width, height};

  std::vector<Npp8u> hostSrc(width * height);
  std::vector<Npp8u> hostDst(width * height, 0xA5);
  fillU8(hostSrc, param.seed);

  int srcStep = 0;
  int dstStep = 0;
  Npp8u *d_src = nppiMalloc_8u_C1(width, height, &srcStep);
  if (!d_src) {
    GTEST_SKIP() << "nppiMalloc_8u_C1 failed";
  }
  Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
  if (!d_dst) {
    nppiFree(d_src);
    GTEST_SKIP() << "nppiMalloc_8u_C1 failed";
  }

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width, width, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), width, width, height, cudaMemcpyHostToDevice);

  std::vector<Npp32s> values = {0, 255};
  std::vector<Npp32s> levelPos = {0, 255};
  const int levels = static_cast<int>(values.size());

  Npp32s *d_values = nullptr;
  Npp32s *d_levels = nullptr;
  if (!allocAndCopyToDevice(values, &d_values) || !allocAndCopyToDevice(levelPos, &d_levels)) {
    if (d_values) {
      cudaFree(d_values);
    }
    if (d_levels) {
      cudaFree(d_levels);
    }
    nppiFree(d_src);
    nppiFree(d_dst);
    GTEST_SKIP() << "Failed to allocate LUT device memory";
  }

  NppStatus status = nppiLUT_8u_C1R(d_src, srcStep, d_dst, dstStep, roi, d_values, d_levels, levels);
#ifdef USE_NVIDIA_NPP_TESTS
  EXPECT_EQ(status, NPP_NO_ERROR);
#else
  EXPECT_EQ(status, NPP_NOT_IMPLEMENTED_ERROR);
#endif

  std::vector<Npp8u> out(width * height);
  cudaMemcpy2D(out.data(), width, d_dst, dstStep, width, height, cudaMemcpyDeviceToHost);
#ifdef USE_NVIDIA_NPP_TESTS
  std::vector<Npp8u> expected(width * height);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = applyLutNoInterpolation(hostSrc[i], values, levelPos);
  }
  EXPECT_EQ(out, expected);
#else
  EXPECT_EQ(out, hostDst);
#endif

  cudaFree(d_values);
  cudaFree(d_levels);
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_P(Lut8uC3RParamTest, ReturnNotImplementedAndNoWrite) {
  const auto param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const NppiSize roi = {width, height};

  std::vector<Npp8u> hostSrc(width * height * 3);
  std::vector<Npp8u> hostDst(width * height * 3, 0x5A);
  fillU8(hostSrc, param.seed);

  int srcStep = 0;
  int dstStep = 0;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  if (!d_src) {
    GTEST_SKIP() << "nppiMalloc_8u_C3 failed";
  }
  Npp8u *d_dst = nppiMalloc_8u_C3(width, height, &dstStep);
  if (!d_dst) {
    nppiFree(d_src);
    GTEST_SKIP() << "nppiMalloc_8u_C3 failed";
  }

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);

  int levels[3] = {2, 2, 2};
  std::vector<Npp32s> values0 = {0, 255};
  std::vector<Npp32s> values1 = {0, 255};
  std::vector<Npp32s> values2 = {0, 255};
  std::vector<Npp32s> levelPos0 = {0, 255};
  std::vector<Npp32s> levelPos1 = {0, 255};
  std::vector<Npp32s> levelPos2 = {0, 255};

  Npp32s *d_values0 = nullptr;
  Npp32s *d_values1 = nullptr;
  Npp32s *d_values2 = nullptr;
  Npp32s *d_levelPos0 = nullptr;
  Npp32s *d_levelPos1 = nullptr;
  Npp32s *d_levelPos2 = nullptr;
  if (!allocAndCopyToDevice(values0, &d_values0) || !allocAndCopyToDevice(values1, &d_values1) ||
      !allocAndCopyToDevice(values2, &d_values2) || !allocAndCopyToDevice(levelPos0, &d_levelPos0) ||
      !allocAndCopyToDevice(levelPos1, &d_levelPos1) || !allocAndCopyToDevice(levelPos2, &d_levelPos2)) {
    if (d_values0) cudaFree(d_values0);
    if (d_values1) cudaFree(d_values1);
    if (d_values2) cudaFree(d_values2);
    if (d_levelPos0) cudaFree(d_levelPos0);
    if (d_levelPos1) cudaFree(d_levelPos1);
    if (d_levelPos2) cudaFree(d_levelPos2);
    nppiFree(d_src);
    nppiFree(d_dst);
    GTEST_SKIP() << "Failed to allocate LUT device memory";
  }

  const Npp32s *values[3] = {d_values0, d_values1, d_values2};
  const Npp32s *levelPos[3] = {d_levelPos0, d_levelPos1, d_levelPos2};

  NppStatus status = nppiLUT_8u_C3R(d_src, srcStep, d_dst, dstStep, roi, values, levelPos, levels);
#ifdef USE_NVIDIA_NPP_TESTS
  EXPECT_EQ(status, NPP_NO_ERROR);
#else
  EXPECT_EQ(status, NPP_NOT_IMPLEMENTED_ERROR);
#endif

  std::vector<Npp8u> out(width * height * 3);
  cudaMemcpy2D(out.data(), width * 3, d_dst, dstStep, width * 3, height, cudaMemcpyDeviceToHost);
#ifdef USE_NVIDIA_NPP_TESTS
  std::vector<Npp8u> expected(width * height * 3);
  for (size_t i = 0; i < expected.size(); i += 3) {
    expected[i] = applyLutNoInterpolation(hostSrc[i], values0, levelPos0);
    expected[i + 1] = applyLutNoInterpolation(hostSrc[i + 1], values1, levelPos1);
    expected[i + 2] = applyLutNoInterpolation(hostSrc[i + 2], values2, levelPos2);
  }
  EXPECT_EQ(out, expected);
#else
  EXPECT_EQ(out, hostDst);
#endif

  cudaFree(d_values0);
  cudaFree(d_values1);
  cudaFree(d_values2);
  cudaFree(d_levelPos0);
  cudaFree(d_levelPos1);
  cudaFree(d_levelPos2);
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_P(Lut8uC4RParamTest, ReturnNotImplementedAndNoWrite) {
  const auto param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const NppiSize roi = {width, height};

  std::vector<Npp8u> hostSrc(width * height * 4);
  std::vector<Npp8u> hostDst(width * height * 4, 0x3C);
  fillU8(hostSrc, param.seed);

  int srcStep = 0;
  int dstStep = 0;
  Npp8u *d_src = nppiMalloc_8u_C4(width, height, &srcStep);
  if (!d_src) {
    GTEST_SKIP() << "nppiMalloc_8u_C4 failed";
  }
  Npp8u *d_dst = nppiMalloc_8u_C4(width, height, &dstStep);
  if (!d_dst) {
    nppiFree(d_src);
    GTEST_SKIP() << "nppiMalloc_8u_C4 failed";
  }

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 4, width * 4, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), width * 4, width * 4, height, cudaMemcpyHostToDevice);

  int levels[4] = {2, 2, 2, 2};
  std::vector<Npp32s> values0 = {0, 255};
  std::vector<Npp32s> values1 = {0, 255};
  std::vector<Npp32s> values2 = {0, 255};
  std::vector<Npp32s> values3 = {0, 255};
  std::vector<Npp32s> levelPos0 = {0, 255};
  std::vector<Npp32s> levelPos1 = {0, 255};
  std::vector<Npp32s> levelPos2 = {0, 255};
  std::vector<Npp32s> levelPos3 = {0, 255};

  Npp32s *d_values0 = nullptr;
  Npp32s *d_values1 = nullptr;
  Npp32s *d_values2 = nullptr;
  Npp32s *d_values3 = nullptr;
  Npp32s *d_levelPos0 = nullptr;
  Npp32s *d_levelPos1 = nullptr;
  Npp32s *d_levelPos2 = nullptr;
  Npp32s *d_levelPos3 = nullptr;
  if (!allocAndCopyToDevice(values0, &d_values0) || !allocAndCopyToDevice(values1, &d_values1) ||
      !allocAndCopyToDevice(values2, &d_values2) || !allocAndCopyToDevice(values3, &d_values3) ||
      !allocAndCopyToDevice(levelPos0, &d_levelPos0) || !allocAndCopyToDevice(levelPos1, &d_levelPos1) ||
      !allocAndCopyToDevice(levelPos2, &d_levelPos2) || !allocAndCopyToDevice(levelPos3, &d_levelPos3)) {
    if (d_values0) cudaFree(d_values0);
    if (d_values1) cudaFree(d_values1);
    if (d_values2) cudaFree(d_values2);
    if (d_values3) cudaFree(d_values3);
    if (d_levelPos0) cudaFree(d_levelPos0);
    if (d_levelPos1) cudaFree(d_levelPos1);
    if (d_levelPos2) cudaFree(d_levelPos2);
    if (d_levelPos3) cudaFree(d_levelPos3);
    nppiFree(d_src);
    nppiFree(d_dst);
    GTEST_SKIP() << "Failed to allocate LUT device memory";
  }

  const Npp32s *values[4] = {d_values0, d_values1, d_values2, d_values3};
  const Npp32s *levelPos[4] = {d_levelPos0, d_levelPos1, d_levelPos2, d_levelPos3};

  NppStatus status = nppiLUT_8u_C4R(d_src, srcStep, d_dst, dstStep, roi, values, levelPos, levels);
#ifdef USE_NVIDIA_NPP_TESTS
  EXPECT_EQ(status, NPP_NO_ERROR);
#else
  EXPECT_EQ(status, NPP_NOT_IMPLEMENTED_ERROR);
#endif

  std::vector<Npp8u> out(width * height * 4);
  cudaMemcpy2D(out.data(), width * 4, d_dst, dstStep, width * 4, height, cudaMemcpyDeviceToHost);
#ifdef USE_NVIDIA_NPP_TESTS
  std::vector<Npp8u> expected(width * height * 4);
  for (size_t i = 0; i < expected.size(); i += 4) {
    expected[i] = applyLutNoInterpolation(hostSrc[i], values0, levelPos0);
    expected[i + 1] = applyLutNoInterpolation(hostSrc[i + 1], values1, levelPos1);
    expected[i + 2] = applyLutNoInterpolation(hostSrc[i + 2], values2, levelPos2);
    expected[i + 3] = applyLutNoInterpolation(hostSrc[i + 3], values3, levelPos3);
  }
  EXPECT_EQ(out, expected);
#else
  EXPECT_EQ(out, hostDst);
#endif

  cudaFree(d_values0);
  cudaFree(d_values1);
  cudaFree(d_values2);
  cudaFree(d_values3);
  cudaFree(d_levelPos0);
  cudaFree(d_levelPos1);
  cudaFree(d_levelPos2);
  cudaFree(d_levelPos3);
  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_P(Lut16uC1RParamTest, ReturnNotImplementedAndNoWrite) {
  const auto param = GetParam();
  const int width = param.width;
  const int height = param.height;
  const NppiSize roi = {width, height};

  std::vector<Npp16u> hostSrc(width * height);
  std::vector<Npp16u> hostDst(width * height, 0xBEEF);
  fillU16(hostSrc, param.seed);

  int srcStep = 0;
  int dstStep = 0;
  Npp16u *d_src = nppiMalloc_16u_C1(width, height, &srcStep);
  if (!d_src) {
    GTEST_SKIP() << "nppiMalloc_16u_C1 failed";
  }
  Npp16u *d_dst = nppiMalloc_16u_C1(width, height, &dstStep);
  if (!d_dst) {
    nppiFree(d_src);
    GTEST_SKIP() << "nppiMalloc_16u_C1 failed";
  }

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * sizeof(Npp16u), width * sizeof(Npp16u), height,
               cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), width * sizeof(Npp16u), width * sizeof(Npp16u), height,
               cudaMemcpyHostToDevice);

  std::vector<Npp32s> values = {0, 65535};
  std::vector<Npp32s> levelPos = {0, 65535};
  const int levels = static_cast<int>(values.size());

  Npp32s *d_values = nullptr;
  Npp32s *d_levels = nullptr;
  if (!allocAndCopyToDevice(values, &d_values) || !allocAndCopyToDevice(levelPos, &d_levels)) {
    if (d_values) {
      cudaFree(d_values);
    }
    if (d_levels) {
      cudaFree(d_levels);
    }
    nppiFree(d_src);
    nppiFree(d_dst);
    GTEST_SKIP() << "Failed to allocate LUT device memory";
  }

  NppStatus status = nppiLUT_16u_C1R(d_src, srcStep, d_dst, dstStep, roi, d_values, d_levels, levels);
#ifdef USE_NVIDIA_NPP_TESTS
  EXPECT_EQ(status, NPP_NO_ERROR);
#else
  EXPECT_EQ(status, NPP_NOT_IMPLEMENTED_ERROR);
#endif

  std::vector<Npp16u> out(width * height);
  cudaMemcpy2D(out.data(), width * sizeof(Npp16u), d_dst, dstStep, width * sizeof(Npp16u), height,
               cudaMemcpyDeviceToHost);
#ifdef USE_NVIDIA_NPP_TESTS
  std::vector<Npp16u> expected(width * height);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = applyLutNoInterpolation(hostSrc[i], values, levelPos);
  }
  EXPECT_EQ(out, expected);
#else
  EXPECT_EQ(out, hostDst);
#endif

  cudaFree(d_values);
  cudaFree(d_levels);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(FunctionalCases, Lut8uC1RParamTest,
                         ::testing::Values(LutCase{8, 8, 123}, LutCase{16, 4, 456}));
INSTANTIATE_TEST_SUITE_P(PrecisionCases, Lut8uC1RParamTest,
                         ::testing::Values(LutCase{32, 6, 789}, LutCase{64, 2, 321}));

INSTANTIATE_TEST_SUITE_P(FunctionalCases, Lut8uC3RParamTest,
                         ::testing::Values(LutCase{8, 8, 123}, LutCase{16, 4, 456}));
INSTANTIATE_TEST_SUITE_P(PrecisionCases, Lut8uC3RParamTest,
                         ::testing::Values(LutCase{32, 6, 789}, LutCase{64, 2, 321}));

INSTANTIATE_TEST_SUITE_P(FunctionalCases, Lut8uC4RParamTest,
                         ::testing::Values(LutCase{8, 8, 123}, LutCase{16, 4, 456}));
INSTANTIATE_TEST_SUITE_P(PrecisionCases, Lut8uC4RParamTest,
                         ::testing::Values(LutCase{32, 6, 789}, LutCase{64, 2, 321}));

INSTANTIATE_TEST_SUITE_P(FunctionalCases, Lut16uC1RParamTest,
                         ::testing::Values(LutCase{8, 8, 123}, LutCase{16, 4, 456}));
INSTANTIATE_TEST_SUITE_P(PrecisionCases, Lut16uC1RParamTest,
                         ::testing::Values(LutCase{32, 6, 789}, LutCase{64, 2, 321}));
