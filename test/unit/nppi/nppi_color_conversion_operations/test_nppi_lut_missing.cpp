#include "npp.h"
#include "npp_test_base.h"
#include <cstring>
#include <gtest/gtest.h>
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

  const int levels = 2;
  const Npp32s values[2] = {0, 255};
  const Npp32s levelPos[2] = {0, 255};

  NppStatus status = nppiLUT_8u_C1R(d_src, srcStep, d_dst, dstStep, roi, values, levelPos, levels);
  EXPECT_EQ(status, NPP_NOT_IMPLEMENTED_ERROR);

  std::vector<Npp8u> out(width * height);
  cudaMemcpy2D(out.data(), width, d_dst, dstStep, width, height, cudaMemcpyDeviceToHost);
  EXPECT_EQ(out, hostDst);

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
  const Npp32s values0[2] = {0, 255};
  const Npp32s values1[2] = {0, 255};
  const Npp32s values2[2] = {0, 255};
  const Npp32s levelPos0[2] = {0, 255};
  const Npp32s levelPos1[2] = {0, 255};
  const Npp32s levelPos2[2] = {0, 255};
  const Npp32s *values[3] = {values0, values1, values2};
  const Npp32s *levelPos[3] = {levelPos0, levelPos1, levelPos2};

  NppStatus status = nppiLUT_8u_C3R(d_src, srcStep, d_dst, dstStep, roi, values, levelPos, levels);
  EXPECT_EQ(status, NPP_NOT_IMPLEMENTED_ERROR);

  std::vector<Npp8u> out(width * height * 3);
  cudaMemcpy2D(out.data(), width * 3, d_dst, dstStep, width * 3, height, cudaMemcpyDeviceToHost);
  EXPECT_EQ(out, hostDst);

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
  const Npp32s values0[2] = {0, 255};
  const Npp32s values1[2] = {0, 255};
  const Npp32s values2[2] = {0, 255};
  const Npp32s values3[2] = {0, 255};
  const Npp32s levelPos0[2] = {0, 255};
  const Npp32s levelPos1[2] = {0, 255};
  const Npp32s levelPos2[2] = {0, 255};
  const Npp32s levelPos3[2] = {0, 255};
  const Npp32s *values[4] = {values0, values1, values2, values3};
  const Npp32s *levelPos[4] = {levelPos0, levelPos1, levelPos2, levelPos3};

  NppStatus status = nppiLUT_8u_C4R(d_src, srcStep, d_dst, dstStep, roi, values, levelPos, levels);
  EXPECT_EQ(status, NPP_NOT_IMPLEMENTED_ERROR);

  std::vector<Npp8u> out(width * height * 4);
  cudaMemcpy2D(out.data(), width * 4, d_dst, dstStep, width * 4, height, cudaMemcpyDeviceToHost);
  EXPECT_EQ(out, hostDst);

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

  const int levels = 2;
  const Npp32s values[2] = {0, 65535};
  const Npp32s levelPos[2] = {0, 65535};

  NppStatus status = nppiLUT_16u_C1R(d_src, srcStep, d_dst, dstStep, roi, values, levelPos, levels);
  EXPECT_EQ(status, NPP_NOT_IMPLEMENTED_ERROR);

  std::vector<Npp16u> out(width * height);
  cudaMemcpy2D(out.data(), width * sizeof(Npp16u), d_dst, dstStep, width * sizeof(Npp16u), height,
               cudaMemcpyDeviceToHost);
  EXPECT_EQ(out, hostDst);

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
