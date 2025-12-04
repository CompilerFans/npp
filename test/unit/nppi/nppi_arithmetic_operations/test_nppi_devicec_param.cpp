// DeviceC 16s parameterized tests for nppi arithmetic operations
// DeviceC functions use device pointer for constant value instead of host value
#include "npp_test_base.h"
#include "nppi_arithmetic_test_framework.h"

using namespace npp_functional_test;
using namespace npp_arithmetic_test;

// ==================== Common Param Structures ====================

struct DeviceC16sParam {
  int width;
  int height;
  int scaleFactor;
  std::string name;
};

// ==================== AddDeviceC 16s C3RSfs TEST_P ====================

class AddDeviceC16sC3RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(AddDeviceC16sC3RSfsParamTest, AddDeviceC_16s_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s* d_src = nppiMalloc_16s_C1(width * 3, height, &srcStep);
  Npp16s* d_dst = nppiMalloc_16s_C1(width * 3, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(1), static_cast<Npp16s>(100), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 3 * sizeof(Npp16s),
               width * 3 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[3] = {10, 20, 30};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 3 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 3 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiAddDeviceC_16s_C3RSfs_Ctx(d_src, srcStep, d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(AddDeviceC16sC3RSfs, AddDeviceC16sC3RSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== AddDeviceC 16s C3IRSfs TEST_P ====================

class AddDeviceC16sC3IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(AddDeviceC16sC3IRSfsParamTest, AddDeviceC_16s_C3IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s* d_dst = nppiMalloc_16s_C1(width * 3, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height * 3);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(1), static_cast<Npp16s>(100), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 3 * sizeof(Npp16s),
               width * 3 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[3] = {10, 20, 30};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 3 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 3 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiAddDeviceC_16s_C3IRSfs_Ctx(d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(AddDeviceC16sC3IRSfs, AddDeviceC16sC3IRSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== AddDeviceC 16s C4RSfs TEST_P ====================

class AddDeviceC16sC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(AddDeviceC16sC4RSfsParamTest, AddDeviceC_16s_C4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s* d_src = nppiMalloc_16s_C4(width, height, &srcStep);
  Npp16s* d_dst = nppiMalloc_16s_C4(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(1), static_cast<Npp16s>(100), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 4 * sizeof(Npp16s),
               width * 4 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[4] = {10, 20, 30, 40};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 4 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 4 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiAddDeviceC_16s_C4RSfs_Ctx(d_src, srcStep, d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(AddDeviceC16sC4RSfs, AddDeviceC16sC4RSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== AddDeviceC 16s C4IRSfs TEST_P ====================

class AddDeviceC16sC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(AddDeviceC16sC4IRSfsParamTest, AddDeviceC_16s_C4IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s* d_dst = nppiMalloc_16s_C4(width, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height * 4);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(1), static_cast<Npp16s>(100), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 4 * sizeof(Npp16s),
               width * 4 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[4] = {10, 20, 30, 40};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 4 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 4 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiAddDeviceC_16s_C4IRSfs_Ctx(d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(AddDeviceC16sC4IRSfs, AddDeviceC16sC4IRSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== AddDeviceC 16s AC4RSfs TEST_P ====================

class AddDeviceC16sAC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(AddDeviceC16sAC4RSfsParamTest, AddDeviceC_16s_AC4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s* d_src = nppiMalloc_16s_C4(width, height, &srcStep);
  Npp16s* d_dst = nppiMalloc_16s_C4(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(1), static_cast<Npp16s>(100), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 4 * sizeof(Npp16s),
               width * 4 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[3] = {10, 20, 30};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 3 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 3 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiAddDeviceC_16s_AC4RSfs_Ctx(d_src, srcStep, d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(AddDeviceC16sAC4RSfs, AddDeviceC16sAC4RSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== AddDeviceC 16s AC4IRSfs TEST_P ====================

class AddDeviceC16sAC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(AddDeviceC16sAC4IRSfsParamTest, AddDeviceC_16s_AC4IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s* d_dst = nppiMalloc_16s_C4(width, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height * 4);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(1), static_cast<Npp16s>(100), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 4 * sizeof(Npp16s),
               width * 4 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[3] = {10, 20, 30};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 3 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 3 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiAddDeviceC_16s_AC4IRSfs_Ctx(d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(AddDeviceC16sAC4IRSfs, AddDeviceC16sAC4IRSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== SubDeviceC 16s C3RSfs TEST_P ====================

class SubDeviceC16sC3RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(SubDeviceC16sC3RSfsParamTest, SubDeviceC_16s_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s* d_src = nppiMalloc_16s_C1(width * 3, height, &srcStep);
  Npp16s* d_dst = nppiMalloc_16s_C1(width * 3, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(50), static_cast<Npp16s>(200), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 3 * sizeof(Npp16s),
               width * 3 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[3] = {10, 20, 30};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 3 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 3 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiSubDeviceC_16s_C3RSfs_Ctx(d_src, srcStep, d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(SubDeviceC16sC3RSfs, SubDeviceC16sC3RSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== SubDeviceC 16s C3IRSfs TEST_P ====================

class SubDeviceC16sC3IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(SubDeviceC16sC3IRSfsParamTest, SubDeviceC_16s_C3IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s* d_dst = nppiMalloc_16s_C1(width * 3, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height * 3);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(50), static_cast<Npp16s>(200), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 3 * sizeof(Npp16s),
               width * 3 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[3] = {10, 20, 30};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 3 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 3 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiSubDeviceC_16s_C3IRSfs_Ctx(d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(SubDeviceC16sC3IRSfs, SubDeviceC16sC3IRSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== SubDeviceC 16s C4RSfs TEST_P ====================

class SubDeviceC16sC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(SubDeviceC16sC4RSfsParamTest, SubDeviceC_16s_C4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s* d_src = nppiMalloc_16s_C4(width, height, &srcStep);
  Npp16s* d_dst = nppiMalloc_16s_C4(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(50), static_cast<Npp16s>(200), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 4 * sizeof(Npp16s),
               width * 4 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[4] = {10, 20, 30, 40};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 4 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 4 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiSubDeviceC_16s_C4RSfs_Ctx(d_src, srcStep, d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(SubDeviceC16sC4RSfs, SubDeviceC16sC4RSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== SubDeviceC 16s C4IRSfs TEST_P ====================

class SubDeviceC16sC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(SubDeviceC16sC4IRSfsParamTest, SubDeviceC_16s_C4IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s* d_dst = nppiMalloc_16s_C4(width, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height * 4);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(50), static_cast<Npp16s>(200), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 4 * sizeof(Npp16s),
               width * 4 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[4] = {10, 20, 30, 40};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 4 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 4 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiSubDeviceC_16s_C4IRSfs_Ctx(d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(SubDeviceC16sC4IRSfs, SubDeviceC16sC4IRSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== SubDeviceC 16s AC4RSfs TEST_P ====================

class SubDeviceC16sAC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(SubDeviceC16sAC4RSfsParamTest, SubDeviceC_16s_AC4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s* d_src = nppiMalloc_16s_C4(width, height, &srcStep);
  Npp16s* d_dst = nppiMalloc_16s_C4(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(50), static_cast<Npp16s>(200), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 4 * sizeof(Npp16s),
               width * 4 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[3] = {10, 20, 30};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 3 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 3 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiSubDeviceC_16s_AC4RSfs_Ctx(d_src, srcStep, d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(SubDeviceC16sAC4RSfs, SubDeviceC16sAC4RSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== SubDeviceC 16s AC4IRSfs TEST_P ====================

class SubDeviceC16sAC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(SubDeviceC16sAC4IRSfsParamTest, SubDeviceC_16s_AC4IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s* d_dst = nppiMalloc_16s_C4(width, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height * 4);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(50), static_cast<Npp16s>(200), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 4 * sizeof(Npp16s),
               width * 4 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[3] = {10, 20, 30};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 3 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 3 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiSubDeviceC_16s_AC4IRSfs_Ctx(d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(SubDeviceC16sAC4IRSfs, SubDeviceC16sAC4IRSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== MulDeviceC 16s C3RSfs TEST_P ====================

class MulDeviceC16sC3RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(MulDeviceC16sC3RSfsParamTest, MulDeviceC_16s_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s* d_src = nppiMalloc_16s_C1(width * 3, height, &srcStep);
  Npp16s* d_dst = nppiMalloc_16s_C1(width * 3, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(1), static_cast<Npp16s>(50), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 3 * sizeof(Npp16s),
               width * 3 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[3] = {2, 3, 4};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 3 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 3 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiMulDeviceC_16s_C3RSfs_Ctx(d_src, srcStep, d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(MulDeviceC16sC3RSfs, MulDeviceC16sC3RSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== MulDeviceC 16s C3IRSfs TEST_P ====================

class MulDeviceC16sC3IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(MulDeviceC16sC3IRSfsParamTest, MulDeviceC_16s_C3IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s* d_dst = nppiMalloc_16s_C1(width * 3, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height * 3);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(1), static_cast<Npp16s>(50), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 3 * sizeof(Npp16s),
               width * 3 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[3] = {2, 3, 4};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 3 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 3 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiMulDeviceC_16s_C3IRSfs_Ctx(d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(MulDeviceC16sC3IRSfs, MulDeviceC16sC3IRSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== MulDeviceC 16s C4RSfs TEST_P ====================

class MulDeviceC16sC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(MulDeviceC16sC4RSfsParamTest, MulDeviceC_16s_C4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s* d_src = nppiMalloc_16s_C4(width, height, &srcStep);
  Npp16s* d_dst = nppiMalloc_16s_C4(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(1), static_cast<Npp16s>(50), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 4 * sizeof(Npp16s),
               width * 4 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[4] = {2, 3, 4, 5};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 4 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 4 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiMulDeviceC_16s_C4RSfs_Ctx(d_src, srcStep, d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(MulDeviceC16sC4RSfs, MulDeviceC16sC4RSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== MulDeviceC 16s C4IRSfs TEST_P ====================

class MulDeviceC16sC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(MulDeviceC16sC4IRSfsParamTest, MulDeviceC_16s_C4IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s* d_dst = nppiMalloc_16s_C4(width, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height * 4);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(1), static_cast<Npp16s>(50), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 4 * sizeof(Npp16s),
               width * 4 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[4] = {2, 3, 4, 5};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 4 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 4 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiMulDeviceC_16s_C4IRSfs_Ctx(d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(MulDeviceC16sC4IRSfs, MulDeviceC16sC4IRSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== MulDeviceC 16s AC4RSfs TEST_P ====================

class MulDeviceC16sAC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(MulDeviceC16sAC4RSfsParamTest, MulDeviceC_16s_AC4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s* d_src = nppiMalloc_16s_C4(width, height, &srcStep);
  Npp16s* d_dst = nppiMalloc_16s_C4(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(1), static_cast<Npp16s>(50), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 4 * sizeof(Npp16s),
               width * 4 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[3] = {2, 3, 4};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 3 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 3 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiMulDeviceC_16s_AC4RSfs_Ctx(d_src, srcStep, d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(MulDeviceC16sAC4RSfs, MulDeviceC16sAC4RSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== MulDeviceC 16s AC4IRSfs TEST_P ====================

class MulDeviceC16sAC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(MulDeviceC16sAC4IRSfsParamTest, MulDeviceC_16s_AC4IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s* d_dst = nppiMalloc_16s_C4(width, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height * 4);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(1), static_cast<Npp16s>(50), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 4 * sizeof(Npp16s),
               width * 4 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[3] = {2, 3, 4};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 3 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 3 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiMulDeviceC_16s_AC4IRSfs_Ctx(d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(MulDeviceC16sAC4IRSfs, MulDeviceC16sAC4IRSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== DivDeviceC 16s C3RSfs TEST_P ====================

class DivDeviceC16sC3RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(DivDeviceC16sC3RSfsParamTest, DivDeviceC_16s_C3RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s* d_src = nppiMalloc_16s_C1(width * 3, height, &srcStep);
  Npp16s* d_dst = nppiMalloc_16s_C1(width * 3, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height * 3);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(50), static_cast<Npp16s>(200), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 3 * sizeof(Npp16s),
               width * 3 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[3] = {2, 3, 4};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 3 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 3 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiDivDeviceC_16s_C3RSfs_Ctx(d_src, srcStep, d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(DivDeviceC16sC3RSfs, DivDeviceC16sC3RSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== DivDeviceC 16s C3IRSfs TEST_P ====================

class DivDeviceC16sC3IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(DivDeviceC16sC3IRSfsParamTest, DivDeviceC_16s_C3IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s* d_dst = nppiMalloc_16s_C1(width * 3, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height * 3);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(50), static_cast<Npp16s>(200), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 3 * sizeof(Npp16s),
               width * 3 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[3] = {2, 3, 4};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 3 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 3 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiDivDeviceC_16s_C3IRSfs_Ctx(d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(DivDeviceC16sC3IRSfs, DivDeviceC16sC3IRSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== DivDeviceC 16s C4RSfs TEST_P ====================

class DivDeviceC16sC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(DivDeviceC16sC4RSfsParamTest, DivDeviceC_16s_C4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s* d_src = nppiMalloc_16s_C4(width, height, &srcStep);
  Npp16s* d_dst = nppiMalloc_16s_C4(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(50), static_cast<Npp16s>(200), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 4 * sizeof(Npp16s),
               width * 4 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[4] = {2, 3, 4, 5};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 4 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 4 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiDivDeviceC_16s_C4RSfs_Ctx(d_src, srcStep, d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(DivDeviceC16sC4RSfs, DivDeviceC16sC4RSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== DivDeviceC 16s C4IRSfs TEST_P ====================

class DivDeviceC16sC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(DivDeviceC16sC4IRSfsParamTest, DivDeviceC_16s_C4IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s* d_dst = nppiMalloc_16s_C4(width, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height * 4);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(50), static_cast<Npp16s>(200), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 4 * sizeof(Npp16s),
               width * 4 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[4] = {2, 3, 4, 5};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 4 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 4 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiDivDeviceC_16s_C4IRSfs_Ctx(d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(DivDeviceC16sC4IRSfs, DivDeviceC16sC4IRSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== DivDeviceC 16s AC4RSfs TEST_P ====================

class DivDeviceC16sAC4RSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(DivDeviceC16sAC4RSfsParamTest, DivDeviceC_16s_AC4RSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int srcStep, dstStep;
  Npp16s* d_src = nppiMalloc_16s_C4(width, height, &srcStep);
  Npp16s* d_dst = nppiMalloc_16s_C4(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> srcData(width * height * 4);
  TestDataGenerator::generateRandom(srcData, static_cast<Npp16s>(50), static_cast<Npp16s>(200), 12345);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 4 * sizeof(Npp16s),
               width * 4 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[3] = {2, 3, 4};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 3 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 3 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiDivDeviceC_16s_AC4RSfs_Ctx(d_src, srcStep, d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_src);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(DivDeviceC16sAC4RSfs, DivDeviceC16sAC4RSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });

// ==================== DivDeviceC 16s AC4IRSfs TEST_P ====================

class DivDeviceC16sAC4IRSfsParamTest : public NppTestBase, public ::testing::WithParamInterface<DeviceC16sParam> {};

TEST_P(DivDeviceC16sAC4IRSfsParamTest, DivDeviceC_16s_AC4IRSfs) {
  const auto &param = GetParam();
  const int width = param.width;
  const int height = param.height;

  int dstStep;
  Npp16s* d_dst = nppiMalloc_16s_C4(width, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  std::vector<Npp16s> dstData(width * height * 4);
  TestDataGenerator::generateRandom(dstData, static_cast<Npp16s>(50), static_cast<Npp16s>(200), 12345);
  cudaMemcpy2D(d_dst, dstStep, dstData.data(), width * 4 * sizeof(Npp16s),
               width * 4 * sizeof(Npp16s), height, cudaMemcpyHostToDevice);

  Npp16s h_constants[3] = {2, 3, 4};
  Npp16s* d_constants;
  cudaMalloc(&d_constants, 3 * sizeof(Npp16s));
  cudaMemcpy(d_constants, h_constants, 3 * sizeof(Npp16s), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx{};
  ctx.hStream = 0;

  NppStatus status = nppiDivDeviceC_16s_AC4IRSfs_Ctx(d_constants, d_dst, dstStep, roi, param.scaleFactor, ctx);
  ASSERT_EQ(status, NPP_NO_ERROR);

  cudaFree(d_constants);
  nppiFree(d_dst);
}

INSTANTIATE_TEST_SUITE_P(DivDeviceC16sAC4IRSfs, DivDeviceC16sAC4IRSfsParamTest,
                         ::testing::Values(DeviceC16sParam{32, 32, 0, "32x32"}),
                         [](const ::testing::TestParamInfo<DeviceC16sParam> &info) { return info.param.name; });
