#include "npp.h"
#include <gtest/gtest.h>
#include <vector>

class LogicalFunctionalTest : public ::testing::Test {
protected:
  void SetUp() override {
    // 创建测试数据
    width = 32;
    height = 32;
    size.width = width;
    size.height = height;

    // 使用NPP内置内存管理函数分配设备内存
    d_src1 = nppiMalloc_8u_C1(width, height, &step1);
    d_src2 = nppiMalloc_8u_C1(width, height, &step2);
    d_dst = nppiMalloc_8u_C1(width, height, &step_dst);

    ASSERT_NE(d_src1, nullptr) << "Failed to allocate src1 memory";
    ASSERT_NE(d_src2, nullptr) << "Failed to allocate src2 memory";
    ASSERT_NE(d_dst, nullptr) << "Failed to allocate dst memory";

    // prepare test data
    h_src1.resize(width * height);
    h_src2.resize(width * height);
    h_dst.resize(width * height);

    // 初始化测试模式
    for (int i = 0; i < width * height; ++i) {
      h_src1[i] = 0xAA; // 10101010
      h_src2[i] = 0x55; // 01010101
    }

    // 上传到设备
    cudaError_t err1 = cudaMemcpy2D(d_src1, step1, h_src1.data(), width, width, height, cudaMemcpyHostToDevice);
    cudaError_t err2 = cudaMemcpy2D(d_src2, step2, h_src2.data(), width, width, height, cudaMemcpyHostToDevice);
    ASSERT_EQ(err1, cudaSuccess) << "Failed to copy src1 data";
    ASSERT_EQ(err2, cudaSuccess) << "Failed to copy src2 data";
  }

  void TearDown() override {
    // 使用NPP内置函数释放内存
    if (d_src1)
      nppiFree(d_src1);
    if (d_src2)
      nppiFree(d_src2);
    if (d_dst)
      nppiFree(d_dst);
  }

  int width, height;
  int step1, step2, step_dst;
  NppiSize size;
  Npp8u *d_src1, *d_src2, *d_dst;
  std::vector<Npp8u> h_src1, h_src2, h_dst;
};

TEST_F(LogicalFunctionalTest, And_8u_C1R) {
  // 执行AND操作
  NppStatus status = nppiAnd_8u_C1R(d_src1, step1, d_src2, step2, d_dst, step_dst, size);
  ASSERT_EQ(status, NPP_SUCCESS);

  // 下载结果
  cudaError_t err = cudaMemcpy2D(h_dst.data(), width, d_dst, step_dst, width, height, cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess);

  // Validate结果 (0xAA & 0x55 = 0x00)
  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(h_dst[i], 0x00);
  }
}

TEST_F(LogicalFunctionalTest, AndC_8u_C1R) {
  const Npp8u constant = 0x0F; // 00001111

  // 执行AND常数操作
  NppStatus status = nppiAndC_8u_C1R(d_src1, step1, constant, d_dst, step_dst, size);
  ASSERT_EQ(status, NPP_SUCCESS);

  // 下载结果
  cudaError_t err = cudaMemcpy2D(h_dst.data(), width, d_dst, step_dst, width, height, cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess);

  // Validate结果 (0xAA & 0x0F = 0x0A)
  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(h_dst[i], 0x0A);
  }
}

TEST_F(LogicalFunctionalTest, Or_8u_C1R) {
  // 执行OR操作
  NppStatus status = nppiOr_8u_C1R(d_src1, step1, d_src2, step2, d_dst, step_dst, size);
  ASSERT_EQ(status, NPP_SUCCESS);

  // 下载结果
  cudaError_t err = cudaMemcpy2D(h_dst.data(), width, d_dst, step_dst, width, height, cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess);

  // Validate结果 (0xAA | 0x55 = 0xFF)
  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(h_dst[i], 0xFF);
  }
}

TEST_F(LogicalFunctionalTest, OrC_8u_C1R) {
  const Npp8u constant = 0x0F; // 00001111

  // 执行OR常数操作
  NppStatus status = nppiOrC_8u_C1R(d_src1, step1, constant, d_dst, step_dst, size);
  ASSERT_EQ(status, NPP_SUCCESS);

  // 下载结果
  cudaError_t err = cudaMemcpy2D(h_dst.data(), width, d_dst, step_dst, width, height, cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess);

  // Validate结果 (0xAA | 0x0F = 0xAF)
  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(h_dst[i], 0xAF);
  }
}

// Parameter validation测试
// NOTE: 测试已被禁用 - vendor NPP对无效参数的处理会污染GPU上下文
TEST(LogicalParameterTest, DISABLED_NullPointerError) {
  NppiSize size = {10, 10};
  int step = 10;
  Npp8u dummy = 0;

  // AND测试
  EXPECT_EQ(nppiAnd_8u_C1R(nullptr, step, &dummy, step, &dummy, step, size), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiAnd_8u_C1R(&dummy, step, nullptr, step, &dummy, step, size), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiAnd_8u_C1R(&dummy, step, &dummy, step, nullptr, step, size), NPP_NULL_POINTER_ERROR);

  // OR测试
  EXPECT_EQ(nppiOr_8u_C1R(nullptr, step, &dummy, step, &dummy, step, size), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiOr_8u_C1R(&dummy, step, nullptr, step, &dummy, step, size), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiOr_8u_C1R(&dummy, step, &dummy, step, nullptr, step, size), NPP_NULL_POINTER_ERROR);
}

// NOTE: 测试已被禁用 - vendor NPP对无效参数的处理会污染GPU上下文
TEST(LogicalParameterTest, DISABLED_InvalidSize) {
  Npp8u dummy = 0;
  int step = 10;

  // 测试无效宽度
  NppiSize size = {0, 10};
  EXPECT_EQ(nppiAnd_8u_C1R(&dummy, step, &dummy, step, &dummy, step, size), NPP_SIZE_ERROR);

  // 测试无效高度
  size = {10, 0};
  EXPECT_EQ(nppiOr_8u_C1R(&dummy, step, &dummy, step, &dummy, step, size), NPP_SIZE_ERROR);
}

// NOTE: 测试已被禁用 - vendor NPP对无效参数的处理会污染GPU上下文
TEST(LogicalParameterTest, DISABLED_InvalidStep) {
  Npp8u dummy = 0;
  NppiSize size = {10, 10};

  // 测试无效步长
  EXPECT_EQ(nppiAnd_8u_C1R(&dummy, 5, &dummy, 10, &dummy, 10, size), NPP_STEP_ERROR);
  EXPECT_EQ(nppiOr_8u_C1R(&dummy, 10, &dummy, 5, &dummy, 10, size), NPP_STEP_ERROR);
}
