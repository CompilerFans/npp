#include "npp.h"
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

class NPPICompareCTest : public ::testing::Test {
protected:
  void SetUp() override {
    // 设置随机种子
    srand(time(nullptr));

    // 初始化测试图像尺寸
    width = 32;
    height = 24;
    roi.width = width;
    roi.height = height;
  }

  void TearDown() override {
    // 清理会在各测试函数中处理
  }

  int width, height;
  NppiSize roi;

  // 辅助函数：生成随机数据
  template <typename T> void generateRandomData(std::vector<T> &data, size_t size, T minVal, T maxVal) {
    data.resize(size);
    for (size_t i = 0; i < size; i++) {
      data[i] = static_cast<T>(rand() % (maxVal - minVal + 1) + minVal);
    }
  }

  // 辅助函数：ValidateCompareC结果
  template <typename T>
  void verifyCompareC(const std::vector<T> &src, const std::vector<Npp8u> &result, T constant, NppCmpOp operation) {
    for (size_t i = 0; i < result.size(); i++) {
      bool expected = false;
      switch (operation) {
      case NPP_CMP_LESS:
        expected = src[i] < constant;
        break;
      case NPP_CMP_LESS_EQ:
        expected = src[i] <= constant;
        break;
      case NPP_CMP_EQ:
        expected = src[i] == constant;
        break;
      case NPP_CMP_GREATER_EQ:
        expected = src[i] >= constant;
        break;
      case NPP_CMP_GREATER:
        expected = src[i] > constant;
        break;
      }
      Npp8u expectedValue = expected ? 255 : 0;
      EXPECT_EQ(result[i], expectedValue)
          << "Mismatch at index " << i << " src=" << (int)src[i] << " const=" << (int)constant << " op=" << operation;
    }
  }
};

// 测试8位无符号单通道与常数比较 - 小于
TEST_F(NPPICompareCTest, CompareC_8u_C1R_Less) {
  size_t dataSize = width * height;
  std::vector<Npp8u> srcData(dataSize), dstData(dataSize);

  // 生成测试数据
  generateRandomData(srcData, dataSize, Npp8u(0), Npp8u(255));
  Npp8u constant = 128;

  // 分配GPU内存
  Npp8u *d_src, *d_dst;
  int srcStep = width * sizeof(Npp8u);
  int dstStep = width * sizeof(Npp8u);

  cudaMalloc(&d_src, dataSize * sizeof(Npp8u));
  cudaMalloc(&d_dst, dataSize * sizeof(Npp8u));

  // 拷贝数据到GPU
  cudaMemcpy(d_src, srcData.data(), dataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // CallNPP函数
  NppStatus status = nppiCompareC_8u_C1R(d_src, srcStep, constant, d_dst, dstStep, roi, NPP_CMP_LESS);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  cudaMemcpy(dstData.data(), d_dst, dataSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);

  // Validate结果
  verifyCompareC(srcData, dstData, constant, NPP_CMP_LESS);

  // 清理GPU内存
  cudaFree(d_src);
  cudaFree(d_dst);
}

// 测试8位无符号单通道与常数比较 - 等于
TEST_F(NPPICompareCTest, CompareC_8u_C1R_Equal) {
  size_t dataSize = width * height;
  std::vector<Npp8u> srcData(dataSize), dstData(dataSize);

  // 生成测试数据并确保有相等值
  generateRandomData(srcData, dataSize, Npp8u(0), Npp8u(255));
  Npp8u constant = 100;

  // 确保至少有一些等于常数的值
  for (size_t i = 0; i < dataSize; i += 10) {
    srcData[i] = constant;
  }

  // 分配GPU内存
  Npp8u *d_src, *d_dst;
  int srcStep = width * sizeof(Npp8u);
  int dstStep = width * sizeof(Npp8u);

  cudaMalloc(&d_src, dataSize * sizeof(Npp8u));
  cudaMalloc(&d_dst, dataSize * sizeof(Npp8u));

  // 拷贝数据到GPU
  cudaMemcpy(d_src, srcData.data(), dataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // CallNPP函数
  NppStatus status = nppiCompareC_8u_C1R(d_src, srcStep, constant, d_dst, dstStep, roi, NPP_CMP_EQ);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  cudaMemcpy(dstData.data(), d_dst, dataSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);

  // Validate结果
  verifyCompareC(srcData, dstData, constant, NPP_CMP_EQ);

  // 清理GPU内存
  cudaFree(d_src);
  cudaFree(d_dst);
}

// 测试8位无符号单通道与常数比较 - 大于
TEST_F(NPPICompareCTest, CompareC_8u_C1R_Greater) {
  size_t dataSize = width * height;
  std::vector<Npp8u> srcData(dataSize), dstData(dataSize);

  // 生成测试数据
  generateRandomData(srcData, dataSize, Npp8u(0), Npp8u(255));
  Npp8u constant = 128;

  // 分配GPU内存
  Npp8u *d_src, *d_dst;
  int srcStep = width * sizeof(Npp8u);
  int dstStep = width * sizeof(Npp8u);

  cudaMalloc(&d_src, dataSize * sizeof(Npp8u));
  cudaMalloc(&d_dst, dataSize * sizeof(Npp8u));

  // 拷贝数据到GPU
  cudaMemcpy(d_src, srcData.data(), dataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // CallNPP函数
  NppStatus status = nppiCompareC_8u_C1R(d_src, srcStep, constant, d_dst, dstStep, roi, NPP_CMP_GREATER);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  cudaMemcpy(dstData.data(), d_dst, dataSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);

  // Validate结果
  verifyCompareC(srcData, dstData, constant, NPP_CMP_GREATER);

  // 清理GPU内存
  cudaFree(d_src);
  cudaFree(d_dst);
}

// 测试16位有符号单通道与常数比较
TEST_F(NPPICompareCTest, CompareC_16s_C1R) {
  size_t dataSize = width * height;
  std::vector<Npp16s> srcData(dataSize);
  std::vector<Npp8u> dstData(dataSize);

  // 生成测试数据（包括负数）
  generateRandomData(srcData, dataSize, Npp16s(-1000), Npp16s(1000));
  Npp16s constant = 0;

  // 分配GPU内存
  Npp16s *d_src;
  Npp8u *d_dst;
  int srcStep = width * sizeof(Npp16s);
  int dstStep = width * sizeof(Npp8u);

  cudaMalloc(&d_src, dataSize * sizeof(Npp16s));
  cudaMalloc(&d_dst, dataSize * sizeof(Npp8u));

  // 拷贝数据到GPU
  cudaMemcpy(d_src, srcData.data(), dataSize * sizeof(Npp16s), cudaMemcpyHostToDevice);

  // CallNPP函数 - 测试小于等于
  NppStatus status = nppiCompareC_16s_C1R(d_src, srcStep, constant, d_dst, dstStep, roi, NPP_CMP_LESS_EQ);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  cudaMemcpy(dstData.data(), d_dst, dataSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);

  // Validate结果
  verifyCompareC(srcData, dstData, constant, NPP_CMP_LESS_EQ);

  // 清理GPU内存
  cudaFree(d_src);
  cudaFree(d_dst);
}

// 测试32位浮点单通道与常数比较
TEST_F(NPPICompareCTest, CompareC_32f_C1R) {
  size_t dataSize = width * height;
  std::vector<Npp32f> srcData(dataSize);
  std::vector<Npp8u> dstData(dataSize);

  // 生成浮点测试数据
  for (size_t i = 0; i < dataSize; i++) {
    srcData[i] = static_cast<Npp32f>((rand() / (float)RAND_MAX) * 2.0f - 1.0f); // [-1, 1]
  }
  Npp32f constant = 0.0f;

  // 分配GPU内存
  Npp32f *d_src;
  Npp8u *d_dst;
  int srcStep = width * sizeof(Npp32f);
  int dstStep = width * sizeof(Npp8u);

  cudaMalloc(&d_src, dataSize * sizeof(Npp32f));
  cudaMalloc(&d_dst, dataSize * sizeof(Npp8u));

  // 拷贝数据到GPU
  cudaMemcpy(d_src, srcData.data(), dataSize * sizeof(Npp32f), cudaMemcpyHostToDevice);

  // CallNPP函数 - 测试大于等于
  NppStatus status = nppiCompareC_32f_C1R(d_src, srcStep, constant, d_dst, dstStep, roi, NPP_CMP_GREATER_EQ);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  cudaMemcpy(dstData.data(), d_dst, dataSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);

  // Validate结果
  verifyCompareC(srcData, dstData, constant, NPP_CMP_GREATER_EQ);

  // 清理GPU内存
  cudaFree(d_src);
  cudaFree(d_dst);
}

// 测试带上下文的版本
TEST_F(NPPICompareCTest, CompareC_8u_C1R_Ctx) {
  size_t dataSize = width * height;
  std::vector<Npp8u> srcData(dataSize), dstData(dataSize);

  // 生成测试数据
  generateRandomData(srcData, dataSize, Npp8u(0), Npp8u(255));
  Npp8u constant = 200;

  // 分配GPU内存
  Npp8u *d_src, *d_dst;
  int srcStep = width * sizeof(Npp8u);
  int dstStep = width * sizeof(Npp8u);

  cudaMalloc(&d_src, dataSize * sizeof(Npp8u));
  cudaMalloc(&d_dst, dataSize * sizeof(Npp8u));

  // 拷贝数据到GPU
  cudaMemcpy(d_src, srcData.data(), dataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // 创建流上下文
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0; // 使用Default stream

  // CallNPP函数
  NppStatus status = nppiCompareC_8u_C1R_Ctx(d_src, srcStep, constant, d_dst, dstStep, roi, NPP_CMP_LESS, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  cudaMemcpy(dstData.data(), d_dst, dataSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);

  // Validate结果
  verifyCompareC(srcData, dstData, constant, NPP_CMP_LESS);

  // 清理GPU内存
  cudaFree(d_src);
  cudaFree(d_dst);
}

// 测试所有比较操作
TEST_F(NPPICompareCTest, CompareC_AllOperations) {
  std::vector<Npp8u> srcData = {50, 100, 150, 200, 100}; // 简单测试数据
  std::vector<Npp8u> dstData(5);
  Npp8u constant = 100;

  // 创建小尺寸测试
  NppiSize smallRoi = {5, 1};

  // 分配GPU内存
  Npp8u *d_src, *d_dst;
  int srcStep = 5 * sizeof(Npp8u);
  int dstStep = 5 * sizeof(Npp8u);

  cudaMalloc(&d_src, 5 * sizeof(Npp8u));
  cudaMalloc(&d_dst, 5 * sizeof(Npp8u));

  // 拷贝数据到GPU
  cudaMemcpy(d_src, srcData.data(), 5 * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // 测试所有比较操作
  NppCmpOp operations[] = {NPP_CMP_LESS, NPP_CMP_LESS_EQ, NPP_CMP_EQ, NPP_CMP_GREATER_EQ, NPP_CMP_GREATER};

  for (NppCmpOp op : operations) {
    NppStatus status = nppiCompareC_8u_C1R(d_src, srcStep, constant, d_dst, dstStep, smallRoi, op);
    EXPECT_EQ(status, NPP_SUCCESS) << "Operation " << op << " failed";

    // 拷贝结果回主机
    cudaMemcpy(dstData.data(), d_dst, 5 * sizeof(Npp8u), cudaMemcpyDeviceToHost);

    // Validate结果
    verifyCompareC(srcData, dstData, constant, op);
  }

  // 清理GPU内存
  cudaFree(d_src);
  cudaFree(d_dst);
}
