#include "npp.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

class NPPIAbsDiffTest : public ::testing::Test {
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

  // 辅助函数：ValidateAbsDiff结果
  template <typename T>
  void verifyAbsDiff(const std::vector<T> &src1, const std::vector<T> &src2, const std::vector<T> &result,
                     int channels = 1) {
    // 使用channels参数以避免警告
    (void)channels;
    for (size_t i = 0; i < result.size(); i++) {
      //
      long long diff = static_cast<long long>(src1[i]) - static_cast<long long>(src2[i]);
      if (diff < 0)
        diff = -diff;
      T expected = static_cast<T>(diff);
      EXPECT_EQ(result[i], expected) << "Mismatch at index " << i;
    }
  }
};

// 测试8位无符号单通道绝对差值
TEST_F(NPPIAbsDiffTest, AbsDiff_8u_C1R) {
  size_t dataSize = width * height;
  std::vector<Npp8u> src1Data(dataSize), src2Data(dataSize), dstData(dataSize);

  // 生成测试数据
  generateRandomData(src1Data, dataSize, Npp8u(0), Npp8u(255));
  generateRandomData(src2Data, dataSize, Npp8u(0), Npp8u(255));

  // 分配GPU内存
  Npp8u *d_src1, *d_src2, *d_dst;
  int srcStep = width * sizeof(Npp8u);
  int dstStep = width * sizeof(Npp8u);

  cudaMalloc(&d_src1, dataSize * sizeof(Npp8u));
  cudaMalloc(&d_src2, dataSize * sizeof(Npp8u));
  cudaMalloc(&d_dst, dataSize * sizeof(Npp8u));

  // 拷贝数据到GPU
  cudaMemcpy(d_src1, src1Data.data(), dataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);
  cudaMemcpy(d_src2, src2Data.data(), dataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // CallNPP函数
  NppStatus status = nppiAbsDiff_8u_C1R(d_src1, srcStep, d_src2, srcStep, d_dst, dstStep, roi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  cudaMemcpy(dstData.data(), d_dst, dataSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);

  // Validate结果
  verifyAbsDiff(src1Data, src2Data, dstData);

  // 清理GPU内存
  cudaFree(d_src1);
  cudaFree(d_src2);
  cudaFree(d_dst);
}

// 测试8位无符号三通道绝对差值
TEST_F(NPPIAbsDiffTest, AbsDiff_8u_C3R) {
  size_t dataSize = width * height * 3;
  std::vector<Npp8u> src1Data(dataSize), src2Data(dataSize), dstData(dataSize);

  // 生成测试数据
  generateRandomData(src1Data, dataSize, Npp8u(0), Npp8u(255));
  generateRandomData(src2Data, dataSize, Npp8u(0), Npp8u(255));

  // 分配GPU内存
  Npp8u *d_src1, *d_src2, *d_dst;
  int srcStep = width * 3 * sizeof(Npp8u);
  int dstStep = width * 3 * sizeof(Npp8u);

  cudaMalloc(&d_src1, dataSize * sizeof(Npp8u));
  cudaMalloc(&d_src2, dataSize * sizeof(Npp8u));
  cudaMalloc(&d_dst, dataSize * sizeof(Npp8u));

  // 拷贝数据到GPU
  cudaMemcpy(d_src1, src1Data.data(), dataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);
  cudaMemcpy(d_src2, src2Data.data(), dataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // CallNPP函数
  NppStatus status = nppiAbsDiff_8u_C3R(d_src1, srcStep, d_src2, srcStep, d_dst, dstStep, roi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  cudaMemcpy(dstData.data(), d_dst, dataSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);

  // Validate结果
  verifyAbsDiff(src1Data, src2Data, dstData, 3);

  // 清理GPU内存
  cudaFree(d_src1);
  cudaFree(d_src2);
  cudaFree(d_dst);
}

// 测试32位浮点单通道绝对差值
TEST_F(NPPIAbsDiffTest, AbsDiff_32f_C1R) {
  size_t dataSize = width * height;
  std::vector<Npp32f> src1Data(dataSize), src2Data(dataSize), dstData(dataSize);

  // 生成浮点测试数据
  for (size_t i = 0; i < dataSize; i++) {
    src1Data[i] = static_cast<Npp32f>((rand() / (float)RAND_MAX) * 2.0f - 1.0f); // [-1, 1]
    src2Data[i] = static_cast<Npp32f>((rand() / (float)RAND_MAX) * 2.0f - 1.0f); // [-1, 1]
  }

  // 分配GPU内存
  Npp32f *d_src1, *d_src2, *d_dst;
  int srcStep = width * sizeof(Npp32f);
  int dstStep = width * sizeof(Npp32f);

  cudaMalloc(&d_src1, dataSize * sizeof(Npp32f));
  cudaMalloc(&d_src2, dataSize * sizeof(Npp32f));
  cudaMalloc(&d_dst, dataSize * sizeof(Npp32f));

  // 拷贝数据到GPU
  cudaMemcpy(d_src1, src1Data.data(), dataSize * sizeof(Npp32f), cudaMemcpyHostToDevice);
  cudaMemcpy(d_src2, src2Data.data(), dataSize * sizeof(Npp32f), cudaMemcpyHostToDevice);

  // CallNPP函数
  NppStatus status = nppiAbsDiff_32f_C1R(d_src1, srcStep, d_src2, srcStep, d_dst, dstStep, roi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  cudaMemcpy(dstData.data(), d_dst, dataSize * sizeof(Npp32f), cudaMemcpyDeviceToHost);

  // Validate结果（浮点数需要容差）
  for (size_t i = 0; i < dstData.size(); i++) {
    float expected = fabs(src1Data[i] - src2Data[i]);
    EXPECT_NEAR(dstData[i], expected, 1e-6f) << "Mismatch at index " << i;
  }

  // 清理GPU内存
  cudaFree(d_src1);
  cudaFree(d_src2);
  cudaFree(d_dst);
}

// 测试带上下文的版本
TEST_F(NPPIAbsDiffTest, AbsDiff_8u_C1R_Ctx) {
  size_t dataSize = width * height;
  std::vector<Npp8u> src1Data(dataSize), src2Data(dataSize), dstData(dataSize);

  // 生成测试数据
  generateRandomData(src1Data, dataSize, Npp8u(0), Npp8u(255));
  generateRandomData(src2Data, dataSize, Npp8u(0), Npp8u(255));

  // 分配GPU内存
  Npp8u *d_src1, *d_src2, *d_dst;
  int srcStep = width * sizeof(Npp8u);
  int dstStep = width * sizeof(Npp8u);

  cudaMalloc(&d_src1, dataSize * sizeof(Npp8u));
  cudaMalloc(&d_src2, dataSize * sizeof(Npp8u));
  cudaMalloc(&d_dst, dataSize * sizeof(Npp8u));

  // 拷贝数据到GPU
  cudaMemcpy(d_src1, src1Data.data(), dataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);
  cudaMemcpy(d_src2, src2Data.data(), dataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // 创建流上下文
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0; // 使用Default stream

  // CallNPP函数
  NppStatus status = nppiAbsDiff_8u_C1R_Ctx(d_src1, srcStep, d_src2, srcStep, d_dst, dstStep, roi, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  cudaMemcpy(dstData.data(), d_dst, dataSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);

  // Validate结果
  verifyAbsDiff(src1Data, src2Data, dstData);

  // 清理GPU内存
  cudaFree(d_src1);
  cudaFree(d_src2);
  cudaFree(d_dst);
}
