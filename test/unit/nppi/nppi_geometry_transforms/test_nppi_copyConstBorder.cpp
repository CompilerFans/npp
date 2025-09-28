#include "npp.h"
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

class NPPICopyConstBorderTest : public ::testing::Test {
protected:
  void SetUp() override {
    // 设置随机种子
    srand(time(nullptr));

    // 初始化测试图像尺寸
    srcWidth = 16;
    srcHeight = 12;
    oSrcSizeROI.width = srcWidth;
    oSrcSizeROI.height = srcHeight;

    // 边界尺寸
    nTopBorderHeight = 4;
    nLeftBorderWidth = 3;

    // 目标尺寸（包含边界）
    dstWidth = srcWidth + nLeftBorderWidth + 5;   // 右边界5像素
    dstHeight = srcHeight + nTopBorderHeight + 6; // 底部边界6像素
    oDstSizeROI.width = dstWidth;
    oDstSizeROI.height = dstHeight;
  }

  void TearDown() override {
    // 清理会在各测试函数中处理
  }

  int srcWidth, srcHeight, dstWidth, dstHeight;
  int nTopBorderHeight, nLeftBorderWidth;
  NppiSize oSrcSizeROI, oDstSizeROI;

  // 辅助函数：生成随机数据
  template <typename T> void generateRandomData(std::vector<T> &data, size_t size, T minVal, T maxVal) {
    data.resize(size);
    for (size_t i = 0; i < size; i++) {
      data[i] = static_cast<T>(rand() % (maxVal - minVal + 1) + minVal);
    }
  }

  // 辅助函数：ValidateCopyConstBorder结果（单通道）
  template <typename T>
  void verifyCopyConstBorder_C1(const std::vector<T> &srcData, const std::vector<T> &dstData, T borderValue) {
    for (int y = 0; y < dstHeight; y++) {
      for (int x = 0; x < dstWidth; x++) {
        int dstIdx = y * dstWidth + x;

        // 检查是否在源区域内
        if (x >= nLeftBorderWidth && x < nLeftBorderWidth + srcWidth && y >= nTopBorderHeight &&
            y < nTopBorderHeight + srcHeight) {
          // 在源区域内，应该复制源数据
          int srcX = x - nLeftBorderWidth;
          int srcY = y - nTopBorderHeight;
          int srcIdx = srcY * srcWidth + srcX;
          EXPECT_EQ(dstData[dstIdx], srcData[srcIdx])
              << "Source copy mismatch at dst(" << x << "," << y << ") src(" << srcX << "," << srcY << ")";
        } else {
          // 在边界区域，应该是边界值
          EXPECT_EQ(dstData[dstIdx], borderValue) << "Border value mismatch at (" << x << "," << y << ")";
        }
      }
    }
  }

  // 辅助函数：ValidateCopyConstBorder结果（三通道）
  void verifyCopyConstBorder_C3(const std::vector<Npp8u> &srcData, const std::vector<Npp8u> &dstData,
                                const Npp8u borderValues[3]) {
    for (int y = 0; y < dstHeight; y++) {
      for (int x = 0; x < dstWidth; x++) {
        // 检查是否在源区域内
        if (x >= nLeftBorderWidth && x < nLeftBorderWidth + srcWidth && y >= nTopBorderHeight &&
            y < nTopBorderHeight + srcHeight) {
          // 在源区域内，应该复制源数据
          int srcX = x - nLeftBorderWidth;
          int srcY = y - nTopBorderHeight;
          for (int c = 0; c < 3; c++) {
            int dstIdx = (y * dstWidth + x) * 3 + c;
            int srcIdx = (srcY * srcWidth + srcX) * 3 + c;
            EXPECT_EQ(dstData[dstIdx], srcData[srcIdx])
                << "Source copy mismatch at dst(" << x << "," << y << ") channel " << c;
          }
        } else {
          // 在边界区域，应该是边界值
          for (int c = 0; c < 3; c++) {
            int dstIdx = (y * dstWidth + x) * 3 + c;
            EXPECT_EQ(dstData[dstIdx], borderValues[c])
                << "Border value mismatch at (" << x << "," << y << ") channel " << c;
          }
        }
      }
    }
  }
};

// 测试8位无符号单通道常量边界拷贝
TEST_F(NPPICopyConstBorderTest, CopyConstBorder_8u_C1R) {
  size_t srcDataSize = srcWidth * srcHeight;
  size_t dstDataSize = dstWidth * dstHeight;
  std::vector<Npp8u> srcData(srcDataSize), dstData(dstDataSize);

  // 生成源数据
  generateRandomData(srcData, srcDataSize, Npp8u(0), Npp8u(255));
  Npp8u borderValue = 128;

  // 分配GPU内存
  Npp8u *d_src, *d_dst;
  int nSrcStep = srcWidth * sizeof(Npp8u);
  int nDstStep = dstWidth * sizeof(Npp8u);

  cudaMalloc(&d_src, srcDataSize * sizeof(Npp8u));
  cudaMalloc(&d_dst, dstDataSize * sizeof(Npp8u));

  // 拷贝数据到GPU
  cudaMemcpy(d_src, srcData.data(), srcDataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // CallNPP函数
  NppStatus status = nppiCopyConstBorder_8u_C1R(d_src, nSrcStep, oSrcSizeROI, d_dst, nDstStep, oDstSizeROI,
                                                nTopBorderHeight, nLeftBorderWidth, borderValue);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  cudaMemcpy(dstData.data(), d_dst, dstDataSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);

  // Validate结果
  verifyCopyConstBorder_C1(srcData, dstData, borderValue);

  // 清理GPU内存
  cudaFree(d_src);
  cudaFree(d_dst);
}

// 测试8位无符号三通道常量边界拷贝
TEST_F(NPPICopyConstBorderTest, CopyConstBorder_8u_C3R) {
  size_t srcDataSize = srcWidth * srcHeight * 3;
  size_t dstDataSize = dstWidth * dstHeight * 3;
  std::vector<Npp8u> srcData(srcDataSize), dstData(dstDataSize);

  // 生成源数据
  generateRandomData(srcData, srcDataSize, Npp8u(0), Npp8u(255));
  Npp8u borderValues[3] = {100, 150, 200};

  // 分配GPU内存
  Npp8u *d_src, *d_dst;
  int nSrcStep = srcWidth * 3 * sizeof(Npp8u);
  int nDstStep = dstWidth * 3 * sizeof(Npp8u);

  cudaMalloc(&d_src, srcDataSize * sizeof(Npp8u));
  cudaMalloc(&d_dst, dstDataSize * sizeof(Npp8u));

  // 拷贝数据到GPU
  cudaMemcpy(d_src, srcData.data(), srcDataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // CallNPP函数
  NppStatus status = nppiCopyConstBorder_8u_C3R(d_src, nSrcStep, oSrcSizeROI, d_dst, nDstStep, oDstSizeROI,
                                                nTopBorderHeight, nLeftBorderWidth, borderValues);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  cudaMemcpy(dstData.data(), d_dst, dstDataSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);

  // Validate结果
  verifyCopyConstBorder_C3(srcData, dstData, borderValues);

  // 清理GPU内存
  cudaFree(d_src);
  cudaFree(d_dst);
}

// 测试16位有符号单通道常量边界拷贝
TEST_F(NPPICopyConstBorderTest, CopyConstBorder_16s_C1R) {
  size_t srcDataSize = srcWidth * srcHeight;
  size_t dstDataSize = dstWidth * dstHeight;
  std::vector<Npp16s> srcData(srcDataSize), dstData(dstDataSize);

  // 生成源数据（包括负数）
  generateRandomData(srcData, srcDataSize, Npp16s(-1000), Npp16s(1000));
  Npp16s borderValue = -500;

  // 分配GPU内存
  Npp16s *d_src, *d_dst;
  int nSrcStep = srcWidth * sizeof(Npp16s);
  int nDstStep = dstWidth * sizeof(Npp16s);

  cudaMalloc(&d_src, srcDataSize * sizeof(Npp16s));
  cudaMalloc(&d_dst, dstDataSize * sizeof(Npp16s));

  // 拷贝数据到GPU
  cudaMemcpy(d_src, srcData.data(), srcDataSize * sizeof(Npp16s), cudaMemcpyHostToDevice);

  // CallNPP函数
  NppStatus status = nppiCopyConstBorder_16s_C1R(d_src, nSrcStep, oSrcSizeROI, d_dst, nDstStep, oDstSizeROI,
                                                 nTopBorderHeight, nLeftBorderWidth, borderValue);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  cudaMemcpy(dstData.data(), d_dst, dstDataSize * sizeof(Npp16s), cudaMemcpyDeviceToHost);

  // Validate结果
  verifyCopyConstBorder_C1(srcData, dstData, borderValue);

  // 清理GPU内存
  cudaFree(d_src);
  cudaFree(d_dst);
}

// 测试32位浮点单通道常量边界拷贝
TEST_F(NPPICopyConstBorderTest, CopyConstBorder_32f_C1R) {
  size_t srcDataSize = srcWidth * srcHeight;
  size_t dstDataSize = dstWidth * dstHeight;
  std::vector<Npp32f> srcData(srcDataSize), dstData(dstDataSize);

  // 生成浮点源数据
  for (size_t i = 0; i < srcDataSize; i++) {
    srcData[i] = static_cast<Npp32f>((rand() / (float)RAND_MAX) * 2.0f - 1.0f); // [-1, 1]
  }
  Npp32f borderValue = -0.5f;

  // 分配GPU内存
  Npp32f *d_src, *d_dst;
  int nSrcStep = srcWidth * sizeof(Npp32f);
  int nDstStep = dstWidth * sizeof(Npp32f);

  cudaMalloc(&d_src, srcDataSize * sizeof(Npp32f));
  cudaMalloc(&d_dst, dstDataSize * sizeof(Npp32f));

  // 拷贝数据到GPU
  cudaMemcpy(d_src, srcData.data(), srcDataSize * sizeof(Npp32f), cudaMemcpyHostToDevice);

  // CallNPP函数
  NppStatus status = nppiCopyConstBorder_32f_C1R(d_src, nSrcStep, oSrcSizeROI, d_dst, nDstStep, oDstSizeROI,
                                                 nTopBorderHeight, nLeftBorderWidth, borderValue);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  cudaMemcpy(dstData.data(), d_dst, dstDataSize * sizeof(Npp32f), cudaMemcpyDeviceToHost);

  // Validate结果（浮点数需要精确匹配）
  verifyCopyConstBorder_C1(srcData, dstData, borderValue);

  // 清理GPU内存
  cudaFree(d_src);
  cudaFree(d_dst);
}

// 测试带上下文的版本
TEST_F(NPPICopyConstBorderTest, CopyConstBorder_8u_C1R_Ctx) {
  size_t srcDataSize = srcWidth * srcHeight;
  size_t dstDataSize = dstWidth * dstHeight;
  std::vector<Npp8u> srcData(srcDataSize), dstData(dstDataSize);

  // 生成源数据
  generateRandomData(srcData, srcDataSize, Npp8u(0), Npp8u(255));
  Npp8u borderValue = 64;

  // 分配GPU内存
  Npp8u *d_src, *d_dst;
  int nSrcStep = srcWidth * sizeof(Npp8u);
  int nDstStep = dstWidth * sizeof(Npp8u);

  cudaMalloc(&d_src, srcDataSize * sizeof(Npp8u));
  cudaMalloc(&d_dst, dstDataSize * sizeof(Npp8u));

  // 拷贝数据到GPU
  cudaMemcpy(d_src, srcData.data(), srcDataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // 创建流上下文
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0; // 使用Default stream

  // CallNPP函数
  NppStatus status = nppiCopyConstBorder_8u_C1R_Ctx(d_src, nSrcStep, oSrcSizeROI, d_dst, nDstStep, oDstSizeROI,
                                                    nTopBorderHeight, nLeftBorderWidth, borderValue, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  cudaMemcpy(dstData.data(), d_dst, dstDataSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);

  // Validate结果
  verifyCopyConstBorder_C1(srcData, dstData, borderValue);

  // 清理GPU内存
  cudaFree(d_src);
  cudaFree(d_dst);
}

// 测试零边界情况
TEST_F(NPPICopyConstBorderTest, CopyConstBorder_ZeroBorders) {
  // 设置零边界
  nTopBorderHeight = 0;
  nLeftBorderWidth = 0;
  dstWidth = srcWidth;
  dstHeight = srcHeight;
  oDstSizeROI.width = dstWidth;
  oDstSizeROI.height = dstHeight;

  size_t dataSize = srcWidth * srcHeight;
  std::vector<Npp8u> srcData(dataSize), dstData(dataSize);

  // 生成源数据
  generateRandomData(srcData, dataSize, Npp8u(0), Npp8u(255));
  Npp8u borderValue = 255; // 应该不会使用

  // 分配GPU内存
  Npp8u *d_src, *d_dst;
  int nSrcStep = srcWidth * sizeof(Npp8u);
  int nDstStep = dstWidth * sizeof(Npp8u);

  cudaMalloc(&d_src, dataSize * sizeof(Npp8u));
  cudaMalloc(&d_dst, dataSize * sizeof(Npp8u));

  // 拷贝数据到GPU
  cudaMemcpy(d_src, srcData.data(), dataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);

  // CallNPP函数
  NppStatus status = nppiCopyConstBorder_8u_C1R(d_src, nSrcStep, oSrcSizeROI, d_dst, nDstStep, oDstSizeROI,
                                                nTopBorderHeight, nLeftBorderWidth, borderValue);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 拷贝结果回主机
  cudaMemcpy(dstData.data(), d_dst, dataSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);

  // Validate结果（应该完全相同）
  for (size_t i = 0; i < dataSize; i++) {
    EXPECT_EQ(dstData[i], srcData[i]) << "Data mismatch at index " << i;
  }

  // 清理GPU内存
  cudaFree(d_src);
  cudaFree(d_dst);
}
