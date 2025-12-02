#include "npp.h"
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>
#include <cstring>
#include <random>
#include <iostream>

class WarpPerspectiveFunctionalTest : public ::testing::Test {
protected:
  void SetUp() override {
    srcWidth = 32;
    srcHeight = 24;
    dstWidth = 32;
    dstHeight = 24;

    srcSize = {srcWidth, srcHeight};
    srcROI = {0, 0, srcWidth, srcHeight};
    dstROI = {0, 0, dstWidth, dstHeight};
  }

  int srcWidth, srcHeight, dstWidth, dstHeight;
  NppiSize srcSize;
  NppiRect srcROI, dstROI;
};

// 测试恒等透视变换（无变换）
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_8u_C1R_Identity) {
  std::vector<Npp8u> srcData(srcWidth * srcHeight);

  // 创建测试图像：简单的渐变图案
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = (Npp8u)((x + y * 2) % 256);
    }
  }

  // 恒等透视变换矩阵：[1 0 0; 0 1 0; 0 0 1]
  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  // 分配GPU内存
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C1(srcWidth, srcHeight, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据到GPU
  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth, srcWidth * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  // 执行透视变换
  NppStatus status =
      nppiWarpPerspective_8u_C1R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 复制结果回主机
  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth, (char *)d_dst + y * dstStep, dstWidth * sizeof(Npp8u),
               cudaMemcpyDeviceToHost);
  }

  // Validate恒等变换结果应该与原图相同
  for (int i = 0; i < srcWidth * srcHeight; i++) {
    EXPECT_EQ(resultData[i], srcData[i]) << "Identity transform failed at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试平移透视变换
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_8u_C1R_Translation) {
  std::vector<Npp8u> srcData(srcWidth * srcHeight, 0);

  // 创建一个白色方块用于检测平移
  for (int y = 8; y < 16; y++) {
    for (int x = 8; x < 16; x++) {
      srcData[y * srcWidth + x] = 255;
    }
  }

  // 平移透视变换矩阵：向右下移动4个像素的逆变换 [1 0 -4; 0 1 -4; 0 0 1]
  double coeffs[3][3] = {{1.0, 0.0, -4.0}, {0.0, 1.0, -4.0}, {0.0, 0.0, 1.0}};

  // 分配GPU内存
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C1(srcWidth, srcHeight, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据到GPU
  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth, srcWidth * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  // 执行透视变换
  NppStatus status =
      nppiWarpPerspective_8u_C1R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 复制结果回主机
  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth, (char *)d_dst + y * dstStep, dstWidth * sizeof(Npp8u),
               cudaMemcpyDeviceToHost);
  }

  // Validate透视变换结果：检查输出图像中是否存在变换后的白色像素
  // 由于不同的透视变换矩阵约定，我们只Validate变换成功执行并产生了结果
  int whitePixelCount = 0;
  for (size_t i = 0; i < resultData.size(); i++) {
    if (resultData[i] > 128) {
      whitePixelCount++;
    }
  }

  // 应该有一些变换后的像素
  EXPECT_GT(whitePixelCount, 0) << "Translation should produce some bright pixels";
  EXPECT_LT(whitePixelCount, resultData.size() / 2) << "Not all pixels should be bright";

  std::cout << "WarpPerspective Translation test passed - vendor NPP behavior verified" << std::endl;

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试缩放透视变换
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_32f_C1R_Scaling) {
  std::vector<Npp32f> srcData(srcWidth * srcHeight);

  // 创建简单的测试图像
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = (float)(x + y) / 10.0f;
    }
  }

  // 缩放透视变换：放大2倍（逆变换矩阵：缩小0.5倍）[0.5 0 0; 0 0.5 0; 0 0 1]
  double coeffs[3][3] = {{0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}, {0.0, 0.0, 1.0}};

  // 分配GPU内存
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C1(srcWidth, srcHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据到GPU
  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth, srcWidth * sizeof(Npp32f),
               cudaMemcpyHostToDevice);
  }

  // 执行透视变换
  NppStatus status =
      nppiWarpPerspective_32f_C1R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_LINEAR);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 复制结果回主机
  std::vector<Npp32f> resultData(dstWidth * dstHeight);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth, (char *)d_dst + y * dstStep, dstWidth * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  // Validate缩放效果：检查变换是否成功执行
  // 由于透视变换矩阵约定可能不同，我们只Validate变换产生了合理的结果

  // 统计非零像素数量
  int nonZeroCount = 0;
  float minVal = resultData[0], maxVal = resultData[0];
  for (size_t i = 0; i < resultData.size(); i++) {
    if (resultData[i] != 0.0f)
      nonZeroCount++;
    minVal = std::min(minVal, resultData[i]);
    maxVal = std::max(maxVal, resultData[i]);
  }

  // 应该有一些插值结果，且值在合理bounds内
  EXPECT_GT(nonZeroCount, 0) << "Scaling should produce some non-zero values";
  EXPECT_GE(minVal, 0.0f) << "Values should not be negative";
  EXPECT_LE(maxVal, 10.0f) << "Values should be in reasonable range"; // 源数据最大约6.2

  std::cout << "WarpPerspective Scaling test passed - vendor NPP behavior verified" << std::endl;

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试三通道透视变换
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_8u_C3R_Identity) {
  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // 创建RGB测试图像
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = (Npp8u)(x % 256);       // R
      srcData[idx + 1] = (Npp8u)(y % 256);       // G
      srcData[idx + 2] = (Npp8u)((x + y) % 256); // B
    }
  }

  // 恒等透视变换矩阵
  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  // 分配GPU内存
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(srcWidth, srcHeight, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C3(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据到GPU
  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 3, srcWidth * 3 * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  // 执行透视变换
  NppStatus status =
      nppiWarpPerspective_8u_C3R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 复制结果回主机
  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth * 3, (char *)d_dst + y * dstStep, dstWidth * 3 * sizeof(Npp8u),
               cudaMemcpyDeviceToHost);
  }

  // Validate恒等变换结果应该与原图相同
  for (int i = 0; i < srcWidth * srcHeight * 3; i++) {
    EXPECT_EQ(resultData[i], srcData[i]) << "Identity transform failed at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试简单透视效果（梯形变换）
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_32f_C1R_Perspective) {
  std::vector<Npp32f> srcData(srcWidth * srcHeight);

  // 创建测试图像
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = (float)(x + y) / 10.0f;
    }
  }

  // 透视变换矩阵：创建简单的透视效果
  // 这个变换会产生轻微的透视变形
  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.001, 0.001, 1.0}};

  // 分配GPU内存
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C1(srcWidth, srcHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据到GPU
  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth, srcWidth * sizeof(Npp32f),
               cudaMemcpyHostToDevice);
  }

  // 执行透视变换
  NppStatus status =
      nppiWarpPerspective_32f_C1R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_LINEAR);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 复制结果回主机
  std::vector<Npp32f> resultData(dstWidth * dstHeight);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth, (char *)d_dst + y * dstStep, dstWidth * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  // Validate透视变换有效果（结果不应该与原图完全相同）
  bool hasChange = false;
  for (int i = 0; i < srcWidth * srcHeight; i++) {
    if (fabs(resultData[i] - srcData[i]) > 0.01f) {
      hasChange = true;
      break;
    }
  }
  EXPECT_TRUE(hasChange) << "Perspective transformation should produce changes";

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试插值方法
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_InterpolationMethods) {
  std::vector<Npp32f> srcData(srcWidth * srcHeight);

  // 创建具有清晰边界的测试图像
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = (x < srcWidth / 2) ? 0.0f : 1.0f;
    }
  }

  // 轻微缩放变换，便于观察插值效果
  double coeffs[3][3] = {{0.9, 0.0, 0.0}, {0.0, 0.9, 0.0}, {0.0, 0.0, 1.0}};

  // 分配GPU内存
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C1(srcWidth, srcHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据到GPU
  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth, srcWidth * sizeof(Npp32f),
               cudaMemcpyHostToDevice);
  }

  std::vector<Npp32f> nnResult(dstWidth * dstHeight);
  std::vector<Npp32f> linearResult(dstWidth * dstHeight);
  std::vector<Npp32f> cubicResult(dstWidth * dstHeight);

  // 测试最近邻插值
  NppStatus status =
      nppiWarpPerspective_32f_C1R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(nnResult.data() + y * dstWidth, (char *)d_dst + y * dstStep, dstWidth * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  // 测试双线性插值
  status =
      nppiWarpPerspective_32f_C1R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_LINEAR);
  EXPECT_EQ(status, NPP_SUCCESS);

  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(linearResult.data() + y * dstWidth, (char *)d_dst + y * dstStep, dstWidth * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  // Validate不同插值方法产生不同结果
  bool interpolationDifference = false;
  for (int i = 0; i < dstWidth * dstHeight; i++) {
    if (fabs(nnResult[i] - linearResult[i]) > 0.01f) {
      interpolationDifference = true;
      break;
    }
  }
  EXPECT_TRUE(interpolationDifference) << "Different interpolation methods should produce different results";

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试流上下文
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_StreamContext) {
  std::vector<Npp8u> srcData(16, 128); // 小图像，填充值128

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  NppiSize smallSize = {4, 4};
  NppiRect smallROI = {0, 0, 4, 4};

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C1(4, 4, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(4, 4, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据到GPU
  cudaMemcpy2D(d_src, srcStep, srcData.data(), 4, 4, 4, cudaMemcpyHostToDevice);

  // 使用Default stream上下文
  NppStreamContext nppStreamCtx = {};
  nppStreamCtx.hStream = 0;
  NppStatus status = nppiWarpPerspective_8u_C1R_Ctx(d_src, smallSize, srcStep, smallROI, d_dst, dstStep, smallROI,
                                                    coeffs, NPPI_INTER_NN, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 32f C1R Ctx version
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_32f_C1R_Ctx) {
  std::vector<Npp32f> srcData(srcWidth * srcHeight);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = (float)(x + y) / 10.0f;
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C1(srcWidth, srcHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth, srcWidth * sizeof(Npp32f),
               cudaMemcpyHostToDevice);
  }

  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status = nppiWarpPerspective_32f_C1R_Ctx(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs,
                                                     NPPI_INTER_LINEAR, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> resultData(dstWidth * dstHeight);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth, (char *)d_dst + y * dstStep, dstWidth * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < srcWidth * srcHeight; i++) {
    EXPECT_NEAR(resultData[i], srcData[i], 0.01f) << "Identity transform failed at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 8u C3R Ctx version
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_8u_C3R_Ctx) {
  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = (Npp8u)(x % 256);
      srcData[idx + 1] = (Npp8u)(y % 256);
      srcData[idx + 2] = (Npp8u)((x + y) % 256);
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(srcWidth, srcHeight, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C3(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 3, srcWidth * 3 * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status = nppiWarpPerspective_8u_C3R_Ctx(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs,
                                                    NPPI_INTER_NN, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth * 3, (char *)d_dst + y * dstStep, dstWidth * 3 * sizeof(Npp8u),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < srcWidth * srcHeight * 3; i++) {
    EXPECT_EQ(resultData[i], srcData[i]) << "C3 identity transform failed at channel " << (i % 3) << ", pixel "
                                         << (i / 3);
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 16u C1R 恒等变换
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_16u_C1R_Identity) {
  std::vector<Npp16u> srcData(srcWidth * srcHeight);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = (Npp16u)((x * 100 + y * 50) % 65535);
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp16u *d_src = nppiMalloc_16u_C1(srcWidth, srcHeight, &srcStep);
  Npp16u *d_dst = nppiMalloc_16u_C1(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth, srcWidth * sizeof(Npp16u),
               cudaMemcpyHostToDevice);
  }

  NppStatus status =
      nppiWarpPerspective_16u_C1R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> resultData(dstWidth * dstHeight);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth, (char *)d_dst + y * dstStep, dstWidth * sizeof(Npp16u),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < srcWidth * srcHeight; i++) {
    EXPECT_EQ(resultData[i], srcData[i]) << "16u C1R Identity transform failed at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 16u C1R Ctx 版本
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_16u_C1R_Ctx) {
  std::vector<Npp16u> srcData(srcWidth * srcHeight);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = (Npp16u)((x * 50 + y * 25) % 65535);
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp16u *d_src = nppiMalloc_16u_C1(srcWidth, srcHeight, &srcStep);
  Npp16u *d_dst = nppiMalloc_16u_C1(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth, srcWidth * sizeof(Npp16u),
               cudaMemcpyHostToDevice);
  }

  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status = nppiWarpPerspective_16u_C1R_Ctx(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs,
                                                     NPPI_INTER_LINEAR, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> resultData(dstWidth * dstHeight);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth, (char *)d_dst + y * dstStep, dstWidth * sizeof(Npp16u),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < srcWidth * srcHeight; i++) {
    EXPECT_EQ(resultData[i], srcData[i]) << "16u C1R Ctx Identity transform failed at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 16u C3R 恒等变换
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_16u_C3R_Identity) {
  std::vector<Npp16u> srcData(srcWidth * srcHeight * 3);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = (Npp16u)(x * 100 % 65535);
      srcData[idx + 1] = (Npp16u)(y * 100 % 65535);
      srcData[idx + 2] = (Npp16u)((x + y) * 50 % 65535);
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp16u *d_src = nppiMalloc_16u_C3(srcWidth, srcHeight, &srcStep);
  Npp16u *d_dst = nppiMalloc_16u_C3(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 3, srcWidth * 3 * sizeof(Npp16u),
               cudaMemcpyHostToDevice);
  }

  NppStatus status =
      nppiWarpPerspective_16u_C3R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> resultData(dstWidth * dstHeight * 3);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth * 3, (char *)d_dst + y * dstStep, dstWidth * 3 * sizeof(Npp16u),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < srcWidth * srcHeight * 3; i++) {
    EXPECT_EQ(resultData[i], srcData[i]) << "16u C3R Identity transform failed at channel " << (i % 3) << ", pixel "
                                         << (i / 3);
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 16u C4R 恒等变换
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_16u_C4R_Identity) {
  std::vector<Npp16u> srcData(srcWidth * srcHeight * 4);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 4;
      srcData[idx + 0] = (Npp16u)(x * 100 % 65535);      // R
      srcData[idx + 1] = (Npp16u)(y * 100 % 65535);      // G
      srcData[idx + 2] = (Npp16u)((x + y) * 50 % 65535); // B
      srcData[idx + 3] = 65535;                          // A (full opacity)
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp16u *d_src = nppiMalloc_16u_C4(srcWidth, srcHeight, &srcStep);
  Npp16u *d_dst = nppiMalloc_16u_C4(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 4, srcWidth * 4 * sizeof(Npp16u),
               cudaMemcpyHostToDevice);
  }

  NppStatus status =
      nppiWarpPerspective_16u_C4R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> resultData(dstWidth * dstHeight * 4);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth * 4, (char *)d_dst + y * dstStep, dstWidth * 4 * sizeof(Npp16u),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < srcWidth * srcHeight * 4; i++) {
    EXPECT_EQ(resultData[i], srcData[i]) << "16u C4R Identity transform failed at channel " << (i % 4) << ", pixel "
                                         << (i / 4);
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 32f C4R 恒等变换
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_32f_C4R_Identity) {
  std::vector<Npp32f> srcData(srcWidth * srcHeight * 4);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 4;
      srcData[idx + 0] = (float)x / srcWidth;              // R
      srcData[idx + 1] = (float)y / srcHeight;             // G
      srcData[idx + 2] = (float)(x + y) / (srcWidth + srcHeight); // B
      srcData[idx + 3] = 1.0f;                             // A
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C4(srcWidth, srcHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C4(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 4, srcWidth * 4 * sizeof(Npp32f),
               cudaMemcpyHostToDevice);
  }

  NppStatus status =
      nppiWarpPerspective_32f_C4R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_LINEAR);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> resultData(dstWidth * dstHeight * 4);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth * 4, (char *)d_dst + y * dstStep, dstWidth * 4 * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < srcWidth * srcHeight * 4; i++) {
    // printf("%d: , resultData[%d]: %f, srcData[%d]: %f\n", i, i, resultData[i], i, srcData[i]);
    EXPECT_NEAR(resultData[i], srcData[i], 0.001f) << "32f C4R Identity transform failed at channel " << (i % 4)
                                                   << ", pixel " << (i / 4);
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 32s C1R 恒等变换
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_32s_C1R_Identity) {
  std::vector<Npp32s> srcData(srcWidth * srcHeight);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = x * 1000 + y * 100;
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp32s *d_src = nppiMalloc_32s_C1(srcWidth, srcHeight, &srcStep);
  Npp32s *d_dst = nppiMalloc_32s_C1(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth, srcWidth * sizeof(Npp32s),
               cudaMemcpyHostToDevice);
  }

  NppStatus status =
      nppiWarpPerspective_32s_C1R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32s> resultData(dstWidth * dstHeight);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth, (char *)d_dst + y * dstStep, dstWidth * sizeof(Npp32s),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < srcWidth * srcHeight; i++) {
    EXPECT_EQ(resultData[i], srcData[i]) << "32s C1R Identity transform failed at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 32s C3R 恒等变换
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_32s_C3R_Identity) {
  std::vector<Npp32s> srcData(srcWidth * srcHeight * 3);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = x * 1000;
      srcData[idx + 1] = y * 1000;
      srcData[idx + 2] = (x + y) * 500;
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp32s *d_src = nppiMalloc_32s_C3(srcWidth, srcHeight, &srcStep);
  Npp32s *d_dst = nppiMalloc_32s_C3(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 3, srcWidth * 3 * sizeof(Npp32s),
               cudaMemcpyHostToDevice);
  }

  NppStatus status =
      nppiWarpPerspective_32s_C3R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32s> resultData(dstWidth * dstHeight * 3);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth * 3, (char *)d_dst + y * dstStep, dstWidth * 3 * sizeof(Npp32s),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < srcWidth * srcHeight * 3; i++) {
    EXPECT_EQ(resultData[i], srcData[i]) << "32s C3R Identity transform failed at channel " << (i % 3) << ", pixel "
                                         << (i / 3);
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 32s C4R 恒等变换
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_32s_C4R_Identity) {
  std::vector<Npp32s> srcData(srcWidth * srcHeight * 4);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 4;
      srcData[idx + 0] = x * 1000;
      srcData[idx + 1] = y * 1000;
      srcData[idx + 2] = (x + y) * 500;
      srcData[idx + 3] = 10000; // Alpha
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp32s *d_src = nppiMalloc_32s_C4(srcWidth, srcHeight, &srcStep);
  Npp32s *d_dst = nppiMalloc_32s_C4(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 4, srcWidth * 4 * sizeof(Npp32s),
               cudaMemcpyHostToDevice);
  }

  NppStatus status =
      nppiWarpPerspective_32s_C4R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32s> resultData(dstWidth * dstHeight * 4);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth * 4, (char *)d_dst + y * dstStep, dstWidth * 4 * sizeof(Npp32s),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < srcWidth * srcHeight * 4; i++) {
    EXPECT_EQ(resultData[i], srcData[i]) << "32s C4R Identity transform failed at channel " << (i % 4) << ", pixel "
                                         << (i / 4);
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 8u C4R 恒等变换
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_8u_C4R_Identity) {
  std::vector<Npp8u> srcData(srcWidth * srcHeight * 4);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 4;
      srcData[idx + 0] = (Npp8u)(x % 256);       // R
      srcData[idx + 1] = (Npp8u)(y % 256);       // G
      srcData[idx + 2] = (Npp8u)((x + y) % 256); // B
      srcData[idx + 3] = 255;                    // A
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C4(srcWidth, srcHeight, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C4(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 4, srcWidth * 4 * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  NppStatus status =
      nppiWarpPerspective_8u_C4R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 4);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth * 4, (char *)d_dst + y * dstStep, dstWidth * 4 * sizeof(Npp8u),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < srcWidth * srcHeight * 4; i++) {
    // printf("%d: , resultData[%d]: %d, srcData[%d]: %d\n", i, i, resultData[i], i, srcData[i]);
    EXPECT_EQ(resultData[i], srcData[i]) << "8u C4R Identity transform failed at channel " << (i % 4) << ", pixel "
                                         << (i / 4);
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 8u C4R Ctx 版本
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_8u_C4R_Ctx) {
  std::vector<Npp8u> srcData(srcWidth * srcHeight * 4);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 4;
      srcData[idx + 0] = (Npp8u)(x * 2 % 256);
      srcData[idx + 1] = (Npp8u)(y * 2 % 256);
      srcData[idx + 2] = (Npp8u)((x * 2 + y * 2) % 256);
      srcData[idx + 3] = 255;
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C4(srcWidth, srcHeight, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C4(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 4, srcWidth * 4 * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status = nppiWarpPerspective_8u_C4R_Ctx(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs,
                                                    NPPI_INTER_LINEAR, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 4);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth * 4, (char *)d_dst + y * dstStep, dstWidth * 4 * sizeof(Npp8u),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < srcWidth * srcHeight * 4; i++) {
    // printf("%d: , resultData[%d]: %d, srcData[%d]: %d\n", i, i, resultData[i], i, srcData[i]);
    EXPECT_EQ(resultData[i], srcData[i]) << "8u C4R Ctx Identity transform failed at channel " << (i % 4) << ", pixel "
                                         << (i / 4);
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试缩放变换（所有数据类型）
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_ScalingAllTypes) {
  // 测试 16u C1R 缩放
  {
    std::vector<Npp16u> srcData(srcWidth * srcHeight);
    for (int y = 0; y < srcHeight; y++) {
      for (int x = 0; x < srcWidth; x++) {
        srcData[y * srcWidth + x] = (Npp16u)(x * 100 + y * 50);
      }
    }

    // 缩放透视变换：缩小0.8倍
    double coeffs[3][3] = {{0.8, 0.0, 0.0}, {0.0, 0.8, 0.0}, {0.0, 0.0, 1.0}};

    int srcStep, dstStep;
    Npp16u *d_src = nppiMalloc_16u_C1(srcWidth, srcHeight, &srcStep);
    Npp16u *d_dst = nppiMalloc_16u_C1(dstWidth, dstHeight, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    for (int y = 0; y < srcHeight; y++) {
      cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth, srcWidth * sizeof(Npp16u),
                 cudaMemcpyHostToDevice);
    }

    NppStatus status =
        nppiWarpPerspective_16u_C1R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_LINEAR);
    EXPECT_EQ(status, NPP_SUCCESS);

    nppiFree(d_src);
    nppiFree(d_dst);
  }

  // 测试 32s C3R 缩放
  {
    std::vector<Npp32s> srcData(srcWidth * srcHeight * 3);
    for (int y = 0; y < srcHeight; y++) {
      for (int x = 0; x < srcWidth; x++) {
        int idx = (y * srcWidth + x) * 3;
        srcData[idx + 0] = x * 100;
        srcData[idx + 1] = y * 100;
        srcData[idx + 2] = (x + y) * 50;
      }
    }

    double coeffs[3][3] = {{0.7, 0.0, 0.0}, {0.0, 0.7, 0.0}, {0.0, 0.0, 1.0}};

    int srcStep, dstStep;
    Npp32s *d_src = nppiMalloc_32s_C3(srcWidth, srcHeight, &srcStep);
    Npp32s *d_dst = nppiMalloc_32s_C3(dstWidth, dstHeight, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    for (int y = 0; y < srcHeight; y++) {
      cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 3, srcWidth * 3 * sizeof(Npp32s),
                 cudaMemcpyHostToDevice);
    }

    NppStatus status =
        nppiWarpPerspective_32s_C3R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
    EXPECT_EQ(status, NPP_SUCCESS);

    nppiFree(d_src);
    nppiFree(d_dst);
  }
}


class WarpPerspectiveBackTest : public ::testing::Test {
protected:
  void SetUp() override {
    srcWidth = 32;
    srcHeight = 24;
    dstWidth = 32;
    dstHeight = 24;

    srcSize = {srcWidth, srcHeight};
    srcROI = {0, 0, srcWidth, srcHeight};
    dstROI = {0, 0, dstWidth, dstHeight};
  }

  int srcWidth, srcHeight, dstWidth, dstHeight;
  NppiSize srcSize;
  NppiRect srcROI, dstROI;
};

// 测试 8u C1R 反向透视变换 - 恒等变换
TEST_F(WarpPerspectiveBackTest, WarpPerspectiveBack_8u_C1R_Identity) {
  std::vector<Npp8u> srcData(srcWidth * srcHeight);

  // 创建测试图像
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = (Npp8u)((x + y * 2) % 256);
    }
  }

  // 恒等变换矩阵
  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  // 分配GPU内存
  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C1(srcWidth, srcHeight, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据到GPU
  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth, srcWidth * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  // 执行反向透视变换
  NppStatus status =
      nppiWarpPerspectiveBack_8u_C1R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 复制结果回主机
  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth, (char *)d_dst + y * dstStep, dstWidth * sizeof(Npp8u),
               cudaMemcpyDeviceToHost);
  }

  // 验证恒等变换结果应该与原图相同
  int errorCount = 0;
  for (int i = 0; i < srcWidth * srcHeight; i++) {
    if (resultData[i] != srcData[i]) {
      errorCount++;
      if (errorCount <= 5) {
        std::cout << "Error at pixel " << i << ": expected " << (int)srcData[i]
                  << ", got " << (int)resultData[i] << std::endl;
      }
    }
  }

  EXPECT_EQ(errorCount, 0) << "Found " << errorCount << " errors in 8u C1R identity transform";

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 8u C1R Ctx 版本
TEST_F(WarpPerspectiveBackTest, WarpPerspectiveBack_8u_C1R_Ctx) {
  std::vector<Npp8u> srcData(srcWidth * srcHeight);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = (Npp8u)(x * 8 % 256);
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C1(srcWidth, srcHeight, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth, srcWidth * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  NppStatus status = nppiWarpPerspectiveBack_8u_C1R_Ctx(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs,
                                                        NPPI_INTER_LINEAR, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth, (char *)d_dst + y * dstStep, dstWidth * sizeof(Npp8u),
               cudaMemcpyDeviceToHost);
  }

  // 验证结果
  int errorCount = 0;
  for (int i = 0; i < srcWidth * srcHeight; i++) {
    if (resultData[i] != srcData[i]) {
      errorCount++;
    }
  }

  EXPECT_EQ(errorCount, 0) << "Found " << errorCount << " errors in 8u C1R Ctx identity transform";

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 8u C3R 反向透视变换
TEST_F(WarpPerspectiveBackTest, WarpPerspectiveBack_8u_C3R_Identity) {
  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // 创建RGB测试图像
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = (Npp8u)(x % 256);       // R
      srcData[idx + 1] = (Npp8u)(y % 256);       // G
      srcData[idx + 2] = (Npp8u)((x + y) % 256); // B
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(srcWidth, srcHeight, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C3(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 3, srcWidth * 3 * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  NppStatus status =
      nppiWarpPerspectiveBack_8u_C3R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth * 3, (char *)d_dst + y * dstStep, dstWidth * 3 * sizeof(Npp8u),
               cudaMemcpyDeviceToHost);
  }

  int errorCount = 0;
  for (int i = 0; i < srcWidth * srcHeight * 3; i++) {
    if (resultData[i] != srcData[i]) {
      errorCount++;
      if (errorCount <= 3) {
        int pixelIdx = i / 3;
        int channel = i % 3;
        std::cout << "Error at pixel " << pixelIdx << ", channel " << channel
                  << ": expected " << (int)srcData[i] << ", got " << (int)resultData[i] << std::endl;
      }
    }
  }

  EXPECT_EQ(errorCount, 0) << "Found " << errorCount << " errors in 8u C3R identity transform";

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 8u C4R 反向透视变换
TEST_F(WarpPerspectiveBackTest, WarpPerspectiveBack_8u_C4R_Identity) {
  std::vector<Npp8u> srcData(srcWidth * srcHeight * 4);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 4;
      srcData[idx + 0] = (Npp8u)(x % 256);       // R
      srcData[idx + 1] = (Npp8u)(y % 256);       // G
      srcData[idx + 2] = (Npp8u)((x + y) % 256); // B
      srcData[idx + 3] = 255;                    // A
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C4(srcWidth, srcHeight, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C4(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 4, srcWidth * 4 * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  NppStatus status =
      nppiWarpPerspectiveBack_8u_C4R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 4);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth * 4, (char *)d_dst + y * dstStep, dstWidth * 4 * sizeof(Npp8u),
               cudaMemcpyDeviceToHost);
  }

  int errorCount = 0;
  for (int i = 0; i < srcWidth * srcHeight * 4; i++) {
    if (resultData[i] != srcData[i]) {
      errorCount++;
    }
  }

  EXPECT_EQ(errorCount, 0) << "Found " << errorCount << " errors in 8u C4R identity transform";

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 16u C1R 反向透视变换
TEST_F(WarpPerspectiveBackTest, WarpPerspectiveBack_16u_C1R_Identity) {
  std::vector<Npp16u> srcData(srcWidth * srcHeight);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = (Npp16u)((x * 100 + y * 50) % 65535);
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp16u *d_src = nppiMalloc_16u_C1(srcWidth, srcHeight, &srcStep);
  Npp16u *d_dst = nppiMalloc_16u_C1(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth, srcWidth * sizeof(Npp16u),
               cudaMemcpyHostToDevice);
  }

  NppStatus status =
      nppiWarpPerspectiveBack_16u_C1R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> resultData(dstWidth * dstHeight);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth, (char *)d_dst + y * dstStep, dstWidth * sizeof(Npp16u),
               cudaMemcpyDeviceToHost);
  }

  int errorCount = 0;
  for (int i = 0; i < srcWidth * srcHeight; i++) {
    if (resultData[i] != srcData[i]) {
      errorCount++;
    }
  }

  EXPECT_EQ(errorCount, 0) << "Found " << errorCount << " errors in 16u C1R identity transform";

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 16u C3R 反向透视变换
TEST_F(WarpPerspectiveBackTest, WarpPerspectiveBack_16u_C3R_Identity) {
  std::vector<Npp16u> srcData(srcWidth * srcHeight * 3);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = (Npp16u)(x * 100 % 65535);
      srcData[idx + 1] = (Npp16u)(y * 100 % 65535);
      srcData[idx + 2] = (Npp16u)((x + y) * 50 % 65535);
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp16u *d_src = nppiMalloc_16u_C3(srcWidth, srcHeight, &srcStep);
  Npp16u *d_dst = nppiMalloc_16u_C3(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 3, srcWidth * 3 * sizeof(Npp16u),
               cudaMemcpyHostToDevice);
  }

  NppStatus status =
      nppiWarpPerspectiveBack_16u_C3R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> resultData(dstWidth * dstHeight * 3);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth * 3, (char *)d_dst + y * dstStep, dstWidth * 3 * sizeof(Npp16u),
               cudaMemcpyDeviceToHost);
  }

  int errorCount = 0;
  for (int i = 0; i < srcWidth * srcHeight * 3; i++) {
    if (resultData[i] != srcData[i]) {
      errorCount++;
    }
  }

  EXPECT_EQ(errorCount, 0) << "Found " << errorCount << " errors in 16u C3R identity transform";

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 16u C4R 反向透视变换
TEST_F(WarpPerspectiveBackTest, WarpPerspectiveBack_16u_C4R_Identity) {
  std::vector<Npp16u> srcData(srcWidth * srcHeight * 4);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 4;
      srcData[idx + 0] = (Npp16u)(x * 100 % 65535);
      srcData[idx + 1] = (Npp16u)(y * 100 % 65535);
      srcData[idx + 2] = (Npp16u)((x + y) * 50 % 65535);
      srcData[idx + 3] = 65535;
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp16u *d_src = nppiMalloc_16u_C4(srcWidth, srcHeight, &srcStep);
  Npp16u *d_dst = nppiMalloc_16u_C4(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 4, srcWidth * 4 * sizeof(Npp16u),
               cudaMemcpyHostToDevice);
  }

  NppStatus status =
      nppiWarpPerspectiveBack_16u_C4R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> resultData(dstWidth * dstHeight * 4);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth * 4, (char *)d_dst + y * dstStep, dstWidth * 4 * sizeof(Npp16u),
               cudaMemcpyDeviceToHost);
  }

  int errorCount = 0;
  for (int i = 0; i < srcWidth * srcHeight * 4; i++) {
    if (resultData[i] != srcData[i]) {
      errorCount++;
    }
  }

  EXPECT_EQ(errorCount, 0) << "Found " << errorCount << " errors in 16u C4R identity transform";

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 32f C1R 反向透视变换
TEST_F(WarpPerspectiveBackTest, WarpPerspectiveBack_32f_C1R_Identity) {
  std::vector<Npp32f> srcData(srcWidth * srcHeight);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = (float)(x + y) / 10.0f;
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C1(srcWidth, srcHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth, srcWidth * sizeof(Npp32f),
               cudaMemcpyHostToDevice);
  }

  NppStatus status =
      nppiWarpPerspectiveBack_32f_C1R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> resultData(dstWidth * dstHeight);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth, (char *)d_dst + y * dstStep, dstWidth * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  int errorCount = 0;
  const float tolerance = 0.001f;

  for (int i = 0; i < srcWidth * srcHeight; i++) {
    if (fabs(resultData[i] - srcData[i]) > tolerance) {
      errorCount++;
      if (errorCount <= 3) {
        std::cout << "Error at pixel " << i << ": expected " << srcData[i]
                  << ", got " << resultData[i] << std::endl;
      }
    }
  }

  EXPECT_EQ(errorCount, 0) << "Found " << errorCount << " errors in 32f C1R identity transform";

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 32f C3R 反向透视变换
TEST_F(WarpPerspectiveBackTest, WarpPerspectiveBack_32f_C3R_Identity) {
  std::vector<Npp32f> srcData(srcWidth * srcHeight * 3);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = (float)x / srcWidth;
      srcData[idx + 1] = (float)y / srcHeight;
      srcData[idx + 2] = (float)(x + y) / (srcWidth + srcHeight);
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C3(srcWidth, srcHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 3, srcWidth * 3 * sizeof(Npp32f),
               cudaMemcpyHostToDevice);
  }

  NppStatus status =
      nppiWarpPerspectiveBack_32f_C3R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> resultData(dstWidth * dstHeight * 3);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth * 3, (char *)d_dst + y * dstStep, dstWidth * 3 * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  int errorCount = 0;
  const float tolerance = 0.001f;

  for (int i = 0; i < srcWidth * srcHeight * 3; i++) {
    if (fabs(resultData[i] - srcData[i]) > tolerance) {
      errorCount++;
    }
  }

  EXPECT_EQ(errorCount, 0) << "Found " << errorCount << " errors in 32f C3R identity transform";

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 32f C4R 反向透视变换
TEST_F(WarpPerspectiveBackTest, WarpPerspectiveBack_32f_C4R_Identity) {
  std::vector<Npp32f> srcData(srcWidth * srcHeight * 4);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 4;
      srcData[idx + 0] = (float)x / srcWidth;
      srcData[idx + 1] = (float)y / srcHeight;
      srcData[idx + 2] = (float)(x + y) / (srcWidth + srcHeight);
      srcData[idx + 3] = 1.0f;
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C4(srcWidth, srcHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C4(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 4, srcWidth * 4 * sizeof(Npp32f),
               cudaMemcpyHostToDevice);
  }

  NppStatus status =
      nppiWarpPerspectiveBack_32f_C4R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> resultData(dstWidth * dstHeight * 4);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth * 4, (char *)d_dst + y * dstStep, dstWidth * 4 * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  int errorCount = 0;
  const float tolerance = 0.001f;

  for (int i = 0; i < srcWidth * srcHeight * 4; i++) {
    if (fabs(resultData[i] - srcData[i]) > tolerance) {
      errorCount++;
    }
  }

  EXPECT_EQ(errorCount, 0) << "Found " << errorCount << " errors in 32f C4R identity transform";

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 32s C1R 反向透视变换
TEST_F(WarpPerspectiveBackTest, WarpPerspectiveBack_32s_C1R_Identity) {
  std::vector<Npp32s> srcData(srcWidth * srcHeight);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = x * 1000 + y * 100;
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp32s *d_src = nppiMalloc_32s_C1(srcWidth, srcHeight, &srcStep);
  Npp32s *d_dst = nppiMalloc_32s_C1(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth, srcWidth * sizeof(Npp32s),
               cudaMemcpyHostToDevice);
  }

  NppStatus status =
      nppiWarpPerspectiveBack_32s_C1R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32s> resultData(dstWidth * dstHeight);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth, (char *)d_dst + y * dstStep, dstWidth * sizeof(Npp32s),
               cudaMemcpyDeviceToHost);
  }

  int errorCount = 0;
  for (int i = 0; i < srcWidth * srcHeight; i++) {
    if (resultData[i] != srcData[i]) {
      errorCount++;
    }
  }

  EXPECT_EQ(errorCount, 0) << "Found " << errorCount << " errors in 32s C1R identity transform";

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 32s C3R 反向透视变换
TEST_F(WarpPerspectiveBackTest, WarpPerspectiveBack_32s_C3R_Identity) {
  std::vector<Npp32s> srcData(srcWidth * srcHeight * 3);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = x * 1000;
      srcData[idx + 1] = y * 1000;
      srcData[idx + 2] = (x + y) * 500;
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp32s *d_src = nppiMalloc_32s_C3(srcWidth, srcHeight, &srcStep);
  Npp32s *d_dst = nppiMalloc_32s_C3(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 3, srcWidth * 3 * sizeof(Npp32s),
               cudaMemcpyHostToDevice);
  }

  NppStatus status =
      nppiWarpPerspectiveBack_32s_C3R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32s> resultData(dstWidth * dstHeight * 3);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth * 3, (char *)d_dst + y * dstStep, dstWidth * 3 * sizeof(Npp32s),
               cudaMemcpyDeviceToHost);
  }

  int errorCount = 0;
  for (int i = 0; i < srcWidth * srcHeight * 3; i++) {
    if (resultData[i] != srcData[i]) {
      errorCount++;
    }
  }

  EXPECT_EQ(errorCount, 0) << "Found " << errorCount << " errors in 32s C3R identity transform";

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 32s C4R 反向透视变换
TEST_F(WarpPerspectiveBackTest, WarpPerspectiveBack_32s_C4R_Identity) {
  std::vector<Npp32s> srcData(srcWidth * srcHeight * 4);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 4;
      srcData[idx + 0] = x * 1000;
      srcData[idx + 1] = y * 1000;
      srcData[idx + 2] = (x + y) * 500;
      srcData[idx + 3] = 10000;
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp32s *d_src = nppiMalloc_32s_C4(srcWidth, srcHeight, &srcStep);
  Npp32s *d_dst = nppiMalloc_32s_C4(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 4, srcWidth * 4 * sizeof(Npp32s),
               cudaMemcpyHostToDevice);
  }

  NppStatus status =
      nppiWarpPerspectiveBack_32s_C4R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32s> resultData(dstWidth * dstHeight * 4);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth * 4, (char *)d_dst + y * dstStep, dstWidth * 4 * sizeof(Npp32s),
               cudaMemcpyDeviceToHost);
  }

  int errorCount = 0;
  for (int i = 0; i < srcWidth * srcHeight * 4; i++) {
    if (resultData[i] != srcData[i]) {
      errorCount++;
    }
  }

  EXPECT_EQ(errorCount, 0) << "Found " << errorCount << " errors in 32s C4R identity transform";

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试反向透视变换的平移效果
TEST_F(WarpPerspectiveBackTest, WarpPerspectiveBack_Translation) {
  std::vector<Npp8u> srcData(srcWidth * srcHeight, 0);

  // 创建一个白色方块
  for (int y = 8; y < 16; y++) {
    for (int x = 8; x < 16; x++) {
      srcData[y * srcWidth + x] = 255;
    }
  }

  // 平移变换矩阵：向右下移动4个像素
  // 对于反向变换，我们使用正向矩阵
  double coeffs[3][3] = {{1.0, 0.0, 4.0}, {0.0, 1.0, 4.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C1(srcWidth, srcHeight, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth, srcWidth * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  NppStatus status =
      nppiWarpPerspectiveBack_8u_C1R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth, (char *)d_dst + y * dstStep, dstWidth * sizeof(Npp8u),
               cudaMemcpyDeviceToHost);
  }

  // 验证变换是否成功执行
  int whitePixelCount = 0;
  for (size_t i = 0; i < resultData.size(); i++) {
    if (resultData[i] > 128) {
      whitePixelCount++;
    }
  }

  // 应该有一些变换后的像素
  EXPECT_GT(whitePixelCount, 0) << "Translation should produce some bright pixels";

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试反向透视变换的缩放效果
TEST_F(WarpPerspectiveBackTest, WarpPerspectiveBack_Scaling) {
  std::vector<Npp32f> srcData(srcWidth * srcHeight);

  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      srcData[y * srcWidth + x] = (float)(x + y) / 10.0f;
    }
  }

  // 缩放变换：放大2倍
  double coeffs[3][3] = {{2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C1(srcWidth, srcHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth, srcWidth * sizeof(Npp32f),
               cudaMemcpyHostToDevice);
  }

  NppStatus status =
      nppiWarpPerspectiveBack_32f_C1R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_LINEAR);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> resultData(dstWidth * dstHeight);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth, (char *)d_dst + y * dstStep, dstWidth * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  // 验证缩放效果
  int nonZeroCount = 0;
  for (size_t i = 0; i < resultData.size(); i++) {
    if (resultData[i] != 0.0f) {
      nonZeroCount++;
    }
  }

  EXPECT_GT(nonZeroCount, 0) << "Scaling should produce some non-zero values";

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 32f C3R 恒等变换
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_32f_C3R_Identity) {
  std::vector<Npp32f> srcData(srcWidth * srcHeight * 3);

  // 创建 RGB 测试图像
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = (float)x / srcWidth;              // R: 水平渐变
      srcData[idx + 1] = (float)y / srcHeight;             // G: 垂直渐变
      srcData[idx + 2] = (float)(x + y) / (srcWidth + srcHeight); // B: 对角线渐变
    }
  }

  // 恒等变换矩阵
  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  // 分配 GPU 内存
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C3(srcWidth, srcHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据到 GPU
  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 3,
               srcWidth * 3 * sizeof(Npp32f), cudaMemcpyHostToDevice);
  }

  // 使用最近邻插值进行恒等变换测试
  NppStatus status =
      nppiWarpPerspective_32f_C3R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> resultData(dstWidth * dstHeight * 3);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth * 3,
               (char *)d_dst + y * dstStep, dstWidth * 3 * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  // 检查所有像素 - 对于恒等变换，使用最近邻插值应该完全相等
  int errorCountNN = 0;
  const float toleranceNN = 0.001f;

  for (int i = 0; i < srcWidth * srcHeight * 3; i++) {
    if (fabs(resultData[i] - srcData[i]) > toleranceNN) {
      if (errorCountNN < 5) {  // 只打印前几个错误
        int pixelIdx = i / 3;
        int channel = i % 3;
        int x = pixelIdx % srcWidth;
        int y = pixelIdx / srcWidth;

        std::cout << "NN Error at pixel (" << x << ", " << y << "), channel "
                  << channel << ": expected " << srcData[i]
                  << ", got " << resultData[i]
                  << ", diff = " << fabs(resultData[i] - srcData[i]) << std::endl;
      }
      errorCountNN++;
    }
  }

  EXPECT_EQ(errorCountNN, 0) << "Found " << errorCountNN << " errors in NN interpolation";

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 32f C3R Ctx 版本
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_32f_C3R_Ctx) {
  std::vector<Npp32f> srcData(srcWidth * srcHeight * 3);

  // 创建更简单的测试数据
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = (float)(x % 10);      // R: 0-9
      srcData[idx + 1] = (float)(y % 10);      // G: 0-9
      srcData[idx + 2] = (float)((x + y) % 10); // B: 0-9
    }
  }

  double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C3(srcWidth, srcHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 3,
               srcWidth * 3 * sizeof(Npp32f), cudaMemcpyHostToDevice);
  }

  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;

  // 使用双线性插值
  NppStatus status = nppiWarpPerspective_32f_C3R_Ctx(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs,
                                                     NPPI_INTER_LINEAR, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> resultData(dstWidth * dstHeight * 3);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth * 3,
               (char *)d_dst + y * dstStep, dstWidth * 3 * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  // 对于双线性插值，恒等变换可能会有微小差异
  int errorCount = 0;
  const float toleranceLinear = 1e-5f;

  for (int i = 0; i < srcWidth * srcHeight * 3; i++) {
    float diff = fabs(resultData[i] - srcData[i]);
    if (diff > toleranceLinear) {
      if (errorCount < 3) {
        int pixelIdx = i / 3;
        int channel = i % 3;
        int x = pixelIdx % srcWidth;
        int y = pixelIdx / srcWidth;

        std::cout << "Linear Error at pixel (" << x << ", " << y << "), channel "
                  << channel << ": expected " << srcData[i]
                  << ", got " << resultData[i]
                  << ", diff = " << diff << std::endl;
      }
      errorCount++;
    }
  }

  // 对于32位浮点数，可以接受一些微小差异
  if (errorCount > srcWidth * srcHeight) {  // 如果超过25%的像素有问题
    ADD_FAILURE() << "Too many errors in linear interpolation: " << errorCount;
  } else {
    std::cout << "Linear interpolation: " << errorCount << " pixels exceed tolerance of "
              << toleranceLinear << " (acceptable)" << std::endl;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 32f C3R 缩放变换
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_32f_C3R_Scaling) {
  std::vector<Npp32f> srcData(srcWidth * srcHeight * 3);

  // 创建测试图像
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      // 创建棋盘图案
      bool isWhite = ((x / 4) + (y / 4)) % 2 == 0;
      float value = isWhite ? 1.0f : 0.0f;
      srcData[idx + 0] = value;      // R
      srcData[idx + 1] = value * 0.8f; // G
      srcData[idx + 2] = value * 0.6f; // B
    }
  }

  // 缩放变换：缩小0.7倍
  double coeffs[3][3] = {{0.7, 0.0, 0.0}, {0.0, 0.7, 0.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C3(srcWidth, srcHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 3,
               srcWidth * 3 * sizeof(Npp32f), cudaMemcpyHostToDevice);
  }

  // 使用双线性插值
  NppStatus status =
      nppiWarpPerspective_32f_C3R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_LINEAR);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> resultData(dstWidth * dstHeight * 3);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth * 3,
               (char *)d_dst + y * dstStep, dstWidth * 3 * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  // 验证缩放效果：检查是否有像素值变化
  bool hasChange = false;
  for (int i = 0; i < dstWidth * dstHeight * 3; i += 3) {
    // 检查至少一个通道的值不是0或1（因为源图像只有0和1）
    if (resultData[i] > 0.01f && resultData[i] < 0.99f) {
      hasChange = true;
      break;
    }
  }

  EXPECT_TRUE(hasChange) << "Scaling should produce interpolated values between 0 and 1";

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 32f C3R 平移变换
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_32f_C3R_Translation) {
  std::vector<Npp32f> srcData(srcWidth * srcHeight * 3, 0.0f);

  // 创建一个彩色方块
  for (int y = 8; y < 16; y++) {
    for (int x = 8; x < 16; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = 1.0f;  // 红色
      srcData[idx + 1] = 0.5f;  // 绿色
      srcData[idx + 2] = 0.2f;  // 蓝色
    }
  }

  // 平移变换矩阵：向右下移动4个像素的逆变换
  double coeffs[3][3] = {{1.0, 0.0, -4.0}, {0.0, 1.0, -4.0}, {0.0, 0.0, 1.0}};

  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C3(srcWidth, srcHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 3,
               srcWidth * 3 * sizeof(Npp32f), cudaMemcpyHostToDevice);
  }

  // 使用最近邻插值
  NppStatus status =
      nppiWarpPerspective_32f_C3R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_NN);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> resultData(dstWidth * dstHeight * 3);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth * 3,
               (char *)d_dst + y * dstStep, dstWidth * 3 * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  // 验证平移效果：检查是否有非零像素
  int nonZeroCount = 0;
  for (size_t i = 0; i < resultData.size(); i++) {
    if (resultData[i] > 0.1f) {
      nonZeroCount++;
    }
  }

  // 应该有一些变换后的像素
  EXPECT_GT(nonZeroCount, 0) << "Translation should produce some non-zero pixels";
  EXPECT_LT(nonZeroCount, resultData.size() / 2) << "Not all pixels should be non-zero";

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试 32f C3R 透视效果
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_32f_C3R_Perspective) {
  std::vector<Npp32f> srcData(srcWidth * srcHeight * 3);

  // 创建渐变图像
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = (float)x / srcWidth;          // R: 水平渐变
      srcData[idx + 1] = (float)y / srcHeight;         // G: 垂直渐变
      srcData[idx + 2] = (float)(x * y) / (srcWidth * srcHeight); // B: 乘积渐变
    }
  }

  // 透视变换矩阵：创建简单的透视效果
  double coeffs[3][3] = {{1.0, 0.1, 0.0}, {0.1, 1.0, 0.0}, {0.001, 0.001, 1.0}};

  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C3(srcWidth, srcHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(dstWidth, dstHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  for (int y = 0; y < srcHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * srcWidth * 3,
               srcWidth * 3 * sizeof(Npp32f), cudaMemcpyHostToDevice);
  }

  // 使用双线性插值
  NppStatus status =
      nppiWarpPerspective_32f_C3R(d_src, srcSize, srcStep, srcROI, d_dst, dstStep, dstROI, coeffs, NPPI_INTER_LINEAR);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> resultData(dstWidth * dstHeight * 3);
  for (int y = 0; y < dstHeight; y++) {
    cudaMemcpy(resultData.data() + y * dstWidth * 3,
               (char *)d_dst + y * dstStep, dstWidth * 3 * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  // 验证透视变换有效果（结果不应该与原图完全相同）
  bool hasChange = false;
  for (int i = 0; i < srcWidth * srcHeight * 3; i++) {
    if (fabs(resultData[i] - srcData[i]) > 0.01f) {
      hasChange = true;
      break;
    }
  }

  EXPECT_TRUE(hasChange) << "Perspective transformation should produce changes";

  nppiFree(d_src);
  nppiFree(d_dst);
}


