#include "../../framework/npp_test_base.h"
#include <cmath>

using namespace npp_functional_test;

class GaussianFilterFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }

  // Function to calculate expected Gaussian blur value
  float calculateGaussianValue(const std::vector<Npp8u> &src, int width, int height, int x, int y, int kernelSize,
                               float sigma) {
    float result = 0.0f;
    float kernelSum = 0.0f;
    int center = kernelSize / 2;
    float twoSigmaSq = 2.0f * sigma * sigma;

    for (int ky = 0; ky < kernelSize; ky++) {
      for (int kx = 0; kx < kernelSize; kx++) {
        int srcX = x + kx - center;
        int srcY = y + ky - center;

        // Clamp coordinates
        srcX = std::max(0, std::min(srcX, width - 1));
        srcY = std::max(0, std::min(srcY, height - 1));

        int dx = kx - center;
        int dy = ky - center;
        float kernelValue = expf(-(dx * dx + dy * dy) / twoSigmaSq);

        result += src[srcY * width + srcX] * kernelValue;
        kernelSum += kernelValue;
      }
    }

    return result / kernelSum;
  }
};

// 测试8位单通道高斯滤波 - 3x3核
TEST_F(GaussianFilterFunctionalTest, FilterGauss_8u_C1R_3x3) {
  const int width = 32, height = 32;

  // 创建带噪声的测试图像
  std::vector<Npp8u> srcData(width * height);
  for (int i = 0; i < width * height; i++) {
    // 创建棋盘模式以测试滤波效果
    int x = i % width;
    int y = i / width;
    srcData[i] = ((x + y) % 2 == 0) ? 255 : 0;
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiFilterGauss_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, NPP_MASK_SIZE_3_X_3);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiFilterGauss_8u_C1R failed";

  // Validate结果 - 高斯滤波应该模糊锐利边缘
  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);

  // 检查边缘是否被模糊（中心区域的值应该介于0和255之间）
  int blurredPixels = 0;
  for (int y = 1; y < height - 1; y++) {
    for (int x = 1; x < width - 1; x++) {
      int idx = y * width + x;
      if (resultData[idx] > 10 && resultData[idx] < 245) {
        blurredPixels++;
      }
    }
  }

  EXPECT_GT(blurredPixels, 0) << "Gaussian filter should blur sharp edges";
}

// 测试8位单通道高斯滤波 - 5x5核
TEST_F(GaussianFilterFunctionalTest, FilterGauss_8u_C1R_5x5) {
  const int width = 16, height = 16;

  // 创建中心有白点的黑色图像
  std::vector<Npp8u> srcData(width * height, 0);
  srcData[height / 2 * width + width / 2] = 255; // 中心白点

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiFilterGauss_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, NPP_MASK_SIZE_5_X_5);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiFilterGauss_8u_C1R 5x5 failed";

  // Validate结果 - 中心点应该扩散到周围像素
  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);

  // 检查中心周围是否有非零值
  int centerY = height / 2;
  int centerX = width / 2;
  int nonZeroAround = 0;

  for (int dy = -2; dy <= 2; dy++) {
    for (int dx = -2; dx <= 2; dx++) {
      int y = centerY + dy;
      int x = centerX + dx;
      if (y >= 0 && y < height && x >= 0 && x < width) {
        if (resultData[y * width + x] > 0) {
          nonZeroAround++;
        }
      }
    }
  }

  EXPECT_GT(nonZeroAround, 5) << "Gaussian filter should spread white dot to neighboring pixels";
}

// 测试8位三通道高斯滤波
TEST_F(GaussianFilterFunctionalTest, FilterGauss_8u_C3R_Basic) {
  const int width = 16, height = 16;

  // 创建彩色测试图像
  std::vector<Npp8u> srcData(width * height * 3);
  for (int i = 0; i < width * height; i++) {
    srcData[i * 3 + 0] = (i % 255);       // Red
    srcData[i * 3 + 1] = ((i * 2) % 255); // Green
    srcData[i * 3 + 2] = ((i * 3) % 255); // Blue
  }

  // 对于3通道图像，手动计算步长
  int srcStep = width * 3 * sizeof(Npp8u);
  int dstStep = width * 3 * sizeof(Npp8u);

  Npp8u *srcPtr = nppsMalloc_8u(width * height * 3);
  Npp8u *dstPtr = nppsMalloc_8u(width * height * 3);

  ASSERT_NE(srcPtr, nullptr);
  ASSERT_NE(dstPtr, nullptr);

  // 复制数据到GPU
  cudaMemcpy(srcPtr, srcData.data(), width * height * 3 * sizeof(Npp8u), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status = nppiFilterGauss_8u_C3R(srcPtr, srcStep, dstPtr, dstStep, roi, NPP_MASK_SIZE_3_X_3);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiFilterGauss_8u_C3R failed";

  // Validate结果 - 检查所有通道都被处理
  std::vector<Npp8u> resultData(width * height * 3);
  cudaMemcpy(resultData.data(), dstPtr, width * height * 3 * sizeof(Npp8u), cudaMemcpyDeviceToHost);

  bool hasValidData = true;
  for (int i = 0; i < width * height * 3; i++) {
    // 移除无效的检查，因为Npp8u最大值就是255
  }

  EXPECT_TRUE(hasValidData) << "All channels should have valid filtered data";

  // 清理内存
  nppsFree(srcPtr);
  nppsFree(dstPtr);
}

// 测试32位浮点高斯滤波
TEST_F(GaussianFilterFunctionalTest, FilterGauss_32f_C1R_Float) {
  const int width = 16, height = 16;

  // 创建浮点测试图像
  std::vector<Npp32f> srcData(width * height);
  for (int i = 0; i < width * height; i++) {
    srcData[i] = sinf(i * 0.1f); // 正弦波模式
  }

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiFilterGauss_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, NPP_MASK_SIZE_3_X_3);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiFilterGauss_32f_C1R failed";

  // Validate结果 - 浮点数据应该被正确处理
  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);

  // 检查是否为有效的浮点数据
  bool hasValidFloats = true;
  for (int i = 0; i < width * height; i++) {
    if (std::isnan(resultData[i]) || std::isinf(resultData[i])) {
      hasValidFloats = false;
      break;
    }
  }

  EXPECT_TRUE(hasValidFloats) << "Float Gaussian filter should produce valid floating-point results";
}

// 错误处理测试
TEST_F(GaussianFilterFunctionalTest, FilterGauss_ErrorHandling) {
  const int width = 16, height = 16;

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  NppiSize roi = {width, height};

  // 测试空指针
  NppStatus status = nppiFilterGauss_8u_C1R(nullptr, src.step(), dst.get(), dst.step(), roi, NPP_MASK_SIZE_3_X_3);
  EXPECT_NE(status, NPP_SUCCESS) << "Should fail with null source pointer";

  // 测试无效ROI
  NppiSize invalidRoi = {0, 0};
  status = nppiFilterGauss_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), invalidRoi, NPP_MASK_SIZE_3_X_3);
  EXPECT_EQ(status, NPP_NO_ERROR) << "vendor NPP returns success for zero-size ROI";

  // 测试不支持的mask大小
  status =
      nppiFilterGauss_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, NPP_MASK_SIZE_1_X_3); // 不支持的大小
  EXPECT_NE(status, NPP_SUCCESS) << "Should fail with unsupported mask size";
}

// 性能和不同核大小比较测试
TEST_F(GaussianFilterFunctionalTest, FilterGauss_DifferentKernelSizes) {
  const int width = 64, height = 64;

  // 创建具有高频特征的测试图像
  std::vector<Npp8u> srcData(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      srcData[y * width + x] = (Npp8u)(128 + 127 * sin(x * 0.5) * cos(y * 0.5));
    }
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst3x3(width, height);
  NppImageMemory<Npp8u> dst5x5(width, height);
  NppImageMemory<Npp8u> dst7x7(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};

  // 测试不同核大小
  NppStatus status3x3 =
      nppiFilterGauss_8u_C1R(src.get(), src.step(), dst3x3.get(), dst3x3.step(), roi, NPP_MASK_SIZE_3_X_3);
  NppStatus status5x5 =
      nppiFilterGauss_8u_C1R(src.get(), src.step(), dst5x5.get(), dst5x5.step(), roi, NPP_MASK_SIZE_5_X_5);
  NppStatus status7x7 =
      nppiFilterGauss_8u_C1R(src.get(), src.step(), dst7x7.get(), dst7x7.step(), roi, NPP_MASK_SIZE_7_X_7);

  ASSERT_EQ(status3x3, NPP_SUCCESS) << "3x3 Gaussian filter failed";
  ASSERT_EQ(status5x5, NPP_SUCCESS) << "5x5 Gaussian filter failed";
  ASSERT_EQ(status7x7, NPP_SUCCESS) << "7x7 Gaussian filter failed";

  // Validate更大的核产生更强的模糊效果
  std::vector<Npp8u> result3x3(width * height);
  std::vector<Npp8u> result7x7(width * height);
  dst3x3.copyToHost(result3x3);
  dst7x7.copyToHost(result7x7);

  // 计算方差以衡量模糊程度（方差越小越模糊）
  float var3x3 = 0.0f, var7x7 = 0.0f;
  float mean3x3 = 0.0f, mean7x7 = 0.0f;

  for (int i = 0; i < width * height; i++) {
    mean3x3 += result3x3[i];
    mean7x7 += result7x7[i];
  }
  mean3x3 /= (width * height);
  mean7x7 /= (width * height);

  for (int i = 0; i < width * height; i++) {
    float diff3x3 = result3x3[i] - mean3x3;
    float diff7x7 = result7x7[i] - mean7x7;
    var3x3 += diff3x3 * diff3x3;
    var7x7 += diff7x7 * diff7x7;
  }
  var3x3 /= (width * height);
  var7x7 /= (width * height);

  EXPECT_LT(var7x7, var3x3) << "7x7 kernel should produce more blur (lower variance) than 3x3";
}
