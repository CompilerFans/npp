#include "npp_test_base.h"
#include <cmath>

using namespace npp_functional_test;

class FilterConvolutionFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }

  // Function to create common kernels
  std::vector<Npp32s> createEdgeDetectionKernel() {
    // 3x3 Sobel X edge detection kernel
    return {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  }

  std::vector<Npp32s> createSharpenKernel() {
    // 3x3 sharpening kernel
    return {0, -1, 0, -1, 5, -1, 0, -1, 0};
  }

  std::vector<Npp32f> createGaussianKernel3x3() {
    // 3x3 Gaussian kernel (normalized)
    return {0.0625f, 0.125f, 0.0625f, 0.125f, 0.25f, 0.125f, 0.0625f, 0.125f, 0.0625f};
  }
};

// 尝试修复GPU上下文损坏问题后重新启用
TEST_F(FilterConvolutionFunctionalTest, Filter_8u_C1R_EdgeDetection) {
  const int width = 16, height = 16;

  // 创建垂直条纹测试图像
  std::vector<Npp8u> srcData(width * height);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      srcData[y * width + x] = (x < width / 2) ? 0 : 255;
    }
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  // 创建边缘检测卷积核
  std::vector<Npp32s> kernel = createEdgeDetectionKernel();
  NppiSize kernelSize = {3, 3};
  NppiPoint anchor = {1, 1}; // 中心anchor
  Npp32s divisor = 1;

  NppiSize roi = {width, height};

// 根据构建配置选择不同的Call方式
#ifdef USE_vendor_NPP
  // vendor NPP使用主机内存中的kernel
  NppStatus status =
      nppiFilter_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, kernel.data(), kernelSize, anchor, divisor);
#else
  // MPP需要设备内存中的kernel
  Npp32s *d_kernel;
  size_t kernelBytes = kernel.size() * sizeof(Npp32s);
  cudaMalloc(&d_kernel, kernelBytes);
  cudaMemcpy(d_kernel, kernel.data(), kernelBytes, cudaMemcpyHostToDevice);

  NppStatus status =
      nppiFilter_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, d_kernel, kernelSize, anchor, divisor);

  cudaFree(d_kernel); // 清理设备内存
#endif

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiFilter_8u_C1R edge detection failed";

  // Validate结果 - 边缘区域应该有非零响应
  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);

  // 检查边缘响应（在整个图像中寻找非零值）
  int edgeResponses = 0;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      if (resultData[y * width + x] > 10) { // 设置合理阈值
        edgeResponses++;
      }
    }
  }

  EXPECT_GT(edgeResponses, 0) << "Edge detection should produce responses at vertical edges";
}

// 第二个测试：锐化滤波
TEST_F(FilterConvolutionFunctionalTest, Filter_8u_C1R_Sharpen) {
  const int width = 16, height = 16;

  // 创建模糊的中心点图像
  std::vector<Npp8u> srcData(width * height, 50);
  srcData[height / 2 * width + width / 2] = 128; // 中心亮点

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  // 创建锐化卷积核
  std::vector<Npp32s> kernel = createSharpenKernel();
  NppiSize kernelSize = {3, 3};
  NppiPoint anchor = {1, 1};
  Npp32s divisor = 1;

  NppiSize roi = {width, height};

// 根据构建配置选择不同的Call方式
#ifdef USE_vendor_NPP
  // vendor NPP使用主机内存中的kernel
  NppStatus status =
      nppiFilter_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, kernel.data(), kernelSize, anchor, divisor);
#else
  // MPP需要设备内存中的kernel
  Npp32s *d_kernel;
  size_t kernelBytes = kernel.size() * sizeof(Npp32s);
  cudaMalloc(&d_kernel, kernelBytes);
  cudaMemcpy(d_kernel, kernel.data(), kernelBytes, cudaMemcpyHostToDevice);

  NppStatus status =
      nppiFilter_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, d_kernel, kernelSize, anchor, divisor);

  cudaFree(d_kernel); // 清理设备内存
#endif

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiFilter_8u_C1R sharpen failed";

  // Validate结果 - 中心点应该更亮，周围有对比增强
  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);

  int centerX = width / 2;
  int centerY = height / 2;
  int centerIdx = centerY * width + centerX;

  EXPECT_GT(resultData[centerIdx], srcData[centerIdx]) << "Sharpening should enhance the center bright point";
}

// NOTE: 重新启用测试 - 检查GPU上下文损坏问题是否已解决
TEST_F(FilterConvolutionFunctionalTest, DISABLED_Filter_8u_C3R_Basic) {
  const int width = 8, height = 8;

  // 创建彩色测试图像
  std::vector<Npp8u> srcData(width * height * 3);
  for (int i = 0; i < width * height; i++) {
    srcData[i * 3 + 0] = 255; // Red channel - full intensity
    srcData[i * 3 + 1] = 128; // Green channel - half intensity
    srcData[i * 3 + 2] = 0;   // Blue channel - zero intensity
  }

  // 使用正确的NPP内存分配，考虑内存对齐
  int srcStep, dstStep;
  Npp8u *srcPtr = nppiMalloc_8u_C3(width, height, &srcStep);
  Npp8u *dstPtr = nppiMalloc_8u_C3(width, height, &dstStep);

  ASSERT_NE(srcPtr, nullptr);
  ASSERT_NE(dstPtr, nullptr);

  // 复制数据到GPU，考虑步长对齐
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)srcPtr + y * srcStep, srcData.data() + y * width * 3, width * 3 * sizeof(Npp8u),
               cudaMemcpyHostToDevice);
  }

  // 创建简单的平均滤波核
  std::vector<Npp32s> kernel = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  NppiSize kernelSize = {3, 3};
  NppiPoint anchor = {1, 1};
  Npp32s divisor = 9; // 平均化

  NppiSize roi = {width, height};
  NppStatus status =
      nppiFilter_8u_C3R(srcPtr, srcStep, dstPtr, dstStep, roi, kernel.data(), kernelSize, anchor, divisor);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiFilter_8u_C3R failed";

  // Validate结果 - 检查所有通道都被处理，考虑步长对齐
  std::vector<Npp8u> resultData(width * height * 3);
  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultData.data() + y * width * 3, (char *)dstPtr + y * dstStep, width * 3 * sizeof(Npp8u),
               cudaMemcpyDeviceToHost);
  }

  // 检查中心区域的结果（避免边界效应）
  int centerX = width / 2;
  int centerY = height / 2;
  int centerIdx = (centerY * width + centerX) * 3;

  EXPECT_GT(resultData[centerIdx + 0], 200) << "Red channel should be processed correctly";
  EXPECT_GT(resultData[centerIdx + 1], 100) << "Green channel should be processed correctly";
  EXPECT_LT(resultData[centerIdx + 2], 50) << "Blue channel should be processed correctly";

  // 清理内存
  nppiFree(srcPtr);
  nppiFree(dstPtr);
}

// 第三个测试：32位浮点高斯滤波
TEST_F(FilterConvolutionFunctionalTest, Filter_32f_C1R_Gaussian) {
  const int width = 8, height = 8;

  // 创建脉冲信号
  std::vector<Npp32f> srcData(width * height, 0.0f);
  srcData[height / 2 * width + width / 2] = 1.0f; // 中心脉冲

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(srcData);

  // 创建高斯卷积核
  std::vector<Npp32f> kernel = createGaussianKernel3x3();
  NppiSize kernelSize = {3, 3};
  NppiPoint anchor = {1, 1};

  NppiSize roi = {width, height};

// 根据构建配置选择不同的Call方式
#ifdef USE_vendor_NPP
  // vendor NPP使用主机内存中的kernel
  NppStatus status =
      nppiFilter_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, kernel.data(), kernelSize, anchor);
#else
  // MPP需要设备内存中的kernel
  Npp32f *d_kernel;
  size_t kernelBytes = kernel.size() * sizeof(Npp32f);
  cudaMalloc(&d_kernel, kernelBytes);
  cudaMemcpy(d_kernel, kernel.data(), kernelBytes, cudaMemcpyHostToDevice);

  NppStatus status =
      nppiFilter_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi, d_kernel, kernelSize, anchor);

  cudaFree(d_kernel); // 清理设备内存
#endif

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiFilter_32f_C1R Gaussian failed";

  // Validate结果 - 脉冲应该被高斯核扩散
  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);

  int centerX = width / 2;
  int centerY = height / 2;

  // 检查中心和周围像素
  EXPECT_GT(resultData[centerY * width + centerX], 0.2f) << "Center should have significant response";
  EXPECT_GT(resultData[centerY * width + centerX - 1], 0.05f) << "Adjacent pixels should have some response";
  EXPECT_GT(resultData[centerY * width + centerX + 1], 0.05f) << "Adjacent pixels should have some response";
}
