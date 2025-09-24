#include "../../framework/npp_test_base.h"
#include "npp.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <thread>
#include <vector>

using namespace npp_functional_test;

class CopyFunctionalTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }

  // Function to verify copy results
  template <typename T> bool verifyCopyResult(const std::vector<T> &src, const std::vector<T> &dst) {
    if (src.size() != dst.size())
      return false;
    for (size_t i = 0; i < src.size(); i++) {
      if (src[i] != dst[i])
        return false;
    }
    return true;
  }
};

// 测试8位单通道拷贝
TEST_F(CopyFunctionalTest, Copy_8u_C1R_Basic) {
  const int width = 32, height = 32;

  // 创建测试数据
  std::vector<Npp8u> srcData(width * height);
  for (int i = 0; i < width * height; i++) {
    srcData[i] = (Npp8u)(i % 256);
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiCopy_8u_C1R failed";

  // Validate结果
  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);

  EXPECT_TRUE(verifyCopyResult(srcData, resultData)) << "Copy result does not match source data";
}

// 测试8位三通道拷贝
TEST_F(CopyFunctionalTest, Copy_8u_C3R_RGB) {
  const int width = 16, height = 16;

  // 创建RGB测试数据
  std::vector<Npp8u> srcData(width * height * 3);
  for (int i = 0; i < width * height; i++) {
    srcData[i * 3 + 0] = (Npp8u)(i % 256);       // Red
    srcData[i * 3 + 1] = (Npp8u)((i * 2) % 256); // Green
    srcData[i * 3 + 2] = (Npp8u)((i * 3) % 256); // Blue
  }

  // 使用手动内存分配
  int srcStep = width * 3 * sizeof(Npp8u);
  int dstStep = width * 3 * sizeof(Npp8u);

  Npp8u *srcPtr = nppsMalloc_8u(width * height * 3);
  Npp8u *dstPtr = nppsMalloc_8u(width * height * 3);

  ASSERT_NE(srcPtr, nullptr);
  ASSERT_NE(dstPtr, nullptr);

  // 复制数据到GPU
  cudaMemcpy(srcPtr, srcData.data(), width * height * 3 * sizeof(Npp8u), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_8u_C3R(srcPtr, srcStep, dstPtr, dstStep, roi);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiCopy_8u_C3R failed";

  // Validate结果
  std::vector<Npp8u> resultData(width * height * 3);
  cudaMemcpy(resultData.data(), dstPtr, width * height * 3 * sizeof(Npp8u), cudaMemcpyDeviceToHost);

  EXPECT_TRUE(verifyCopyResult(srcData, resultData)) << "RGB copy result does not match source data";

  // 清理内存
  nppsFree(srcPtr);
  nppsFree(dstPtr);
}

// 测试8位四通道拷贝
TEST_F(CopyFunctionalTest, Copy_8u_C4R_RGBA) {
  const int width = 8, height = 8;

  // 创建RGBA测试数据
  std::vector<Npp8u> srcData(width * height * 4);
  for (int i = 0; i < width * height; i++) {
    srcData[i * 4 + 0] = (Npp8u)(i % 256);       // Red
    srcData[i * 4 + 1] = (Npp8u)((i * 2) % 256); // Green
    srcData[i * 4 + 2] = (Npp8u)((i * 3) % 256); // Blue
    srcData[i * 4 + 3] = (Npp8u)((i * 4) % 256); // Alpha
  }

  // 使用手动内存分配
  int srcStep = width * 4 * sizeof(Npp8u);
  int dstStep = width * 4 * sizeof(Npp8u);

  Npp8u *srcPtr = nppsMalloc_8u(width * height * 4);
  Npp8u *dstPtr = nppsMalloc_8u(width * height * 4);

  ASSERT_NE(srcPtr, nullptr);
  ASSERT_NE(dstPtr, nullptr);

  // 复制数据到GPU
  cudaMemcpy(srcPtr, srcData.data(), width * height * 4 * sizeof(Npp8u), cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_8u_C4R(srcPtr, srcStep, dstPtr, dstStep, roi);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiCopy_8u_C4R failed";

  // Validate结果
  std::vector<Npp8u> resultData(width * height * 4);
  cudaMemcpy(resultData.data(), dstPtr, width * height * 4 * sizeof(Npp8u), cudaMemcpyDeviceToHost);

  EXPECT_TRUE(verifyCopyResult(srcData, resultData)) << "RGBA copy result does not match source data";

  // 清理内存
  nppsFree(srcPtr);
  nppsFree(dstPtr);
}

// 测试32位浮点拷贝
TEST_F(CopyFunctionalTest, Copy_32f_C1R_Float) {
  const int width = 16, height = 16;

  // 创建浮点测试数据
  std::vector<Npp32f> srcData(width * height);
  for (int i = 0; i < width * height; i++) {
    srcData[i] = sinf(i * 0.1f) * cosf(i * 0.05f); // 复杂浮点模式
  }

  NppImageMemory<Npp32f> src(width, height);
  NppImageMemory<Npp32f> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_32f_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);

  ASSERT_EQ(status, NPP_SUCCESS) << "nppiCopy_32f_C1R failed";

  // Validate结果
  std::vector<Npp32f> resultData(width * height);
  dst.copyToHost(resultData);

  // 检查浮点精度
  bool isEqual = true;
  for (size_t i = 0; i < srcData.size(); i++) {
    if (fabsf(srcData[i] - resultData[i]) > 1e-6f) {
      isEqual = false;
      break;
    }
  }

  EXPECT_TRUE(isEqual) << "Float copy result does not match source data";
}

// 测试不同ROI大小
TEST_F(CopyFunctionalTest, Copy_8u_C1R_DifferentROI) {
  const int width = 64, height = 64;

  // 创建大图像
  std::vector<Npp8u> srcData(width * height);
  for (int i = 0; i < width * height; i++) {
    srcData[i] = (Npp8u)(i % 256);
  }

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  // 测试不同的ROI大小
  std::vector<NppiSize> roiSizes = {{16, 16}, {32, 32}, {48, 48}, {64, 64}};

  for (const auto &roi : roiSizes) {
    NppStatus status = nppiCopy_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);

    ASSERT_EQ(status, NPP_SUCCESS) << "nppiCopy_8u_C1R failed for ROI " << roi.width << "x" << roi.height;

    // Validate拷贝区域
    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);

    bool roiCorrect = true;
    for (int y = 0; y < roi.height; y++) {
      for (int x = 0; x < roi.width; x++) {
        int idx = y * width + x;
        if (srcData[idx] != resultData[idx]) {
          roiCorrect = false;
          break;
        }
      }
      if (!roiCorrect)
        break;
    }

    EXPECT_TRUE(roiCorrect) << "ROI copy incorrect for size " << roi.width << "x" << roi.height;
  }
}

// 错误处理测试
// NOTE: 测试已被禁用 - vendor NPP对无效参数的错误检测行为与预期不符
TEST_F(CopyFunctionalTest, DISABLED_Copy_ErrorHandling) {
  const int width = 16, height = 16;

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  NppiSize roi = {width, height};

  // 测试空指针
  NppStatus status = nppiCopy_8u_C1R(nullptr, src.step(), dst.get(), dst.step(), roi);
  EXPECT_NE(status, NPP_SUCCESS) << "Should fail with null source pointer";

  status = nppiCopy_8u_C1R(src.get(), src.step(), nullptr, dst.step(), roi);
  EXPECT_NE(status, NPP_SUCCESS) << "Should fail with null destination pointer";

  // 测试无效ROI
  NppiSize invalidRoi = {0, 0};
  status = nppiCopy_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), invalidRoi);
  EXPECT_NE(status, NPP_SUCCESS) << "Should fail with invalid ROI";

  // 测试负步长
  status = nppiCopy_8u_C1R(src.get(), -1, dst.get(), dst.step(), roi);
  EXPECT_NE(status, NPP_SUCCESS) << "Should fail with negative source step";

  status = nppiCopy_8u_C1R(src.get(), src.step(), dst.get(), -1, roi);
  EXPECT_NE(status, NPP_SUCCESS) << "Should fail with negative destination step";
}

// 性能基准测试
TEST_F(CopyFunctionalTest, Copy_Performance_Benchmark) {
  const int width = 1024, height = 1024;

  std::vector<Npp8u> srcData(width * height, 128);

  NppImageMemory<Npp8u> src(width, height);
  NppImageMemory<Npp8u> dst(width, height);

  src.copyFromHost(srcData);

  NppiSize roi = {width, height};

  // 预热
  for (int i = 0; i < 5; i++) {
    nppiCopy_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);
  }

  // 同步GPU确保准确计时
  cudaDeviceSynchronize();

  auto start = std::chrono::high_resolution_clock::now();

  // 执行多次拷贝以获得稳定的测量结果
  const int iterations = 100;
  for (int i = 0; i < iterations; i++) {
    NppStatus status = nppiCopy_8u_C1R(src.get(), src.step(), dst.get(), dst.step(), roi);
    ASSERT_EQ(status, NPP_SUCCESS) << "Copy failed in performance test";
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  double avgTime = duration.count() / (double)iterations;
  double throughput = (width * height * sizeof(Npp8u) * 2) / (avgTime * 1e-6) / (1024 * 1024 * 1024); // GB/s

  std::cout << "Copy Performance: " << avgTime << " μs per operation, " << throughput << " GB/s throughput"
            << std::endl;

  // Validate结果仍然正确
  std::vector<Npp8u> resultData(width * height);
  dst.copyToHost(resultData);
  EXPECT_TRUE(verifyCopyResult(srcData, resultData)) << "Performance test result verification failed";
}

class CopyExtendedFunctionalTest : public ::testing::Test {
protected:
  void SetUp() override {
    width = 8;
    height = 6;
    roi.width = width;
    roi.height = height;
  }

  int width, height;
  NppiSize roi;
};

// 测试nppiCopy_32f_C3R函数
TEST_F(CopyExtendedFunctionalTest, Copy_32f_C3R_BasicOperation) {
  const int channels = 3;
  std::vector<Npp32f> srcData(width * height * channels);

  // 创建测试数据：RGB渐变模式
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * channels;
      srcData[idx + 0] = (float)x / (width - 1);  // R通道: 0.0-1.0渐变
      srcData[idx + 1] = (float)y / (height - 1); // G通道: 0.0-1.0渐变
      srcData[idx + 2] = 0.5f;                    // B通道: 固定0.5
    }
  }

  // 分配GPU内存
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制输入数据到GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * channels, width * channels * sizeof(Npp32f),
               cudaMemcpyHostToDevice);
  }

  // 执行复制
  NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // 复制结果回主机
  std::vector<Npp32f> resultData(width * height * channels);
  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultData.data() + y * width * channels, (char *)d_dst + y * dstStep, width * channels * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  // Validate结果：应该完全一致
  for (int i = 0; i < width * height * channels; i++) {
    EXPECT_FLOAT_EQ(resultData[i], srcData[i])
        << "复制失败 at index " << i << ": got " << resultData[i] << ", expected " << srcData[i];
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 测试nppiCopy_32f_C3P3R函数（packed到planar转换）
TEST_F(CopyExtendedFunctionalTest, Copy_32f_C3P3R_BasicOperation) {
  const int channels = 3;

  // 创建packed格式的源数据（RGB交错）
  std::vector<Npp32f> srcData(width * height * channels);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * channels;
      srcData[idx + 0] = (float)(x + y) / (width + height - 2); // R通道
      srcData[idx + 1] = (float)x / (width - 1);                // G通道
      srcData[idx + 2] = (float)y / (height - 1);               // B通道
    }
  }

  // 分配GPU内存 - 源是packed格式，目标是planar格式
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  Npp32f *d_dstR = nppiMalloc_32f_C1(width, height, &dstStep);
  Npp32f *d_dstG = nppiMalloc_32f_C1(width, height, &dstStep);
  Npp32f *d_dstB = nppiMalloc_32f_C1(width, height, &dstStep);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dstR, nullptr);
  ASSERT_NE(d_dstG, nullptr);
  ASSERT_NE(d_dstB, nullptr);

  // 复制packed源数据到GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * channels, width * channels * sizeof(Npp32f),
               cudaMemcpyHostToDevice);
  }

  // 设置目标指针数组
  Npp32f *dstPtrs[3] = {d_dstR, d_dstG, d_dstB};

  // 执行packed到planar转换
  NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, dstPtrs, dstStep, roi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Validate结果
  std::vector<Npp32f> resultR(width * height);
  std::vector<Npp32f> resultG(width * height);
  std::vector<Npp32f> resultB(width * height);

  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultR.data() + y * width, (char *)d_dstR + y * dstStep, width * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(resultG.data() + y * width, (char *)d_dstG + y * dstStep, width * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(resultB.data() + y * width, (char *)d_dstB + y * dstStep, width * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  // Validate每个通道的数据
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int packedIdx = (y * width + x) * channels;
      int planarIdx = y * width + x;

      EXPECT_FLOAT_EQ(resultR[planarIdx], srcData[packedIdx + 0]) << "R通道转换失败 at (" << x << "," << y << ")";
      EXPECT_FLOAT_EQ(resultG[planarIdx], srcData[packedIdx + 1]) << "G通道转换失败 at (" << x << "," << y << ")";
      EXPECT_FLOAT_EQ(resultB[planarIdx], srcData[packedIdx + 2]) << "B通道转换失败 at (" << x << "," << y << ")";
    }
  }

  // 清理内存
  nppiFree(d_src);
  nppiFree(d_dstR);
  nppiFree(d_dstG);
  nppiFree(d_dstB);
}

// 测试边界条件
TEST_F(CopyExtendedFunctionalTest, Copy_32f_C3R_BoundaryValues) {
  std::vector<Npp32f> srcData = {
      -1.0f,    0.0f,      1.0f,  // 负值，零，正值
      3.14159f, -2.71828f, 0.5f,  // 数学常数
      1e6f,     -1e-6f,    1e-38f // 极值
  };

  NppiSize testRoi = {3, 1};

  // 分配GPU内存
  int step = 3 * 3 * sizeof(Npp32f);
  Npp32f *d_src = nppsMalloc_32f(3 * 3);
  Npp32f *d_dst = nppsMalloc_32f(3 * 3);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据
  cudaMemcpy(d_src, srcData.data(), srcData.size() * sizeof(Npp32f), cudaMemcpyHostToDevice);

  // 执行复制
  NppStatus status = nppiCopy_32f_C3R(d_src, step, d_dst, step, testRoi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Validate结果
  std::vector<Npp32f> resultData(3 * 3);
  cudaMemcpy(resultData.data(), d_dst, resultData.size() * sizeof(Npp32f), cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < srcData.size(); i++) {
    EXPECT_FLOAT_EQ(resultData[i], srcData[i]) << "边界值复制失败 at index " << i;
  }

  nppsFree(d_src);
  nppsFree(d_dst);
}

// 测试流上下文版本
TEST_F(CopyExtendedFunctionalTest, Copy_32f_C3R_StreamContext) {
  const int channels = 3;
  std::vector<Npp32f> srcData(width * height * channels, 1.5f); // 所有像素值为1.5f

  // 分配GPU内存
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * channels, width * channels * sizeof(Npp32f),
               cudaMemcpyHostToDevice);
  }

  // 创建流上下文
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // 执行复制
  NppStatus status = nppiCopy_32f_C3R_Ctx(d_src, srcStep, d_dst, dstStep, roi, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Validate结果
  std::vector<Npp32f> resultData(width * height * channels);
  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultData.data() + y * width * channels, (char *)d_dst + y * dstStep, width * channels * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  // 所有值应该都是1.5f
  for (int i = 0; i < width * height * channels; i++) {
    EXPECT_FLOAT_EQ(resultData[i], 1.5f) << "流上下文测试失败 at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 参数化测试：测试不同的图像尺寸
struct ImageSizeParams {
  int width;
  int height;
  std::string description;
};

class CopyParameterizedTest : public ::testing::TestWithParam<ImageSizeParams> {
protected:
  void SetUp() override {
    auto params = GetParam();
    width = params.width;
    height = params.height;
    roi.width = width;
    roi.height = height;
  }

  int width, height;
  NppiSize roi;
};

// 定义测试参数
INSTANTIATE_TEST_SUITE_P(DifferentImageSizes, CopyParameterizedTest,
                         ::testing::Values(ImageSizeParams{1, 1, "SinglePixel"}, ImageSizeParams{2, 3, "SmallRect"},
                                           ImageSizeParams{4, 4, "SmallSquare"}, ImageSizeParams{7, 5, "OddSize"},
                                           ImageSizeParams{16, 8, "Rect16x8"}, ImageSizeParams{32, 32, "Square32x32"},
                                           ImageSizeParams{64, 16, "WideRect"}, ImageSizeParams{8, 64, "TallRect"},
                                           ImageSizeParams{128, 96, "MediumSize"},
                                           ImageSizeParams{8192, 4096, "Large"}),
                         [](const ::testing::TestParamInfo<ImageSizeParams> &info) {
                           return info.param.description + "_" + std::to_string(info.param.width) + "x" +
                                  std::to_string(info.param.height);
                         });

// 参数化测试nppiCopy_32f_C3R
TEST_P(CopyParameterizedTest, Copy_32f_C3R_ParameterizedSizes) {
  const int channels = 3;
  std::vector<Npp32f> srcData(width * height * channels);

  // 创建测试数据：每个像素用坐标编码
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * channels;
      srcData[idx + 0] = (float)(x + y * 100);  // R: 位置编码
      srcData[idx + 1] = (float)(x * y + 1000); // G: 乘积编码
      srcData[idx + 2] = (float)(x - y + 2000); // B: 差值编码
    }
  }

  // 分配GPU内存
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据到GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * channels, width * channels * sizeof(Npp32f),
               cudaMemcpyHostToDevice);
  }

  // 执行拷贝
  NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Validate结果
  std::vector<Npp32f> resultData(width * height * channels);
  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultData.data() + y * width * channels, (char *)d_dst + y * dstStep, width * channels * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < width * height * channels; i++) {
    EXPECT_FLOAT_EQ(resultData[i], srcData[i])
        << "参数化测试失败 at index " << i << " for size " << width << "x" << height;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// 数据模式参数化测试
struct DataPatternParams {
  std::string name;
  std::function<float(int, int, int, int, int)> generator; // x, y, width, height, channel
};

class CopyDataPatternTest : public ::testing::TestWithParam<DataPatternParams> {
protected:
  void SetUp() override {
    width = 16;
    height = 12;
    roi.width = width;
    roi.height = height;
  }

  int width, height;
  NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(DifferentDataPatterns, CopyDataPatternTest,
                         ::testing::Values(DataPatternParams{"Constant",
                                                             [](int x, int y, int w, int h, int c) -> float {
                                                               (void)x;
                                                               (void)y;
                                                               (void)w;
                                                               (void)h;
                                                               (void)c;
                                                               return 42.0f;
                                                             }},
                                           DataPatternParams{"HorizontalGrad",
                                                             [](int x, int y, int w, int h, int c) -> float {
                                                               (void)y;
                                                               (void)h;
                                                               (void)c;
                                                               return (float)x / (w - 1);
                                                             }},
                                           DataPatternParams{"VerticalGrad",
                                                             [](int x, int y, int w, int h, int c) -> float {
                                                               (void)x;
                                                               (void)w;
                                                               (void)c;
                                                               return (float)y / (h - 1);
                                                             }},
                                           DataPatternParams{"Checkerboard",
                                                             [](int x, int y, int w, int h, int c) -> float {
                                                               (void)w;
                                                               (void)h;
                                                               (void)c;
                                                               return ((x + y) % 2) ? 1.0f : 0.0f;
                                                             }},
                                           DataPatternParams{"DiagonalGrad",
                                                             [](int x, int y, int w, int h, int c) -> float {
                                                               (void)c;
                                                               return (float)(x + y) / (w + h - 2);
                                                             }},
                                           DataPatternParams{"ChannelRelated",
                                                             [](int x, int y, int w, int h, int c) -> float {
                                                               return (float)(c + 1) * ((float)x / w + (float)y / h);
                                                             }},
                                           DataPatternParams{"SineWave",
                                                             [](int x, int y, int w, int h, int c) -> float {
                                                               (void)c;
                                                               return sin((float)x * 3.14159f / w) *
                                                                      cos((float)y * 3.14159f / h);
                                                             }}),
                         [](const ::testing::TestParamInfo<DataPatternParams> &info) { return info.param.name; });

// 参数化测试不同数据模式的nppiCopy_32f_C3R
TEST_P(CopyDataPatternTest, Copy_32f_C3R_DataPatterns) {
  auto params = GetParam();
  const int channels = 3;
  std::vector<Npp32f> srcData(width * height * channels);

  // 使用参数化的数据生成器
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * channels;
      for (int c = 0; c < channels; c++) {
        srcData[idx + c] = params.generator(x, y, width, height, c);
      }
    }
  }

  // 分配GPU内存
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据到GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * channels, width * channels * sizeof(Npp32f),
               cudaMemcpyHostToDevice);
  }

  // 执行拷贝
  NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Validate结果
  std::vector<Npp32f> resultData(width * height * channels);
  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultData.data() + y * width * channels, (char *)d_dst + y * dstStep, width * channels * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < width * height * channels; i++) {
    EXPECT_FLOAT_EQ(resultData[i], srcData[i]) << "数据模式测试失败 at index " << i << " for pattern: " << params.name;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Enhanced parameterized tests for nppiCopy_32f_C3R
class CopyC3REnhancedTest : public ::testing::TestWithParam<ImageSizeParams> {
protected:
  void SetUp() override {
    auto params = GetParam();
    width = params.width;
    height = params.height;
    roi.width = width;
    roi.height = height;
  }

  int width, height;
  NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(C3REnhancedSizes, CopyC3REnhancedTest,
                         ::testing::Values(ImageSizeParams{1, 1, "SinglePixel"}, ImageSizeParams{2, 3, "SmallRect"},
                                           ImageSizeParams{3, 3, "SmallSquare"}, ImageSizeParams{5, 7, "OddSize"},
                                           ImageSizeParams{16, 8, "Rect16x8"}, ImageSizeParams{32, 32, "Square32x32"},
                                           ImageSizeParams{64, 16, "WideRect"}, ImageSizeParams{8, 64, "TallRect"},
                                           ImageSizeParams{128, 96, "MediumSize"},
                                           ImageSizeParams{256, 192, "LargeSize"},
                                           ImageSizeParams{8192, 4096, "Large"}),
                         [](const ::testing::TestParamInfo<ImageSizeParams> &info) {
                           return info.param.description + "_" + std::to_string(info.param.width) + "x" +
                                  std::to_string(info.param.height);
                         });

TEST_P(CopyC3REnhancedTest, Copy_32f_C3R_EnhancedSizes) {
  const int channels = 3;
  std::vector<Npp32f> srcData(width * height * channels);

  // Create RGB channel-specific test data
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * channels;
      srcData[idx + 0] = (float)(x + y * 1000.0f);    // R: position encoding
      srcData[idx + 1] = (float)(x * y + 10000.0f);   // G: product encoding
      srcData[idx + 2] = (float)((x ^ y) + 20000.0f); // B: XOR encoding
    }
  }

  // Allocate GPU memory
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * channels, width * channels * sizeof(Npp32f),
               cudaMemcpyHostToDevice);
  }

  // Execute copy
  NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Verify result
  std::vector<Npp32f> resultData(width * height * channels);
  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultData.data() + y * width * channels, (char *)d_dst + y * dstStep, width * channels * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < width * height * channels; i++) {
    EXPECT_FLOAT_EQ(resultData[i], srcData[i])
        << "Enhanced C3R copy failed at index " << i << " for size " << width << "x" << height;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Data pattern tests for nppiCopy_32f_C3R
class CopyC3RDataPatternTest : public ::testing::TestWithParam<DataPatternParams> {
protected:
  void SetUp() override {
    width = 16;
    height = 12;
    roi.width = width;
    roi.height = height;
  }

  int width, height;
  NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(
    C3RDataPatterns, CopyC3RDataPatternTest,
    ::testing::Values(DataPatternParams{"Constant",
                                        [](int x, int y, int w, int h, int c) -> float {
                                          (void)x;
                                          (void)y;
                                          (void)w;
                                          (void)h;
                                          return (float)(c + 1) * 100.0f;
                                        }},
                      DataPatternParams{"RGBGradients",
                                        [](int x, int y, int w, int h, int c) -> float {
                                          if (c == 0)
                                            return (float)x / (w - 1) * 255.0f; // R: horizontal gradient
                                          if (c == 1)
                                            return (float)y / (h - 1) * 255.0f;         // G: vertical gradient
                                          return (float)(x + y) / (w + h - 2) * 255.0f; // B: diagonal gradient
                                        }},
                      DataPatternParams{"RGBCheckers",
                                        [](int x, int y, int w, int h, int c) -> float {
                                          (void)w;
                                          (void)h;
                                          bool checker = ((x + y) % 2) == 0;
                                          if (c == 0)
                                            return checker ? 255.0f : 0.0f; // R: checkerboard
                                          if (c == 1)
                                            return checker ? 0.0f : 255.0f; // G: inverted checkerboard
                                          return checker == ((x + y + 1) % 2) ? 128.0f : 64.0f; // B: offset pattern
                                        }},
                      DataPatternParams{"RGBSines",
                                        [](int x, int y, int w, int h, int c) -> float {
                                          float pi = 3.14159f;
                                          if (c == 0)
                                            return sin((float)x * pi / w) * 127.5f + 127.5f; // R: horizontal sine
                                          if (c == 1)
                                            return cos((float)y * pi / h) * 127.5f + 127.5f; // G: vertical cosine
                                          return sin((float)(x + y) * pi / (w + h)) * 127.5f +
                                                 127.5f; // B: diagonal sine
                                        }},
                      DataPatternParams{"RGBCircles",
                                        [](int x, int y, int w, int h, int c) -> float {
                                          float cx = w / 2.0f, cy = h / 2.0f;
                                          float dx = x - cx, dy = y - cy;
                                          float dist = sqrt(dx * dx + dy * dy);
                                          float maxDist = sqrt(cx * cx + cy * cy);
                                          if (c == 0)
                                            return (1.0f - dist / maxDist) * 255.0f; // R: radial gradient
                                          if (c == 1)
                                            return sin(dist * 0.5f) * 127.5f + 127.5f;   // G: concentric circles
                                          return cos(dist * 0.3f + c) * 127.5f + 127.5f; // B: phase-shifted circles
                                        }},
                      DataPatternParams{"RGBNoise",
                                        [](int x, int y, int w, int h, int c) -> float {
                                          // Deterministic pseudo-random based on position and channel
                                          unsigned int seed = (unsigned int)(x * 1000 + y * 100 + c * 10 + w + h);
                                          seed = seed * 1103515245 + 12345;
                                          return (float)(seed % 256);
                                        }},
                      DataPatternParams{"RGBBoundaries",
                                        [](int x, int y, int w, int h, int c) -> float {
                                          bool isEdge = (x == 0 || y == 0 || x == w - 1 || y == h - 1);
                                          bool isCorner = (x == 0 || x == w - 1) && (y == 0 || y == h - 1);
                                          if (c == 0)
                                            return isCorner ? 255.0f
                                                            : (isEdge ? 128.0f : 64.0f); // R: corner/edge detection
                                          if (c == 1)
                                            return isEdge ? 255.0f : 0.0f; // G: edge detection
                                          return (float)((x + y) % 256);   // B: position-based
                                        }}),
    [](const ::testing::TestParamInfo<DataPatternParams> &info) { return info.param.name; });

TEST_P(CopyC3RDataPatternTest, Copy_32f_C3R_DataPatterns) {
  auto params = GetParam();
  const int channels = 3;
  std::vector<Npp32f> srcData(width * height * channels);

  // Generate RGB test data using pattern
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * channels;
      for (int c = 0; c < channels; c++) {
        srcData[idx + c] = params.generator(x, y, width, height, c);
      }
    }
  }

  // Allocate GPU memory
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * channels, width * channels * sizeof(Npp32f),
               cudaMemcpyHostToDevice);
  }

  // Execute copy
  NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Verify result
  std::vector<Npp32f> resultData(width * height * channels);
  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultData.data() + y * width * channels, (char *)d_dst + y * dstStep, width * channels * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < width * height * channels; i++) {
    EXPECT_FLOAT_EQ(resultData[i], srcData[i])
        << "C3R pattern test failed at index " << i << " for pattern: " << params.name;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Stream context tests for nppiCopy_32f_C3R
class CopyC3RStreamTest : public ::testing::TestWithParam<ImageSizeParams> {
protected:
  void SetUp() override {
    auto params = GetParam();
    width = params.width;
    height = params.height;
    roi.width = width;
    roi.height = height;
  }

  int width, height;
  NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(C3RStreamSizes, CopyC3RStreamTest,
                         ::testing::Values(ImageSizeParams{4, 4, "Small"}, ImageSizeParams{16, 16, "Medium"},
                                           ImageSizeParams{64, 64, "Large"}, ImageSizeParams{4096, 2048, "Large2"}),
                         [](const ::testing::TestParamInfo<ImageSizeParams> &info) {
                           return info.param.description + "_" + std::to_string(info.param.width) + "x" +
                                  std::to_string(info.param.height);
                         });

TEST_P(CopyC3RStreamTest, Copy_32f_C3R_StreamContext) {
  const int channels = 3;
  std::vector<Npp32f> srcData(width * height * channels);

  // Create test data with RGB patterns
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * channels;
      srcData[idx + 0] = (float)(x % 256);       // R: horizontal pattern
      srcData[idx + 1] = (float)(y % 256);       // G: vertical pattern
      srcData[idx + 2] = (float)((x + y) % 256); // B: diagonal pattern
    }
  }

  // Allocate GPU memory
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * channels, width * channels * sizeof(Npp32f),
               cudaMemcpyHostToDevice);
  }

  // Create stream context
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  // Execute copy with stream context
  NppStatus status = nppiCopy_32f_C3R_Ctx(d_src, srcStep, d_dst, dstStep, roi, nppStreamCtx);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Verify result
  std::vector<Npp32f> resultData(width * height * channels);
  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultData.data() + y * width * channels, (char *)d_dst + y * dstStep, width * channels * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < width * height * channels; i++) {
    EXPECT_FLOAT_EQ(resultData[i], srcData[i]) << "C3R stream context test failed at index " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Parameterized tests for nppiCopy_32f_C1R
class CopyC1RParameterizedTest : public ::testing::TestWithParam<ImageSizeParams> {
protected:
  void SetUp() override {
    auto params = GetParam();
    width = params.width;
    height = params.height;
    roi.width = width;
    roi.height = height;
  }

  int width, height;
  NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(C1RImageSizes, CopyC1RParameterizedTest,
                         ::testing::Values(ImageSizeParams{1, 1, "SinglePixel"}, ImageSizeParams{2, 3, "SmallRect"},
                                           ImageSizeParams{4, 4, "SmallSquare"}, ImageSizeParams{7, 5, "OddSize"},
                                           ImageSizeParams{16, 8, "Rect16x8"}, ImageSizeParams{32, 32, "Square32x32"},
                                           ImageSizeParams{64, 16, "WideRect"}, ImageSizeParams{8, 64, "TallRect"},
                                           ImageSizeParams{128, 96, "MediumSize"},
                                           ImageSizeParams{8192, 4096, "Large"}),
                         [](const ::testing::TestParamInfo<ImageSizeParams> &info) {
                           return info.param.description + "_" + std::to_string(info.param.width) + "x" +
                                  std::to_string(info.param.height);
                         });

TEST_P(CopyC1RParameterizedTest, Copy_32f_C1R_ParameterizedSizes) {
  std::vector<Npp32f> srcData(width * height);

  // Create position-encoded test data
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      srcData[idx] = (float)(x + y * 100.0f + 1000.0f);
    }
  }

  // Allocate GPU memory
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C1(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width, width * sizeof(Npp32f), cudaMemcpyHostToDevice);
  }

  // Execute copy
  NppStatus status = nppiCopy_32f_C1R(d_src, srcStep, d_dst, dstStep, roi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Verify result
  std::vector<Npp32f> resultData(width * height);
  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultData.data() + y * width, (char *)d_dst + y * dstStep, width * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < width * height; i++) {
    EXPECT_FLOAT_EQ(resultData[i], srcData[i])
        << "C1R copy failed at index " << i << " for size " << width << "x" << height;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Data pattern tests for nppiCopy_32f_C1R
class CopyC1RDataPatternTest : public ::testing::TestWithParam<DataPatternParams> {
protected:
  void SetUp() override {
    width = 16;
    height = 12;
    roi.width = width;
    roi.height = height;
  }

  int width, height;
  NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(C1RDataPatterns, CopyC1RDataPatternTest,
                         ::testing::Values(DataPatternParams{"Constant",
                                                             [](int x, int y, int w, int h, int c) -> float {
                                                               (void)x;
                                                               (void)y;
                                                               (void)w;
                                                               (void)h;
                                                               (void)c;
                                                               return 42.0f;
                                                             }},
                                           DataPatternParams{"HorizontalGrad",
                                                             [](int x, int y, int w, int h, int c) -> float {
                                                               (void)y;
                                                               (void)h;
                                                               (void)c;
                                                               return (float)x / (w - 1);
                                                             }},
                                           DataPatternParams{"VerticalGrad",
                                                             [](int x, int y, int w, int h, int c) -> float {
                                                               (void)x;
                                                               (void)w;
                                                               (void)c;
                                                               return (float)y / (h - 1);
                                                             }},
                                           DataPatternParams{"Checkerboard",
                                                             [](int x, int y, int w, int h, int c) -> float {
                                                               (void)w;
                                                               (void)h;
                                                               (void)c;
                                                               return ((x + y) % 2) ? 1.0f : 0.0f;
                                                             }},
                                           DataPatternParams{"DiagonalGrad",
                                                             [](int x, int y, int w, int h, int c) -> float {
                                                               (void)c;
                                                               return (float)(x + y) / (w + h - 2);
                                                             }},
                                           DataPatternParams{"CirclePattern",
                                                             [](int x, int y, int w, int h, int c) -> float {
                                                               (void)c;
                                                               float cx = w / 2.0f, cy = h / 2.0f;
                                                               float dx = x - cx, dy = y - cy;
                                                               float dist = sqrt(dx * dx + dy * dy);
                                                               return sin(dist * 0.5f);
                                                             }}),
                         [](const ::testing::TestParamInfo<DataPatternParams> &info) { return info.param.name; });

TEST_P(CopyC1RDataPatternTest, Copy_32f_C1R_DataPatterns) {
  auto params = GetParam();
  std::vector<Npp32f> srcData(width * height);

  // Generate test data using pattern
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      srcData[idx] = params.generator(x, y, width, height, 0);
    }
  }

  // Allocate GPU memory
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C1(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data to GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width, width * sizeof(Npp32f), cudaMemcpyHostToDevice);
  }

  // Execute copy
  NppStatus status = nppiCopy_32f_C1R(d_src, srcStep, d_dst, dstStep, roi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Verify result
  std::vector<Npp32f> resultData(width * height);
  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultData.data() + y * width, (char *)d_dst + y * dstStep, width * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  for (int i = 0; i < width * height; i++) {
    EXPECT_FLOAT_EQ(resultData[i], srcData[i])
        << "C1R pattern test failed at index " << i << " for pattern: " << params.name;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Parameterized tests for nppiCopy_32f_C3P3R
class CopyC3P3RParameterizedTest : public ::testing::TestWithParam<ImageSizeParams> {
protected:
  void SetUp() override {
    auto params = GetParam();
    width = params.width;
    height = params.height;
    roi.width = width;
    roi.height = height;
  }

  int width, height;
  NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(C3P3RImageSizes, CopyC3P3RParameterizedTest,
                         ::testing::Values(ImageSizeParams{1, 1, "SinglePixel"}, ImageSizeParams{2, 3, "SmallRect"},
                                           ImageSizeParams{4, 4, "SmallSquare"}, ImageSizeParams{7, 5, "OddSize"},
                                           ImageSizeParams{16, 8, "Rect16x8"}, ImageSizeParams{32, 32, "Square32x32"},
                                           ImageSizeParams{64, 16, "WideRect"}, ImageSizeParams{8, 64, "TallRect"},
                                           ImageSizeParams{128, 96, "MediumSize"},
                                           ImageSizeParams{8192, 4096, "Large"}),
                         [](const ::testing::TestParamInfo<ImageSizeParams> &info) {
                           return info.param.description + "_" + std::to_string(info.param.width) + "x" +
                                  std::to_string(info.param.height);
                         });

TEST_P(CopyC3P3RParameterizedTest, Copy_32f_C3P3R_ParameterizedSizes) {
  const int channels = 3;
  std::vector<Npp32f> srcData(width * height * channels);

  // Create channel-encoded packed test data
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * channels;
      srcData[idx + 0] = (float)(x + y * 200.0f);  // R channel
      srcData[idx + 1] = (float)(y + x * 300.0f);  // G channel
      srcData[idx + 2] = (float)(x * y + 1000.0f); // B channel
    }
  }

  // Allocate GPU memory - source is packed, dest is planar
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  Npp32f *d_dstR = nppiMalloc_32f_C1(width, height, &dstStep);
  Npp32f *d_dstG = nppiMalloc_32f_C1(width, height, &dstStep);
  Npp32f *d_dstB = nppiMalloc_32f_C1(width, height, &dstStep);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dstR, nullptr);
  ASSERT_NE(d_dstG, nullptr);
  ASSERT_NE(d_dstB, nullptr);

  // Copy packed source data to GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * channels, width * channels * sizeof(Npp32f),
               cudaMemcpyHostToDevice);
  }

  // Setup destination pointer array
  Npp32f *dstPtrs[3] = {d_dstR, d_dstG, d_dstB};

  // Execute packed to planar conversion
  NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, dstPtrs, dstStep, roi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Verify results
  std::vector<Npp32f> resultR(width * height);
  std::vector<Npp32f> resultG(width * height);
  std::vector<Npp32f> resultB(width * height);

  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultR.data() + y * width, (char *)d_dstR + y * dstStep, width * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(resultG.data() + y * width, (char *)d_dstG + y * dstStep, width * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(resultB.data() + y * width, (char *)d_dstB + y * dstStep, width * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  // Verify each channel
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int packedIdx = (y * width + x) * channels;
      int planarIdx = y * width + x;

      EXPECT_FLOAT_EQ(resultR[planarIdx], srcData[packedIdx + 0])
          << "R channel failed at (" << x << "," << y << ") for size " << width << "x" << height;
      EXPECT_FLOAT_EQ(resultG[planarIdx], srcData[packedIdx + 1])
          << "G channel failed at (" << x << "," << y << ") for size " << width << "x" << height;
      EXPECT_FLOAT_EQ(resultB[planarIdx], srcData[packedIdx + 2])
          << "B channel failed at (" << x << "," << y << ") for size " << width << "x" << height;
    }
  }

  nppiFree(d_src);
  nppiFree(d_dstR);
  nppiFree(d_dstG);
  nppiFree(d_dstB);
}

// Data pattern tests for nppiCopy_32f_C3P3R
class CopyC3P3RDataPatternTest : public ::testing::TestWithParam<DataPatternParams> {
protected:
  void SetUp() override {
    width = 16;
    height = 12;
    roi.width = width;
    roi.height = height;
  }

  int width, height;
  NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(C3P3RDataPatterns, CopyC3P3RDataPatternTest,
                         ::testing::Values(DataPatternParams{"Constant",
                                                             [](int x, int y, int w, int h, int c) -> float {
                                                               (void)x;
                                                               (void)y;
                                                               (void)w;
                                                               (void)h;
                                                               return (float)(c + 1) * 50.0f;
                                                             }},
                                           DataPatternParams{"ChannelGradients",
                                                             [](int x, int y, int w, int h, int c) -> float {
                                                               if (c == 0)
                                                                 return (float)x / (w - 1);
                                                               if (c == 1)
                                                                 return (float)y / (h - 1);
                                                               return (float)(x + y) / (w + h - 2);
                                                             }},
                                           DataPatternParams{"ChannelCheckers",
                                                             [](int x, int y, int w, int h, int c) -> float {
                                                               (void)w;
                                                               (void)h;
                                                               return ((x + y + c) % 2) ? 255.0f : 0.0f;
                                                             }},
                                           DataPatternParams{"ChannelSines",
                                                             [](int x, int y, int w, int h, int c) -> float {
                                                               float freq = (c + 1) * 0.5f;
                                                               return sin((float)x * freq * 3.14159f / w) *
                                                                      cos((float)y * freq * 3.14159f / h);
                                                             }},
                                           DataPatternParams{"ChannelScaled",
                                                             [](int x, int y, int w, int h, int c) -> float {
                                                               return (float)(c + 1) * ((float)x / w + (float)y / h) *
                                                                      100.0f;
                                                             }},
                                           DataPatternParams{"ChannelCircles",
                                                             [](int x, int y, int w, int h, int c) -> float {
                                                               float cx = w / 2.0f, cy = h / 2.0f;
                                                               float dx = x - cx, dy = y - cy;
                                                               float dist = sqrt(dx * dx + dy * dy);
                                                               return sin(dist * (c + 1) * 0.3f) * 128.0f + 128.0f;
                                                             }}),
                         [](const ::testing::TestParamInfo<DataPatternParams> &info) { return info.param.name; });

TEST_P(CopyC3P3RDataPatternTest, Copy_32f_C3P3R_DataPatterns) {
  auto params = GetParam();
  const int channels = 3;
  std::vector<Npp32f> srcData(width * height * channels);

  // Generate multi-channel test data
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * channels;
      for (int c = 0; c < channels; c++) {
        srcData[idx + c] = params.generator(x, y, width, height, c);
      }
    }
  }

  // Allocate GPU memory
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  Npp32f *d_dstR = nppiMalloc_32f_C1(width, height, &dstStep);
  Npp32f *d_dstG = nppiMalloc_32f_C1(width, height, &dstStep);
  Npp32f *d_dstB = nppiMalloc_32f_C1(width, height, &dstStep);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dstR, nullptr);
  ASSERT_NE(d_dstG, nullptr);
  ASSERT_NE(d_dstB, nullptr);

  // Copy data to GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * width * channels, width * channels * sizeof(Npp32f),
               cudaMemcpyHostToDevice);
  }

  // Setup destination pointer array
  Npp32f *dstPtrs[3] = {d_dstR, d_dstG, d_dstB};

  // Execute conversion
  NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, dstPtrs, dstStep, roi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // Verify results
  std::vector<Npp32f> resultR(width * height);
  std::vector<Npp32f> resultG(width * height);
  std::vector<Npp32f> resultB(width * height);

  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultR.data() + y * width, (char *)d_dstR + y * dstStep, width * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(resultG.data() + y * width, (char *)d_dstG + y * dstStep, width * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(resultB.data() + y * width, (char *)d_dstB + y * dstStep, width * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  // Verify each channel
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int packedIdx = (y * width + x) * channels;
      int planarIdx = y * width + x;

      EXPECT_FLOAT_EQ(resultR[planarIdx], srcData[packedIdx + 0])
          << "R channel pattern test failed at (" << x << "," << y << ") for pattern: " << params.name;
      EXPECT_FLOAT_EQ(resultG[planarIdx], srcData[packedIdx + 1])
          << "G channel pattern test failed at (" << x << "," << y << ") for pattern: " << params.name;
      EXPECT_FLOAT_EQ(resultB[planarIdx], srcData[packedIdx + 2])
          << "B channel pattern test failed at (" << x << "," << y << ") for pattern: " << params.name;
    }
  }

  nppiFree(d_src);
  nppiFree(d_dstR);
  nppiFree(d_dstG);
  nppiFree(d_dstB);
}

// 转换和拷贝组合参数化测试
struct ConvertCopyParams {
  std::string name;
  int width;
  int height;
  std::vector<Npp8u> inputPattern;
};

class ConvertCopyChainTest : public ::testing::TestWithParam<ConvertCopyParams> {
protected:
  void SetUp() override {
    auto params = GetParam();
    width = params.width;
    height = params.height;
    roi.width = width;
    roi.height = height;
  }

  int width, height;
  NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(
    ConvertCopyChains, ConvertCopyChainTest,
    ::testing::Values(ConvertCopyParams{"Boundary2x2", 2, 2, {0, 128, 255, 64, 192, 32, 96, 160, 224, 16, 48, 80}},
                      ConvertCopyParams{"Gradient4x3", 4, 3, {0,   0,   0,   85,  85,  85,  170, 170, 170,
                                                              255, 255, 255, 64,  64,  64,  128, 128, 128,
                                                              192, 192, 192, 224, 224, 224, 32,  96,  160,
                                                              48,  112, 176, 64,  128, 192, 80,  144, 208}}),
    [](const ::testing::TestParamInfo<ConvertCopyParams> &info) { return info.param.name; });

// 参数化测试转换和拷贝的组合操作
TEST_P(ConvertCopyChainTest, ConvertThenCopy_8u32f_C3R) {
  auto params = GetParam();
  const int channels = 3;

  // 分配内存
  int srcStep, tmpStep, dstStep;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  Npp32f *d_tmp = nppiMalloc_32f_C3(width, height, &tmpStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(width, height, &dstStep);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_tmp, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制输入数据到GPU
  for (int y = 0; y < height; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, params.inputPattern.data() + y * width * channels,
               width * channels * sizeof(Npp8u), cudaMemcpyHostToDevice);
  }

  // 步骤1: 转换 8u -> 32f
  NppStatus status1 = nppiConvert_8u32f_C3R(d_src, srcStep, d_tmp, tmpStep, roi);
  EXPECT_EQ(status1, NPP_SUCCESS);

  // 步骤2: 拷贝 32f -> 32f
  NppStatus status2 = nppiCopy_32f_C3R(d_tmp, tmpStep, d_dst, dstStep, roi);
  EXPECT_EQ(status2, NPP_SUCCESS);

  // Validate最终结果
  std::vector<Npp32f> resultData(width * height * channels);
  for (int y = 0; y < height; y++) {
    cudaMemcpy(resultData.data() + y * width * channels, (char *)d_dst + y * dstStep, width * channels * sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
  }

  // Validate转换正确性
  for (int i = 0; i < width * height * channels; i++) {
    float expected = (float)params.inputPattern[i];
    EXPECT_FLOAT_EQ(resultData[i], expected) << "转换拷贝链测试失败 at index " << i << " for " << params.name;
  }

  nppiFree(d_src);
  nppiFree(d_tmp);
  nppiFree(d_dst);
}

// ROI尺寸参数化测试
struct ROIParams {
  std::string name;
  int fullWidth;
  int fullHeight;
  int roiWidth;
  int roiHeight;
};

class CopyROITest : public ::testing::TestWithParam<ROIParams> {
protected:
  void SetUp() override {
    auto params = GetParam();
    fullWidth = params.fullWidth;
    fullHeight = params.fullHeight;
    roi.width = params.roiWidth;
    roi.height = params.roiHeight;
  }

  int fullWidth, fullHeight;
  NppiSize roi;
};

INSTANTIATE_TEST_SUITE_P(DifferentROISizes, CopyROITest,
                         ::testing::Values(ROIParams{"FullImage", 8, 6, 8, 6}, ROIParams{"TopLeft", 8, 6, 4, 3},
                                           ROIParams{"SingleRow", 8, 6, 8, 1}, ROIParams{"SingleCol", 8, 6, 1, 6},
                                           ROIParams{"CenterArea", 16, 12, 8, 6},
                                           ROIParams{"SmallBlock", 32, 32, 4, 4}),
                         [](const ::testing::TestParamInfo<ROIParams> &info) {
                           return info.param.name + "_" + std::to_string(info.param.roiWidth) + "x" +
                                  std::to_string(info.param.roiHeight);
                         });

// 参数化测试不同ROI尺寸的拷贝
TEST_P(CopyROITest, Copy_32f_C3R_DifferentROI) {
  const int channels = 3;
  std::vector<Npp32f> srcData(fullWidth * fullHeight * channels);

  // 创建唯一标识每个像素的数据
  for (int y = 0; y < fullHeight; y++) {
    for (int x = 0; x < fullWidth; x++) {
      int idx = (y * fullWidth + x) * channels;
      srcData[idx + 0] = (float)(y * 1000 + x); // R: 唯一ID
      srcData[idx + 1] = (float)(x);            // G: X坐标
      srcData[idx + 2] = (float)(y);            // B: Y坐标
    }
  }

  // 分配GPU内存（使用全图尺寸）
  int srcStep, dstStep;
  Npp32f *d_src = nppiMalloc_32f_C3(fullWidth, fullHeight, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C3(fullWidth, fullHeight, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // 复制数据到GPU
  for (int y = 0; y < fullHeight; y++) {
    cudaMemcpy((char *)d_src + y * srcStep, srcData.data() + y * fullWidth * channels,
               fullWidth * channels * sizeof(Npp32f), cudaMemcpyHostToDevice);
  }

  // 执行ROI拷贝
  NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  EXPECT_EQ(status, NPP_SUCCESS);

  // ValidateROI区域的结果
  std::vector<Npp32f> resultData(roi.width * roi.height * channels);
  for (int y = 0; y < roi.height; y++) {
    cudaMemcpy(resultData.data() + y * roi.width * channels, (char *)d_dst + y * dstStep,
               roi.width * channels * sizeof(Npp32f), cudaMemcpyDeviceToHost);
  }

  // ValidateROI内的数据正确性
  for (int y = 0; y < roi.height; y++) {
    for (int x = 0; x < roi.width; x++) {
      int resultIdx = (y * roi.width + x) * channels;
      int srcIdx = (y * fullWidth + x) * channels;

      for (int c = 0; c < channels; c++) {
        EXPECT_FLOAT_EQ(resultData[resultIdx + c], srcData[srcIdx + c])
            << "ROI拷贝失败 at (" << x << "," << y << ") channel " << c;
      }
    }
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

class NppiCopy32fC1RTest : public ::testing::Test {
protected:
  void SetUp() override {
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess);
  }

  void TearDown() override {
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
  }

  // Helper to allocate aligned memory
  void allocateImage(Npp32f **ppImg, int width, int height, int *pStep) {
    *ppImg = nppiMalloc_32f_C1(width, height, pStep);
    ASSERT_NE(*ppImg, nullptr);
  }

  // Helper to verify data integrity
  bool verifyData(const std::vector<Npp32f> &expected, const std::vector<Npp32f> &actual, float tolerance = 0.0f) {
    if (expected.size() != actual.size())
      return false;

    for (size_t i = 0; i < expected.size(); i++) {
      if (std::abs(expected[i] - actual[i]) > tolerance) {
        return false;
      }
    }
    return true;
  }

  // Generate test pattern
  void generatePattern(std::vector<Npp32f> &data, int width, int height, float scale = 1.0f) {
    data.resize(width * height);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        data[y * width + x] = ((y * width + x) % 1000) * scale;
      }
    }
  }
};

// Test 1: Basic copy functionality
TEST_F(NppiCopy32fC1RTest, BasicCopy) {
  const int width = 256;
  const int height = 256;

  std::vector<Npp32f> hostSrc(width * height);
  generatePattern(hostSrc, width, height);

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  // Upload source data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  // Perform copy
  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_32f_C1R(d_src, srcStep, d_dst, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Verify result
  std::vector<Npp32f> hostDst(width * height);
  cudaMemcpy2D(hostDst.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  EXPECT_TRUE(verifyData(hostSrc, hostDst));

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 2: Partial ROI copy
TEST_F(NppiCopy32fC1RTest, PartialROICopy) {
  const int width = 512;
  const int height = 512;
  const int roiX = 128;
  const int roiY = 128;
  const int roiWidth = 256;
  const int roiHeight = 256;

  std::vector<Npp32f> hostSrc(width * height);
  std::vector<Npp32f> hostDst(width * height, -999.0f);
  generatePattern(hostSrc, width, height);

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  // Upload data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  // Copy ROI
  Npp32f *pSrcROI = (Npp32f *)((Npp8u *)d_src + roiY * srcStep) + roiX;
  Npp32f *pDstROI = (Npp32f *)((Npp8u *)d_dst + roiY * dstStep) + roiX;

  NppiSize roi = {roiWidth, roiHeight};
  NppStatus status = nppiCopy_32f_C1R(pSrcROI, srcStep, pDstROI, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Verify result
  cudaMemcpy2D(hostDst.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = y * width + x;
      if (x >= roiX && x < roiX + roiWidth && y >= roiY && y < roiY + roiHeight) {
        EXPECT_FLOAT_EQ(hostDst[idx], hostSrc[idx]);
      } else {
        EXPECT_FLOAT_EQ(hostDst[idx], -999.0f);
      }
    }
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 4: Different source and destination strides
TEST_F(NppiCopy32fC1RTest, DifferentStrides) {
  const int width = 200;
  const int height = 200;
  const int srcExtraBytes = 256;
  const int dstExtraBytes = 128;

  int srcStep = width * sizeof(Npp32f) + srcExtraBytes;
  int dstStep = width * sizeof(Npp32f) + dstExtraBytes;

  // Allocate with custom strides
  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  cudaMalloc(&d_src, srcStep * height);
  cudaMalloc(&d_dst, dstStep * height);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Create test data
  std::vector<Npp8u> hostSrcBuffer(srcStep * height);
  for (int y = 0; y < height; y++) {
    Npp32f *row = (Npp32f *)(hostSrcBuffer.data() + y * srcStep);
    for (int x = 0; x < width; x++) {
      row[x] = y * 1000.0f + x;
    }
  }

  cudaMemcpy(d_src, hostSrcBuffer.data(), srcStep * height, cudaMemcpyHostToDevice);

  // Perform copy
  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_32f_C1R(d_src, srcStep, d_dst, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Verify
  std::vector<Npp8u> hostDstBuffer(dstStep * height);
  cudaMemcpy(hostDstBuffer.data(), d_dst, dstStep * height, cudaMemcpyDeviceToHost);

  for (int y = 0; y < height; y++) {
    Npp32f *srcRow = (Npp32f *)(hostSrcBuffer.data() + y * srcStep);
    Npp32f *dstRow = (Npp32f *)(hostDstBuffer.data() + y * dstStep);
    for (int x = 0; x < width; x++) {
      EXPECT_FLOAT_EQ(dstRow[x], srcRow[x]);
    }
  }

  cudaFree(d_src);
  cudaFree(d_dst);
}

// Test 5: Large image copy performance
TEST_F(NppiCopy32fC1RTest, LargeImagePerformance) {
  const int width = 8192;
  const int height = 4096;
  const int iterations = 10;

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  // Fill source with test pattern
  cudaMemset(d_src, 0x3F, srcStep * height); // ~0.5f pattern

  NppiSize roi = {width, height};

  // Warm up
  nppiCopy_32f_C1R(d_src, srcStep, d_dst, dstStep, roi);
  cudaDeviceSynchronize();

  // Measure performance
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    NppStatus status = nppiCopy_32f_C1R(d_src, srcStep, d_dst, dstStep, roi);
    ASSERT_EQ(status, NPP_SUCCESS);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  double avgTime = duration / (double)iterations;
  double dataSize = width * height * sizeof(Npp32f) * 2 / 1024.0 / 1024.0 / 1024.0; // GB
  double bandwidth = dataSize * 1000000.0 / avgTime;                                // GB/s

  std::cout << "Large image copy performance:" << std::endl;
  std::cout << "  Image size: " << width << "x" << height << std::endl;
  std::cout << "  Average time: " << avgTime << " us" << std::endl;
  std::cout << "  Bandwidth: " << bandwidth << " GB/s" << std::endl;

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 6: Concurrent stream operations
TEST_F(NppiCopy32fC1RTest, DISABLED_ConcurrentStreams) {
  const int numStreams = 8;
  const int width = 1024;
  const int height = 1024;

  std::vector<cudaStream_t> streams(numStreams);
  std::vector<Npp32f *> srcBuffers(numStreams);
  std::vector<Npp32f *> dstBuffers(numStreams);
  std::vector<int> srcSteps(numStreams);
  std::vector<int> dstSteps(numStreams);

  // Create streams and allocate buffers
  for (int i = 0; i < numStreams; i++) {
    cudaStreamCreate(&streams[i]);
    allocateImage(&srcBuffers[i], width, height, &srcSteps[i]);
    allocateImage(&dstBuffers[i], width, height, &dstSteps[i]);

    // Initialize with different patterns
    std::vector<Npp32f> pattern(width * height);
    generatePattern(pattern, width, height, static_cast<float>(i + 1));
    cudaMemcpy2DAsync(srcBuffers[i], srcSteps[i], pattern.data(), width * sizeof(Npp32f), width * sizeof(Npp32f),
                      height, cudaMemcpyHostToDevice, streams[i]);
  }

  // Launch concurrent copies
  NppiSize roi = {width, height};
  for (int i = 0; i < numStreams; i++) {
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);
    ctx.hStream = streams[i];

    NppStatus status = nppiCopy_32f_C1R_Ctx(srcBuffers[i], srcSteps[i], dstBuffers[i], dstSteps[i], roi, ctx);
    EXPECT_EQ(status, NPP_SUCCESS);
  }

  // Verify results
  for (int i = 0; i < numStreams; i++) {
    cudaStreamSynchronize(streams[i]);

    std::vector<Npp32f> result(width * height);
    cudaMemcpy2D(result.data(), width * sizeof(Npp32f), dstBuffers[i], dstSteps[i], width * sizeof(Npp32f), height,
                 cudaMemcpyDeviceToHost);

    // Check first and last values
    float expectedFirst = 0.0f;
    float expectedLast = ((height - 1) * width + (width - 1)) * (i + 1);
    EXPECT_FLOAT_EQ(result[0], expectedFirst);
    EXPECT_NEAR(result[width * height - 1], expectedLast, 0.01f);
  }

  // Cleanup
  for (int i = 0; i < numStreams; i++) {
    cudaStreamDestroy(streams[i]);
    nppiFree(srcBuffers[i]);
    nppiFree(dstBuffers[i]);
  }
}

// Test 7: Edge alignment cases
TEST_F(NppiCopy32fC1RTest, EdgeAlignmentCases) {
  const int testWidths[] = {1, 3, 7, 13, 17, 31, 63, 127, 251, 509, 1021, 2047};
  const int height = 17; // Prime number for extra edge case

  for (int width : testWidths) {
    Npp32f *d_src = nullptr;
    Npp32f *d_dst = nullptr;
    int srcStep, dstStep;

    d_src = nppiMalloc_32f_C1(width, height, &srcStep);
    d_dst = nppiMalloc_32f_C1(width, height, &dstStep);

    ASSERT_NE(d_src, nullptr) << "Allocation failed for width " << width;
    ASSERT_NE(d_dst, nullptr) << "Allocation failed for width " << width;

    // Create unique pattern
    std::vector<Npp32f> pattern(width * height);
    for (int i = 0; i < width * height; i++) {
      pattern[i] = static_cast<float>(i * 7 + width);
    }

    cudaMemcpy2D(d_src, srcStep, pattern.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
                 cudaMemcpyHostToDevice);

    NppiSize roi = {width, height};
    NppStatus status = nppiCopy_32f_C1R(d_src, srcStep, d_dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS) << "Copy failed for width " << width;

    // Verify
    std::vector<Npp32f> result(width * height);
    cudaMemcpy2D(result.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f), height,
                 cudaMemcpyDeviceToHost);

    EXPECT_TRUE(verifyData(pattern, result)) << "Data mismatch for width " << width;

    nppiFree(d_src);
    nppiFree(d_dst);
  }
}

// Test 8: Error handling
TEST_F(NppiCopy32fC1RTest, DISABLED_ErrorHandling) {
  const int width = 100;
  const int height = 100;

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  NppiSize roi = {width, height};

  // Test null source
  EXPECT_EQ(nppiCopy_32f_C1R(nullptr, srcStep, d_dst, dstStep, roi), NPP_NULL_POINTER_ERROR);

  // Test null destination
  EXPECT_EQ(nppiCopy_32f_C1R(d_src, srcStep, nullptr, dstStep, roi), NPP_NULL_POINTER_ERROR);

  // Test invalid ROI
  NppiSize invalidRoi = {0, height};
  EXPECT_EQ(nppiCopy_32f_C1R(d_src, srcStep, d_dst, dstStep, invalidRoi), NPP_SUCCESS);

  invalidRoi = {width, -5};
  EXPECT_EQ(nppiCopy_32f_C1R(d_src, srcStep, d_dst, dstStep, invalidRoi), NPP_SIZE_ERROR);

  // Test invalid step
  EXPECT_EQ(nppiCopy_32f_C1R(d_src, 0, d_dst, dstStep, roi), NPP_SUCCESS);
  EXPECT_NE(nppiCopy_32f_C1R(d_src, srcStep, d_dst, -1, roi), NPP_SUCCESS);

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 9: Stress test with random operations
TEST_F(NppiCopy32fC1RTest, StressTest) {
  const int numOperations = 100;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> sizeDist(1, 100);
  std::uniform_int_distribution<> coordDist(0, 50);

  for (int op = 0; op < numOperations; op++) {
    // Random dimensions
    int srcWidth = sizeDist(gen) * 4;
    int srcHeight = sizeDist(gen) * 4;
    int dstWidth = srcWidth + coordDist(gen) * 4;
    int dstHeight = srcHeight + coordDist(gen) * 4;

    // Random ROI
    int roiWidth = std::min(srcWidth, sizeDist(gen) * 2);
    int roiHeight = std::min(srcHeight, sizeDist(gen) * 2);
    int roiX = coordDist(gen) % (srcWidth - roiWidth + 1);
    int roiY = coordDist(gen) % (srcHeight - roiHeight + 1);

    Npp32f *d_src = nullptr;
    Npp32f *d_dst = nullptr;
    int srcStep, dstStep;

    d_src = nppiMalloc_32f_C1(srcWidth, srcHeight, &srcStep);
    d_dst = nppiMalloc_32f_C1(dstWidth, dstHeight, &dstStep);

    if (d_src && d_dst) {
      // Initialize with pattern
      std::vector<Npp32f> pattern(srcWidth * srcHeight);
      generatePattern(pattern, srcWidth, srcHeight, op * 0.1f);
      cudaMemcpy2D(d_src, srcStep, pattern.data(), srcWidth * sizeof(Npp32f), srcWidth * sizeof(Npp32f), srcHeight,
                   cudaMemcpyHostToDevice);

      // Perform copy
      Npp32f *pSrcROI = (Npp32f *)((Npp8u *)d_src + roiY * srcStep) + roiX;
      Npp32f *pDstROI = d_dst;

      NppiSize roi = {roiWidth, roiHeight};
      NppStatus status = nppiCopy_32f_C1R(pSrcROI, srcStep, pDstROI, dstStep, roi);
      EXPECT_EQ(status, NPP_SUCCESS) << "Operation " << op << " failed";

      nppiFree(d_src);
      nppiFree(d_dst);
    }
  }
}

// Test 10: Zero-copy scenarios (device to device)
TEST_F(NppiCopy32fC1RTest, DeviceToDeviceCopy) {
  const int width = 512;
  const int height = 512;

  // Create multiple GPU buffers
  Npp32f *d_buf1 = nullptr;
  Npp32f *d_buf2 = nullptr;
  Npp32f *d_buf3 = nullptr;
  int step1, step2, step3;

  allocateImage(&d_buf1, width, height, &step1);
  allocateImage(&d_buf2, width, height, &step2);
  allocateImage(&d_buf3, width, height, &step3);

  // Initialize first buffer
  std::vector<Npp32f> initialData(width * height);
  generatePattern(initialData, width, height, 3.14f);
  cudaMemcpy2D(d_buf1, step1, initialData.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};

  // Chain of copies: buf1 -> buf2 -> buf3
  NppStatus status1 = nppiCopy_32f_C1R(d_buf1, step1, d_buf2, step2, roi);
  NppStatus status2 = nppiCopy_32f_C1R(d_buf2, step2, d_buf3, step3, roi);

  ASSERT_EQ(status1, NPP_SUCCESS);
  ASSERT_EQ(status2, NPP_SUCCESS);

  // Verify final result
  std::vector<Npp32f> finalData(width * height);
  cudaMemcpy2D(finalData.data(), width * sizeof(Npp32f), d_buf3, step3, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  EXPECT_TRUE(verifyData(initialData, finalData, 0.001f));

  nppiFree(d_buf1);
  nppiFree(d_buf2);
  nppiFree(d_buf3);
}

// Test 11: Special Float Values Handling
TEST_F(NppiCopy32fC1RTest, SpecialFloatValuesHandling) {
  const int width = 64, height = 64;

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  std::vector<Npp32f> srcData(width * height);

  // Fill with special float values including infinity and extreme values
  for (int i = 0; i < width * height; i++) {
    switch (i % 8) {
    case 0:
      srcData[i] = 0.0f;
      break;
    case 1:
      srcData[i] = -0.0f;
      break;
    case 2:
      srcData[i] = std::numeric_limits<float>::max();
      break;
    case 3:
      srcData[i] = std::numeric_limits<float>::lowest();
      break;
    case 4:
      srcData[i] = std::numeric_limits<float>::infinity();
      break;
    case 5:
      srcData[i] = -std::numeric_limits<float>::infinity();
      break;
    case 6:
      srcData[i] = std::numeric_limits<float>::epsilon();
      break;
    case 7:
      srcData[i] = std::numeric_limits<float>::denorm_min();
      break;
    }
  }

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_32f_C1R(d_src, srcStep, d_dst, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> dstData(width * height);
  cudaMemcpy2D(dstData.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  // Verify special values are preserved
  for (int i = 0; i < width * height; i++) {
    if (std::isfinite(srcData[i])) {
      EXPECT_FLOAT_EQ(dstData[i], srcData[i]) << "Finite value mismatch at index " << i;
    } else {
      EXPECT_EQ(std::signbit(dstData[i]), std::signbit(srcData[i])) << "Sign mismatch at index " << i;
      EXPECT_EQ(std::isinf(dstData[i]), std::isinf(srcData[i])) << "Infinity state mismatch at index " << i;
    }
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 12: Stream Context Performance
TEST_F(NppiCopy32fC1RTest, StreamContextPerformance) {
  const int width = 1024, height = 1024;
  const int iterations = 10;

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  std::vector<Npp32f> srcData(width * height);
  generatePattern(srcData, width, height, 1.5f);
  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp32f), width * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  nppGetStreamContext(&ctx);

  // Measure performance with stream context
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    NppStatus status = nppiCopy_32f_C1R_Ctx(d_src, srcStep, d_dst, dstStep, roi, ctx);
    ASSERT_EQ(status, NPP_SUCCESS);
  }
  cudaStreamSynchronize(ctx.hStream);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  double avgTime = duration / (double)iterations;
  double dataSize = width * height * sizeof(Npp32f) * 2 / 1024.0 / 1024.0; // MB (read + write)
  double bandwidth = dataSize * 1000000.0 / avgTime;                       // MB/s

  std::cout << "Stream context copy performance:" << std::endl;
  std::cout << "  Average time: " << avgTime << " μs" << std::endl;
  std::cout << "  Bandwidth: " << bandwidth << " MB/s" << std::endl;

  // Verify correctness
  std::vector<Npp32f> dstData(width * height);
  cudaMemcpy2D(dstData.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);
  EXPECT_TRUE(verifyData(srcData, dstData));

  nppiFree(d_src);
  nppiFree(d_dst);
}

class NppiCopy32fC3P3RTest : public ::testing::Test {
protected:
  void SetUp() override {
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess);
  }

  void TearDown() override {
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
  }

  // Helper to generate test pattern for 3-channel interleaved data
  void generateInterleavedPattern(std::vector<Npp32f> &data, int width, int height) {
    data.resize(width * height * 3);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (y * width + x) * 3;
        data[idx] = x * 1.0f;           // R channel
        data[idx + 1] = y * 2.0f;       // G channel
        data[idx + 2] = (x + y) * 3.0f; // B channel
      }
    }
  }

  // Helper to verify planar data
  bool verifyPlanarData(const std::vector<Npp32f> &interleaved, const std::vector<Npp32f> &planarR,
                        const std::vector<Npp32f> &planarG, const std::vector<Npp32f> &planarB, int width, int height) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int interleavedIdx = (y * width + x) * 3;
        int planarIdx = y * width + x;

        if (planarR[planarIdx] != interleaved[interleavedIdx] ||
            planarG[planarIdx] != interleaved[interleavedIdx + 1] ||
            planarB[planarIdx] != interleaved[interleavedIdx + 2]) {
          return false;
        }
      }
    }
    return true;
  }
};

// Test 1: Basic interleaved to planar conversion
TEST_F(NppiCopy32fC3P3RTest, BasicInterleavedToPlanar) {
  const int width = 256;
  const int height = 256;

  std::vector<Npp32f> hostSrc(width * height * 3);
  generateInterleavedPattern(hostSrc, width, height);

  // Allocate GPU memory
  Npp32f *d_src = nullptr;
  int srcStep;
  d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);

  Npp32f *d_dstPlanes[3];
  int dstStep;
  for (int i = 0; i < 3; i++) {
    d_dstPlanes[i] = nppiMalloc_32f_C1(width, height, &dstStep);
    ASSERT_NE(d_dstPlanes[i], nullptr);
  }

  // Upload interleaved data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  // Perform copy
  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Download planar data
  std::vector<Npp32f> hostR(width * height), hostG(width * height), hostB(width * height);
  cudaMemcpy2D(hostR.data(), width * sizeof(Npp32f), d_dstPlanes[0], dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);
  cudaMemcpy2D(hostG.data(), width * sizeof(Npp32f), d_dstPlanes[1], dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);
  cudaMemcpy2D(hostB.data(), width * sizeof(Npp32f), d_dstPlanes[2], dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  // Verify
  EXPECT_TRUE(verifyPlanarData(hostSrc, hostR, hostG, hostB, width, height));

  // Cleanup
  nppiFree(d_src);
  for (int i = 0; i < 3; i++) {
    nppiFree(d_dstPlanes[i]);
  }
}

// Test 2: ROI operations
TEST_F(NppiCopy32fC3P3RTest, ROIOperations) {
  const int width = 512;
  const int height = 512;
  const int roiX = 128;
  const int roiY = 128;
  const int roiWidth = 256;
  const int roiHeight = 256;

  std::vector<Npp32f> hostSrc(width * height * 3);
  generateInterleavedPattern(hostSrc, width, height);

  // Allocate GPU memory
  Npp32f *d_src = nullptr;
  int srcStep;
  d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);

  Npp32f *d_dstPlanes[3];
  int dstStep;
  for (int i = 0; i < 3; i++) {
    d_dstPlanes[i] = nppiMalloc_32f_C1(width, height, &dstStep);
    ASSERT_NE(d_dstPlanes[i], nullptr);
    // Initialize with -999 to verify ROI boundaries
    cudaMemset(d_dstPlanes[i], 0xFF, dstStep * height);
  }

  // Upload source data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  // Set up ROI pointers
  Npp32f *pSrcROI = (Npp32f *)((Npp8u *)d_src + roiY * srcStep) + roiX * 3;
  Npp32f *pDstPlanesROI[3];
  for (int i = 0; i < 3; i++) {
    pDstPlanesROI[i] = (Npp32f *)((Npp8u *)d_dstPlanes[i] + roiY * dstStep) + roiX;
  }

  // Perform ROI copy
  NppiSize roi = {roiWidth, roiHeight};
  NppStatus status = nppiCopy_32f_C3P3R(pSrcROI, srcStep, pDstPlanesROI, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Verify ROI data
  std::vector<Npp32f> hostR(width * height), hostG(width * height), hostB(width * height);
  cudaMemcpy2D(hostR.data(), width * sizeof(Npp32f), d_dstPlanes[0], dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);
  cudaMemcpy2D(hostG.data(), width * sizeof(Npp32f), d_dstPlanes[1], dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);
  cudaMemcpy2D(hostB.data(), width * sizeof(Npp32f), d_dstPlanes[2], dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  // Check ROI region
  for (int y = roiY; y < roiY + roiHeight; y++) {
    for (int x = roiX; x < roiX + roiWidth; x++) {
      int interleavedIdx = (y * width + x) * 3;
      int planarIdx = y * width + x;
      EXPECT_FLOAT_EQ(hostR[planarIdx], hostSrc[interleavedIdx]);
      EXPECT_FLOAT_EQ(hostG[planarIdx], hostSrc[interleavedIdx + 1]);
      EXPECT_FLOAT_EQ(hostB[planarIdx], hostSrc[interleavedIdx + 2]);
    }
  }

  // Cleanup
  nppiFree(d_src);
  for (int i = 0; i < 3; i++) {
    nppiFree(d_dstPlanes[i]);
  }
}

// Test 3: Various image dimensions
TEST_F(NppiCopy32fC3P3RTest, VariousDimensions) {
  struct TestDimension {
    int width, height;
    const char *description;
  };

  TestDimension dimensions[] = {{1, 1, "Single pixel"},  {17, 13, "Prime dimensions"}, {640, 480, "VGA"},
                                {1920, 1080, "Full HD"}, {31, 1, "Single row"},        {1, 100, "Single column"}};

  for (const auto &dim : dimensions) {
    Npp32f *d_src = nullptr;
    int srcStep;
    d_src = nppiMalloc_32f_C3(dim.width, dim.height, &srcStep);

    if (d_src == nullptr) {
      continue; // Skip if allocation fails
    }

    Npp32f *d_dstPlanes[3];
    int dstStep;
    bool allocationSuccess = true;
    for (int i = 0; i < 3; i++) {
      d_dstPlanes[i] = nppiMalloc_32f_C1(dim.width, dim.height, &dstStep);
      if (d_dstPlanes[i] == nullptr) {
        allocationSuccess = false;
        // Free already allocated planes
        for (int j = 0; j < i; j++) {
          nppiFree(d_dstPlanes[j]);
        }
        break;
      }
    }

    if (!allocationSuccess) {
      nppiFree(d_src);
      continue;
    }

    // Test pattern
    std::vector<Npp32f> pattern(dim.width * dim.height * 3);
    generateInterleavedPattern(pattern, dim.width, dim.height);
    cudaMemcpy2D(d_src, srcStep, pattern.data(), dim.width * 3 * sizeof(Npp32f), dim.width * 3 * sizeof(Npp32f),
                 dim.height, cudaMemcpyHostToDevice);

    NppiSize roi = {dim.width, dim.height};
    NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS) << "Failed for " << dim.description << " (" << dim.width << "x" << dim.height << ")";

    // Cleanup
    nppiFree(d_src);
    for (int i = 0; i < 3; i++) {
      nppiFree(d_dstPlanes[i]);
    }
  }
}

// Test 4: Performance with large images
TEST_F(NppiCopy32fC3P3RTest, LargeImagePerformance) {
  const int width = 4096;
  const int height = 2160;
  const int iterations = 10;

  Npp32f *d_src = nullptr;
  int srcStep;
  d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);

  Npp32f *d_dstPlanes[3];
  int dstStep;
  for (int i = 0; i < 3; i++) {
    d_dstPlanes[i] = nppiMalloc_32f_C1(width, height, &dstStep);
    ASSERT_NE(d_dstPlanes[i], nullptr);
  }

  // Initialize with pattern
  cudaMemset(d_src, 0x3F, srcStep * height);

  NppiSize roi = {width, height};

  // Warm up
  nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, dstStep, roi);
  cudaDeviceSynchronize();

  // Measure performance
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, dstStep, roi);
    ASSERT_EQ(status, NPP_SUCCESS);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  double avgTime = duration / (double)iterations;
  double dataSize = width * height * 3 * sizeof(Npp32f) * 2 / 1024.0 / 1024.0 / 1024.0; // GB
  double bandwidth = dataSize * 1000000.0 / avgTime;                                    // GB/s

  std::cout << "Interleaved to planar copy performance:" << std::endl;
  std::cout << "  Image size: " << width << "x" << height << std::endl;
  std::cout << "  Average time: " << avgTime << " us" << std::endl;
  std::cout << "  Bandwidth: " << bandwidth << " GB/s" << std::endl;

  // Cleanup
  nppiFree(d_src);
  for (int i = 0; i < 3; i++) {
    nppiFree(d_dstPlanes[i]);
  }
}

// Test 5: Memory alignment edge cases
TEST_F(NppiCopy32fC3P3RTest, MemoryAlignmentEdgeCases) {
  const int testWidths[] = {1, 3, 7, 13, 31, 63, 127, 255, 333, 511, 1023};
  const int height = 17;

  for (int width : testWidths) {
    Npp32f *d_src = nullptr;
    int srcStep;
    d_src = nppiMalloc_32f_C3(width, height, &srcStep);

    if (d_src == nullptr)
      continue;

    Npp32f *d_dstPlanes[3];
    int dstStep;
    bool allocSuccess = true;

    for (int i = 0; i < 3; i++) {
      d_dstPlanes[i] = nppiMalloc_32f_C1(width, height, &dstStep);
      if (d_dstPlanes[i] == nullptr) {
        allocSuccess = false;
        for (int j = 0; j < i; j++) {
          nppiFree(d_dstPlanes[j]);
        }
        break;
      }
    }

    if (!allocSuccess) {
      nppiFree(d_src);
      continue;
    }

    // Create unique pattern
    std::vector<Npp32f> pattern(width * height * 3);
    for (int i = 0; i < width * height; i++) {
      pattern[i * 3] = static_cast<float>(i);
      pattern[i * 3 + 1] = static_cast<float>(i * 2);
      pattern[i * 3 + 2] = static_cast<float>(i * 3);
    }

    cudaMemcpy2D(d_src, srcStep, pattern.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
                 cudaMemcpyHostToDevice);

    NppiSize roi = {width, height};
    NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS) << "Failed for width " << width;

    // Verify first pixel of each plane
    Npp32f firstPixelR, firstPixelG, firstPixelB;
    cudaMemcpy(&firstPixelR, d_dstPlanes[0], sizeof(Npp32f), cudaMemcpyDeviceToHost);
    cudaMemcpy(&firstPixelG, d_dstPlanes[1], sizeof(Npp32f), cudaMemcpyDeviceToHost);
    cudaMemcpy(&firstPixelB, d_dstPlanes[2], sizeof(Npp32f), cudaMemcpyDeviceToHost);

    EXPECT_FLOAT_EQ(firstPixelR, 0.0f) << "Wrong R value for width " << width;
    EXPECT_FLOAT_EQ(firstPixelG, 0.0f) << "Wrong G value for width " << width;
    EXPECT_FLOAT_EQ(firstPixelB, 0.0f) << "Wrong B value for width " << width;

    // Cleanup
    nppiFree(d_src);
    for (int i = 0; i < 3; i++) {
      nppiFree(d_dstPlanes[i]);
    }
  }
}

// Test 6: Different source and destination strides
TEST_F(NppiCopy32fC3P3RTest, DifferentStrides) {
  const int width = 200;
  const int height = 200;
  const int srcExtraBytes = 256;
  const int dstExtraBytes = 128;

  int srcStep = width * 3 * sizeof(Npp32f) + srcExtraBytes;
  int dstStep = width * sizeof(Npp32f) + dstExtraBytes;

  // Allocate with custom strides
  Npp32f *d_src = nullptr;
  Npp32f *d_dstPlanes[3];

  cudaMalloc(&d_src, srcStep * height);
  ASSERT_NE(d_src, nullptr);

  for (int i = 0; i < 3; i++) {
    cudaMalloc(&d_dstPlanes[i], dstStep * height);
    ASSERT_NE(d_dstPlanes[i], nullptr);
  }

  // Create test data with custom stride
  std::vector<Npp8u> hostSrcBuffer(srcStep * height);
  for (int y = 0; y < height; y++) {
    Npp32f *row = (Npp32f *)(hostSrcBuffer.data() + y * srcStep);
    for (int x = 0; x < width; x++) {
      row[x * 3] = y * 1000.0f + x;
      row[x * 3 + 1] = y * 1000.0f + x + 0.1f;
      row[x * 3 + 2] = y * 1000.0f + x + 0.2f;
    }
  }

  cudaMemcpy(d_src, hostSrcBuffer.data(), srcStep * height, cudaMemcpyHostToDevice);

  // Perform copy
  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Verify first row of each plane
  std::vector<Npp32f> firstRowR(width);
  std::vector<Npp32f> firstRowG(width);
  std::vector<Npp32f> firstRowB(width);

  cudaMemcpy(firstRowR.data(), d_dstPlanes[0], width * sizeof(Npp32f), cudaMemcpyDeviceToHost);
  cudaMemcpy(firstRowG.data(), d_dstPlanes[1], width * sizeof(Npp32f), cudaMemcpyDeviceToHost);
  cudaMemcpy(firstRowB.data(), d_dstPlanes[2], width * sizeof(Npp32f), cudaMemcpyDeviceToHost);

  for (int x = 0; x < width; x++) {
    EXPECT_FLOAT_EQ(firstRowR[x], x);
    EXPECT_FLOAT_EQ(firstRowG[x], x + 0.1f);
    EXPECT_FLOAT_EQ(firstRowB[x], x + 0.2f);
  }

  // Cleanup
  cudaFree(d_src);
  for (int i = 0; i < 3; i++) {
    cudaFree(d_dstPlanes[i]);
  }
}

// Test 7: Error handling
TEST_F(NppiCopy32fC3P3RTest, DISABLED_ErrorHandling) {
  const int width = 100;
  const int height = 100;

  Npp32f *d_src = nullptr;
  int srcStep, dstStep;
  d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);

  Npp32f *d_dstPlanes[3];
  for (int i = 0; i < 3; i++) {
    d_dstPlanes[i] = nppiMalloc_32f_C1(width, height, &dstStep);
    ASSERT_NE(d_dstPlanes[i], nullptr);
  }

  NppiSize roi = {width, height};

  // Test null source
  EXPECT_EQ(nppiCopy_32f_C3P3R(nullptr, srcStep, d_dstPlanes, dstStep, roi), NPP_NULL_POINTER_ERROR);

  // Test null destination array
  EXPECT_EQ(nppiCopy_32f_C3P3R(d_src, srcStep, nullptr, dstStep, roi), NPP_NULL_POINTER_ERROR);

  // Test null destination plane
  Npp32f *badPlanes[3] = {d_dstPlanes[0], nullptr, d_dstPlanes[2]};
  EXPECT_EQ(nppiCopy_32f_C3P3R(d_src, srcStep, badPlanes, dstStep, roi), NPP_NULL_POINTER_ERROR);

  // Test invalid ROI
  NppiSize invalidRoi = {0, height};
  EXPECT_EQ(nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, dstStep, invalidRoi), NPP_SIZE_ERROR);

  invalidRoi = {width, -5};
  EXPECT_EQ(nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, dstStep, invalidRoi), NPP_SIZE_ERROR);

  // Test invalid step
  EXPECT_EQ(nppiCopy_32f_C3P3R(d_src, 0, d_dstPlanes, dstStep, roi), NPP_STEP_ERROR);
  EXPECT_EQ(nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, -1, roi), NPP_STEP_ERROR);

  // Cleanup
  nppiFree(d_src);
  for (int i = 0; i < 3; i++) {
    nppiFree(d_dstPlanes[i]);
  }
}

// Test 8: Concurrent stream operations
TEST_F(NppiCopy32fC3P3RTest, ConcurrentStreams) {
  const int numStreams = 4;
  const int width = 1024;
  const int height = 1024;

  std::vector<cudaStream_t> streams(numStreams);
  std::vector<Npp32f *> srcBuffers(numStreams);
  std::vector<std::vector<Npp32f *>> dstBuffers(numStreams, std::vector<Npp32f *>(3));
  std::vector<int> srcSteps(numStreams);
  std::vector<int> dstSteps(numStreams);

  // Create streams and allocate buffers
  for (int i = 0; i < numStreams; i++) {
    cudaStreamCreate(&streams[i]);
    srcBuffers[i] = nppiMalloc_32f_C3(width, height, &srcSteps[i]);
    ASSERT_NE(srcBuffers[i], nullptr);

    for (int j = 0; j < 3; j++) {
      dstBuffers[i][j] = nppiMalloc_32f_C1(width, height, &dstSteps[i]);
      ASSERT_NE(dstBuffers[i][j], nullptr);
    }

    // Initialize with different patterns
    std::vector<Npp32f> pattern(width * height * 3);
    generateInterleavedPattern(pattern, width, height);
    for (size_t k = 0; k < pattern.size(); k++) {
      pattern[k] *= (i + 1);
    }

    cudaMemcpy2DAsync(srcBuffers[i], srcSteps[i], pattern.data(), width * 3 * sizeof(Npp32f),
                      width * 3 * sizeof(Npp32f), height, cudaMemcpyHostToDevice, streams[i]);
  }

  // Launch concurrent copies
  NppiSize roi = {width, height};
  for (int i = 0; i < numStreams; i++) {
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);
    ctx.hStream = streams[i];

    Npp32f *planes[3] = {dstBuffers[i][0], dstBuffers[i][1], dstBuffers[i][2]};
    NppStatus status = nppiCopy_32f_C3P3R_Ctx(srcBuffers[i], srcSteps[i], planes, dstSteps[i], roi, ctx);
    EXPECT_EQ(status, NPP_SUCCESS);
  }

  // Wait for all to complete and verify
  for (int i = 0; i < numStreams; i++) {
    cudaStreamSynchronize(streams[i]);

    // Verify first pixel of each plane
    Npp32f sampleR, sampleG, sampleB;
    cudaMemcpy(&sampleR, dstBuffers[i][0], sizeof(Npp32f), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sampleG, dstBuffers[i][1], sizeof(Npp32f), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sampleB, dstBuffers[i][2], sizeof(Npp32f), cudaMemcpyDeviceToHost);

    EXPECT_FLOAT_EQ(sampleR, 0.0f) << "Stream " << i << " R channel failed";
    EXPECT_FLOAT_EQ(sampleG, 0.0f) << "Stream " << i << " G channel failed";
    EXPECT_FLOAT_EQ(sampleB, 0.0f) << "Stream " << i << " B channel failed";
  }

  // Cleanup
  for (int i = 0; i < numStreams; i++) {
    cudaStreamDestroy(streams[i]);
    nppiFree(srcBuffers[i]);
    for (int j = 0; j < 3; j++) {
      nppiFree(dstBuffers[i][j]);
    }
  }
}

// Test 9: Stress test with random operations
TEST_F(NppiCopy32fC3P3RTest, StressTest) {
  const int numOperations = 50;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> sizeDist(1, 50);
  std::uniform_int_distribution<> coordDist(0, 20);

  for (int op = 0; op < numOperations; op++) {
    // Random dimensions
    int width = sizeDist(gen) * 4;
    int height = sizeDist(gen) * 4;

    Npp32f *d_src = nullptr;
    int srcStep;
    d_src = nppiMalloc_32f_C3(width, height, &srcStep);

    if (d_src == nullptr)
      continue;

    Npp32f *d_dstPlanes[3];
    int dstStep;
    bool allocSuccess = true;

    for (int i = 0; i < 3; i++) {
      d_dstPlanes[i] = nppiMalloc_32f_C1(width, height, &dstStep);
      if (d_dstPlanes[i] == nullptr) {
        allocSuccess = false;
        for (int j = 0; j < i; j++) {
          nppiFree(d_dstPlanes[j]);
        }
        break;
      }
    }

    if (!allocSuccess) {
      nppiFree(d_src);
      continue;
    }

    // Initialize with pattern
    std::vector<Npp32f> pattern(width * height * 3);
    generateInterleavedPattern(pattern, width, height);
    cudaMemcpy2D(d_src, srcStep, pattern.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
                 cudaMemcpyHostToDevice);

    // Random ROI
    int roiWidth = std::min(width, sizeDist(gen) * 2);
    int roiHeight = std::min(height, sizeDist(gen) * 2);

    NppiSize roi = {roiWidth, roiHeight};
    NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS) << "Operation " << op << " failed";

    // Cleanup
    nppiFree(d_src);
    for (int i = 0; i < 3; i++) {
      nppiFree(d_dstPlanes[i]);
    }
  }
}

// Test 10: Special patterns verification
TEST_F(NppiCopy32fC3P3RTest, SpecialPatterns) {
  const int width = 256;
  const int height = 256;

  struct PatternTest {
    const char *name;
    std::function<void(std::vector<Npp32f> &, int, int)> generator;
  };

  PatternTest patterns[] = {{"Gradient",
                             [](std::vector<Npp32f> &data, int w, int h) {
                               for (int y = 0; y < h; y++) {
                                 for (int x = 0; x < w; x++) {
                                   int idx = (y * w + x) * 3;
                                   data[idx] = x / (float)w * 255.0f;
                                   data[idx + 1] = y / (float)h * 255.0f;
                                   data[idx + 2] = (x + y) / (float)(w + h) * 255.0f;
                                 }
                               }
                             }},
                            {"Checkerboard",
                             [](std::vector<Npp32f> &data, int w, int h) {
                               for (int y = 0; y < h; y++) {
                                 for (int x = 0; x < w; x++) {
                                   int idx = (y * w + x) * 3;
                                   float val = ((x / 32 + y / 32) % 2) * 255.0f;
                                   data[idx] = val;
                                   data[idx + 1] = val;
                                   data[idx + 2] = val;
                                 }
                               }
                             }},
                            {"Diagonal stripes", [](std::vector<Npp32f> &data, int w, int h) {
                               for (int y = 0; y < h; y++) {
                                 for (int x = 0; x < w; x++) {
                                   int idx = (y * w + x) * 3;
                                   float val = ((x + y) % 64 < 32) ? 255.0f : 0.0f;
                                   data[idx] = val;
                                   data[idx + 1] = val * 0.5f;
                                   data[idx + 2] = val * 0.25f;
                                 }
                               }
                             }}};

  for (const auto &pattern : patterns) {
    std::vector<Npp32f> hostSrc(width * height * 3);
    pattern.generator(hostSrc, width, height);

    // Allocate GPU memory
    Npp32f *d_src = nullptr;
    int srcStep;
    d_src = nppiMalloc_32f_C3(width, height, &srcStep);
    ASSERT_NE(d_src, nullptr);

    Npp32f *d_dstPlanes[3];
    int dstStep;
    for (int i = 0; i < 3; i++) {
      d_dstPlanes[i] = nppiMalloc_32f_C1(width, height, &dstStep);
      ASSERT_NE(d_dstPlanes[i], nullptr);
    }

    // Upload and process
    cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
                 cudaMemcpyHostToDevice);

    NppiSize roi = {width, height};
    NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, dstStep, roi);
    ASSERT_EQ(status, NPP_SUCCESS) << "Failed for pattern: " << pattern.name;

    // Verify center pixel
    int centerX = width / 2;
    int centerY = height / 2;
    int centerIdx = (centerY * width + centerX) * 3;

    Npp32f centerR, centerG, centerB;
    cudaMemcpy(&centerR, (Npp32f *)((Npp8u *)d_dstPlanes[0] + centerY * dstStep) + centerX, sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&centerG, (Npp32f *)((Npp8u *)d_dstPlanes[1] + centerY * dstStep) + centerX, sizeof(Npp32f),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&centerB, (Npp32f *)((Npp8u *)d_dstPlanes[2] + centerY * dstStep) + centerX, sizeof(Npp32f),
               cudaMemcpyDeviceToHost);

    EXPECT_FLOAT_EQ(centerR, hostSrc[centerIdx]) << "Pattern: " << pattern.name;
    EXPECT_FLOAT_EQ(centerG, hostSrc[centerIdx + 1]) << "Pattern: " << pattern.name;
    EXPECT_FLOAT_EQ(centerB, hostSrc[centerIdx + 2]) << "Pattern: " << pattern.name;

    // Cleanup
    nppiFree(d_src);
    for (int i = 0; i < 3; i++) {
      nppiFree(d_dstPlanes[i]);
    }
  }
}

// Test 11: Non-Contiguous Memory Layout Verification
TEST_F(NppiCopy32fC3P3RTest, NonContiguousMemoryLayoutVerification) {
  const int width = 128, height = 128;

  Npp32f *d_src = nullptr;
  int srcStep;
  d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);

  // Allocate destination planes with different strides to simulate non-contiguous layout
  Npp32f *d_dstPlanes[3];
  int dstSteps[3];

  for (int i = 0; i < 3; i++) {
    d_dstPlanes[i] = nppiMalloc_32f_C1(width, height, &dstSteps[i]);
    ASSERT_NE(d_dstPlanes[i], nullptr);
  }

  // Create test data with distinct channel values
  std::vector<Npp32f> hostSrc(width * height * 3);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 3;
      hostSrc[idx] = 100.0f + x;           // R channel: base 100 + x coordinate
      hostSrc[idx + 1] = 200.0f + y;       // G channel: base 200 + y coordinate
      hostSrc[idx + 2] = 300.0f + (x + y); // B channel: base 300 + diagonal
    }
  }

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, d_dstPlanes, dstSteps[0], roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Verify each plane separately with different memory layouts
  for (int plane = 0; plane < 3; plane++) {
    std::vector<Npp32f> hostPlane(width * height);
    cudaMemcpy2D(hostPlane.data(), width * sizeof(Npp32f), d_dstPlanes[plane], dstSteps[plane], width * sizeof(Npp32f),
                 height, cudaMemcpyDeviceToHost);

    // Verify several sample points across the image
    int samplePoints[][2] = {{0, 0},
                             {width - 1, 0},
                             {0, height - 1},
                             {width - 1, height - 1},
                             {width / 4, height / 4},
                             {3 * width / 4, 3 * height / 4}};

    for (auto &pt : samplePoints) {
      int x = pt[0], y = pt[1];
      int srcIdx = (y * width + x) * 3 + plane;
      int dstIdx = y * width + x;

      EXPECT_FLOAT_EQ(hostPlane[dstIdx], hostSrc[srcIdx])
          << "Plane " << plane << " mismatch at (" << x << "," << y << ")";
    }
  }

  nppiFree(d_src);
  for (int i = 0; i < 3; i++) {
    nppiFree(d_dstPlanes[i]);
  }
}

// Test 12: Stream Context Synchronization
TEST_F(NppiCopy32fC3P3RTest, StreamContextSynchronization) {
  const int width = 512, height = 512;
  const int numStreams = 4;

  std::vector<cudaStream_t> streams(numStreams);
  std::vector<Npp32f *> srcBuffers(numStreams);
  std::vector<std::vector<Npp32f *>> dstBuffers(numStreams, std::vector<Npp32f *>(3));
  std::vector<int> srcSteps(numStreams);
  std::vector<int> dstSteps(numStreams);

  // Create streams and allocate memory
  for (int i = 0; i < numStreams; i++) {
    cudaStreamCreate(&streams[i]);
    srcBuffers[i] = nppiMalloc_32f_C3(width, height, &srcSteps[i]);
    ASSERT_NE(srcBuffers[i], nullptr);

    for (int j = 0; j < 3; j++) {
      dstBuffers[i][j] = nppiMalloc_32f_C1(width, height, &dstSteps[i]);
      ASSERT_NE(dstBuffers[i][j], nullptr);
    }

    // Initialize with stream-specific pattern
    std::vector<Npp32f> pattern(width * height * 3);
    for (int p = 0; p < width * height; p++) {
      pattern[p * 3] = (float)(i * 1000 + p % 256);             // R
      pattern[p * 3 + 1] = (float)(i * 1000 + (p + 100) % 256); // G
      pattern[p * 3 + 2] = (float)(i * 1000 + (p + 200) % 256); // B
    }

    cudaMemcpy2DAsync(srcBuffers[i], srcSteps[i], pattern.data(), width * 3 * sizeof(Npp32f),
                      width * 3 * sizeof(Npp32f), height, cudaMemcpyHostToDevice, streams[i]);
  }

  // Launch operations on different streams
  NppiSize roi = {width, height};
  for (int i = 0; i < numStreams; i++) {
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);
    ctx.hStream = streams[i];

    Npp32f *planes[3] = {dstBuffers[i][0], dstBuffers[i][1], dstBuffers[i][2]};
    NppStatus status = nppiCopy_32f_C3P3R_Ctx(srcBuffers[i], srcSteps[i], planes, dstSteps[i], roi, ctx);
    EXPECT_EQ(status, NPP_SUCCESS) << "Stream " << i << " failed";
  }

  // Wait for all streams and verify cross-stream independence
  for (int i = 0; i < numStreams; i++) {
    cudaStreamSynchronize(streams[i]);

    // Verify each plane has correct stream-specific data
    for (int plane = 0; plane < 3; plane++) {
      Npp32f sampleValue;
      cudaMemcpy(&sampleValue, dstBuffers[i][plane], sizeof(Npp32f), cudaMemcpyDeviceToHost);

      float expected = (float)(i * 1000 + (plane * 100) % 256);
      EXPECT_FLOAT_EQ(sampleValue, expected) << "Stream " << i << " plane " << plane << " incorrect value";
    }
  }

  // Cleanup
  for (int i = 0; i < numStreams; i++) {
    cudaStreamDestroy(streams[i]);
    nppiFree(srcBuffers[i]);
    for (int j = 0; j < 3; j++) {
      nppiFree(dstBuffers[i][j]);
    }
  }
}

class NppiCopy32fC3RTest : public ::testing::Test {
protected:
  void SetUp() override {
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess);
  }

  void TearDown() override {
    cudaError_t err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
  }

  // Helper to allocate aligned memory
  void allocateImage(Npp32f **ppImg, int width, int height, int *pStep) {
    *ppImg = nppiMalloc_32f_C3(width, height, pStep);
    ASSERT_NE(*ppImg, nullptr);
  }

  // Helper to verify data integrity
  bool verifyData(const std::vector<Npp32f> &expected, const std::vector<Npp32f> &actual, float tolerance = 0.0f) {
    if (expected.size() != actual.size())
      return false;

    for (size_t i = 0; i < expected.size(); i++) {
      if (std::abs(expected[i] - actual[i]) > tolerance) {
        return false;
      }
    }
    return true;
  }

  // Generate test pattern for 3-channel data
  void generatePattern(std::vector<Npp32f> &data, int width, int height, float scale = 1.0f) {
    data.resize(width * height * 3);
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (y * width + x) * 3;
        data[idx] = ((y * width + x) % 1000) * scale;            // R
        data[idx + 1] = ((y * width + x) % 1000) * scale + 0.1f; // G
        data[idx + 2] = ((y * width + x) % 1000) * scale + 0.2f; // B
      }
    }
  }
};

// Test 1: Basic 3-channel copy functionality
TEST_F(NppiCopy32fC3RTest, Basic3ChannelCopy) {
  const int width = 256;
  const int height = 256;

  std::vector<Npp32f> hostSrc(width * height * 3);
  generatePattern(hostSrc, width, height);

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  // Upload source data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  // Perform copy
  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Verify result
  std::vector<Npp32f> hostDst(width * height * 3);
  cudaMemcpy2D(hostDst.data(), width * 3 * sizeof(Npp32f), d_dst, dstStep, width * 3 * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  EXPECT_TRUE(verifyData(hostSrc, hostDst));

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 2: Partial ROI copy with 3 channels
TEST_F(NppiCopy32fC3RTest, PartialROI3ChannelCopy) {
  const int width = 512;
  const int height = 512;
  const int roiX = 100;
  const int roiY = 100;
  const int roiWidth = 300;
  const int roiHeight = 300;

  std::vector<Npp32f> hostSrc(width * height * 3);
  std::vector<Npp32f> hostDst(width * height * 3, -999.0f);
  generatePattern(hostSrc, width, height);

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  // Upload data
  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_dst, dstStep, hostDst.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  // Copy ROI
  Npp32f *pSrcROI = (Npp32f *)((Npp8u *)d_src + roiY * srcStep) + roiX * 3;
  Npp32f *pDstROI = (Npp32f *)((Npp8u *)d_dst + roiY * dstStep) + roiX * 3;

  NppiSize roi = {roiWidth, roiHeight};
  NppStatus status = nppiCopy_32f_C3R(pSrcROI, srcStep, pDstROI, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Verify result
  cudaMemcpy2D(hostDst.data(), width * 3 * sizeof(Npp32f), d_dst, dstStep, width * 3 * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 3;
      if (x >= roiX && x < roiX + roiWidth && y >= roiY && y < roiY + roiHeight) {
        // Inside ROI - all 3 channels should be copied
        EXPECT_FLOAT_EQ(hostDst[idx], hostSrc[idx]);
        EXPECT_FLOAT_EQ(hostDst[idx + 1], hostSrc[idx + 1]);
        EXPECT_FLOAT_EQ(hostDst[idx + 2], hostSrc[idx + 2]);
      } else {
        // Outside ROI - should remain -999
        EXPECT_FLOAT_EQ(hostDst[idx], -999.0f);
        EXPECT_FLOAT_EQ(hostDst[idx + 1], -999.0f);
        EXPECT_FLOAT_EQ(hostDst[idx + 2], -999.0f);
      }
    }
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 3: Color patterns verification
TEST_F(NppiCopy32fC3RTest, ColorPatternsVerification) {
  const int width = 256;
  const int height = 256;

  // Create distinct color patterns
  std::vector<Npp32f> hostSrc(width * height * 3);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 3;
      // Create gradient in each channel
      hostSrc[idx] = (x / (float)width) * 255.0f;                      // R gradient left to right
      hostSrc[idx + 1] = (y / (float)height) * 255.0f;                 // G gradient top to bottom
      hostSrc[idx + 2] = ((x + y) / (float)(width + height)) * 255.0f; // B diagonal
    }
  }

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> hostDst(width * height * 3);
  cudaMemcpy2D(hostDst.data(), width * 3 * sizeof(Npp32f), d_dst, dstStep, width * 3 * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  // Verify specific points
  int testPoints[][2] = {{0, 0}, {width - 1, 0}, {0, height - 1}, {width - 1, height - 1}, {width / 2, height / 2}};
  for (auto &pt : testPoints) {
    int x = pt[0], y = pt[1];
    int idx = (y * width + x) * 3;
    EXPECT_FLOAT_EQ(hostDst[idx], hostSrc[idx]) << "R channel mismatch at (" << x << "," << y << ")";
    EXPECT_FLOAT_EQ(hostDst[idx + 1], hostSrc[idx + 1]) << "G channel mismatch at (" << x << "," << y << ")";
    EXPECT_FLOAT_EQ(hostDst[idx + 2], hostSrc[idx + 2]) << "B channel mismatch at (" << x << "," << y << ")";
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 4: Different source and destination strides
TEST_F(NppiCopy32fC3RTest, DifferentStrides3Channel) {
  const int width = 200;
  const int height = 200;
  const int srcExtraBytes = 512; // Extra padding for source
  const int dstExtraBytes = 256; // Different padding for destination

  int srcStep = width * 3 * sizeof(Npp32f) + srcExtraBytes;
  int dstStep = width * 3 * sizeof(Npp32f) + dstExtraBytes;

  // Allocate with custom strides
  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  cudaMalloc(&d_src, srcStep * height);
  cudaMalloc(&d_dst, dstStep * height);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Create test data
  std::vector<Npp8u> hostSrcBuffer(srcStep * height, 0);
  for (int y = 0; y < height; y++) {
    Npp32f *row = (Npp32f *)(hostSrcBuffer.data() + y * srcStep);
    for (int x = 0; x < width; x++) {
      row[x * 3] = y * 1000.0f + x;
      row[x * 3 + 1] = y * 1000.0f + x + 500.0f;
      row[x * 3 + 2] = y * 1000.0f + x + 1000.0f;
    }
  }

  cudaMemcpy(d_src, hostSrcBuffer.data(), srcStep * height, cudaMemcpyHostToDevice);

  // Clear destination
  cudaMemset(d_dst, 0, dstStep * height);

  // Perform copy
  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  // Verify
  std::vector<Npp8u> hostDstBuffer(dstStep * height);
  cudaMemcpy(hostDstBuffer.data(), d_dst, dstStep * height, cudaMemcpyDeviceToHost);

  for (int y = 0; y < height; y++) {
    Npp32f *srcRow = (Npp32f *)(hostSrcBuffer.data() + y * srcStep);
    Npp32f *dstRow = (Npp32f *)(hostDstBuffer.data() + y * dstStep);
    for (int x = 0; x < width; x++) {
      EXPECT_FLOAT_EQ(dstRow[x * 3], srcRow[x * 3]);
      EXPECT_FLOAT_EQ(dstRow[x * 3 + 1], srcRow[x * 3 + 1]);
      EXPECT_FLOAT_EQ(dstRow[x * 3 + 2], srcRow[x * 3 + 2]);
    }
  }

  cudaFree(d_src);
  cudaFree(d_dst);
}

// Test 5: Large image copy performance for 3 channels
TEST_F(NppiCopy32fC3RTest, LargeImage3ChannelPerformance) {
  const int width = 3840;  // 4K width
  const int height = 2160; // 4K height
  const int iterations = 10;

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  // Fill source with test pattern
  cudaMemset(d_src, 0x42, srcStep * height);

  NppiSize roi = {width, height};

  // Warm up
  nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  cudaDeviceSynchronize();

  // Measure performance
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
    ASSERT_EQ(status, NPP_SUCCESS);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  double avgTime = duration / (double)iterations;
  double dataSize = width * height * 3 * sizeof(Npp32f) * 2 / 1024.0 / 1024.0 / 1024.0; // GB
  double bandwidth = dataSize * 1000000.0 / avgTime;                                    // GB/s

  std::cout << "3-channel large image copy performance:" << std::endl;
  std::cout << "  Image size: " << width << "x" << height << " (3 channels)" << std::endl;
  std::cout << "  Average time: " << avgTime << " us" << std::endl;
  std::cout << "  Bandwidth: " << bandwidth << " GB/s" << std::endl;

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 7: Edge alignment cases for 3-channel data
TEST_F(NppiCopy32fC3RTest, EdgeAlignmentCases3Channel) {
  const int testWidths[] = {1, 3, 5, 7, 13, 17, 31, 63, 85, 127, 171, 255, 341, 511};
  const int height = 13; // Prime number for edge case

  for (int width : testWidths) {
    Npp32f *d_src = nullptr;
    Npp32f *d_dst = nullptr;
    int srcStep, dstStep;

    d_src = nppiMalloc_32f_C3(width, height, &srcStep);
    d_dst = nppiMalloc_32f_C3(width, height, &dstStep);

    if (!d_src || !d_dst) {
      if (d_src)
        nppiFree(d_src);
      if (d_dst)
        nppiFree(d_dst);
      continue;
    }

    // Create unique pattern
    std::vector<Npp32f> pattern(width * height * 3);
    for (int i = 0; i < width * height; i++) {
      pattern[i * 3] = static_cast<float>(i * 7 + width);
      pattern[i * 3 + 1] = static_cast<float>(i * 11 + width);
      pattern[i * 3 + 2] = static_cast<float>(i * 13 + width);
    }

    cudaMemcpy2D(d_src, srcStep, pattern.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
                 cudaMemcpyHostToDevice);

    NppiSize roi = {width, height};
    NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS) << "Copy failed for width " << width;

    // Verify
    std::vector<Npp32f> result(width * height * 3);
    cudaMemcpy2D(result.data(), width * 3 * sizeof(Npp32f), d_dst, dstStep, width * 3 * sizeof(Npp32f), height,
                 cudaMemcpyDeviceToHost);

    EXPECT_TRUE(verifyData(pattern, result)) << "Data mismatch for width " << width;

    nppiFree(d_src);
    nppiFree(d_dst);
  }
}

// Test 8: Error handling for 3-channel copy
TEST_F(NppiCopy32fC3RTest, DISABLED_ErrorHandling3Channel) {
  const int width = 100;
  const int height = 100;

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  NppiSize roi = {width, height};

  // Test null source
  EXPECT_EQ(nppiCopy_32f_C3R(nullptr, srcStep, d_dst, dstStep, roi), NPP_NULL_POINTER_ERROR);

  // Test null destination
  EXPECT_EQ(nppiCopy_32f_C3R(d_src, srcStep, nullptr, dstStep, roi), NPP_NULL_POINTER_ERROR);

  // Test invalid ROI
  NppiSize invalidRoi = {0, height};
  EXPECT_EQ(nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, invalidRoi), NPP_SUCCESS);

  invalidRoi = {width, -5};
  EXPECT_EQ(nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, invalidRoi), NPP_SIZE_ERROR);

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 9: Concurrent stream operations with 3 channels
TEST_F(NppiCopy32fC3RTest, DISABLED_ConcurrentStreams3Channel) {
  const int numStreams = 6;
  const int width = 800;
  const int height = 600;

  std::vector<cudaStream_t> streams(numStreams);
  std::vector<Npp32f *> srcBuffers(numStreams);
  std::vector<Npp32f *> dstBuffers(numStreams);
  std::vector<int> srcSteps(numStreams);
  std::vector<int> dstSteps(numStreams);

  // Create streams and allocate buffers
  for (int i = 0; i < numStreams; i++) {
    cudaStreamCreate(&streams[i]);
    srcBuffers[i] = nppiMalloc_32f_C3(width, height, &srcSteps[i]);
    dstBuffers[i] = nppiMalloc_32f_C3(width, height, &dstSteps[i]);
    ASSERT_NE(srcBuffers[i], nullptr);
    ASSERT_NE(dstBuffers[i], nullptr);

    // Initialize with different patterns
    std::vector<Npp32f> pattern(width * height * 3);
    generatePattern(pattern, width, height, static_cast<float>(i + 1));
    cudaMemcpy2DAsync(srcBuffers[i], srcSteps[i], pattern.data(), width * 3 * sizeof(Npp32f),
                      width * 3 * sizeof(Npp32f), height, cudaMemcpyHostToDevice, streams[i]);
  }

  // Launch concurrent copies
  NppiSize roi = {width, height};
  for (int i = 0; i < numStreams; i++) {
    NppStreamContext ctx;
    nppGetStreamContext(&ctx);
    ctx.hStream = streams[i];

    NppStatus status = nppiCopy_32f_C3R_Ctx(srcBuffers[i], srcSteps[i], dstBuffers[i], dstSteps[i], roi, ctx);
    EXPECT_EQ(status, NPP_SUCCESS);
  }

  // Verify results
  for (int i = 0; i < numStreams; i++) {
    cudaStreamSynchronize(streams[i]);

    std::vector<Npp32f> result(width * height * 3);
    cudaMemcpy2D(result.data(), width * 3 * sizeof(Npp32f), dstBuffers[i], dstSteps[i], width * 3 * sizeof(Npp32f),
                 height, cudaMemcpyDeviceToHost);

    // Check first and last pixel (3 channels each)
    float expectedFirst[3] = {0.0f, 0.1f, 0.2f};
    float expectedLast[3];
    for (int c = 0; c < 3; c++) {
      expectedLast[c] = ((height - 1) * width + (width - 1)) * (i + 1) + c * 0.1f;
      EXPECT_FLOAT_EQ(result[c], expectedFirst[c] * (i + 1))
          << "Stream " << i << " channel " << c << " first pixel failed";
      EXPECT_NEAR(result[(width * height - 1) * 3 + c], expectedLast[c], 0.1f)
          << "Stream " << i << " channel " << c << " last pixel failed";
    }
  }

  // Cleanup
  for (int i = 0; i < numStreams; i++) {
    cudaStreamDestroy(streams[i]);
    nppiFree(srcBuffers[i]);
    nppiFree(dstBuffers[i]);
  }
}

// Test 10: Extreme dimensions for 3-channel data
TEST_F(NppiCopy32fC3RTest, DISABLED_ExtremeDimensions3Channel) {
  // Test very wide but short image
  {
    const int width = 8192;
    const int height = 2;

    Npp32f *d_src = nullptr;
    Npp32f *d_dst = nullptr;
    int srcStep, dstStep;

    d_src = nppiMalloc_32f_C3(width, height, &srcStep);
    d_dst = nppiMalloc_32f_C3(width, height, &dstStep);

    if (d_src && d_dst) {
      // Initialize with pattern
      std::vector<Npp32f> pattern(width * height * 3);
      for (int i = 0; i < width * height * 3; i++) {
        pattern[i] = i * 0.1f;
      }

      cudaMemcpy2D(d_src, srcStep, pattern.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
                   cudaMemcpyHostToDevice);

      NppiSize roi = {width, height};
      NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
      EXPECT_EQ(status, NPP_SUCCESS);

      // Verify corner pixels
      Npp32f corners[12]; // 4 corners * 3 channels
      cudaMemcpy(&corners[0], d_dst, 3 * sizeof(Npp32f), cudaMemcpyDeviceToHost);
      cudaMemcpy(&corners[3], (Npp32f *)((Npp8u *)d_dst) + (width - 1) * 3 * sizeof(Npp32f), 3 * sizeof(Npp32f),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(&corners[6], (Npp32f *)((Npp8u *)d_dst + dstStep) + 0, 3 * sizeof(Npp32f), cudaMemcpyDeviceToHost);
      cudaMemcpy(&corners[9], (Npp32f *)((Npp8u *)d_dst + dstStep) + (width - 1) * 3 * sizeof(Npp32f),
                 3 * sizeof(Npp32f), cudaMemcpyDeviceToHost);

      // Verify expected values
      EXPECT_FLOAT_EQ(corners[0], 0.0f);
      EXPECT_FLOAT_EQ(corners[1], 0.1f);
      EXPECT_FLOAT_EQ(corners[2], 0.2f);

      nppiFree(d_src);
      nppiFree(d_dst);
    }
  }

  // Test very tall but narrow image
  {
    const int width = 3;
    const int height = 8192;

    Npp32f *d_src = nullptr;
    Npp32f *d_dst = nullptr;
    int srcStep, dstStep;

    d_src = nppiMalloc_32f_C3(width, height, &srcStep);
    d_dst = nppiMalloc_32f_C3(width, height, &dstStep);

    if (d_src && d_dst) {
      NppiSize roi = {width, height};
      NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
      EXPECT_EQ(status, NPP_SUCCESS);

      nppiFree(d_src);
      nppiFree(d_dst);
    }
  }
}

// Test 11: Channel-Specific Pattern Verification
TEST_F(NppiCopy32fC3RTest, ChannelSpecificPatternVerification) {
  const int width = 64, height = 64;

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  // Create distinct patterns for each channel to verify channel integrity
  std::vector<Npp32f> hostSrc(width * height * 3);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 3;
      // R channel: sine wave pattern
      hostSrc[idx] = 128.0f + 127.0f * std::sin(2.0f * M_PI * x / width);
      // G channel: cosine wave pattern
      hostSrc[idx + 1] = 128.0f + 127.0f * std::cos(2.0f * M_PI * y / height);
      // B channel: checkerboard pattern
      hostSrc[idx + 2] = ((x / 8 + y / 8) % 2) ? 255.0f : 0.0f;
    }
  }

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> hostDst(width * height * 3);
  cudaMemcpy2D(hostDst.data(), width * 3 * sizeof(Npp32f), d_dst, dstStep, width * 3 * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  // Verify pattern integrity at strategic points
  int testPoints[][2] = {{0, 0},
                         {width / 4, height / 4},
                         {width / 2, height / 2},
                         {3 * width / 4, 3 * height / 4},
                         {width - 1, height - 1}};

  for (auto &pt : testPoints) {
    int x = pt[0], y = pt[1];
    int idx = (y * width + x) * 3;

    // Verify R channel (sine wave)
    float expectedR = 128.0f + 127.0f * std::sin(2.0f * M_PI * x / width);
    EXPECT_NEAR(hostDst[idx], expectedR, 0.01f) << "R channel pattern mismatch at (" << x << "," << y << ")";

    // Verify G channel (cosine wave)
    float expectedG = 128.0f + 127.0f * std::cos(2.0f * M_PI * y / height);
    EXPECT_NEAR(hostDst[idx + 1], expectedG, 0.01f) << "G channel pattern mismatch at (" << x << "," << y << ")";

    // Verify B channel (checkerboard)
    float expectedB = ((x / 8 + y / 8) % 2) ? 255.0f : 0.0f;
    EXPECT_FLOAT_EQ(hostDst[idx + 2], expectedB) << "B channel pattern mismatch at (" << x << "," << y << ")";
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

// Test 12: Multi-ROI Performance and Correctness
TEST_F(NppiCopy32fC3RTest, MultiROIPerformanceAndCorrectness) {
  const int width = 512, height = 512;
  const int numROIs = 9; // 3x3 grid of ROIs

  Npp32f *d_src = nullptr;
  Npp32f *d_dst = nullptr;
  int srcStep, dstStep;
  allocateImage(&d_src, width, height, &srcStep);
  allocateImage(&d_dst, width, height, &dstStep);

  // Initialize source with position-dependent values
  std::vector<Npp32f> hostSrc(width * height * 3);
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int idx = (y * width + x) * 3;
      hostSrc[idx] = y * 1000.0f + x;            // R: position encoding
      hostSrc[idx + 1] = y * 1000.0f + x + 0.5f; // G: slight offset
      hostSrc[idx + 2] = y * 1000.0f + x + 1.0f; // B: larger offset
    }
  }

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * 3 * sizeof(Npp32f), width * 3 * sizeof(Npp32f), height,
               cudaMemcpyHostToDevice);

  // Clear destination
  cudaMemset(d_dst, 0, dstStep * height);

  // Define 3x3 grid of ROIs
  const int roiWidth = width / 3;
  const int roiHeight = height / 3;

  auto start = std::chrono::high_resolution_clock::now();

  // Copy each ROI
  for (int roiY = 0; roiY < 3; roiY++) {
    for (int roiX = 0; roiX < 3; roiX++) {
      int startX = roiX * roiWidth;
      int startY = roiY * roiHeight;

      // Handle edge cases for last ROI
      int actualWidth = (roiX == 2) ? width - startX : roiWidth;
      int actualHeight = (roiY == 2) ? height - startY : roiHeight;

      Npp32f *pSrcROI = (Npp32f *)((Npp8u *)d_src + startY * srcStep) + startX * 3;
      Npp32f *pDstROI = (Npp32f *)((Npp8u *)d_dst + startY * dstStep) + startX * 3;

      NppiSize roi = {actualWidth, actualHeight};
      NppStatus status = nppiCopy_32f_C3R(pSrcROI, srcStep, pDstROI, dstStep, roi);
      ASSERT_EQ(status, NPP_SUCCESS) << "ROI (" << roiX << "," << roiY << ") failed";
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "Multi-ROI copy (" << numROIs << " ROIs) completed in " << duration << " microseconds" << std::endl;

  // Verify correctness
  std::vector<Npp32f> hostDst(width * height * 3);
  cudaMemcpy2D(hostDst.data(), width * 3 * sizeof(Npp32f), d_dst, dstStep, width * 3 * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  // Sample verification across all ROIs
  for (int roiY = 0; roiY < 3; roiY++) {
    for (int roiX = 0; roiX < 3; roiX++) {
      int startX = roiX * roiWidth;
      int startY = roiY * roiHeight;
      int endX = (roiX == 2) ? width : (roiX + 1) * roiWidth;
      int endY = (roiY == 2) ? height : (roiY + 1) * roiHeight;

      // Test a few sample points in each ROI
      for (int sy = startY; sy < endY; sy += (endY - startY) / 3 + 1) {
        for (int sx = startX; sx < endX; sx += (endX - startX) / 3 + 1) {
          int idx = (sy * width + sx) * 3;

          EXPECT_FLOAT_EQ(hostDst[idx], hostSrc[idx])
              << "R mismatch at ROI(" << roiX << "," << roiY << ") pos(" << sx << "," << sy << ")";
          EXPECT_FLOAT_EQ(hostDst[idx + 1], hostSrc[idx + 1])
              << "G mismatch at ROI(" << roiX << "," << roiY << ") pos(" << sx << "," << sy << ")";
          EXPECT_FLOAT_EQ(hostDst[idx + 2], hostSrc[idx + 2])
              << "B mismatch at ROI(" << roiX << "," << roiY << ") pos(" << sx << "," << sy << ")";
        }
      }
    }
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}
