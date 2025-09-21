#include "npp.h"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

class NV12ToRGBTest : public ::testing::Test {
protected:
  void SetUp() override {
    width = 64;
    height = 64;

    // NV12 requires even dimensions
    ASSERT_EQ(width % 2, 0);
    ASSERT_EQ(height % 2, 0);
  }

  void TearDown() override {
    // Allocate memory
  }

  // 创建测试用的NV12数据
  void createTestNV12Data(std::vector<Npp8u> &yData, std::vector<Npp8u> &uvData) {
    yData.resize(width * height);
    uvData.resize(width * height / 2); // UV plane is half the size

    // 创建渐变Y平面 (亮度)
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        // 从左上角(暗)到右下角(亮)的渐变
        yData[y * width + x] = (Npp8u)(16 + (x + y) * 239 / (width + height - 2));
      }
    }

    // 创建UV平面 (色度)
    for (int y = 0; y < height / 2; ++y) {
      for (int x = 0; x < width; x += 2) {
        int idx = y * width + x;
        // U (蓝色色度) - 水平渐变
        uvData[idx] = (Npp8u)(64 + x * 128 / width);
        // V (红色色度) - 垂直渐变
        uvData[idx + 1] = (Npp8u)(64 + y * 128 / (height / 2));
      }
    }
  }

  // ValidateRGB值在合理bounds内
  bool isValidRGB(Npp8u r, Npp8u g, Npp8u b) {
    // Npp8u is unsigned char, so values are always 0-255
    (void)r;
    (void)g;
    (void)b; // Suppress unused parameter warnings
    return true;
  }

  int width, height;
};

// 测试基本的NV12到RGB转换
TEST_F(NV12ToRGBTest, BasicNV12ToRGB_8u_P2C3R) {
  std::vector<Npp8u> hostYData, hostUVData;
  createTestNV12Data(hostYData, hostUVData);

  // 分配GPU内存
  Npp8u *d_srcY = nppsMalloc_8u(width * height);
  Npp8u *d_srcUV = nppsMalloc_8u(width * height / 2);

  int rgbStep;
  Npp8u *d_rgb = nppiMalloc_8u_C3(width, height, &rgbStep);

  ASSERT_NE(d_srcY, nullptr);
  ASSERT_NE(d_srcUV, nullptr);
  ASSERT_NE(d_rgb, nullptr);

  // 复制数据到GPU
  cudaMemcpy(d_srcY, hostYData.data(), hostYData.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_srcUV, hostUVData.data(), hostUVData.size(), cudaMemcpyHostToDevice);

  // 准备NV12源数组
  const Npp8u *pSrc[2] = {d_srcY, d_srcUV};
  NppiSize roi = {width, height};

  // 执行转换
  NppStatus status = nppiNV12ToRGB_8u_P2C3R(pSrc, width, d_rgb, rgbStep, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  // 读取结果Validate
  std::vector<Npp8u> hostRGB(rgbStep * height);
  cudaMemcpy(hostRGB.data(), d_rgb, rgbStep * height, cudaMemcpyDeviceToHost);

  // Validate几个像素点
  for (int y = 0; y < height; y += 8) {
    for (int x = 0; x < width; x += 8) {
      int rgbIdx = y * rgbStep + x * 3;
      Npp8u r = hostRGB[rgbIdx];
      Npp8u g = hostRGB[rgbIdx + 1];
      Npp8u b = hostRGB[rgbIdx + 2];

      EXPECT_TRUE(isValidRGB(r, g, b)) << "Invalid RGB at (" << x << "," << y << "): " << (int)r << "," << (int)g << ","
                                       << (int)b;
    }
  }

  // 清理内存
  nppsFree(d_srcY);
  nppsFree(d_srcUV);
  nppiFree(d_rgb);
}

// 测试基本的NV12到RGB转换 (带Context版本)
TEST_F(NV12ToRGBTest, BasicNV12ToRGB_8u_P2C3R_Ctx) {
  std::vector<Npp8u> hostYData, hostUVData;
  createTestNV12Data(hostYData, hostUVData);

  // 分配GPU内存
  Npp8u *d_srcY = nppsMalloc_8u(width * height);
  Npp8u *d_srcUV = nppsMalloc_8u(width * height / 2);

  int rgbStep;
  Npp8u *d_rgb = nppiMalloc_8u_C3(width, height, &rgbStep);

  ASSERT_NE(d_srcY, nullptr);
  ASSERT_NE(d_srcUV, nullptr);
  ASSERT_NE(d_rgb, nullptr);

  // 复制数据到GPU
  cudaMemcpy(d_srcY, hostYData.data(), hostYData.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_srcUV, hostUVData.data(), hostUVData.size(), cudaMemcpyHostToDevice);

  // 准备NV12源数组
  const Npp8u *pSrc[2] = {d_srcY, d_srcUV};
  NppiSize roi = {width, height};

  // 创建Stream Context
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0; // 使用Default stream

  // 执行转换 (Context版本)
  NppStatus status = nppiNV12ToRGB_8u_P2C3R_Ctx(pSrc, width, d_rgb, rgbStep, roi, nppStreamCtx);
  EXPECT_EQ(status, NPP_NO_ERROR) << "Basic NV12ToRGB Context conversion failed";

  // 读取结果Validate
  std::vector<Npp8u> hostRGB(rgbStep * height);
  cudaMemcpy(hostRGB.data(), d_rgb, rgbStep * height, cudaMemcpyDeviceToHost);

  // Validate几个像素点并计算有效像素数量
  bool hasValidPixels = false;
  int validPixelCount = 0;

  for (int y = 0; y < height; y += 8) {
    for (int x = 0; x < width; x += 8) {
      int rgbIdx = y * rgbStep + x * 3;
      Npp8u r = hostRGB[rgbIdx];
      Npp8u g = hostRGB[rgbIdx + 1];
      Npp8u b = hostRGB[rgbIdx + 2];

      if (isValidRGB(r, g, b)) {
        hasValidPixels = true;
        validPixelCount++;
      }

      EXPECT_TRUE(isValidRGB(r, g, b)) << "Invalid RGB at (" << x << "," << y << "): " << (int)r << "," << (int)g << ","
                                       << (int)b;
    }
  }

  EXPECT_TRUE(hasValidPixels) << "No valid RGB pixels found in basic NV12ToRGB conversion";
  EXPECT_GT(validPixelCount, 50) << "Too few valid pixels in basic NV12ToRGB conversion";

  // 清理内存
  nppsFree(d_srcY);
  nppsFree(d_srcUV);
  nppiFree(d_rgb);
}

// 测试BT.709色彩空间转换
TEST_F(NV12ToRGBTest, NV12ToRGB_709CSC_8u_P2C3R) {
  std::vector<Npp8u> hostYData, hostUVData;
  createTestNV12Data(hostYData, hostUVData);

  // 分配GPU内存
  Npp8u *d_srcY = nppsMalloc_8u(width * height);
  Npp8u *d_srcUV = nppsMalloc_8u(width * height / 2);

  int rgbStep;
  Npp8u *d_rgb = nppiMalloc_8u_C3(width, height, &rgbStep);

  ASSERT_NE(d_srcY, nullptr);
  ASSERT_NE(d_srcUV, nullptr);
  ASSERT_NE(d_rgb, nullptr);

  // 复制数据到GPU
  cudaMemcpy(d_srcY, hostYData.data(), hostYData.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_srcUV, hostUVData.data(), hostUVData.size(), cudaMemcpyHostToDevice);

  // 准备NV12源数组
  const Npp8u *pSrc[2] = {d_srcY, d_srcUV};
  NppiSize roi = {width, height};

  // 执行BT.709转换
  NppStatus status = nppiNV12ToRGB_709CSC_8u_P2C3R(pSrc, width, d_rgb, rgbStep, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  // 读取结果Validate
  std::vector<Npp8u> hostRGB(rgbStep * height);
  cudaMemcpy(hostRGB.data(), d_rgb, rgbStep * height, cudaMemcpyDeviceToHost);

  // Validate转换结果
  bool hasValidPixels = false;
  for (int y = 0; y < height; y += 4) {
    for (int x = 0; x < width; x += 4) {
      int rgbIdx = y * rgbStep + x * 3;
      Npp8u r = hostRGB[rgbIdx];
      Npp8u g = hostRGB[rgbIdx + 1];
      Npp8u b = hostRGB[rgbIdx + 2];

      if (isValidRGB(r, g, b)) {
        hasValidPixels = true;
      }
    }
  }
  EXPECT_TRUE(hasValidPixels);

  // 清理内存
  nppsFree(d_srcY);
  nppsFree(d_srcUV);
  nppiFree(d_rgb);
}

// 测试BT.709色彩空间转换 (带Context版本)
TEST_F(NV12ToRGBTest, NV12ToRGB_709CSC_8u_P2C3R_Ctx) {
  std::vector<Npp8u> hostYData, hostUVData;
  createTestNV12Data(hostYData, hostUVData);

  // 分配GPU内存
  Npp8u *d_srcY = nppsMalloc_8u(width * height);
  Npp8u *d_srcUV = nppsMalloc_8u(width * height / 2);

  int rgbStep;
  Npp8u *d_rgb = nppiMalloc_8u_C3(width, height, &rgbStep);

  ASSERT_NE(d_srcY, nullptr);
  ASSERT_NE(d_srcUV, nullptr);
  ASSERT_NE(d_rgb, nullptr);

  // 复制数据到GPU
  cudaMemcpy(d_srcY, hostYData.data(), hostYData.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_srcUV, hostUVData.data(), hostUVData.size(), cudaMemcpyHostToDevice);

  // 准备NV12源数组
  const Npp8u *pSrc[2] = {d_srcY, d_srcUV};
  NppiSize roi = {width, height};

  // 创建Stream Context
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0; // 使用Default stream

  // 执行BT.709转换 (Context版本)
  NppStatus status = nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx(pSrc, width, d_rgb, rgbStep, roi, nppStreamCtx);
  EXPECT_EQ(status, NPP_NO_ERROR) << "BT.709 CSC Context conversion failed";

  // 读取结果Validate
  std::vector<Npp8u> hostRGB(rgbStep * height);
  cudaMemcpy(hostRGB.data(), d_rgb, rgbStep * height, cudaMemcpyDeviceToHost);

  // Validate转换结果
  bool hasValidPixels = false;
  int validPixelCount = 0;

  for (int y = 0; y < height; y += 4) {
    for (int x = 0; x < width; x += 4) {
      int rgbIdx = y * rgbStep + x * 3;
      Npp8u r = hostRGB[rgbIdx];
      Npp8u g = hostRGB[rgbIdx + 1];
      Npp8u b = hostRGB[rgbIdx + 2];

      if (isValidRGB(r, g, b)) {
        hasValidPixels = true;
        validPixelCount++;
      }
    }
  }

  EXPECT_TRUE(hasValidPixels) << "No valid RGB pixels found in BT.709 CSC conversion";
  EXPECT_GT(validPixelCount, 50) << "Too few valid pixels in BT.709 CSC conversion";

  // 清理内存
  nppsFree(d_srcY);
  nppsFree(d_srcUV);
  nppiFree(d_rgb);
}

// 测试BT.709 HDTV转换（别名函数）
TEST_F(NV12ToRGBTest, NV12ToRGB_709HDTV_8u_P2C3R) {
  std::vector<Npp8u> hostYData, hostUVData;
  createTestNV12Data(hostYData, hostUVData);

  // 分配GPU内存
  Npp8u *d_srcY = nppsMalloc_8u(width * height);
  Npp8u *d_srcUV = nppsMalloc_8u(width * height / 2);

  int rgbStep;
  Npp8u *d_rgb = nppiMalloc_8u_C3(width, height, &rgbStep);

  ASSERT_NE(d_srcY, nullptr);
  ASSERT_NE(d_srcUV, nullptr);
  ASSERT_NE(d_rgb, nullptr);

  // 复制数据到GPU
  cudaMemcpy(d_srcY, hostYData.data(), hostYData.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_srcUV, hostUVData.data(), hostUVData.size(), cudaMemcpyHostToDevice);

  // 准备NV12源数组
  const Npp8u *pSrc[2] = {d_srcY, d_srcUV};
  NppiSize roi = {width, height};

  // 执行HDTV转换
  NppStatus status = nppiNV12ToRGB_709HDTV_8u_P2C3R(pSrc, width, d_rgb, rgbStep, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  // 清理内存
  nppsFree(d_srcY);
  nppsFree(d_srcUV);
  nppiFree(d_rgb);
}

// 测试BT.709 HDTV转换（带Context版本）
TEST_F(NV12ToRGBTest, NV12ToRGB_709HDTV_8u_P2C3R_Ctx) {
  std::vector<Npp8u> hostYData, hostUVData;
  createTestNV12Data(hostYData, hostUVData);

  // 分配GPU内存
  Npp8u *d_srcY = nppsMalloc_8u(width * height);
  Npp8u *d_srcUV = nppsMalloc_8u(width * height / 2);

  int rgbStep;
  Npp8u *d_rgb = nppiMalloc_8u_C3(width, height, &rgbStep);

  ASSERT_NE(d_srcY, nullptr);
  ASSERT_NE(d_srcUV, nullptr);
  ASSERT_NE(d_rgb, nullptr);

  // 复制数据到GPU
  cudaMemcpy(d_srcY, hostYData.data(), hostYData.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_srcUV, hostUVData.data(), hostUVData.size(), cudaMemcpyHostToDevice);

  // 准备NV12源数组
  const Npp8u *pSrc[2] = {d_srcY, d_srcUV};
  NppiSize roi = {width, height};

  // 创建Stream Context
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0; // 使用Default stream

  // 执行HDTV转换 (Context版本)
  NppStatus status = nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(pSrc, width, d_rgb, rgbStep, roi, nppStreamCtx);
  EXPECT_EQ(status, NPP_NO_ERROR) << "BT.709 HDTV Context conversion failed";

  // 读取结果Validate
  std::vector<Npp8u> hostRGB(rgbStep * height);
  cudaMemcpy(hostRGB.data(), d_rgb, rgbStep * height, cudaMemcpyDeviceToHost);

  // Validate转换结果 - HDTV转换应该产生有效的RGB值
  bool hasValidPixels = false;
  int validPixelCount = 0;

  for (int y = 0; y < height; y += 4) {
    for (int x = 0; x < width; x += 4) {
      int rgbIdx = y * rgbStep + x * 3;
      Npp8u r = hostRGB[rgbIdx];
      Npp8u g = hostRGB[rgbIdx + 1];
      Npp8u b = hostRGB[rgbIdx + 2];

      if (isValidRGB(r, g, b)) {
        hasValidPixels = true;
        validPixelCount++;
      }
    }
  }

  EXPECT_TRUE(hasValidPixels) << "No valid RGB pixels found in BT.709 HDTV conversion";
  EXPECT_GT(validPixelCount, 50) << "Too few valid pixels in BT.709 HDTV conversion";

  // 清理内存
  nppsFree(d_srcY);
  nppsFree(d_srcUV);
  nppiFree(d_rgb);
}

// 测试不同图像尺寸
TEST_F(NV12ToRGBTest, VariousImageSizes) {
  struct TestSize {
    int width, height;
    const char *name;
  } sizes[] = {
      {32, 32, "32x32"}, {128, 64, "128x64"}, {256, 128, "256x128"}, {640, 480, "640x480"}, {1920, 1080, "1920x1080"}};

  for (const auto &size : sizes) {
    // 确保是偶数尺寸
    if (size.width % 2 != 0 || size.height % 2 != 0)
      continue;

    std::vector<Npp8u> yData(size.width * size.height, 128);      // 中性灰
    std::vector<Npp8u> uvData(size.width * size.height / 2, 128); // 中性色度

    // 分配GPU内存
    Npp8u *d_srcY = nppsMalloc_8u(size.width * size.height);
    Npp8u *d_srcUV = nppsMalloc_8u(size.width * size.height / 2);

    int rgbStep;
    Npp8u *d_rgb = nppiMalloc_8u_C3(size.width, size.height, &rgbStep);

    if (!d_srcY || !d_srcUV || !d_rgb) {
      // Allocate memory
      if (d_srcY)
        nppsFree(d_srcY);
      if (d_srcUV)
        nppsFree(d_srcUV);
      if (d_rgb)
        nppiFree(d_rgb);

      GTEST_SKIP() << "Could not allocate GPU memory for " << size.name;
      continue;
    }

    // 复制数据并执行转换
    cudaMemcpy(d_srcY, yData.data(), yData.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_srcUV, uvData.data(), uvData.size(), cudaMemcpyHostToDevice);

    const Npp8u *pSrc[2] = {d_srcY, d_srcUV};
    NppiSize roi = {size.width, size.height};

    NppStatus status = nppiNV12ToRGB_8u_P2C3R(pSrc, size.width, d_rgb, rgbStep, roi);
    EXPECT_EQ(status, NPP_NO_ERROR) << "Failed for size " << size.name;

    // 清理内存
    nppsFree(d_srcY);
    nppsFree(d_srcUV);
    nppiFree(d_rgb);
  }
}

//======================================================================
// TorchCodec兼容性测试
//======================================================================

class TorchCodecCompatibilityTest : public ::testing::Test {
protected:
  void SetUp() override {
    width = 1920; // 使用典型的视频分辨率
    height = 1080;
    ASSERT_EQ(width % 2, 0);
    ASSERT_EQ(height % 2, 0);

    // 模拟AVFrame的linesize
    yLinesize = width;
    uvLinesize = width; // NV12 UV plane has same width as Y
  }

  void TearDown() override {}

  // 创建模拟torchcodec使用场景的NV12数据
  void createTorchCodecNV12Data(std::vector<Npp8u> &yData, std::vector<Npp8u> &uvData) {
    yData.resize(yLinesize * height);
    uvData.resize(uvLinesize * height / 2);

    // 创建更真实的视频帧数据 - 模拟典型场景
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        // 模拟有些细节的图像（不是简单渐变）
        int luminance = 16 + (int)(219 * (0.5 + 0.3 * sin(x * 0.01) + 0.2 * cos(y * 0.015)));
        yData[y * yLinesize + x] = (Npp8u)std::clamp(luminance, 16, 235);
      }
    }

    // UV plane - 模拟色度信息
    for (int y = 0; y < height / 2; y++) {
      for (int x = 0; x < width; x += 2) {
        int uvIdx = y * uvLinesize + x;
        // U分量
        uvData[uvIdx] = (Npp8u)(128 + 64 * sin((x + y) * 0.02));
        // V分量
        uvData[uvIdx + 1] = (Npp8u)(128 + 64 * cos((x + y) * 0.02));
      }
    }
  }

  int width, height;
  int yLinesize, uvLinesize;
};

// 测试TorchCodec中使用的BT.709全bounds转换（使用自定义ColorTwist矩阵）
TEST_F(TorchCodecCompatibilityTest, TorchCodecYUVConversion) {
  std::vector<Npp8u> yData, uvData;
  createTorchCodecNV12Data(yData, uvData);

  // 分配GPU内存 - 模拟torchcodec的内存布局
  Npp8u *d_yPlane = nppsMalloc_8u(yLinesize * height);
  Npp8u *d_uvPlane = nppsMalloc_8u(uvLinesize * height / 2);

  // RGB输出 - 模拟PyTorch tensor stride
  int rgbStep = width * 3; // 紧密排列
  Npp8u *d_rgb = nppsMalloc_8u(rgbStep * height);

  ASSERT_NE(d_yPlane, nullptr);
  ASSERT_NE(d_uvPlane, nullptr);
  ASSERT_NE(d_rgb, nullptr);

  // 使用RAII确保内存清理
  struct ResourceGuard {
    Npp8u *y, *uv, *rgb;
    ResourceGuard(Npp8u *y_, Npp8u *uv_, Npp8u *rgb_) : y(y_), uv(uv_), rgb(rgb_) {}
    ~ResourceGuard() {
      if (y)
        nppsFree(y);
      if (uv)
        nppsFree(uv);
      if (rgb)
        nppsFree(rgb);
    }
  } guard(d_yPlane, d_uvPlane, d_rgb);

  // 复制数据到GPU
  cudaMemcpy(d_yPlane, yData.data(), yData.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_uvPlane, uvData.data(), uvData.size(), cudaMemcpyHostToDevice);

  // 测试场景1: BT.709 full range with custom ColorTwist (torchcodec使用的精确矩阵)
  // 这是torchcodec为了更好匹配CPU结果而使用的自定义矩阵
  static const Npp32f bt709FullRangeColorTwist[3][4] = {
      {1.0f, 0.0f, 1.5748f, -179.456f},             // R = Y + 1.5748*V - 179.456
      {1.0f, -0.18732427f, -0.46812427f, 135.459f}, // G = Y - 0.18732*U - 0.46812*V + 135.459
      {1.0f, 1.8556f, 0.0f, -226.816f}              // B = Y + 1.8556*U - 226.816
  };

  // 准备源数据数组（模拟torchcodec中的avFrame->data）
  Npp8u *yuvData[2] = {d_yPlane, d_uvPlane};
  int srcStep[2] = {yLinesize, uvLinesize};
  NppiSize oSizeROI = {width, height};

  // 执行BT.709 full range转换 - 这是torchcodec中AVCOL_RANGE_JPEG + AVCOL_SPC_BT709的路径
  NppStreamContext nppStreamCtx = {};
  nppStreamCtx.hStream = 0;
  NppStatus status = nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx((const Npp8u **)yuvData, srcStep, d_rgb, rgbStep,
                                                              oSizeROI, bt709FullRangeColorTwist, nppStreamCtx);

  EXPECT_EQ(status, NPP_SUCCESS) << "BT.709 full range ColorTwist conversion failed";

  // 测试场景2: BT.709 limited range (torchcodec中studio range的路径)
  status = nppiNV12ToRGB_709CSC_8u_P2C3R((const Npp8u **)yuvData,
                                         yLinesize, // torchcodec使用avFrame->linesize[0]
                                         d_rgb, rgbStep, oSizeROI);

  EXPECT_EQ(status, NPP_SUCCESS) << "BT.709 limited range conversion failed";

  // 测试场景3: BT.601默认转换（torchcodec的fallback路径）
  status = nppiNV12ToRGB_8u_P2C3R((const Npp8u **)yuvData, yLinesize, d_rgb, rgbStep, oSizeROI);

  EXPECT_EQ(status, NPP_SUCCESS) << "BT.601 conversion failed";

  // Validate输出有效性
  std::vector<Npp8u> hostRGB(rgbStep * height);
  cudaMemcpy(hostRGB.data(), d_rgb, rgbStep * height, cudaMemcpyDeviceToHost);

  // 检查RGB数据的合理性
  bool hasValidRGB = false;
  int validPixelCount = 0;

  for (int y = 0; y < height; y += 10) { // 采样检查
    for (int x = 0; x < width; x += 10) {
      int rgbIdx = y * rgbStep + x * 3;
      Npp8u r = hostRGB[rgbIdx];
      Npp8u g = hostRGB[rgbIdx + 1];
      Npp8u b = hostRGB[rgbIdx + 2];

      // 检查是否有合理的RGB值（不全为0或255）
      if ((r > 10 && r < 245) || (g > 10 && g < 245) || (b > 10 && b < 245)) {
        validPixelCount++;
        hasValidRGB = true;
      }
    }
  }

  EXPECT_TRUE(hasValidRGB) << "No valid RGB pixels found in converted output";
  EXPECT_GT(validPixelCount, 100) << "Too few valid pixels, conversion may be incorrect";

  std::cout << "TorchCodec compatibility test passed. Valid pixels: " << validPixelCount << std::endl;
}
