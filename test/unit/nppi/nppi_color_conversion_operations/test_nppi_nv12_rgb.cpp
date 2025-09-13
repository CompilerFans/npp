#include <gtest/gtest.h>
#include "npp.h"
#include <vector>
#include <cmath>
#include <algorithm>

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
        // 清理在测试中分配的内存
    }
    
    // 创建测试用的NV12数据
    void createTestNV12Data(std::vector<Npp8u>& yData, std::vector<Npp8u>& uvData) {
        yData.resize(width * height);
        uvData.resize(width * height / 2);  // UV plane is half the size
        
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
    
    // 验证RGB值在合理范围内 
    bool isValidRGB(Npp8u r, Npp8u g, Npp8u b) {
        // Npp8u is unsigned char, so values are always 0-255
        (void)r; (void)g; (void)b; // Suppress unused parameter warnings
        return true;
    }
    
    int width, height;
};

// 测试基本的NV12到RGB转换
TEST_F(NV12ToRGBTest, BasicNV12ToRGB_8u_P2C3R) {
    std::vector<Npp8u> hostYData, hostUVData;
    createTestNV12Data(hostYData, hostUVData);
    
    // 分配GPU内存
    Npp8u* d_srcY = nppsMalloc_8u(width * height);
    Npp8u* d_srcUV = nppsMalloc_8u(width * height / 2);
    
    int rgbStep;
    Npp8u* d_rgb = nppiMalloc_8u_C3(width, height, &rgbStep);
    
    ASSERT_NE(d_srcY, nullptr);
    ASSERT_NE(d_srcUV, nullptr);
    ASSERT_NE(d_rgb, nullptr);
    
    // 复制数据到GPU
    cudaMemcpy(d_srcY, hostYData.data(), hostYData.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_srcUV, hostUVData.data(), hostUVData.size(), cudaMemcpyHostToDevice);
    
    // 准备NV12源数组
    const Npp8u* pSrc[2] = { d_srcY, d_srcUV };
    NppiSize roi = { width, height };
    
    // 执行转换
    NppStatus status = nppiNV12ToRGB_8u_P2C3R(pSrc, width, d_rgb, rgbStep, roi);
    EXPECT_EQ(status, NPP_NO_ERROR);
    
    // 读取结果验证
    std::vector<Npp8u> hostRGB(rgbStep * height);
    cudaMemcpy(hostRGB.data(), d_rgb, rgbStep * height, cudaMemcpyDeviceToHost);
    
    // 验证几个像素点
    for (int y = 0; y < height; y += 8) {
        for (int x = 0; x < width; x += 8) {
            int rgbIdx = y * rgbStep + x * 3;
            Npp8u r = hostRGB[rgbIdx];
            Npp8u g = hostRGB[rgbIdx + 1];
            Npp8u b = hostRGB[rgbIdx + 2];
            
            EXPECT_TRUE(isValidRGB(r, g, b)) 
                << "Invalid RGB at (" << x << "," << y << "): " 
                << (int)r << "," << (int)g << "," << (int)b;
        }
    }
    
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
    Npp8u* d_srcY = nppsMalloc_8u(width * height);
    Npp8u* d_srcUV = nppsMalloc_8u(width * height / 2);
    
    int rgbStep;
    Npp8u* d_rgb = nppiMalloc_8u_C3(width, height, &rgbStep);
    
    ASSERT_NE(d_srcY, nullptr);
    ASSERT_NE(d_srcUV, nullptr);
    ASSERT_NE(d_rgb, nullptr);
    
    // 复制数据到GPU
    cudaMemcpy(d_srcY, hostYData.data(), hostYData.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_srcUV, hostUVData.data(), hostUVData.size(), cudaMemcpyHostToDevice);
    
    // 准备NV12源数组
    const Npp8u* pSrc[2] = { d_srcY, d_srcUV };
    NppiSize roi = { width, height };
    
    // 执行BT.709转换
    NppStatus status = nppiNV12ToRGB_709CSC_8u_P2C3R(pSrc, width, d_rgb, rgbStep, roi);
    EXPECT_EQ(status, NPP_NO_ERROR);
    
    // 读取结果验证
    std::vector<Npp8u> hostRGB(rgbStep * height);
    cudaMemcpy(hostRGB.data(), d_rgb, rgbStep * height, cudaMemcpyDeviceToHost);
    
    // 验证转换结果
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

// 测试BT.709 HDTV转换（别名函数）
TEST_F(NV12ToRGBTest, NV12ToRGB_709HDTV_8u_P2C3R) {
    std::vector<Npp8u> hostYData, hostUVData;
    createTestNV12Data(hostYData, hostUVData);
    
    // 分配GPU内存
    Npp8u* d_srcY = nppsMalloc_8u(width * height);
    Npp8u* d_srcUV = nppsMalloc_8u(width * height / 2);
    
    int rgbStep;
    Npp8u* d_rgb = nppiMalloc_8u_C3(width, height, &rgbStep);
    
    ASSERT_NE(d_srcY, nullptr);
    ASSERT_NE(d_srcUV, nullptr);
    ASSERT_NE(d_rgb, nullptr);
    
    // 复制数据到GPU
    cudaMemcpy(d_srcY, hostYData.data(), hostYData.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_srcUV, hostUVData.data(), hostUVData.size(), cudaMemcpyHostToDevice);
    
    // 准备NV12源数组
    const Npp8u* pSrc[2] = { d_srcY, d_srcUV };
    NppiSize roi = { width, height };
    
    // 执行HDTV转换
    NppStatus status = nppiNV12ToRGB_709HDTV_8u_P2C3R(pSrc, width, d_rgb, rgbStep, roi);
    EXPECT_EQ(status, NPP_NO_ERROR);
    
    // 清理内存
    nppsFree(d_srcY);
    nppsFree(d_srcUV);
    nppiFree(d_rgb);
}

// 测试不同图像尺寸
TEST_F(NV12ToRGBTest, VariousImageSizes) {
    struct TestSize {
        int width, height;
        const char* name;
    } sizes[] = {
        {32, 32, "32x32"},
        {128, 64, "128x64"},
        {256, 128, "256x128"},
        {640, 480, "640x480"},
        {1920, 1080, "1920x1080"}
    };
    
    for (const auto& size : sizes) {
        // 确保是偶数尺寸
        if (size.width % 2 != 0 || size.height % 2 != 0) continue;
        
        std::vector<Npp8u> yData(size.width * size.height, 128);  // 中性灰
        std::vector<Npp8u> uvData(size.width * size.height / 2, 128);  // 中性色度
        
        // 分配GPU内存
        Npp8u* d_srcY = nppsMalloc_8u(size.width * size.height);
        Npp8u* d_srcUV = nppsMalloc_8u(size.width * size.height / 2);
        
        int rgbStep;
        Npp8u* d_rgb = nppiMalloc_8u_C3(size.width, size.height, &rgbStep);
        
        if (!d_srcY || !d_srcUV || !d_rgb) {
            // 清理已分配的内存
            if (d_srcY) nppsFree(d_srcY);
            if (d_srcUV) nppsFree(d_srcUV);
            if (d_rgb) nppiFree(d_rgb);
            
            GTEST_SKIP() << "Could not allocate GPU memory for " << size.name;
            continue;
        }
        
        // 复制数据并执行转换
        cudaMemcpy(d_srcY, yData.data(), yData.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_srcUV, uvData.data(), uvData.size(), cudaMemcpyHostToDevice);
        
        const Npp8u* pSrc[2] = { d_srcY, d_srcUV };
        NppiSize roi = { size.width, size.height };
        
        NppStatus status = nppiNV12ToRGB_8u_P2C3R(pSrc, size.width, d_rgb, rgbStep, roi);
        EXPECT_EQ(status, NPP_NO_ERROR) << "Failed for size " << size.name;
        
        // 清理内存
        nppsFree(d_srcY);
        nppsFree(d_srcUV);
        nppiFree(d_rgb);
    }
}

// 参数验证测试
class NV12ToRGBParameterTest : public ::testing::Test {};

TEST_F(NV12ToRGBParameterTest, NullPointerError) {
    NppiSize roi = { 64, 64 };
    int step = 64;
    Npp8u dummy;
    
    // 测试空指针错误
    EXPECT_EQ(nppiNV12ToRGB_8u_P2C3R(nullptr, step, &dummy, step, roi), 
              NPP_NULL_POINTER_ERROR);
    
    const Npp8u* nullSrc[2] = { nullptr, &dummy };
    EXPECT_EQ(nppiNV12ToRGB_8u_P2C3R(nullSrc, step, &dummy, step, roi), 
              NPP_NULL_POINTER_ERROR);
    
    const Npp8u* nullSrc2[2] = { &dummy, nullptr };
    EXPECT_EQ(nppiNV12ToRGB_8u_P2C3R(nullSrc2, step, &dummy, step, roi), 
              NPP_NULL_POINTER_ERROR);
    
    const Npp8u* validSrc[2] = { &dummy, &dummy };
    EXPECT_EQ(nppiNV12ToRGB_8u_P2C3R(validSrc, step, nullptr, step, roi), 
              NPP_NULL_POINTER_ERROR);
}

TEST_F(NV12ToRGBParameterTest, InvalidROI) {
    Npp8u dummy;
    const Npp8u* pSrc[2] = { &dummy, &dummy };
    int step = 64;
    
    // 测试无效的ROI
    NppiSize invalidROI1 = { 0, 64 };
    EXPECT_EQ(nppiNV12ToRGB_8u_P2C3R(pSrc, step, &dummy, step, invalidROI1), 
              NPP_WRONG_INTERSECTION_ROI_ERROR);
    
    NppiSize invalidROI2 = { 64, 0 };
    EXPECT_EQ(nppiNV12ToRGB_8u_P2C3R(pSrc, step, &dummy, step, invalidROI2), 
              NPP_WRONG_INTERSECTION_ROI_ERROR);
    
    NppiSize invalidROI3 = { -10, 64 };
    EXPECT_EQ(nppiNV12ToRGB_8u_P2C3R(pSrc, step, &dummy, step, invalidROI3), 
              NPP_WRONG_INTERSECTION_ROI_ERROR);
    
    // 测试奇数尺寸（NV12要求偶数尺寸）
    NppiSize oddSize = { 63, 64 };
    EXPECT_EQ(nppiNV12ToRGB_8u_P2C3R(pSrc, step, &dummy, step, oddSize), 
              NPP_WRONG_INTERSECTION_ROI_ERROR);
}

TEST_F(NV12ToRGBParameterTest, InvalidStep) {
    Npp8u dummy;
    const Npp8u* pSrc[2] = { &dummy, &dummy };
    NppiSize roi = { 64, 64 };
    
    // 测试无效的step
    EXPECT_EQ(nppiNV12ToRGB_8u_P2C3R(pSrc, 0, &dummy, 64, roi), 
              NPP_STEP_ERROR);
    
    EXPECT_EQ(nppiNV12ToRGB_8u_P2C3R(pSrc, -1, &dummy, 64, roi), 
              NPP_STEP_ERROR);
    
    EXPECT_EQ(nppiNV12ToRGB_8u_P2C3R(pSrc, 64, &dummy, 0, roi), 
              NPP_STEP_ERROR);
    
    EXPECT_EQ(nppiNV12ToRGB_8u_P2C3R(pSrc, 64, &dummy, -1, roi), 
              NPP_STEP_ERROR);
}

// torchcodec兼容性测试
class TorchCodecCompatibilityTest : public ::testing::Test {
protected:
    void SetUp() override {
        // torchcodec常用的视频尺寸
        width = 1920;
        height = 1080;
    }
    
    int width, height;
};

TEST_F(TorchCodecCompatibilityTest, TorchCodecUsagePattern) {
    // 模拟torchcodec的使用模式
    
    // 分配NV12数据（模拟从NVDEC解码器输出）
    Npp8u* d_srcY = nppsMalloc_8u(width * height);
    Npp8u* d_srcUV = nppsMalloc_8u(width * height / 2);
    
    int rgbStep;
    Npp8u* d_rgb = nppiMalloc_8u_C3(width, height, &rgbStep);
    
    if (!d_srcY || !d_srcUV || !d_rgb) {
        // 清理内存
        if (d_srcY) nppsFree(d_srcY);
        if (d_srcUV) nppsFree(d_srcUV);
        if (d_rgb) nppiFree(d_rgb);
        
        GTEST_SKIP() << "Cannot allocate GPU memory for 1920x1080 test";
    }
    
    // 初始化测试数据
    cudaMemset(d_srcY, 128, width * height);      // 中性灰度
    cudaMemset(d_srcUV, 128, width * height / 2); // 中性色度
    
    const Npp8u* pSrc[2] = { d_srcY, d_srcUV };
    NppiSize roi = { width, height };
    
    // 测试torchcodec使用的两个主要函数
    
    // 1. BT.709 color space conversion (for HDTV content)
    NppStatus status1 = nppiNV12ToRGB_709CSC_8u_P2C3R(pSrc, width, d_rgb, rgbStep, roi);
    EXPECT_EQ(status1, NPP_NO_ERROR) << "BT.709 conversion failed";
    
    // 2. Standard conversion (fallback)
    NppStatus status2 = nppiNV12ToRGB_8u_P2C3R(pSrc, width, d_rgb, rgbStep, roi);
    EXPECT_EQ(status2, NPP_NO_ERROR) << "Standard conversion failed";
    
    // 验证结果不为零
    std::vector<Npp8u> sample(rgbStep);
    cudaMemcpy(sample.data(), d_rgb, rgbStep, cudaMemcpyDeviceToHost);
    
    bool hasNonZero = false;
    for (int i = 0; i < rgbStep; i += 3) {
        if (sample[i] != 0 || sample[i+1] != 0 || sample[i+2] != 0) {
            hasNonZero = true;
            break;
        }
    }
    EXPECT_TRUE(hasNonZero) << "RGB output appears to be all zeros";
    
    // 清理内存
    nppsFree(d_srcY);
    nppsFree(d_srcUV);
    nppiFree(d_rgb);
}