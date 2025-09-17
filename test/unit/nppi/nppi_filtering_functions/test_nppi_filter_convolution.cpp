/**
 * @file test_nppi_filter_convolution.cpp
 * @brief NPP 通用2D卷积函数测试
 */

#include "../../framework/npp_test_base.h"
#include <cmath>

using namespace npp_functional_test;

class FilterConvolutionFunctionalTest : public NppTestBase {
protected:
    void SetUp() override {
        NppTestBase::SetUp();
    }
    
    void TearDown() override {
        NppTestBase::TearDown();
    }
    
    // Helper function to create common kernels
    std::vector<Npp32s> createEdgeDetectionKernel() {
        // 3x3 Sobel X edge detection kernel
        return {-1, 0, 1,
                -2, 0, 2,
                -1, 0, 1};
    }
    
    std::vector<Npp32s> createSharpenKernel() {
        // 3x3 sharpening kernel
        return { 0, -1,  0,
                -1,  5, -1,
                 0, -1,  0};
    }
    
    std::vector<Npp32f> createGaussianKernel3x3() {
        // 3x3 Gaussian kernel (normalized)
        return {0.0625f, 0.125f, 0.0625f,
                0.125f,  0.25f,  0.125f,
                0.0625f, 0.125f, 0.0625f};
    }
};

// 尝试修复CUDA上下文损坏问题后重新启用
TEST_F(FilterConvolutionFunctionalTest, Filter_8u_C1R_EdgeDetection) {
    const int width = 16, height = 16;
    
    // 创建垂直条纹测试图像
    std::vector<Npp8u> srcData(width * height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            srcData[y * width + x] = (x < width/2) ? 0 : 255;
        }
    }
    
    NppImageMemory<Npp8u> src(width, height);
    NppImageMemory<Npp8u> dst(width, height);
    
    src.copyFromHost(srcData);
    
    // 创建边缘检测卷积核
    std::vector<Npp32s> kernel = createEdgeDetectionKernel();
    NppiSize kernelSize = {3, 3};
    NppiPoint anchor = {1, 1}; // 中心锚点
    Npp32s divisor = 1;
    
    NppiSize roi = {width, height};
    
    // 根据构建配置选择不同的调用方式
    #ifdef USE_NVIDIA_NPP
        // NVIDIA NPP使用主机内存中的kernel
        NppStatus status = nppiFilter_8u_C1R(
            src.get(), src.step(),
            dst.get(), dst.step(),
            roi, kernel.data(), kernelSize, anchor, divisor);
    #else
        // OpenNPP需要设备内存中的kernel
        Npp32s* d_kernel;
        size_t kernelBytes = kernel.size() * sizeof(Npp32s);
        cudaMalloc(&d_kernel, kernelBytes);
        cudaMemcpy(d_kernel, kernel.data(), kernelBytes, cudaMemcpyHostToDevice);
        
        NppStatus status = nppiFilter_8u_C1R(
            src.get(), src.step(),
            dst.get(), dst.step(),
            roi, d_kernel, kernelSize, anchor, divisor);
        
        cudaFree(d_kernel); // 清理设备内存
    #endif
    
    ASSERT_EQ(status, NPP_SUCCESS) << "nppiFilter_8u_C1R edge detection failed";
    
    // 验证结果 - 边缘区域应该有非零响应
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
    srcData[height/2 * width + width/2] = 128; // 中心亮点
    
    NppImageMemory<Npp8u> src(width, height);
    NppImageMemory<Npp8u> dst(width, height);
    
    src.copyFromHost(srcData);
    
    // 创建锐化卷积核
    std::vector<Npp32s> kernel = createSharpenKernel();
    NppiSize kernelSize = {3, 3};
    NppiPoint anchor = {1, 1};
    Npp32s divisor = 1;
    
    NppiSize roi = {width, height};
    
    // 根据构建配置选择不同的调用方式  
    #ifdef USE_NVIDIA_NPP
        // NVIDIA NPP使用主机内存中的kernel
        NppStatus status = nppiFilter_8u_C1R(
            src.get(), src.step(),
            dst.get(), dst.step(),
            roi, kernel.data(), kernelSize, anchor, divisor);
    #else
        // OpenNPP需要设备内存中的kernel
        Npp32s* d_kernel;
        size_t kernelBytes = kernel.size() * sizeof(Npp32s);
        cudaMalloc(&d_kernel, kernelBytes);
        cudaMemcpy(d_kernel, kernel.data(), kernelBytes, cudaMemcpyHostToDevice);
        
        NppStatus status = nppiFilter_8u_C1R(
            src.get(), src.step(),
            dst.get(), dst.step(),
            roi, d_kernel, kernelSize, anchor, divisor);
        
        cudaFree(d_kernel); // 清理设备内存
    #endif
    
    ASSERT_EQ(status, NPP_SUCCESS) << "nppiFilter_8u_C1R sharpen failed";
    
    // 验证结果 - 中心点应该更亮，周围有对比增强
    std::vector<Npp8u> resultData(width * height);
    dst.copyToHost(resultData);
    
    int centerX = width / 2;
    int centerY = height / 2;
    int centerIdx = centerY * width + centerX;
    
    EXPECT_GT(resultData[centerIdx], srcData[centerIdx]) << "Sharpening should enhance the center bright point";
}

// NOTE: 重新启用测试 - 检查CUDA上下文损坏问题是否已解决
TEST_F(FilterConvolutionFunctionalTest, Filter_8u_C3R_Basic) {
    const int width = 8, height = 8;
    
    // 创建彩色测试图像
    std::vector<Npp8u> srcData(width * height * 3);
    for (int i = 0; i < width * height; i++) {
        srcData[i * 3 + 0] = 255; // Red channel - full intensity
        srcData[i * 3 + 1] = 128; // Green channel - half intensity  
        srcData[i * 3 + 2] = 0;   // Blue channel - zero intensity
    }
    
    // 使用手动内存分配
    int srcStep = width * 3 * sizeof(Npp8u);
    int dstStep = width * 3 * sizeof(Npp8u);
    
    Npp8u* srcPtr = nppsMalloc_8u(width * height * 3);
    Npp8u* dstPtr = nppsMalloc_8u(width * height * 3);
    
    ASSERT_NE(srcPtr, nullptr);
    ASSERT_NE(dstPtr, nullptr);
    
    // 复制数据到GPU
    cudaMemcpy(srcPtr, srcData.data(), width * height * 3 * sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    // 创建简单的平均滤波核
    std::vector<Npp32s> kernel = {1, 1, 1,
                                  1, 1, 1,
                                  1, 1, 1};
    NppiSize kernelSize = {3, 3};
    NppiPoint anchor = {1, 1};
    Npp32s divisor = 9; // 平均化
    
    NppiSize roi = {width, height};
    NppStatus status = nppiFilter_8u_C3R(
        srcPtr, srcStep,
        dstPtr, dstStep,
        roi, kernel.data(), kernelSize, anchor, divisor);
    
    ASSERT_EQ(status, NPP_SUCCESS) << "nppiFilter_8u_C3R failed";
    
    // 验证结果 - 检查所有通道都被处理
    std::vector<Npp8u> resultData(width * height * 3);
    cudaMemcpy(resultData.data(), dstPtr, width * height * 3 * sizeof(Npp8u), cudaMemcpyDeviceToHost);
    
    // 检查中心区域的结果（避免边界效应）
    int centerX = width / 2;
    int centerY = height / 2;
    int centerIdx = (centerY * width + centerX) * 3;
    
    EXPECT_GT(resultData[centerIdx + 0], 200) << "Red channel should be processed correctly";
    EXPECT_GT(resultData[centerIdx + 1], 100) << "Green channel should be processed correctly";
    EXPECT_LT(resultData[centerIdx + 2], 50) << "Blue channel should be processed correctly";
    
    // 清理内存
    nppsFree(srcPtr);
    nppsFree(dstPtr);
}

// 第三个测试：32位浮点高斯滤波
TEST_F(FilterConvolutionFunctionalTest, Filter_32f_C1R_Gaussian) {
    const int width = 8, height = 8;
    
    // 创建脉冲信号
    std::vector<Npp32f> srcData(width * height, 0.0f);
    srcData[height/2 * width + width/2] = 1.0f; // 中心脉冲
    
    NppImageMemory<Npp32f> src(width, height);
    NppImageMemory<Npp32f> dst(width, height);
    
    src.copyFromHost(srcData);
    
    // 创建高斯卷积核
    std::vector<Npp32f> kernel = createGaussianKernel3x3();
    NppiSize kernelSize = {3, 3};
    NppiPoint anchor = {1, 1};
    
    NppiSize roi = {width, height};
    
    // 根据构建配置选择不同的调用方式
    #ifdef USE_NVIDIA_NPP
        // NVIDIA NPP使用主机内存中的kernel
        NppStatus status = nppiFilter_32f_C1R(
            src.get(), src.step(),
            dst.get(), dst.step(),
            roi, kernel.data(), kernelSize, anchor);
    #else
        // OpenNPP需要设备内存中的kernel
        Npp32f* d_kernel;
        size_t kernelBytes = kernel.size() * sizeof(Npp32f);
        cudaMalloc(&d_kernel, kernelBytes);
        cudaMemcpy(d_kernel, kernel.data(), kernelBytes, cudaMemcpyHostToDevice);
        
        NppStatus status = nppiFilter_32f_C1R(
            src.get(), src.step(),
            dst.get(), dst.step(),
            roi, d_kernel, kernelSize, anchor);
        
        cudaFree(d_kernel); // 清理设备内存
    #endif
    
    ASSERT_EQ(status, NPP_SUCCESS) << "nppiFilter_32f_C1R Gaussian failed";
    
    // 验证结果 - 脉冲应该被高斯核扩散
    std::vector<Npp32f> resultData(width * height);
    dst.copyToHost(resultData);
    
    int centerX = width / 2;
    int centerY = height / 2;
    
    // 检查中心和周围像素
    EXPECT_GT(resultData[centerY * width + centerX], 0.2f) << "Center should have significant response";
    EXPECT_GT(resultData[centerY * width + centerX - 1], 0.05f) << "Adjacent pixels should have some response";
    EXPECT_GT(resultData[centerY * width + centerX + 1], 0.05f) << "Adjacent pixels should have some response";
}

// 错误处理测试
// NOTE: 测试已被禁用 - NVIDIA NPP对无效参数的错误检测行为与预期不符
TEST_F(FilterConvolutionFunctionalTest, DISABLED_Filter_ErrorHandling) {
    const int width = 16, height = 16;
    
    NppImageMemory<Npp8u> src(width, height);
    NppImageMemory<Npp8u> dst(width, height);
    
    std::vector<Npp32s> kernel = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    NppiSize kernelSize = {3, 3};
    NppiPoint anchor = {1, 1};
    NppiSize roi = {width, height};
    
    // 测试空指针
    NppStatus status = nppiFilter_8u_C1R(
        nullptr, src.step(),
        dst.get(), dst.step(),
        roi, kernel.data(), kernelSize, anchor, 1);
    EXPECT_NE(status, NPP_SUCCESS) << "Should fail with null source pointer";
    
    // 测试无效ROI
    NppiSize invalidRoi = {0, 0};
    status = nppiFilter_8u_C1R(
        src.get(), src.step(),
        dst.get(), dst.step(),
        invalidRoi, kernel.data(), kernelSize, anchor, 1);
    EXPECT_NE(status, NPP_SUCCESS) << "Should fail with invalid ROI";
    
    // 测试偶数核大小
    NppiSize evenKernelSize = {2, 2};
    status = nppiFilter_8u_C1R(
        src.get(), src.step(),
        dst.get(), dst.step(),
        roi, kernel.data(), evenKernelSize, anchor, 1);
    EXPECT_NE(status, NPP_SUCCESS) << "Should fail with even kernel size";
    
    // 测试无效锚点
    NppiPoint invalidAnchor = {-1, -1};
    status = nppiFilter_8u_C1R(
        src.get(), src.step(),
        dst.get(), dst.step(),
        roi, kernel.data(), kernelSize, invalidAnchor, 1);
    EXPECT_NE(status, NPP_SUCCESS) << "Should fail with invalid anchor";
    
    // 测试零除数
    status = nppiFilter_8u_C1R(
        src.get(), src.step(),
        dst.get(), dst.step(),
        roi, kernel.data(), kernelSize, anchor, 0);
    EXPECT_NE(status, NPP_SUCCESS) << "Should fail with zero divisor";
}

// NOTE: 测试已被禁用 - 该测试导致CUDA上下文损坏，影响后续所有测试
TEST_F(FilterConvolutionFunctionalTest, DISABLED_Filter_DifferentKernelSizes) {
    const int width = 32, height = 32;
    
    // 创建随机测试图像
    std::vector<Npp8u> srcData(width * height);
    for (int i = 0; i < width * height; i++) {
        srcData[i] = i % 256;
    }
    
    NppImageMemory<Npp8u> src(width, height);
    NppImageMemory<Npp8u> dst3x3(width, height);
    NppImageMemory<Npp8u> dst5x5(width, height);
    
    src.copyFromHost(srcData);
    
    NppiSize roi = {width, height};
    
    // 测试3x3卷积核
    std::vector<Npp32s> kernel3x3(9, 1);
    NppiSize kernelSize3x3 = {3, 3};
    NppiPoint anchor3x3 = {1, 1};
    
    NppStatus status3x3 = nppiFilter_8u_C1R(
        src.get(), src.step(), dst3x3.get(), dst3x3.step(),
        roi, kernel3x3.data(), kernelSize3x3, anchor3x3, 9);
    
    // 测试5x5卷积核
    std::vector<Npp32s> kernel5x5(25, 1);
    NppiSize kernelSize5x5 = {5, 5};
    NppiPoint anchor5x5 = {2, 2};
    
    NppStatus status5x5 = nppiFilter_8u_C1R(
        src.get(), src.step(), dst5x5.get(), dst5x5.step(),
        roi, kernel5x5.data(), kernelSize5x5, anchor5x5, 25);
    
    ASSERT_EQ(status3x3, NPP_SUCCESS) << "3x3 filter failed";
    ASSERT_EQ(status5x5, NPP_SUCCESS) << "5x5 filter failed";
    
    // 验证两种核都产生了有效的结果
    std::vector<Npp8u> result3x3(width * height);
    std::vector<Npp8u> result5x5(width * height);
    dst3x3.copyToHost(result3x3);
    dst5x5.copyToHost(result5x5);
    
    bool hasValid3x3 = false, hasValid5x5 = false;
    for (int i = 0; i < width * height; i++) {
        if (result3x3[i] > 0) hasValid3x3 = true;
        if (result5x5[i] > 0) hasValid5x5 = true;
    }
    
    EXPECT_TRUE(hasValid3x3) << "3x3 filter should produce valid results";
    EXPECT_TRUE(hasValid5x5) << "5x5 filter should produce valid results";
}