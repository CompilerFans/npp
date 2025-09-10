#include <gtest/gtest.h>
#include "npp.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

class WarpAffineFunctionalTest : public ::testing::Test {
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

// 测试恒等变换（无变换）
TEST_F(WarpAffineFunctionalTest, WarpAffine_8u_C1R_Identity) {
    std::vector<Npp8u> srcData(srcWidth * srcHeight);
    
    // 创建测试图像：简单的渐变图案
    for (int y = 0; y < srcHeight; y++) {
        for (int x = 0; x < srcWidth; x++) {
            srcData[y * srcWidth + x] = (Npp8u)((x + y * 2) % 256);
        }
    }
    
    // 恒等变换矩阵：[1 0 0; 0 1 0]
    double coeffs[2][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    
    // 分配GPU内存
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C1(srcWidth, srcHeight, &srcStep);
    Npp8u* d_dst = nppiMalloc_8u_C1(dstWidth, dstHeight, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 复制数据到GPU
    for (int y = 0; y < srcHeight; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * srcWidth,
                   srcWidth * sizeof(Npp8u),
                   cudaMemcpyHostToDevice);
    }
    
    // 执行仿射变换
    NppStatus status = nppiWarpAffine_8u_C1R(d_src, srcSize, srcStep, srcROI,
                                             d_dst, dstStep, dstROI,
                                             coeffs, NPPI_INTER_NN);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 复制结果回主机
    std::vector<Npp8u> resultData(dstWidth * dstHeight);
    for (int y = 0; y < dstHeight; y++) {
        cudaMemcpy(resultData.data() + y * dstWidth,
                   (char*)d_dst + y * dstStep,
                   dstWidth * sizeof(Npp8u),
                   cudaMemcpyDeviceToHost);
    }
    
    // 验证恒等变换结果应该与原图相同
    for (int i = 0; i < srcWidth * srcHeight; i++) {
        EXPECT_EQ(resultData[i], srcData[i])
            << "Identity transform failed at pixel " << i;
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// 测试平移变换
TEST_F(WarpAffineFunctionalTest, WarpAffine_8u_C1R_Translation) {
    std::vector<Npp8u> srcData(srcWidth * srcHeight, 0);
    
    // 在源图像中心创建一个4x4的白色方块
    int centerX = srcWidth / 2;
    int centerY = srcHeight / 2;
    for (int y = centerY - 2; y <= centerY + 1; y++) {
        for (int x = centerX - 2; x <= centerX + 1; x++) {
            if (x >= 0 && x < srcWidth && y >= 0 && y < srcHeight) {
                srcData[y * srcWidth + x] = 255;
            }
        }
    }
    
    // 平移变换：向右移动5像素，向下移动3像素
    // 逆变换矩阵：[1 0 -5; 0 1 -3]
    double coeffs[2][3] = {{1.0, 0.0, -5.0}, {0.0, 1.0, -3.0}};
    
    // 分配GPU内存
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C1(srcWidth, srcHeight, &srcStep);
    Npp8u* d_dst = nppiMalloc_8u_C1(dstWidth, dstHeight, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 初始化目标图像为0
    cudaMemset(d_dst, 0, dstStep * dstHeight);
    
    // 复制数据到GPU
    for (int y = 0; y < srcHeight; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * srcWidth,
                   srcWidth * sizeof(Npp8u),
                   cudaMemcpyHostToDevice);
    }
    
    // 执行仿射变换
    NppStatus status = nppiWarpAffine_8u_C1R(d_src, srcSize, srcStep, srcROI,
                                             d_dst, dstStep, dstROI,
                                             coeffs, NPPI_INTER_NN);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 复制结果回主机
    std::vector<Npp8u> resultData(dstWidth * dstHeight);
    for (int y = 0; y < dstHeight; y++) {
        cudaMemcpy(resultData.data() + y * dstWidth,
                   (char*)d_dst + y * dstStep,
                   dstWidth * sizeof(Npp8u),
                   cudaMemcpyDeviceToHost);
    }
    
    // 验证平移结果：在新位置应该有白色方块
    int newCenterX = centerX + 5;
    int newCenterY = centerY + 3;
    bool foundWhitePixel = false;
    for (int y = newCenterY - 2; y <= newCenterY + 1; y++) {
        for (int x = newCenterX - 2; x <= newCenterX + 1; x++) {
            if (x >= 0 && x < dstWidth && y >= 0 && y < dstHeight) {
                if (resultData[y * dstWidth + x] == 255) {
                    foundWhitePixel = true;
                }
            }
        }
    }
    EXPECT_TRUE(foundWhitePixel) << "Translation failed: white square not found at expected position";
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// 测试缩放变换
TEST_F(WarpAffineFunctionalTest, WarpAffine_32f_C1R_Scaling) {
    std::vector<Npp32f> srcData(srcWidth * srcHeight);
    
    // 创建简单的测试图像
    for (int y = 0; y < srcHeight; y++) {
        for (int x = 0; x < srcWidth; x++) {
            srcData[y * srcWidth + x] = (float)(x + y) / 10.0f;
        }
    }
    
    // 缩放变换：放大2倍
    // 逆变换矩阵：[0.5 0 0; 0 0.5 0]
    double coeffs[2][3] = {{0.5, 0.0, 0.0}, {0.0, 0.5, 0.0}};
    
    // 分配GPU内存
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C1(srcWidth, srcHeight, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C1(dstWidth, dstHeight, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 复制数据到GPU
    for (int y = 0; y < srcHeight; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * srcWidth,
                   srcWidth * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
    }
    
    // 执行仿射变换
    NppStatus status = nppiWarpAffine_32f_C1R(d_src, srcSize, srcStep, srcROI,
                                              d_dst, dstStep, dstROI,
                                              coeffs, NPPI_INTER_LINEAR);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 复制结果回主机
    std::vector<Npp32f> resultData(dstWidth * dstHeight);
    for (int y = 0; y < dstHeight; y++) {
        cudaMemcpy(resultData.data() + y * dstWidth,
                   (char*)d_dst + y * dstStep,
                   dstWidth * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    // 验证缩放效果：检查几个点的近似值
    // 目标(0,0) -> 源(0,0) = 0.0，这是正确的
    EXPECT_FLOAT_EQ(resultData[0], 0.0f) << "Scaling transformation failed at (0,0)";
    
    // 目标(2,2) -> 源(1,1) = (1+1)/10.0 = 0.2，检查这个位置
    int idx_2_2 = 2 * dstWidth + 2;
    EXPECT_GT(resultData[idx_2_2], 0.1f) << "Scaling transformation failed at (2,2)";
    EXPECT_LT(resultData[idx_2_2], 0.3f) << "Scaling transformation produced incorrect values at (2,2)";
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// 测试三通道图像变换
TEST_F(WarpAffineFunctionalTest, WarpAffine_8u_C3R_Identity) {
    std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);
    
    // 创建RGB测试图像
    for (int y = 0; y < srcHeight; y++) {
        for (int x = 0; x < srcWidth; x++) {
            int idx = (y * srcWidth + x) * 3;
            srcData[idx + 0] = (Npp8u)(x % 256);     // R
            srcData[idx + 1] = (Npp8u)(y % 256);     // G  
            srcData[idx + 2] = (Npp8u)((x + y) % 256); // B
        }
    }
    
    // 恒等变换
    double coeffs[2][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    
    // 分配GPU内存
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C3(srcWidth, srcHeight, &srcStep);
    Npp8u* d_dst = nppiMalloc_8u_C3(dstWidth, dstHeight, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 复制数据到GPU
    for (int y = 0; y < srcHeight; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * srcWidth * 3,
                   srcWidth * 3 * sizeof(Npp8u),
                   cudaMemcpyHostToDevice);
    }
    
    // 执行仿射变换
    NppStatus status = nppiWarpAffine_8u_C3R(d_src, srcSize, srcStep, srcROI,
                                             d_dst, dstStep, dstROI,
                                             coeffs, NPPI_INTER_NN);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 复制结果回主机
    std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
    for (int y = 0; y < dstHeight; y++) {
        cudaMemcpy(resultData.data() + y * dstWidth * 3,
                   (char*)d_dst + y * dstStep,
                   dstWidth * 3 * sizeof(Npp8u),
                   cudaMemcpyDeviceToHost);
    }
    
    // 验证三通道恒等变换
    for (int i = 0; i < srcWidth * srcHeight * 3; i++) {
        EXPECT_EQ(resultData[i], srcData[i])
            << "C3 identity transform failed at channel " << (i % 3) 
            << ", pixel " << (i / 3);
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// 测试旋转变换
TEST_F(WarpAffineFunctionalTest, WarpAffine_32f_C1R_Rotation) {
    const int size = 16; // 使用较小尺寸便于测试
    NppiSize testSize = {size, size};
    NppiRect testROI = {0, 0, size, size};
    
    std::vector<Npp32f> srcData(size * size, 0.0f);
    
    // 在中心创建一个十字图案
    int center = size / 2;
    for (int i = 0; i < size; i++) {
        srcData[center * size + i] = 1.0f; // 水平线
        srcData[i * size + center] = 1.0f; // 垂直线
    }
    
    // 45度旋转矩阵（逆变换）
    double cos45 = cos(M_PI / 4.0);
    double sin45 = sin(M_PI / 4.0);
    double coeffs[2][3] = {{cos45, sin45, 0.0}, {-sin45, cos45, 0.0}};
    
    // 分配GPU内存
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C1(size, size, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C1(size, size, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 初始化目标为0
    cudaMemset(d_dst, 0, dstStep * size);
    
    // 复制数据到GPU
    for (int y = 0; y < size; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * size,
                   size * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
    }
    
    // 执行仿射变换
    NppStatus status = nppiWarpAffine_32f_C1R(d_src, testSize, srcStep, testROI,
                                              d_dst, dstStep, testROI,
                                              coeffs, NPPI_INTER_LINEAR);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 复制结果回主机
    std::vector<Npp32f> resultData(size * size);
    for (int y = 0; y < size; y++) {
        cudaMemcpy(resultData.data() + y * size,
                   (char*)d_dst + y * dstStep,
                   size * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    // 验证旋转效果：检查是否有非零像素（十字变为X）
    bool hasNonZeroPixels = false;
    for (int i = 0; i < size * size; i++) {
        if (resultData[i] > 0.1f) {
            hasNonZeroPixels = true;
            break;
        }
    }
    EXPECT_TRUE(hasNonZeroPixels) << "Rotation transformation produced no output";
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// 测试插值方法
TEST_F(WarpAffineFunctionalTest, WarpAffine_InterpolationMethods) {
    const int size = 8;
    NppiSize testSize = {size, size};
    NppiRect testROI = {0, 0, size, size};
    
    std::vector<Npp8u> srcData(size * size);
    for (int i = 0; i < size * size; i++) {
        srcData[i] = (i % 2 == 0) ? 255 : 0; // 棋盘图案
    }
    
    // 轻微缩放变换
    double coeffs[2][3] = {{0.9, 0.0, 0.0}, {0.0, 0.9, 0.0}};
    
    // 分配GPU内存
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C1(size, size, &srcStep);
    Npp8u* d_dst = nppiMalloc_8u_C1(size, size, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    for (int y = 0; y < size; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * size,
                   size * sizeof(Npp8u),
                   cudaMemcpyHostToDevice);
    }
    
    // 测试最近邻插值
    NppStatus status = nppiWarpAffine_8u_C1R(d_src, testSize, srcStep, testROI,
                                             d_dst, dstStep, testROI,
                                             coeffs, NPPI_INTER_NN);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 测试线性插值
    status = nppiWarpAffine_8u_C1R(d_src, testSize, srcStep, testROI,
                                   d_dst, dstStep, testROI,
                                   coeffs, NPPI_INTER_LINEAR);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 测试三次插值
    status = nppiWarpAffine_8u_C1R(d_src, testSize, srcStep, testROI,
                                   d_dst, dstStep, testROI,
                                   coeffs, NPPI_INTER_CUBIC);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// 测试错误处理
TEST_F(WarpAffineFunctionalTest, WarpAffine_ErrorHandling) {
    double coeffs[2][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    
    // 测试空指针
    NppStatus status = nppiWarpAffine_8u_C1R(nullptr, srcSize, 32, srcROI,
                                             nullptr, 32, dstROI,
                                             coeffs, NPPI_INTER_NN);
    EXPECT_NE(status, NPP_SUCCESS);
    
    // 测试无效的step
    status = nppiWarpAffine_8u_C1R(nullptr, srcSize, 0, srcROI,
                                   nullptr, 32, dstROI,
                                   coeffs, NPPI_INTER_NN);
    EXPECT_NE(status, NPP_SUCCESS);
    
    // 测试无效的尺寸
    NppiSize invalidSize = {0, 0};
    status = nppiWarpAffine_8u_C1R(nullptr, invalidSize, 32, srcROI,
                                   nullptr, 32, dstROI,
                                   coeffs, NPPI_INTER_NN);
    EXPECT_NE(status, NPP_SUCCESS);
    
    // 测试无效的插值方法
    status = nppiWarpAffine_8u_C1R(nullptr, srcSize, 32, srcROI,
                                   nullptr, 32, dstROI,
                                   coeffs, 999); // 无效插值方法
    EXPECT_NE(status, NPP_SUCCESS);
}

// 测试流上下文版本
TEST_F(WarpAffineFunctionalTest, WarpAffine_StreamContext) {
    const int size = 4;
    NppiSize testSize = {size, size};
    NppiRect testROI = {0, 0, size, size};
    
    std::vector<Npp32f> srcData(size * size);
    for (int i = 0; i < size * size; i++) {
        srcData[i] = (float)(i + 1);
    }
    
    double coeffs[2][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};
    
    // 分配GPU内存
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C1(size, size, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C1(size, size, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    for (int y = 0; y < size; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * size,
                   size * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
    }
    
    // 创建流上下文
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    
    // 执行带上下文的变换
    NppStatus status = nppiWarpAffine_32f_C1R_Ctx(d_src, testSize, srcStep, testROI,
                                                  d_dst, dstStep, testROI,
                                                  coeffs, NPPI_INTER_NN, nppStreamCtx);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    nppiFree(d_src);
    nppiFree(d_dst);
}