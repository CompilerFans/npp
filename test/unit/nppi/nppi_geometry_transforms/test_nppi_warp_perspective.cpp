#include <gtest/gtest.h>
#include "npp.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

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
    
    // 执行透视变换
    NppStatus status = nppiWarpPerspective_8u_C1R(d_src, srcSize, srcStep, srcROI,
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
    
    // 执行透视变换
    NppStatus status = nppiWarpPerspective_8u_C1R(d_src, srcSize, srcStep, srcROI,
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
    
    // 验证白色方块是否移动到了正确位置（12-20, 12-20）
    bool foundWhitePixel = false;
    for (int y = 12; y < 20 && y < dstHeight; y++) {
        for (int x = 12; x < 20 && x < dstWidth; x++) {
            if (resultData[y * dstWidth + x] > 128) {
                foundWhitePixel = true;
                break;
            }
        }
    }
    EXPECT_TRUE(foundWhitePixel) << "Translation failed: white square not found at expected position";
    
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
    
    // 执行透视变换
    NppStatus status = nppiWarpPerspective_32f_C1R(d_src, srcSize, srcStep, srcROI,
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

// 测试三通道透视变换
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_8u_C3R_Identity) {
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
    
    // 恒等透视变换矩阵
    double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    
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
    
    // 执行透视变换
    NppStatus status = nppiWarpPerspective_8u_C3R(d_src, srcSize, srcStep, srcROI,
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
    
    // 验证恒等变换结果应该与原图相同
    for (int i = 0; i < srcWidth * srcHeight * 3; i++) {
        EXPECT_EQ(resultData[i], srcData[i])
            << "Identity transform failed at pixel " << i;
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
    
    // 执行透视变换
    NppStatus status = nppiWarpPerspective_32f_C1R(d_src, srcSize, srcStep, srcROI,
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
    
    // 验证透视变换有效果（结果不应该与原图完全相同）
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
            srcData[y * srcWidth + x] = (x < srcWidth/2) ? 0.0f : 1.0f;
        }
    }
    
    // 轻微缩放变换，便于观察插值效果
    double coeffs[3][3] = {{0.9, 0.0, 0.0}, {0.0, 0.9, 0.0}, {0.0, 0.0, 1.0}};
    
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
    
    std::vector<Npp32f> nnResult(dstWidth * dstHeight);
    std::vector<Npp32f> linearResult(dstWidth * dstHeight);
    std::vector<Npp32f> cubicResult(dstWidth * dstHeight);
    
    // 测试最近邻插值
    NppStatus status = nppiWarpPerspective_32f_C1R(d_src, srcSize, srcStep, srcROI,
                                                   d_dst, dstStep, dstROI,
                                                   coeffs, NPPI_INTER_NN);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    for (int y = 0; y < dstHeight; y++) {
        cudaMemcpy(nnResult.data() + y * dstWidth,
                   (char*)d_dst + y * dstStep,
                   dstWidth * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    // 测试双线性插值
    status = nppiWarpPerspective_32f_C1R(d_src, srcSize, srcStep, srcROI,
                                        d_dst, dstStep, dstROI,
                                        coeffs, NPPI_INTER_LINEAR);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    for (int y = 0; y < dstHeight; y++) {
        cudaMemcpy(linearResult.data() + y * dstWidth,
                   (char*)d_dst + y * dstStep,
                   dstWidth * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    // 验证不同插值方法产生不同结果
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

// 测试错误处理
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_ErrorHandling) {
    // 测试空指针
    double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    
    NppStatus status = nppiWarpPerspective_8u_C1R(nullptr, srcSize, 32, srcROI,
                                                  nullptr, 32, dstROI,
                                                  coeffs, NPPI_INTER_NN);
    EXPECT_EQ(status, NPP_NULL_POINTER_ERROR);
    
    // 测试无效矩阵（会导致除零）
    double invalidCoeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.0}};
    
    status = nppiWarpPerspective_8u_C1R((Npp8u*)0x1000, srcSize, 32, srcROI,
                                       (Npp8u*)0x2000, 32, dstROI,
                                       invalidCoeffs, NPPI_INTER_NN);
    EXPECT_EQ(status, NPP_COEFFICIENT_ERROR);
}

// 测试流上下文
TEST_F(WarpPerspectiveFunctionalTest, WarpPerspective_StreamContext) {
    std::vector<Npp8u> srcData(16, 128);  // 小图像，填充值128
    
    double coeffs[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    
    NppiSize smallSize = {4, 4};
    NppiRect smallROI = {0, 0, 4, 4};
    
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C1(4, 4, &srcStep);
    Npp8u* d_dst = nppiMalloc_8u_C1(4, 4, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 复制数据到GPU
    cudaMemcpy2D(d_src, srcStep, srcData.data(), 4, 4, 4, cudaMemcpyHostToDevice);
    
    // 使用默认流上下文
    NppStreamContext nppStreamCtx = {0};
    NppStatus status = nppiWarpPerspective_8u_C1R_Ctx(d_src, smallSize, srcStep, smallROI,
                                                      d_dst, dstStep, smallROI,
                                                      coeffs, NPPI_INTER_NN, nppStreamCtx);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    nppiFree(d_src);
    nppiFree(d_dst);
}