#include <gtest/gtest.h>
#include "npp.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

/**
 * @file test_nppi_copy_extended.cpp
 * @brief 测试新增的nppiCopy API功能
 */

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
            srcData[idx + 0] = (float)x / (width - 1);   // R通道: 0.0-1.0渐变
            srcData[idx + 1] = (float)y / (height - 1);  // G通道: 0.0-1.0渐变
            srcData[idx + 2] = 0.5f;                     // B通道: 固定0.5
        }
    }
    
    // 分配GPU内存
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C3(width, height, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 复制输入数据到GPU
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width * channels,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
    }
    
    // 执行复制
    NppStatus status = nppiCopy_32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 复制结果回主机
    std::vector<Npp32f> resultData(width * height * channels);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width * channels,
                   (char*)d_dst + y * dstStep,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    // 验证结果：应该完全一致
    for (int i = 0; i < width * height * channels; i++) {
        EXPECT_FLOAT_EQ(resultData[i], srcData[i]) 
            << "复制失败 at index " << i << ": got " << resultData[i] 
            << ", expected " << srcData[i];
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
            srcData[idx + 0] = (float)(x + y) / (width + height - 2);  // R通道
            srcData[idx + 1] = (float)x / (width - 1);                 // G通道
            srcData[idx + 2] = (float)y / (height - 1);                // B通道
        }
    }
    
    // 分配GPU内存 - 源是packed格式，目标是planar格式
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C3(width, height, &srcStep);
    Npp32f* d_dstR = nppiMalloc_32f_C1(width, height, &dstStep);
    Npp32f* d_dstG = nppiMalloc_32f_C1(width, height, &dstStep);
    Npp32f* d_dstB = nppiMalloc_32f_C1(width, height, &dstStep);
    
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dstR, nullptr);
    ASSERT_NE(d_dstG, nullptr);
    ASSERT_NE(d_dstB, nullptr);
    
    // 复制packed源数据到GPU
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width * channels,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
    }
    
    // 设置目标指针数组
    Npp32f* dstPtrs[3] = {d_dstR, d_dstG, d_dstB};
    
    // 执行packed到planar转换
    NppStatus status = nppiCopy_32f_C3P3R(d_src, srcStep, dstPtrs, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 验证结果
    std::vector<Npp32f> resultR(width * height);
    std::vector<Npp32f> resultG(width * height);
    std::vector<Npp32f> resultB(width * height);
    
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultR.data() + y * width, (char*)d_dstR + y * dstStep, width * sizeof(Npp32f), cudaMemcpyDeviceToHost);
        cudaMemcpy(resultG.data() + y * width, (char*)d_dstG + y * dstStep, width * sizeof(Npp32f), cudaMemcpyDeviceToHost);
        cudaMemcpy(resultB.data() + y * width, (char*)d_dstB + y * dstStep, width * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    }
    
    // 验证每个通道的数据
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int packedIdx = (y * width + x) * channels;
            int planarIdx = y * width + x;
            
            EXPECT_FLOAT_EQ(resultR[planarIdx], srcData[packedIdx + 0]) 
                << "R通道转换失败 at (" << x << "," << y << ")";
            EXPECT_FLOAT_EQ(resultG[planarIdx], srcData[packedIdx + 1]) 
                << "G通道转换失败 at (" << x << "," << y << ")";
            EXPECT_FLOAT_EQ(resultB[planarIdx], srcData[packedIdx + 2]) 
                << "B通道转换失败 at (" << x << "," << y << ")";
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
        -1.0f, 0.0f, 1.0f,           // 负值，零，正值
        3.14159f, -2.71828f, 0.5f,   // 数学常数
        1e6f, -1e-6f, 1e-38f         // 极值
    };
    
    NppiSize testRoi = {3, 1};
    
    // 分配GPU内存
    int step = 3 * 3 * sizeof(Npp32f);
    Npp32f* d_src = nppsMalloc_32f(3 * 3);
    Npp32f* d_dst = nppsMalloc_32f(3 * 3);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 复制数据
    cudaMemcpy(d_src, srcData.data(), srcData.size() * sizeof(Npp32f), cudaMemcpyHostToDevice);
    
    // 执行复制
    NppStatus status = nppiCopy_32f_C3R(d_src, step, d_dst, step, testRoi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 验证结果
    std::vector<Npp32f> resultData(3 * 3);
    cudaMemcpy(resultData.data(), d_dst, resultData.size() * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < srcData.size(); i++) {
        EXPECT_FLOAT_EQ(resultData[i], srcData[i]) 
            << "边界值复制失败 at index " << i;
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
    Npp32f* d_src = nppiMalloc_32f_C3(width, height, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 复制数据
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width * channels,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
    }
    
    // 创建流上下文
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    // 执行复制
    NppStatus status = nppiCopy_32f_C3R_Ctx(d_src, srcStep, d_dst, dstStep, roi, nppStreamCtx);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 验证结果
    std::vector<Npp32f> resultData(width * height * channels);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width * channels,
                   (char*)d_dst + y * dstStep,
                   width * channels * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    // 所有值应该都是1.5f
    for (int i = 0; i < width * height * channels; i++) {
        EXPECT_FLOAT_EQ(resultData[i], 1.5f) 
            << "流上下文测试失败 at index " << i;
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}