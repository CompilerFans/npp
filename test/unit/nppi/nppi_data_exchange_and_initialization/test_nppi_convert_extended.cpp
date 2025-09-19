#include <gtest/gtest.h>
#include "npp.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

/**
 * @file test_nppi_convert_extended.cpp
 * @brief 测试新增的nppiConvert API功能
 */

class ConvertExtendedFunctionalTest : public ::testing::Test {
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

// 测试nppiConvert_8u32f_C3R函数
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_BasicOperation) {
    // const int 3 = 3;  // Commented out as not used
    std::vector<Npp8u> srcData(width * height * 3);
    std::vector<Npp32f> expectedData(width * height * 3);
    
    // 创建测试数据：RGB渐变模式
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            srcData[idx + 0] = (Npp8u)((x * 255) / (width - 1));  // R通道
            srcData[idx + 1] = (Npp8u)((y * 255) / (height - 1)); // G通道  
            srcData[idx + 2] = (Npp8u)(128);                      // B通道固定值
            
            // 期望结果：8位转32位浮点 (0-255 -> 0.0-255.0)
            expectedData[idx + 0] = (Npp32f)srcData[idx + 0];
            expectedData[idx + 1] = (Npp32f)srcData[idx + 1];
            expectedData[idx + 2] = (Npp32f)srcData[idx + 2];
        }
    }
    
    // 分配GPU内存
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C3(width, height, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 复制输入数据到GPU
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width * 3,
                   width * 3 * sizeof(Npp8u),
                   cudaMemcpyHostToDevice);
    }
    
    // 执行转换
    NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 复制结果回主机
    std::vector<Npp32f> resultData(width * height * 3);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width * 3,
                   (char*)d_dst + y * dstStep,
                   width * 3 * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    // 验证结果
    for (int i = 0; i < width * height * 3; i++) {
        EXPECT_FLOAT_EQ(resultData[i], expectedData[i]) 
            << "转换失败 at index " << i << ": got " << resultData[i] 
            << ", expected " << expectedData[i];
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// 测试nppiConvert_8u32f_C3R边界值
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_BoundaryValues) {
    // const int 3 = 3;  // Commented out as not used
    std::vector<Npp8u> srcData = {
        0, 0, 0,        // 最小值 
        255, 255, 255,  // 最大值
        128, 64, 192    // 中间值
    };
    std::vector<Npp32f> expectedData = {
        0.0f, 0.0f, 0.0f,
        255.0f, 255.0f, 255.0f,
        128.0f, 64.0f, 192.0f
    };
    
    NppiSize testRoi = {3, 1};
    
    // 分配GPU内存
    int srcStep = 3 * 3 * sizeof(Npp8u);
    int dstStep = 3 * 3 * sizeof(Npp32f);
    
    Npp8u* d_src = nppsMalloc_8u(3 * 3);
    Npp32f* d_dst = nppsMalloc_32f(3 * 3);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 复制数据
    cudaMemcpy(d_src, srcData.data(), srcData.size() * sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    // 执行转换
    NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, testRoi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 验证结果
    std::vector<Npp32f> resultData(3 * 3);
    cudaMemcpy(resultData.data(), d_dst, resultData.size() * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < expectedData.size(); i++) {
        EXPECT_FLOAT_EQ(resultData[i], expectedData[i]) 
            << "边界值转换失败 at index " << i;
    }
    
    nppsFree(d_src);
    nppsFree(d_dst);
}

// 测试流上下文版本
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_StreamContext) {
    // const int 3 = 3;  // Commented out as not used
    std::vector<Npp8u> srcData(width * height * 3, 100); // 所有像素值为100
    
    // 分配GPU内存
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C3(width, height, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 复制数据
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width * 3,
                   width * 3 * sizeof(Npp8u),
                   cudaMemcpyHostToDevice);
    }
    
    // 创建流上下文
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    // 执行转换
    NppStatus status = nppiConvert_8u32f_C3R_Ctx(d_src, srcStep, d_dst, dstStep, roi, nppStreamCtx);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 验证结果
    std::vector<Npp32f> resultData(width * height * 3);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width * 3,
                   (char*)d_dst + y * dstStep,
                   width * 3 * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    // 所有值应该都是100.0f
    for (int i = 0; i < width * height * 3; i++) {
        EXPECT_FLOAT_EQ(resultData[i], 100.0f) 
            << "流上下文测试失败 at index " << i;
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}