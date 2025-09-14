/**
 * @file test_nppi_set.cpp
 * @brief NPP Set函数测试（设置图像像素值）
 */

#include "../../framework/npp_test_base.h"
#include <algorithm>

using namespace npp_functional_test;

class SetFunctionalTest : public NppTestBase {
protected:
    void SetUp() override {
        NppTestBase::SetUp();
    }
    
    void TearDown() override {
        NppTestBase::TearDown();
    }
};

// 测试8位单通道Set函数
TEST_F(SetFunctionalTest, Set_8u_C1R_Ctx_Basic) {
    const int width = 64, height = 64;
    const Npp8u setValue = 128;
    
    // 准备内存
    NppImageMemory<Npp8u> dst(width, height);
    
    // 初始化为0
    cudaMemset(dst.get(), 0, dst.size());
    
    // 设置参数
    NppiSize oSizeROI = {width, height};
    
    // 获取流上下文
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    // 执行Set操作
    NppStatus status = nppiSet_8u_C1R_Ctx(
        setValue,
        dst.get(), dst.step(),
        oSizeROI, nppStreamCtx);
    
    ASSERT_EQ(status, NPP_SUCCESS);
    
    // 同步并获取结果
    cudaStreamSynchronize(nppStreamCtx.hStream);
    std::vector<Npp8u> dstData(width * height);
    dst.copyToHost(dstData);
    
    // 验证所有像素都被设置为指定值
    for (const auto& val : dstData) {
        ASSERT_EQ(val, setValue);
    }
}

// 测试8位三通道Set函数
TEST_F(SetFunctionalTest, Set_8u_C3R_Ctx_ColorValues) {
    const int width = 32, height = 32;
    const Npp8u aValue[3] = {255, 128, 64}; // RGB值
    
    // 准备内存
    NppImageMemory<Npp8u> dst(width, height, 3);
    
    // 初始化为0
    cudaMemset(dst.get(), 0, dst.size());
    
    // 设置参数
    NppiSize oSizeROI = {width, height};
    
    // 获取流上下文
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    // 执行Set操作
    NppStatus status = nppiSet_8u_C3R_Ctx(
        aValue,
        dst.get(), dst.step(),
        oSizeROI, nppStreamCtx);
    
    ASSERT_EQ(status, NPP_SUCCESS);
    
    // 同步并获取结果
    cudaStreamSynchronize(nppStreamCtx.hStream);
    std::vector<Npp8u> dstData(width * height * 3);
    dst.copyToHost(dstData);
    
    // 验证所有像素的RGB值
    for (int i = 0; i < width * height; i++) {
        ASSERT_EQ(dstData[i * 3 + 0], aValue[0]); // R
        ASSERT_EQ(dstData[i * 3 + 1], aValue[1]); // G
        ASSERT_EQ(dstData[i * 3 + 2], aValue[2]); // B
    }
}

// 测试32位浮点单通道Set函数
TEST_F(SetFunctionalTest, Set_32f_C1R_Ctx_FloatValue) {
    const int width = 48, height = 48;
    const Npp32f setValue = 3.14159f;
    
    // 准备内存
    NppImageMemory<Npp32f> dst(width, height);
    
    // 初始化为0
    cudaMemset(dst.get(), 0, dst.size());
    
    // 设置参数
    NppiSize oSizeROI = {width, height};
    
    // 获取流上下文
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    // 执行Set操作
    NppStatus status = nppiSet_32f_C1R_Ctx(
        setValue,
        dst.get(), dst.step(),
        oSizeROI, nppStreamCtx);
    
    ASSERT_EQ(status, NPP_SUCCESS);
    
    // 同步并获取结果
    cudaStreamSynchronize(nppStreamCtx.hStream);
    std::vector<Npp32f> dstData(width * height);
    dst.copyToHost(dstData);
    
    // 验证所有像素都被设置为指定值
    for (const auto& val : dstData) {
        ASSERT_FLOAT_EQ(val, setValue);
    }
}

// 测试ROI部分设置
TEST_F(SetFunctionalTest, Set_8u_C1R_Ctx_PartialROI) {
    const int width = 64, height = 64;
    const Npp8u setValue = 200;
    const Npp8u bgValue = 50;
    
    // 准备内存并初始化为背景值
    NppImageMemory<Npp8u> dst(width, height);
    cudaMemset(dst.get(), bgValue, dst.size());
    
    // 设置ROI为中心区域
    NppiSize oSizeROI = {32, 32};
    int xOffset = 16;
    int yOffset = 16;
    
    // 获取流上下文
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    // 执行Set操作（在ROI区域）
    Npp8u* pDstROI = dst.get() + yOffset * dst.step() / sizeof(Npp8u) + xOffset;
    NppStatus status = nppiSet_8u_C1R_Ctx(
        setValue,
        pDstROI, dst.step(),
        oSizeROI, nppStreamCtx);
    
    ASSERT_EQ(status, NPP_SUCCESS);
    
    // 同步并获取结果
    cudaStreamSynchronize(nppStreamCtx.hStream);
    std::vector<Npp8u> dstData(width * height);
    dst.copyToHost(dstData);
    
    // 验证ROI内外的值
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            if (x >= xOffset && x < xOffset + 32 && 
                y >= yOffset && y < yOffset + 32) {
                // ROI内应该是设置值
                ASSERT_EQ(dstData[idx], setValue);
            } else {
                // ROI外应该保持背景值
                ASSERT_EQ(dstData[idx], bgValue);
            }
        }
    }
}