/**
 * @file test_nppi_copy_extended.cpp
 * @brief NPP Copy函数Context版本测试
 */

#include "../../framework/npp_test_base.h"
#include <algorithm>

using namespace npp_functional_test;

class CopyExtendedTest : public NppTestBase {
protected:
    void SetUp() override {
        NppTestBase::SetUp();
    }
    
    void TearDown() override {
        NppTestBase::TearDown();
    }
};

// 测试8位单通道Copy函数Context版本
TEST_F(CopyExtendedTest, Copy_8u_C1R_Ctx_Basic) {
    const int width = 64, height = 64;
    
    // 准备测试数据 - 棋盘图案
    std::vector<Npp8u> srcData(width * height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            srcData[y * width + x] = ((x / 8 + y / 8) % 2) ? 255 : 0;
        }
    }
    
    NppImageMemory<Npp8u> src(width, height);
    NppImageMemory<Npp8u> dst(width, height);
    
    src.copyFromHost(srcData);
    cudaMemset(dst.get(), 0, dst.sizeInBytes()); // 初始化为0
    
    // 设置参数
    NppiSize oSizeROI = {width, height};
    
    // 获取流上下文
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    // 执行Copy操作
    NppStatus status = nppiCopy_8u_C1R_Ctx(
        src.get(), src.step(),
        dst.get(), dst.step(),
        oSizeROI, nppStreamCtx);
    
    ASSERT_EQ(status, NPP_SUCCESS);
    
    // 同步并获取结果
    cudaStreamSynchronize(nppStreamCtx.hStream);
    std::vector<Npp8u> dstData(width * height);
    dst.copyToHost(dstData);
    
    // 验证复制结果
    for (int i = 0; i < width * height; i++) {
        ASSERT_EQ(srcData[i], dstData[i]);
    }
}

// 测试8位三通道Copy函数Context版本
TEST_F(CopyExtendedTest, Copy_8u_C3R_Ctx_ColorImage) {
    const int width = 32, height = 32;
    
    // 准备测试数据 - RGB渐变
    std::vector<Npp8u> srcData(width * height * 3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            srcData[idx + 0] = static_cast<Npp8u>(x * 255 / width);    // R
            srcData[idx + 1] = static_cast<Npp8u>(y * 255 / height);   // G
            srcData[idx + 2] = static_cast<Npp8u>(255 - (x + y) * 255 / (width + height)); // B
        }
    }
    
    NppImageMemory<Npp8u> src(width, height, 3);
    NppImageMemory<Npp8u> dst(width, height, 3);
    
    src.copyFromHost(srcData);
    cudaMemset(dst.get(), 0, dst.sizeInBytes()); // 初始化为0
    
    // 设置参数
    NppiSize oSizeROI = {width, height};
    
    // 获取流上下文
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    // 执行Copy操作
    NppStatus status = nppiCopy_8u_C3R_Ctx(
        src.get(), src.step(),
        dst.get(), dst.step(),
        oSizeROI, nppStreamCtx);
    
    ASSERT_EQ(status, NPP_SUCCESS);
    
    // 同步并获取结果
    cudaStreamSynchronize(nppStreamCtx.hStream);
    std::vector<Npp8u> dstData(width * height * 3);
    dst.copyToHost(dstData);
    
    // 验证复制结果
    for (int i = 0; i < width * height * 3; i++) {
        ASSERT_EQ(srcData[i], dstData[i]);
    }
}

// 测试ROI部分复制
TEST_F(CopyExtendedTest, Copy_8u_C1R_Ctx_PartialROI) {
    const int width = 64, height = 64;
    
    // 准备测试数据
    std::vector<Npp8u> srcData(width * height);
    for (int i = 0; i < width * height; i++) {
        srcData[i] = static_cast<Npp8u>(i % 256);
    }
    
    NppImageMemory<Npp8u> src(width, height);
    NppImageMemory<Npp8u> dst(width, height);
    
    src.copyFromHost(srcData);
    cudaMemset(dst.get(), 0, dst.sizeInBytes()); // 初始化为0
    
    // 设置ROI为中心区域
    NppiSize oSizeROI = {32, 32};
    int xOffset = 16;
    int yOffset = 16;
    
    // 获取流上下文
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    
    // 执行ROI Copy操作
    Npp8u* pSrcROI = src.get() + yOffset * src.step() / sizeof(Npp8u) + xOffset;
    Npp8u* pDstROI = dst.get() + yOffset * dst.step() / sizeof(Npp8u) + xOffset;
    
    NppStatus status = nppiCopy_8u_C1R_Ctx(
        pSrcROI, src.step(),
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
                // ROI内应该复制了源数据
                ASSERT_EQ(dstData[idx], srcData[idx]);
            } else {
                // ROI外应该保持0
                ASSERT_EQ(dstData[idx], 0);
            }
        }
    }
}