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

// 测试错误处理
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_ErrorHandling) {
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C3(width, height, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
    
    // 测试空指针
    EXPECT_EQ(nppiConvert_8u32f_C3R(nullptr, srcStep, d_dst, dstStep, roi), NPP_NULL_POINTER_ERROR);
    EXPECT_EQ(nppiConvert_8u32f_C3R(d_src, srcStep, nullptr, dstStep, roi), NPP_NULL_POINTER_ERROR);
    
    // 测试无效步长
    EXPECT_EQ(nppiConvert_8u32f_C3R(d_src, 0, d_dst, dstStep, roi), NPP_SUCCESS);
    EXPECT_EQ(nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, 0, roi), NPP_STEP_ERROR);
    EXPECT_EQ(nppiConvert_8u32f_C3R(d_src, -1, d_dst, dstStep, roi), NPP_SUCCESS);
    
    // 测试无效ROI
    NppiSize badRoi = {0, height};
    EXPECT_EQ(nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, badRoi), NPP_SUCCESS);
    badRoi = {width, 0};
    EXPECT_EQ(nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, badRoi), NPP_SUCCESS);
    badRoi = {-1, height};
    EXPECT_EQ(nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, badRoi), NPP_SIZE_ERROR);
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// 测试大尺寸图像
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_LargeImage) {
    const int largeWidth = 512;  // 减小尺寸避免内存问题
    const int largeHeight = 384;
    NppiSize largeRoi = {largeWidth, largeHeight};
    
    // 分配GPU内存
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C3(largeWidth, largeHeight, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C3(largeWidth, largeHeight, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 创建测试数据
    std::vector<Npp8u> srcData(largeWidth * largeHeight * 3);
    for (int i = 0; i < largeWidth * largeHeight * 3; i++) {
        srcData[i] = (Npp8u)(i % 256);
    }
    
    // 复制数据到GPU
    for (int y = 0; y < largeHeight; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * largeWidth * 3,
                   largeWidth * 3 * sizeof(Npp8u),
                   cudaMemcpyHostToDevice);
    }
    
    // 执行转换
    NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, largeRoi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 验证部分结果（检查第一行）
    std::vector<Npp32f> resultData(largeWidth * 3);
    cudaMemcpy(resultData.data(), d_dst, largeWidth * 3 * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    
    for (int x = 0; x < largeWidth * 3; x++) {
        EXPECT_FLOAT_EQ(resultData[x], (Npp32f)srcData[x]);
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// 测试零尺寸ROI
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_ZeroROI) {
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C3(width, height, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C3(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    NppiSize zeroRoi = {0, 0};
    NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, zeroRoi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// 测试内存对齐
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_MemoryAlignment) {
    // 使用不对齐的尺寸测试
    const int oddWidth = 7;
    const int oddHeight = 5;
    NppiSize oddRoi = {oddWidth, oddHeight};
    
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C3(oddWidth, oddHeight, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C3(oddWidth, oddHeight, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 创建测试数据
    std::vector<Npp8u> srcData(oddWidth * oddHeight * 3, 42);
    
    // 复制数据到GPU (考虑步长)
    for (int y = 0; y < oddHeight; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * oddWidth * 3,
                   oddWidth * 3 * sizeof(Npp8u),
                   cudaMemcpyHostToDevice);
    }
    
    // 执行转换
    NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, oddRoi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 验证结果
    std::vector<Npp32f> resultData(oddWidth * oddHeight * 3);
    for (int y = 0; y < oddHeight; y++) {
        cudaMemcpy(resultData.data() + y * oddWidth * 3,
                   (char*)d_dst + y * dstStep,
                   oddWidth * 3 * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    for (int i = 0; i < oddWidth * oddHeight * 3; i++) {
        EXPECT_FLOAT_EQ(resultData[i], 42.0f);
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// 测试单像素ROI
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_SinglePixel) {
    NppiSize singlePixelRoi = {1, 1};
    std::vector<Npp8u> srcData = {123, 45, 210};
    std::vector<Npp32f> expectedData = {123.0f, 45.0f, 210.0f};
    
    int srcStep = 3 * sizeof(Npp8u);
    int dstStep = 3 * sizeof(Npp32f);
    
    Npp8u* d_src = nppsMalloc_8u(3);
    Npp32f* d_dst = nppsMalloc_32f(3);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 复制数据
    cudaMemcpy(d_src, srcData.data(), 3 * sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    // 执行转换
    NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, singlePixelRoi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 验证结果
    std::vector<Npp32f> resultData(3);
    cudaMemcpy(resultData.data(), d_dst, 3 * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < 3; i++) {
        EXPECT_FLOAT_EQ(resultData[i], expectedData[i]) 
            << "单像素转换失败 at channel " << i;
    }
    
    nppsFree(d_src);
    nppsFree(d_dst);
}

// 测试颜色渐变图像
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_ColorGradient) {
    const int gradWidth = 16;
    const int gradHeight = 12;
    NppiSize gradRoi = {gradWidth, gradHeight};
    
    // 创建RGB颜色渐变
    std::vector<Npp8u> srcData(gradWidth * gradHeight * 3);
    std::vector<Npp32f> expectedData(gradWidth * gradHeight * 3);
    
    for (int y = 0; y < gradHeight; y++) {
        for (int x = 0; x < gradWidth; x++) {
            int idx = (y * gradWidth + x) * 3;
            
            // R通道: 水平渐变 0->255
            srcData[idx + 0] = (Npp8u)((x * 255) / (gradWidth - 1));
            
            // G通道: 垂直渐变 0->255  
            srcData[idx + 1] = (Npp8u)((y * 255) / (gradHeight - 1));
            
            // B通道: 对角渐变
            int diag = (x + y) * 255 / (gradWidth + gradHeight - 2);
            srcData[idx + 2] = (Npp8u)(diag > 255 ? 255 : diag);
            
            // 期望的32位浮点结果
            expectedData[idx + 0] = (Npp32f)srcData[idx + 0];
            expectedData[idx + 1] = (Npp32f)srcData[idx + 1];
            expectedData[idx + 2] = (Npp32f)srcData[idx + 2];
        }
    }
    
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C3(gradWidth, gradHeight, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C3(gradWidth, gradHeight, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 复制数据到GPU
    for (int y = 0; y < gradHeight; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * gradWidth * 3,
                   gradWidth * 3 * sizeof(Npp8u),
                   cudaMemcpyHostToDevice);
    }
    
    // 执行转换
    NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, gradRoi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 验证结果
    std::vector<Npp32f> resultData(gradWidth * gradHeight * 3);
    for (int y = 0; y < gradHeight; y++) {
        cudaMemcpy(resultData.data() + y * gradWidth * 3,
                   (char*)d_dst + y * dstStep,
                   gradWidth * 3 * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    for (int i = 0; i < gradWidth * gradHeight * 3; i++) {
        EXPECT_FLOAT_EQ(resultData[i], expectedData[i]) 
            << "颜色渐变转换失败 at index " << i;
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// 测试棋盘图案
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_CheckerboardPattern) {
    const int boardWidth = 8;
    const int boardHeight = 8;
    NppiSize boardRoi = {boardWidth, boardHeight};
    
    std::vector<Npp8u> srcData(boardWidth * boardHeight * 3);
    std::vector<Npp32f> expectedData(boardWidth * boardHeight * 3);
    
    // 创建棋盘图案
    for (int y = 0; y < boardHeight; y++) {
        for (int x = 0; x < boardWidth; x++) {
            int idx = (y * boardWidth + x) * 3;
            bool isWhite = ((x + y) % 2) == 0;
            
            if (isWhite) {
                srcData[idx + 0] = 255; // R
                srcData[idx + 1] = 255; // G
                srcData[idx + 2] = 255; // B
            } else {
                srcData[idx + 0] = 0;   // R
                srcData[idx + 1] = 0;   // G
                srcData[idx + 2] = 0;   // B
            }
            
            expectedData[idx + 0] = (Npp32f)srcData[idx + 0];
            expectedData[idx + 1] = (Npp32f)srcData[idx + 1];
            expectedData[idx + 2] = (Npp32f)srcData[idx + 2];
        }
    }
    
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C3(boardWidth, boardHeight, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C3(boardWidth, boardHeight, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 复制数据到GPU
    for (int y = 0; y < boardHeight; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * boardWidth * 3,
                   boardWidth * 3 * sizeof(Npp8u),
                   cudaMemcpyHostToDevice);
    }
    
    // 执行转换
    NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, boardRoi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 验证结果
    std::vector<Npp32f> resultData(boardWidth * boardHeight * 3);
    for (int y = 0; y < boardHeight; y++) {
        cudaMemcpy(resultData.data() + y * boardWidth * 3,
                   (char*)d_dst + y * dstStep,
                   boardWidth * 3 * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    for (int i = 0; i < boardWidth * boardHeight * 3; i++) {
        EXPECT_FLOAT_EQ(resultData[i], expectedData[i]) 
            << "棋盘图案转换失败 at index " << i;
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// 测试随机数据
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_RandomData) {
    const int randWidth = 15;
    const int randHeight = 11;
    NppiSize randRoi = {randWidth, randHeight};
    
    std::vector<Npp8u> srcData(randWidth * randHeight * 3);
    std::vector<Npp32f> expectedData(randWidth * randHeight * 3);
    
    // 生成确定性"随机"数据（使用简单的线性同余生成器）
    unsigned int seed = 12345;
    for (int i = 0; i < randWidth * randHeight * 3; i++) {
        seed = seed * 1103515245 + 12345;
        srcData[i] = (Npp8u)(seed % 256);
        expectedData[i] = (Npp32f)srcData[i];
    }
    
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C3(randWidth, randHeight, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C3(randWidth, randHeight, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 复制数据到GPU
    for (int y = 0; y < randHeight; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * randWidth * 3,
                   randWidth * 3 * sizeof(Npp8u),
                   cudaMemcpyHostToDevice);
    }
    
    // 执行转换
    NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, randRoi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 验证结果
    std::vector<Npp32f> resultData(randWidth * randHeight * 3);
    for (int y = 0; y < randHeight; y++) {
        cudaMemcpy(resultData.data() + y * randWidth * 3,
                   (char*)d_dst + y * dstStep,
                   randWidth * 3 * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    for (int i = 0; i < randWidth * randHeight * 3; i++) {
        EXPECT_FLOAT_EQ(resultData[i], expectedData[i]) 
            << "随机数据转换失败 at index " << i;
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// 测试矩形ROI（非正方形）
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_RectangularROI) {
    const int rectWidth = 32;
    const int rectHeight = 8;
    NppiSize rectRoi = {rectWidth, rectHeight};
    
    std::vector<Npp8u> srcData(rectWidth * rectHeight * 3);
    std::vector<Npp32f> expectedData(rectWidth * rectHeight * 3);
    
    // 创建水平条纹图案
    for (int y = 0; y < rectHeight; y++) {
        for (int x = 0; x < rectWidth; x++) {
            int idx = (y * rectWidth + x) * 3;
            Npp8u value = (Npp8u)((y * 255) / (rectHeight - 1));
            
            srcData[idx + 0] = value;       // R
            srcData[idx + 1] = 255 - value; // G (反转)
            srcData[idx + 2] = 128;         // B (固定)
            
            expectedData[idx + 0] = (Npp32f)srcData[idx + 0];
            expectedData[idx + 1] = (Npp32f)srcData[idx + 1];
            expectedData[idx + 2] = (Npp32f)srcData[idx + 2];
        }
    }
    
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C3(rectWidth, rectHeight, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C3(rectWidth, rectHeight, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 复制数据到GPU
    for (int y = 0; y < rectHeight; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * rectWidth * 3,
                   rectWidth * 3 * sizeof(Npp8u),
                   cudaMemcpyHostToDevice);
    }
    
    // 执行转换
    NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, rectRoi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 验证结果
    std::vector<Npp32f> resultData(rectWidth * rectHeight * 3);
    for (int y = 0; y < rectHeight; y++) {
        cudaMemcpy(resultData.data() + y * rectWidth * 3,
                   (char*)d_dst + y * dstStep,
                   rectWidth * 3 * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    for (int i = 0; i < rectWidth * rectHeight * 3; i++) {
        EXPECT_FLOAT_EQ(resultData[i], expectedData[i]) 
            << "矩形ROI转换失败 at index " << i;
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// 测试性能（大尺寸图像）
TEST_F(ConvertExtendedFunctionalTest, Convert_8u32f_C3R_PerformanceTest) {
    const int perfWidth = 1024;
    const int perfHeight = 768;
    NppiSize perfRoi = {perfWidth, perfHeight};
    
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C3(perfWidth, perfHeight, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C3(perfWidth, perfHeight, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // 创建简单的测试数据（不验证结果，主要测试性能）
    std::vector<Npp8u> srcData(perfWidth * perfHeight * 3, 128);
    
    // 复制数据到GPU
    for (int y = 0; y < perfHeight; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * perfWidth * 3,
                   perfWidth * 3 * sizeof(Npp8u),
                   cudaMemcpyHostToDevice);
    }
    
    // 执行转换（多次测试稳定性）
    for (int i = 0; i < 5; i++) {
        NppStatus status = nppiConvert_8u32f_C3R(d_src, srcStep, d_dst, dstStep, perfRoi);
        EXPECT_EQ(status, NPP_SUCCESS) << "性能测试第 " << i << " 次失败";
    }
    
    // 验证第一个和最后一个像素
    std::vector<Npp32f> firstPixel(3), lastPixel(3);
    cudaMemcpy(firstPixel.data(), d_dst, 3 * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    
    int lastPixelOffset = ((perfHeight - 1) * dstStep) + ((perfWidth - 1) * 3 * sizeof(Npp32f));
    cudaMemcpy(lastPixel.data(), (char*)d_dst + lastPixelOffset, 3 * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    
    for (int c = 0; c < 3; c++) {
        EXPECT_FLOAT_EQ(firstPixel[c], 128.0f) << "第一个像素通道 " << c << " 错误";
        EXPECT_FLOAT_EQ(lastPixel[c], 128.0f) << "最后一个像素通道 " << c << " 错误";
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}