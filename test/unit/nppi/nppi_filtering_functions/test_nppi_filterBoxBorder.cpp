#include <gtest/gtest.h>
#include "npp.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cmath>

class NPPIFilterBoxBorderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 设置随机种子
        srand(time(nullptr));
        
        // 初始化测试图像尺寸
        srcWidth = 16;
        srcHeight = 12;
        oSrcSizeROI.width = srcWidth;
        oSrcSizeROI.height = srcHeight;
        
        // 目标尺寸（通常与源相同或稍小）
        dstWidth = 14;
        dstHeight = 10;
        oDstSizeROI.width = dstWidth;
        oDstSizeROI.height = dstHeight;
        
        // 滤波核尺寸
        oMaskSize.width = 3;
        oMaskSize.height = 3;
        
        // 锚点（核心中心）
        oAnchor.x = 1;
        oAnchor.y = 1;
        
        // 源偏移
        oSrcOffset.x = 1;
        oSrcOffset.y = 1;
    }

    void TearDown() override {
        // 清理会在各测试函数中处理
    }

    int srcWidth, srcHeight, dstWidth, dstHeight;
    NppiSize oSrcSizeROI, oDstSizeROI, oMaskSize;
    NppiPoint oAnchor, oSrcOffset;
    
    // 辅助函数：生成随机数据
    template<typename T>
    void generateRandomData(std::vector<T>& data, size_t size, T minVal, T maxVal) {
        data.resize(size);
        for (size_t i = 0; i < size; i++) {
            data[i] = static_cast<T>(rand() % (maxVal - minVal + 1) + minVal);
        }
    }
    
    // 辅助函数：CPU参考实现（简单的box filter）
    template<typename T>
    void referenceBoxFilter(const std::vector<T>& srcData, std::vector<T>& refResult) {
        refResult.resize(dstWidth * dstHeight);
        
        for (int dy = 0; dy < dstHeight; dy++) {
            for (int dx = 0; dx < dstWidth; dx++) {
                int sum = 0;
                int count = 0;
                
                // 应用滤波核
                for (int ky = 0; ky < oMaskSize.height; ky++) {
                    for (int kx = 0; kx < oMaskSize.width; kx++) {
                        int src_x = dx + oSrcOffset.x + kx - oAnchor.x;
                        int src_y = dy + oSrcOffset.y + ky - oAnchor.y;
                        
                        // 边界处理 - 复制边界
                        if (src_x < 0) src_x = 0;
                        if (src_x >= srcWidth) src_x = srcWidth - 1;
                        if (src_y < 0) src_y = 0;
                        if (src_y >= srcHeight) src_y = srcHeight - 1;
                        
                        sum += srcData[src_y * srcWidth + src_x];
                        count++;
                    }
                }
                
                refResult[dy * dstWidth + dx] = static_cast<T>(sum / count);
            }
        }
    }
    
    // 辅助函数：验证滤波结果（允许一定误差）
    template<typename T>
    void verifyFilterResult(const std::vector<T>& result, const std::vector<T>& reference, 
                           T tolerance = T(1)) {
        ASSERT_EQ(result.size(), reference.size());
        for (size_t i = 0; i < result.size(); i++) {
            T diff = static_cast<T>(abs(static_cast<int>(result[i]) - static_cast<int>(reference[i])));
            EXPECT_LE(diff, tolerance) << "Mismatch at index " << i 
                                      << " result=" << (int)result[i] 
                                      << " reference=" << (int)reference[i];
        }
    }
};

// 测试8位无符号单通道盒式滤波
TEST_F(NPPIFilterBoxBorderTest, FilterBoxBorder_8u_C1R) {
    size_t srcDataSize = srcWidth * srcHeight;
    size_t dstDataSize = dstWidth * dstHeight;
    std::vector<Npp8u> srcData(srcDataSize), dstData(dstDataSize), refResult;
    
    // 生成源数据
    generateRandomData(srcData, srcDataSize, Npp8u(0), Npp8u(255));
    
    // 计算CPU参考结果
    referenceBoxFilter(srcData, refResult);
    
    // 分配GPU内存
    Npp8u *d_src, *d_dst;
    int nSrcStep = srcWidth * sizeof(Npp8u);
    int nDstStep = dstWidth * sizeof(Npp8u);
    
    cudaMalloc(&d_src, srcDataSize * sizeof(Npp8u));
    cudaMalloc(&d_dst, dstDataSize * sizeof(Npp8u));
    
    // 拷贝数据到GPU
    cudaMemcpy(d_src, srcData.data(), srcDataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    // 调用NPP函数
    NppStatus status = nppiFilterBoxBorder_8u_C1R(d_src, nSrcStep, oSrcSizeROI, oSrcOffset,
                                                  d_dst, nDstStep, oDstSizeROI,
                                                  oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 拷贝结果回主机
    cudaMemcpy(dstData.data(), d_dst, dstDataSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);
    
    // 验证结果（允许±2的误差）
    verifyFilterResult(dstData, refResult, Npp8u(2));
    
    // 清理GPU内存
    cudaFree(d_src);
    cudaFree(d_dst);
}

// 测试8位无符号三通道盒式滤波
TEST_F(NPPIFilterBoxBorderTest, FilterBoxBorder_8u_C3R) {
    size_t srcDataSize = srcWidth * srcHeight * 3;
    size_t dstDataSize = dstWidth * dstHeight * 3;
    std::vector<Npp8u> srcData(srcDataSize), dstData(dstDataSize);
    
    // 生成源数据
    generateRandomData(srcData, srcDataSize, Npp8u(0), Npp8u(255));
    
    // 分配GPU内存
    Npp8u *d_src, *d_dst;
    int nSrcStep = srcWidth * 3 * sizeof(Npp8u);
    int nDstStep = dstWidth * 3 * sizeof(Npp8u);
    
    cudaMalloc(&d_src, srcDataSize * sizeof(Npp8u));
    cudaMalloc(&d_dst, dstDataSize * sizeof(Npp8u));
    
    // 拷贝数据到GPU
    cudaMemcpy(d_src, srcData.data(), srcDataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    // 调用NPP函数
    NppStatus status = nppiFilterBoxBorder_8u_C3R(d_src, nSrcStep, oSrcSizeROI, oSrcOffset,
                                                  d_dst, nDstStep, oDstSizeROI,
                                                  oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 拷贝结果回主机
    cudaMemcpy(dstData.data(), d_dst, dstDataSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);
    
    // 验证结果存在性（基本检查）
    bool hasNonZero = false;
    for (size_t i = 0; i < dstDataSize; i++) {
        if (dstData[i] != 0) {
            hasNonZero = true;
            break;
        }
    }
    EXPECT_TRUE(hasNonZero) << "Filter result seems to be all zeros";
    
    // 清理GPU内存
    cudaFree(d_src);
    cudaFree(d_dst);
}

// 测试16位有符号单通道盒式滤波
TEST_F(NPPIFilterBoxBorderTest, FilterBoxBorder_16s_C1R) {
    size_t srcDataSize = srcWidth * srcHeight;
    size_t dstDataSize = dstWidth * dstHeight;
    std::vector<Npp16s> srcData(srcDataSize), dstData(dstDataSize), refResult;
    
    // 生成源数据（包括负数）
    generateRandomData(srcData, srcDataSize, Npp16s(-1000), Npp16s(1000));
    
    // 计算CPU参考结果
    referenceBoxFilter(srcData, refResult);
    
    // 分配GPU内存
    Npp16s *d_src, *d_dst;
    int nSrcStep = srcWidth * sizeof(Npp16s);
    int nDstStep = dstWidth * sizeof(Npp16s);
    
    cudaMalloc(&d_src, srcDataSize * sizeof(Npp16s));
    cudaMalloc(&d_dst, dstDataSize * sizeof(Npp16s));
    
    // 拷贝数据到GPU
    cudaMemcpy(d_src, srcData.data(), srcDataSize * sizeof(Npp16s), cudaMemcpyHostToDevice);
    
    // 调用NPP函数
    NppStatus status = nppiFilterBoxBorder_16s_C1R(d_src, nSrcStep, oSrcSizeROI, oSrcOffset,
                                                   d_dst, nDstStep, oDstSizeROI,
                                                   oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 拷贝结果回主机
    cudaMemcpy(dstData.data(), d_dst, dstDataSize * sizeof(Npp16s), cudaMemcpyDeviceToHost);
    
    // 验证结果（允许±5的误差）
    verifyFilterResult(dstData, refResult, Npp16s(5));
    
    // 清理GPU内存
    cudaFree(d_src);
    cudaFree(d_dst);
}

// 测试32位浮点单通道盒式滤波
TEST_F(NPPIFilterBoxBorderTest, FilterBoxBorder_32f_C1R) {
    size_t srcDataSize = srcWidth * srcHeight;
    size_t dstDataSize = dstWidth * dstHeight;
    std::vector<Npp32f> srcData(srcDataSize), dstData(dstDataSize);
    
    // 生成浮点源数据
    for (size_t i = 0; i < srcDataSize; i++) {
        srcData[i] = static_cast<Npp32f>((rand() / (float)RAND_MAX) * 2.0f - 1.0f);  // [-1, 1]
    }
    
    // 分配GPU内存
    Npp32f *d_src, *d_dst;
    int nSrcStep = srcWidth * sizeof(Npp32f);
    int nDstStep = dstWidth * sizeof(Npp32f);
    
    cudaMalloc(&d_src, srcDataSize * sizeof(Npp32f));
    cudaMalloc(&d_dst, dstDataSize * sizeof(Npp32f));
    
    // 拷贝数据到GPU
    cudaMemcpy(d_src, srcData.data(), srcDataSize * sizeof(Npp32f), cudaMemcpyHostToDevice);
    
    // 调用NPP函数
    NppStatus status = nppiFilterBoxBorder_32f_C1R(d_src, nSrcStep, oSrcSizeROI, oSrcOffset,
                                                   d_dst, nDstStep, oDstSizeROI,
                                                   oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 拷贝结果回主机
    cudaMemcpy(dstData.data(), d_dst, dstDataSize * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    
    // 验证结果存在性和合理性
    float sum = 0.0f;
    float minVal = dstData[0], maxVal = dstData[0];
    for (size_t i = 0; i < dstDataSize; i++) {
        sum += dstData[i];
        minVal = std::min(minVal, dstData[i]);
        maxVal = std::max(maxVal, dstData[i]);
    }
    
    // 滤波后的值应该在合理范围内
    EXPECT_GE(minVal, -2.0f);
    EXPECT_LE(maxVal, 2.0f);
    
    // 清理GPU内存
    cudaFree(d_src);
    cudaFree(d_dst);
}

// 测试不同滤波核尺寸
TEST_F(NPPIFilterBoxBorderTest, FilterBoxBorder_DifferentKernelSizes) {
    size_t srcDataSize = srcWidth * srcHeight;
    size_t dstDataSize = dstWidth * dstHeight;
    std::vector<Npp8u> srcData(srcDataSize), dstData(dstDataSize);
    
    // 生成固定的源数据（便于测试）
    for (size_t i = 0; i < srcDataSize; i++) {
        srcData[i] = static_cast<Npp8u>((i % 10) * 25);  // 0, 25, 50, ..., 225, 0, 25, ...
    }
    
    // 分配GPU内存
    Npp8u *d_src, *d_dst;
    int nSrcStep = srcWidth * sizeof(Npp8u);
    int nDstStep = dstWidth * sizeof(Npp8u);
    
    cudaMalloc(&d_src, srcDataSize * sizeof(Npp8u));
    cudaMalloc(&d_dst, dstDataSize * sizeof(Npp8u));
    
    // 拷贝数据到GPU
    cudaMemcpy(d_src, srcData.data(), srcDataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    // 测试5x5滤波核
    oMaskSize.width = 5;
    oMaskSize.height = 5;
    oAnchor.x = 2;
    oAnchor.y = 2;
    
    NppStatus status = nppiFilterBoxBorder_8u_C1R(d_src, nSrcStep, oSrcSizeROI, oSrcOffset,
                                                  d_dst, nDstStep, oDstSizeROI,
                                                  oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 拷贝结果回主机
    cudaMemcpy(dstData.data(), d_dst, dstDataSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);
    
    // 5x5滤波器应该产生更平滑的结果
    bool hasValidResult = false;
    for (size_t i = 0; i < dstDataSize; i++) {
        if (dstData[i] > 0) {
            hasValidResult = true;
            break;
        }
    }
    EXPECT_TRUE(hasValidResult);
    
    // 清理GPU内存
    cudaFree(d_src);
    cudaFree(d_dst);
}

// 测试不同边界处理模式
TEST_F(NPPIFilterBoxBorderTest, FilterBoxBorder_DifferentBorderTypes) {
    size_t srcDataSize = srcWidth * srcHeight;
    size_t dstDataSize = dstWidth * dstHeight;
    std::vector<Npp8u> srcData(srcDataSize), dstDataReplicate(dstDataSize), dstDataConstant(dstDataSize);
    
    // 生成源数据
    generateRandomData(srcData, srcDataSize, Npp8u(50), Npp8u(200));
    
    // 分配GPU内存
    Npp8u *d_src, *d_dst;
    int nSrcStep = srcWidth * sizeof(Npp8u);
    int nDstStep = dstWidth * sizeof(Npp8u);
    
    cudaMalloc(&d_src, srcDataSize * sizeof(Npp8u));
    cudaMalloc(&d_dst, dstDataSize * sizeof(Npp8u));
    
    // 拷贝数据到GPU
    cudaMemcpy(d_src, srcData.data(), srcDataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    // 测试REPLICATE边界模式
    NppStatus status = nppiFilterBoxBorder_8u_C1R(d_src, nSrcStep, oSrcSizeROI, oSrcOffset,
                                                  d_dst, nDstStep, oDstSizeROI,
                                                  oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
    EXPECT_EQ(status, NPP_SUCCESS);
    cudaMemcpy(dstDataReplicate.data(), d_dst, dstDataSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);
    
    // 测试CONSTANT边界模式
    status = nppiFilterBoxBorder_8u_C1R(d_src, nSrcStep, oSrcSizeROI, oSrcOffset,
                                        d_dst, nDstStep, oDstSizeROI,
                                        oMaskSize, oAnchor, NPP_BORDER_CONSTANT);
    EXPECT_EQ(status, NPP_SUCCESS);
    cudaMemcpy(dstDataConstant.data(), d_dst, dstDataSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);
    
    // 两种边界模式的结果应该不同
    bool isDifferent = false;
    for (size_t i = 0; i < dstDataSize; i++) {
        if (dstDataReplicate[i] != dstDataConstant[i]) {
            isDifferent = true;
            break;
        }
    }
    EXPECT_TRUE(isDifferent) << "Different border types should produce different results";
    
    // 清理GPU内存
    cudaFree(d_src);
    cudaFree(d_dst);
}

// 测试带上下文的版本
TEST_F(NPPIFilterBoxBorderTest, FilterBoxBorder_8u_C1R_Ctx) {
    size_t srcDataSize = srcWidth * srcHeight;
    size_t dstDataSize = dstWidth * dstHeight;
    std::vector<Npp8u> srcData(srcDataSize), dstData(dstDataSize);
    
    // 生成源数据
    generateRandomData(srcData, srcDataSize, Npp8u(0), Npp8u(255));
    
    // 分配GPU内存
    Npp8u *d_src, *d_dst;
    int nSrcStep = srcWidth * sizeof(Npp8u);
    int nDstStep = dstWidth * sizeof(Npp8u);
    
    cudaMalloc(&d_src, srcDataSize * sizeof(Npp8u));
    cudaMalloc(&d_dst, dstDataSize * sizeof(Npp8u));
    
    // 拷贝数据到GPU
    cudaMemcpy(d_src, srcData.data(), srcDataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    // 创建流上下文
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;  // 使用默认流
    
    // 调用NPP函数
    NppStatus status = nppiFilterBoxBorder_8u_C1R_Ctx(d_src, nSrcStep, oSrcSizeROI, oSrcOffset,
                                                      d_dst, nDstStep, oDstSizeROI,
                                                      oMaskSize, oAnchor, NPP_BORDER_REPLICATE, nppStreamCtx);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 拷贝结果回主机
    cudaMemcpy(dstData.data(), d_dst, dstDataSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);
    
    // 验证结果存在性
    bool hasNonZero = false;
    for (size_t i = 0; i < dstDataSize; i++) {
        if (dstData[i] != 0) {
            hasNonZero = true;
            break;
        }
    }
    EXPECT_TRUE(hasNonZero);
    
    // 清理GPU内存
    cudaFree(d_src);
    cudaFree(d_dst);
}

// 测试错误处理
TEST_F(NPPIFilterBoxBorderTest, ErrorHandling) {
    // 测试空指针
    NppStatus status = nppiFilterBoxBorder_8u_C1R(nullptr, 32, oSrcSizeROI, oSrcOffset,
                                                  nullptr, 32, oDstSizeROI,
                                                  oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
    EXPECT_EQ(status, NPP_NULL_POINTER_ERROR);
    
    // 测试无效源尺寸
    NppiSize invalidSrcRoi = {0, 0};
    status = nppiFilterBoxBorder_8u_C1R(nullptr, 32, invalidSrcRoi, oSrcOffset,
                                        nullptr, 32, oDstSizeROI,
                                        oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
    EXPECT_EQ(status, NPP_SIZE_ERROR);
    
    // 测试无效目标尺寸
    NppiSize invalidDstRoi = {0, 0};
    status = nppiFilterBoxBorder_8u_C1R(nullptr, 32, oSrcSizeROI, oSrcOffset,
                                        nullptr, 32, invalidDstRoi,
                                        oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
    EXPECT_EQ(status, NPP_SIZE_ERROR);
    
    // 测试无效步长
    status = nppiFilterBoxBorder_8u_C1R(nullptr, 0, oSrcSizeROI, oSrcOffset,
                                        nullptr, 0, oDstSizeROI,
                                        oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
    EXPECT_EQ(status, NPP_STEP_ERROR);
    
    // 测试无效滤波核尺寸
    NppiSize invalidMaskSize = {0, 0};
    status = nppiFilterBoxBorder_8u_C1R(nullptr, 32, oSrcSizeROI, oSrcOffset,
                                        nullptr, 32, oDstSizeROI,
                                        invalidMaskSize, oAnchor, NPP_BORDER_REPLICATE);
    EXPECT_EQ(status, NPP_MASK_SIZE_ERROR);
    
    // 测试无效锚点
    NppiPoint invalidAnchor = {-1, -1};
    status = nppiFilterBoxBorder_8u_C1R(nullptr, 32, oSrcSizeROI, oSrcOffset,
                                        nullptr, 32, oDstSizeROI,
                                        oMaskSize, invalidAnchor, NPP_BORDER_REPLICATE);
    EXPECT_EQ(status, NPP_ANCHOR_ERROR);
    
    // 测试无效边界类型
    status = nppiFilterBoxBorder_8u_C1R(nullptr, 32, oSrcSizeROI, oSrcOffset,
                                        nullptr, 32, oDstSizeROI,
                                        oMaskSize, oAnchor, static_cast<NppiBorderType>(-1));
    EXPECT_EQ(status, NPP_BAD_ARGUMENT_ERROR);
}