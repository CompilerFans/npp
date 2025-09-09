#include <gtest/gtest.h>
#include "npp.h"
#include <cuda_runtime.h>
#include <vector>

class NPPIHistogramTest : public ::testing::Test {
protected:
    void SetUp() override {
        width = 16;
        height = 12;
        roi.width = width;
        roi.height = height;
    }

    int width, height;
    NppiSize roi;
};

// 测试nppiEvenLevelsHost函数
TEST_F(NPPIHistogramTest, EvenLevelsHost_32s) {
    int nLevels = 5;
    std::vector<Npp32s> pLevels(nLevels);
    Npp32s nLowerBound = 0;
    Npp32s nUpperBound = 255;
    
    NppStatus status = nppiEvenLevelsHost_32s(pLevels.data(), nLevels, nLowerBound, nUpperBound);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 验证生成的等级
    EXPECT_EQ(pLevels[0], 0);
    EXPECT_EQ(pLevels[nLevels-1], 255);
    
    // 验证等级是递增的
    for (int i = 1; i < nLevels; i++) {
        EXPECT_GT(pLevels[i], pLevels[i-1]);
    }
}

// 测试nppiEvenLevelsHost错误处理
TEST_F(NPPIHistogramTest, EvenLevelsHost_ErrorHandling) {
    std::vector<Npp32s> pLevels(5);
    
    // 测试空指针
    NppStatus status = nppiEvenLevelsHost_32s(nullptr, 5, 0, 255);
    EXPECT_EQ(status, NPP_NULL_POINTER_ERROR);
    
    // 测试无效等级数
    status = nppiEvenLevelsHost_32s(pLevels.data(), 1, 0, 255);
    EXPECT_EQ(status, NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR);
    
    // 测试无效边界
    status = nppiEvenLevelsHost_32s(pLevels.data(), 5, 255, 0);
    EXPECT_EQ(status, NPP_BAD_ARGUMENT_ERROR);
}

// 测试直方图缓冲区大小获取
TEST_F(NPPIHistogramTest, HistogramEvenGetBufferSize_8u_C1R) {
    int nLevels = 256;
    int bufferSize = 0;
    
    NppStatus status = nppiHistogramEvenGetBufferSize_8u_C1R(roi, nLevels, &bufferSize);
    EXPECT_EQ(status, NPP_SUCCESS);
    EXPECT_GT(bufferSize, 0);
}

// 测试8位直方图计算
TEST_F(NPPIHistogramTest, HistogramEven_8u_C1R_Basic) {
    size_t dataSize = width * height;
    std::vector<Npp8u> srcData(dataSize);
    
    // 生成测试数据：一半0，一半255
    for (size_t i = 0; i < dataSize/2; i++) {
        srcData[i] = 0;
        srcData[i + dataSize/2] = 255;
    }
    
    int nLevels = 257;  // 256 bins + 1 level
    Npp32s nLowerLevel = 0;
    Npp32s nUpperLevel = 256;
    std::vector<Npp32s> pHist(nLevels - 1);
    
    // 获取缓冲区大小
    int bufferSize = 0;
    NppStatus status = nppiHistogramEvenGetBufferSize_8u_C1R(roi, nLevels, &bufferSize);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 分配GPU内存
    Npp8u *d_src, *d_buffer;
    Npp32s *d_hist;
    int srcStep = width * sizeof(Npp8u);
    
    cudaMalloc(&d_src, dataSize * sizeof(Npp8u));
    cudaMalloc(&d_buffer, bufferSize);
    cudaMalloc(&d_hist, (nLevels - 1) * sizeof(Npp32s));
    
    cudaMemcpy(d_src, srcData.data(), dataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    // 调用NPP函数
    status = nppiHistogramEven_8u_C1R(d_src, srcStep, roi, d_hist, nLevels,
                                     nLowerLevel, nUpperLevel, d_buffer);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 拷贝结果回主机
    cudaMemcpy(pHist.data(), d_hist, (nLevels - 1) * sizeof(Npp32s), cudaMemcpyDeviceToHost);
    
    // 验证结果：bin 0和bin 255应该有计数
    EXPECT_GT(pHist[0], 0);    // bin 0应该有计数
    EXPECT_GT(pHist[255], 0);  // bin 255应该有计数
    
    // 计算总像素数
    int totalCount = 0;
    for (int i = 0; i < nLevels - 1; i++) {
        totalCount += pHist[i];
    }
    EXPECT_EQ(totalCount, (int)dataSize);
    
    cudaFree(d_src);
    cudaFree(d_buffer);
    cudaFree(d_hist);
}

// 测试直方图错误处理
TEST_F(NPPIHistogramTest, HistogramEven_ErrorHandling) {
    int nLevels = 256;
    Npp32s nLowerLevel = 0;
    Npp32s nUpperLevel = 255;
    
    // 测试空指针
    NppStatus status = nppiHistogramEven_8u_C1R(nullptr, 32, roi, nullptr, nLevels,
                                               nLowerLevel, nUpperLevel, nullptr);
    EXPECT_EQ(status, NPP_NULL_POINTER_ERROR);
    
    // 测试无效尺寸
    NppiSize invalidRoi = {0, 0};
    status = nppiHistogramEven_8u_C1R(nullptr, 32, invalidRoi, nullptr, nLevels,
                                     nLowerLevel, nUpperLevel, nullptr);
    EXPECT_EQ(status, NPP_SIZE_ERROR);
    
    // 测试无效等级数
    status = nppiHistogramEven_8u_C1R(nullptr, 32, roi, nullptr, 1,
                                     nLowerLevel, nUpperLevel, nullptr);
    EXPECT_EQ(status, NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR);
    
    // 测试无效边界
    status = nppiHistogramEven_8u_C1R(nullptr, 32, roi, nullptr, nLevels,
                                     255, 0, nullptr);
    EXPECT_EQ(status, NPP_BAD_ARGUMENT_ERROR);
}