#include <gtest/gtest.h>
#include "npp.h"
#include <cuda_runtime.h>
#include <vector>

class NPPIWatershedTest : public ::testing::Test {
protected:
    void SetUp() override {
        width = 16;
        height = 12;
        oSizeROI.width = width;
        oSizeROI.height = height;
    }

    int width, height;
    NppiSize oSizeROI;
};

// 测试缓冲区大小获取
TEST_F(NPPIWatershedTest, SegmentWatershedGetBufferSize_Basic) {
    size_t bufferSize = 0;
    
    NppStatus status = nppiSegmentWatershedGetBufferSize_8u_C1R(oSizeROI, &bufferSize);
    EXPECT_EQ(status, NPP_SUCCESS);
    EXPECT_GT(bufferSize, 0);
}

// 测试缓冲区大小获取错误处理
TEST_F(NPPIWatershedTest, SegmentWatershedGetBufferSize_ErrorHandling) {
    // 测试空指针
    NppStatus status = nppiSegmentWatershedGetBufferSize_8u_C1R(oSizeROI, nullptr);
    EXPECT_EQ(status, NPP_NULL_POINTER_ERROR);
    
    // 测试无效尺寸
    NppiSize invalidROI = {0, 0};
    size_t bufferSize = 0;
    status = nppiSegmentWatershedGetBufferSize_8u_C1R(invalidROI, &bufferSize);
    EXPECT_EQ(status, NPP_SIZE_ERROR);
}

// 测试Watershed分割基础功能
TEST_F(NPPIWatershedTest, SegmentWatershed_8u_C1IR_Basic) {
    size_t dataSize = width * height;
    std::vector<Npp8u> srcData(dataSize);
    std::vector<Npp32u> markerData(dataSize);
    
    // 生成测试图像：简单的梯度
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            srcData[y * width + x] = (Npp8u)((x + y) * 255 / (width + height - 2));
        }
    }
    
    // 初始化标记：左上角和右下角作为种子点
    for (size_t i = 0; i < dataSize; i++) {
        markerData[i] = 0;  // 背景
    }
    markerData[1 * width + 1] = 1;                    // 左上角种子点
    markerData[(height-2) * width + (width-2)] = 2;  // 右下角种子点
    
    // 获取缓冲区大小
    size_t bufferSize = 0;
    NppStatus status = nppiSegmentWatershedGetBufferSize_8u_C1R(oSizeROI, &bufferSize);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 分配GPU内存
    Npp8u *d_src, *d_buffer;
    Npp32u *d_markers;
    int srcStep = width * sizeof(Npp8u);
    int markersStep = width * sizeof(Npp32u);
    
    cudaMalloc(&d_src, dataSize * sizeof(Npp8u));
    cudaMalloc(&d_markers, dataSize * sizeof(Npp32u));
    cudaMalloc(&d_buffer, bufferSize);
    
    cudaMemcpy(d_src, srcData.data(), dataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);
    cudaMemcpy(d_markers, markerData.data(), dataSize * sizeof(Npp32u), cudaMemcpyHostToDevice);
    
    // 调用Watershed分割
    status = nppiSegmentWatershed_8u_C1IR(d_src, srcStep, d_markers, markersStep,
                                         oSizeROI, nppiNormL1, 2, d_buffer);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 拷贝结果回主机
    std::vector<Npp32u> resultData(dataSize);
    cudaMemcpy(resultData.data(), d_markers, dataSize * sizeof(Npp32u), cudaMemcpyDeviceToHost);
    
    // 基本验证：检查是否有标记区域
    int label1Count = 0, label2Count = 0, watershedCount = 0;
    for (size_t i = 0; i < dataSize; i++) {
        if (resultData[i] == 1) label1Count++;
        else if (resultData[i] == 2) label2Count++;
        else if (resultData[i] == -1) watershedCount++;
    }
    
    EXPECT_GT(label1Count, 0);     // 应该有标签1的区域
    EXPECT_GT(label2Count, 0);     // 应该有标签2的区域
    // 注意：分水岭线可能为0，这取决于具体实现
    
    cudaFree(d_src);
    cudaFree(d_markers);
    cudaFree(d_buffer);
}

// 测试错误处理
TEST_F(NPPIWatershedTest, SegmentWatershed_ErrorHandling) {
    // 测试空指针
    NppStatus status = nppiSegmentWatershed_8u_C1IR(nullptr, 32, nullptr, 32,
                                                   oSizeROI, 2, nullptr);
    EXPECT_EQ(status, NPP_NULL_POINTER_ERROR);
    
    // 测试无效尺寸
    NppiSize invalidROI = {0, 0};
    status = nppiSegmentWatershed_8u_C1IR(nullptr, 32, nullptr, 32,
                                         invalidROI, 2, nullptr);
    EXPECT_EQ(status, NPP_SIZE_ERROR);
    
    // 测试无效步长
    status = nppiSegmentWatershed_8u_C1IR(nullptr, -1, nullptr, -1,
                                         oSizeROI, 2, nullptr);
    EXPECT_EQ(status, NPP_STEP_ERROR);
    
    // 测试无效范数
    status = nppiSegmentWatershed_8u_C1IR(nullptr, 32, nullptr, 32,
                                         oSizeROI, 3, nullptr);  // 只支持L1(1)或L2(2)
    EXPECT_EQ(status, NPP_BAD_ARGUMENT_ERROR);
}