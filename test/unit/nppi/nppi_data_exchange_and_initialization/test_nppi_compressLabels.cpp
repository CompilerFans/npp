#include <gtest/gtest.h>
#include "npp.h"
#include <cuda_runtime.h>
#include <vector>

class NPPICompressLabelsTest : public ::testing::Test {
protected:
    void SetUp() override {
        width = 16;
        height = 12;
        oMarkerLabelsROI.width = width;
        oMarkerLabelsROI.height = height;
    }

    int width, height;
    NppiSize oMarkerLabelsROI;
};

// 测试缓冲区大小获取
TEST_F(NPPICompressLabelsTest, CompressMarkerLabelsGetBufferSize_Basic) {
    int nMarkerLabels = 1000;
    int bufferSize = 0;
    
    NppStatus status = nppiCompressMarkerLabelsGetBufferSize_32u_C1R(nMarkerLabels, &bufferSize);
    EXPECT_EQ(status, NPP_SUCCESS);
    EXPECT_GT(bufferSize, 0);
}

// 测试缓冲区大小获取错误处理
TEST_F(NPPICompressLabelsTest, CompressMarkerLabelsGetBufferSize_ErrorHandling) {
    // 测试空指针
    NppStatus status = nppiCompressMarkerLabelsGetBufferSize_32u_C1R(100, nullptr);
    EXPECT_EQ(status, NPP_NULL_POINTER_ERROR);
    
    // 测试无效标签数
    int bufferSize = 0;
    status = nppiCompressMarkerLabelsGetBufferSize_32u_C1R(-1, &bufferSize);
    EXPECT_EQ(status, NPP_BAD_ARGUMENT_ERROR);
}

// 测试标签压缩基础功能
TEST_F(NPPICompressLabelsTest, CompressMarkerLabelsUF_32u_C1IR_Basic) {
    size_t dataSize = width * height;
    std::vector<Npp32u> markerData(dataSize);
    
    // 生成测试数据：创建一些连通组件
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (x < width/3) {
                markerData[y * width + x] = 100;  // 第一个区域
            } else if (x < 2*width/3) {
                markerData[y * width + x] = 0;    // 背景
            } else {
                markerData[y * width + x] = 200;  // 第二个区域
            }
        }
    }
    
    // 获取缓冲区大小
    int nMarkerLabels = 300;
    int bufferSize = 0;
    NppStatus status = nppiCompressMarkerLabelsGetBufferSize_32u_C1R(nMarkerLabels, &bufferSize);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 分配GPU内存
    Npp32u *d_markers;
    Npp8u *d_buffer;
    int *d_newCount;
    int markersStep = width * sizeof(Npp32u);
    
    cudaMalloc(&d_markers, dataSize * sizeof(Npp32u));
    cudaMalloc(&d_buffer, bufferSize);
    cudaMalloc(&d_newCount, sizeof(int));
    
    cudaMemcpy(d_markers, markerData.data(), dataSize * sizeof(Npp32u), cudaMemcpyHostToDevice);
    
    // 调用标签压缩
    int newMarkerLabelsNumber = 0;
    status = nppiCompressMarkerLabelsUF_32u_C1IR(d_markers, markersStep, oMarkerLabelsROI,
                                                 1, &newMarkerLabelsNumber, d_buffer);
    EXPECT_EQ(status, NPP_SUCCESS);
    EXPECT_GT(newMarkerLabelsNumber, 0);  // 应该有压缩后的标签
    
    // 拷贝结果回主机
    std::vector<Npp32u> resultData(dataSize);
    cudaMemcpy(resultData.data(), d_markers, dataSize * sizeof(Npp32u), cudaMemcpyDeviceToHost);
    
    // 验证结果：检查标签是否连续
    std::set<Npp32u> uniqueLabels;
    for (size_t i = 0; i < dataSize; i++) {
        if (resultData[i] > 0) {
            uniqueLabels.insert(resultData[i]);
        }
    }
    
    EXPECT_EQ((int)uniqueLabels.size(), newMarkerLabelsNumber);
    
    cudaFree(d_markers);
    cudaFree(d_buffer);
    cudaFree(d_newCount);
}

// 测试错误处理
TEST_F(NPPICompressLabelsTest, CompressMarkerLabelsUF_ErrorHandling) {
    // 测试空指针
    int newCount = 0;
    NppStatus status = nppiCompressMarkerLabelsUF_32u_C1IR(nullptr, 32, oMarkerLabelsROI,
                                                          1, &newCount, nullptr);
    EXPECT_EQ(status, NPP_NULL_POINTER_ERROR);
    
    // 测试无效尺寸
    NppiSize invalidROI = {0, 0};
    status = nppiCompressMarkerLabelsUF_32u_C1IR(nullptr, 32, invalidROI,
                                                1, &newCount, nullptr);
    EXPECT_EQ(status, NPP_SIZE_ERROR);
    
    // 测试无效步长
    status = nppiCompressMarkerLabelsUF_32u_C1IR(nullptr, -1, oMarkerLabelsROI,
                                                1, &newCount, nullptr);
    EXPECT_EQ(status, NPP_STEP_ERROR);
    
    // 测试无效起始数字
    status = nppiCompressMarkerLabelsUF_32u_C1IR(nullptr, 32, oMarkerLabelsROI,
                                                -1, &newCount, nullptr);
    EXPECT_EQ(status, NPP_BAD_ARGUMENT_ERROR);
}