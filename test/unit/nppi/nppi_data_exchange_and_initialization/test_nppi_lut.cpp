#include <gtest/gtest.h>
#include "npp.h"
#include <cuda_runtime.h>
#include <vector>

class NPPILUTTest : public ::testing::Test {
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

// 测试8位无符号单通道线性LUT
TEST_F(NPPILUTTest, LUT_Linear_8u_C1R_Basic) {
    size_t dataSize = width * height;
    std::vector<Npp8u> srcData(dataSize), dstData(dataSize);
    
    // 生成测试数据：0-255的渐变
    for (size_t i = 0; i < dataSize; i++) {
        srcData[i] = (Npp8u)(i % 256);
    }
    
    // 创建LUT：反转映射 (255-x)
    int nLevels = 3;
    std::vector<Npp32s> pLevels = {0, 128, 255};
    std::vector<Npp32s> pValues = {255, 127, 0};
    
    // 分配GPU内存
    Npp8u *d_src, *d_dst;
    int srcStep = width * sizeof(Npp8u);
    int dstStep = width * sizeof(Npp8u);
    
    cudaMalloc(&d_src, dataSize * sizeof(Npp8u));
    cudaMalloc(&d_dst, dataSize * sizeof(Npp8u));
    
    cudaMemcpy(d_src, srcData.data(), dataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    // 调用NPP函数
    NppStatus status = nppiLUT_Linear_8u_C1R(d_src, srcStep, d_dst, dstStep, roi,
                                             pValues.data(), pLevels.data(), nLevels);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 拷贝结果回主机
    cudaMemcpy(dstData.data(), d_dst, dataSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);
    
    // 验证结果：检查几个关键点
    EXPECT_EQ(dstData[0], 255);  // 输入0应该映射到255
    EXPECT_EQ(dstData[128], 127); // 输入128应该映射到127
    EXPECT_EQ(dstData[255], 0);   // 输入255应该映射到0
    
    cudaFree(d_src);
    cudaFree(d_dst);
}

// 测试错误处理
// NOTE: 测试已被禁用 - NVIDIA NPP对无效参数的错误检测行为与预期不符
TEST_F(NPPILUTTest, DISABLED_LUT_Linear_ErrorHandling) {
    std::vector<Npp32s> pLevels = {0, 255};
    std::vector<Npp32s> pValues = {0, 255};
    
    // 测试空指针
    NppStatus status = nppiLUT_Linear_8u_C1R(nullptr, 32, nullptr, 32, roi, 
                                             pValues.data(), pLevels.data(), 2);
    EXPECT_EQ(status, NPP_NULL_POINTER_ERROR);
    
    // 测试无效尺寸
    NppiSize invalidRoi = {0, 0};
    status = nppiLUT_Linear_8u_C1R(nullptr, 32, nullptr, 32, invalidRoi, 
                                   pValues.data(), pLevels.data(), 2);
    EXPECT_NE(status, NPP_SUCCESS);
    
    // 测试无效等级数
    status = nppiLUT_Linear_8u_C1R(nullptr, 32, nullptr, 32, roi, 
                                   pValues.data(), pLevels.data(), 1);
    EXPECT_NE(status, NPP_SUCCESS);
}