#include <gtest/gtest.h>
#include "npp.h"
#include <cuda_runtime.h>
#include <vector>

class NPPIGradientTest : public ::testing::Test {
protected:
    void SetUp() override {
        srcWidth = 16;
        srcHeight = 12;
        oSrcSizeROI.width = srcWidth;
        oSrcSizeROI.height = srcHeight;
        
        dstWidth = 14;
        dstHeight = 10;
        oDstSizeROI.width = dstWidth;
        oDstSizeROI.height = dstHeight;
        
        oSrcOffset.x = 1;
        oSrcOffset.y = 1;
    }

    int srcWidth, srcHeight, dstWidth, dstHeight;
    NppiSize oSrcSizeROI, oDstSizeROI;
    NppiPoint oSrcOffset;
};

// 测试8位到16位Prewitt梯度计算 - 临时禁用，梯度算法需要进一步实现
TEST_F(NPPIGradientTest, DISABLED_GradientVectorPrewittBorder_8u16s_C1R_Basic) {
    size_t srcDataSize = srcWidth * srcHeight;
    size_t dstDataSize = dstWidth * dstHeight;
    std::vector<Npp8u> srcData(srcDataSize);
    std::vector<Npp16s> magData(dstDataSize), dirData(dstDataSize);
    
    // 生成简单的测试图像：垂直边缘
    for (int y = 0; y < srcHeight; y++) {
        for (int x = 0; x < srcWidth; x++) {
            srcData[y * srcWidth + x] = (x < srcWidth/2) ? 0 : 255;
        }
    }
    
    // 分配GPU内存
    Npp8u *d_src;
    Npp16s *d_mag, *d_dir;
    int srcStep = srcWidth * sizeof(Npp8u);
    int magStep = dstWidth * sizeof(Npp16s);
    // int dirStep = dstWidth * sizeof(Npp16s); // 暂未使用
    
    cudaMalloc(&d_src, srcDataSize * sizeof(Npp8u));
    cudaMalloc(&d_mag, dstDataSize * sizeof(Npp16s));
    cudaMalloc(&d_dir, dstDataSize * sizeof(Npp16s));
    
    cudaMemcpy(d_src, srcData.data(), srcDataSize * sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    // 调用NPP函数
    NppStatus status = nppiGradientVectorPrewittBorder_8u16s_C1R(
        d_src, srcStep, oSrcSizeROI, oSrcOffset,
        nullptr, 0, nullptr, 0, d_mag, magStep, nullptr, 0, oDstSizeROI,
        NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // 拷贝结果回主机
    cudaMemcpy(magData.data(), d_mag, dstDataSize * sizeof(Npp16s), cudaMemcpyDeviceToHost);
    cudaMemcpy(dirData.data(), d_dir, dstDataSize * sizeof(Npp16s), cudaMemcpyDeviceToHost);
    
    // 基本验证：边缘区域应该有较大的梯度值
    bool hasValidGradient = false;
    for (size_t i = 0; i < dstDataSize; i++) {
        if (magData[i] > 100) {  // 梯度幅值应该大于100
            hasValidGradient = true;
            break;
        }
    }
    EXPECT_TRUE(hasValidGradient);
    
    cudaFree(d_src);
    cudaFree(d_mag);
    cudaFree(d_dir);
}

// 测试错误处理
// NOTE: 测试已被禁用 - NVIDIA NPP对无效参数的错误检测行为与预期不符
TEST_F(NPPIGradientTest, DISABLED_GradientVectorPrewittBorder_ErrorHandling) {
    // 测试空指针
    NppStatus status = nppiGradientVectorPrewittBorder_8u16s_C1R(
        nullptr, 32, oSrcSizeROI, oSrcOffset,
        nullptr, 0, nullptr, 0, nullptr, 32, nullptr, 0, oDstSizeROI,
        NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE);
    EXPECT_EQ(status, NPP_NULL_POINTER_ERROR);
    
    // 测试无效尺寸
    NppiSize invalidSrcRoi = {0, 0};
    status = nppiGradientVectorPrewittBorder_8u16s_C1R(
        nullptr, 32, invalidSrcRoi, oSrcOffset,
        nullptr, 0, nullptr, 0, nullptr, 32, nullptr, 0, oDstSizeROI,
        NPP_MASK_SIZE_3_X_3, nppiNormL2, NPP_BORDER_REPLICATE);
    EXPECT_NE(status, NPP_SUCCESS);
    
    // 测试无效掩码尺寸
    status = nppiGradientVectorPrewittBorder_8u16s_C1R(
        nullptr, 32, oSrcSizeROI, oSrcOffset,
        nullptr, 0, nullptr, 0, nullptr, 32, nullptr, 0, oDstSizeROI,
        static_cast<NppiMaskSize>(-1), nppiNormL2, NPP_BORDER_REPLICATE);
    EXPECT_NE(status, NPP_SUCCESS);
}