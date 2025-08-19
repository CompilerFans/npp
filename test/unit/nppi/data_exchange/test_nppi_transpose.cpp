/**
 * @file test_nppi_transpose.cpp
 * @brief NPP Image Transpose Functions Unit Tests
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>
#include "npp.h"

class NPPITransposeTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess);
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    
    template<typename T>
    void generateRandomData(std::vector<T>& data, T minVal, T maxVal) {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        if constexpr (std::is_floating_point<T>::value) {
            std::uniform_real_distribution<T> dis(minVal, maxVal);
            for (auto& val : data) {
                val = dis(gen);
            }
        } else {
            std::uniform_int_distribution<int> dis(minVal, maxVal);
            for (auto& val : data) {
                val = static_cast<T>(dis(gen));
            }
        }
    }
    
    template<typename T>
    void cpuTranspose(const std::vector<T>& src, std::vector<T>& dst, 
                     int srcWidth, int srcHeight, int channels) {
        for (int y = 0; y < srcHeight; y++) {
            for (int x = 0; x < srcWidth; x++) {
                for (int c = 0; c < channels; c++) {
                    int srcIdx = (y * srcWidth + x) * channels + c;
                    int dstIdx = (x * srcHeight + y) * channels + c;
                    dst[dstIdx] = src[srcIdx];
                }
            }
        }
    }
};

// ==================== 8-bit unsigned tests ====================

TEST_F(NPPITransposeTest, Transpose_8u_C1R_Basic) {
    const int srcWidth = 64;
    const int srcHeight = 48;
    const int dstWidth = srcHeight;
    const int dstHeight = srcWidth;
    
    const int srcStep = srcWidth * sizeof(Npp8u);
    const int dstStep = dstWidth * sizeof(Npp8u);
    
    std::vector<Npp8u> hostSrc(srcWidth * srcHeight);
    std::vector<Npp8u> hostDst(dstWidth * dstHeight);
    std::vector<Npp8u> hostRef(dstWidth * dstHeight);
    
    generateRandomData(hostSrc, Npp8u(0), Npp8u(255));
    cpuTranspose(hostSrc, hostRef, srcWidth, srcHeight, 1);
    
    Npp8u* deviceSrc = nullptr;
    Npp8u* deviceDst = nullptr;
    
    ASSERT_EQ(cudaMalloc(&deviceSrc, srcWidth * srcHeight * sizeof(Npp8u)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceDst, dstWidth * dstHeight * sizeof(Npp8u)), cudaSuccess);
    
    ASSERT_EQ(cudaMemcpy(deviceSrc, hostSrc.data(), srcWidth * srcHeight * sizeof(Npp8u), cudaMemcpyHostToDevice), cudaSuccess);
    
    NppiSize oSrcROI = {srcWidth, srcHeight};
    NppStatus status = nppiTranspose_8u_C1R(deviceSrc, srcStep, deviceDst, dstStep, oSrcROI);
    ASSERT_EQ(status, NPP_NO_ERROR);
    
    ASSERT_EQ(cudaMemcpy(hostDst.data(), deviceDst, dstWidth * dstHeight * sizeof(Npp8u), cudaMemcpyDeviceToHost), cudaSuccess);
    
    for (int i = 0; i < dstWidth * dstHeight; i++) {
        EXPECT_EQ(hostDst[i], hostRef[i]) << "Mismatch at index " << i;
    }
    
    cudaFree(deviceSrc);
    cudaFree(deviceDst);
}

TEST_F(NPPITransposeTest, Transpose_8u_C3R_Basic) {
    const int srcWidth = 32;
    const int srcHeight = 24;
    const int dstWidth = srcHeight;
    const int dstHeight = srcWidth;
    const int channels = 3;
    
    const int srcStep = srcWidth * channels * sizeof(Npp8u);
    const int dstStep = dstWidth * channels * sizeof(Npp8u);
    
    std::vector<Npp8u> hostSrc(srcWidth * srcHeight * channels);
    std::vector<Npp8u> hostDst(dstWidth * dstHeight * channels);
    std::vector<Npp8u> hostRef(dstWidth * dstHeight * channels);
    
    generateRandomData(hostSrc, Npp8u(0), Npp8u(255));
    cpuTranspose(hostSrc, hostRef, srcWidth, srcHeight, channels);
    
    Npp8u* deviceSrc = nullptr;
    Npp8u* deviceDst = nullptr;
    
    ASSERT_EQ(cudaMalloc(&deviceSrc, srcWidth * srcHeight * channels * sizeof(Npp8u)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceDst, dstWidth * dstHeight * channels * sizeof(Npp8u)), cudaSuccess);
    
    ASSERT_EQ(cudaMemcpy(deviceSrc, hostSrc.data(), srcWidth * srcHeight * channels * sizeof(Npp8u), cudaMemcpyHostToDevice), cudaSuccess);
    
    NppiSize oSrcROI = {srcWidth, srcHeight};
    NppStatus status = nppiTranspose_8u_C3R(deviceSrc, srcStep, deviceDst, dstStep, oSrcROI);
    ASSERT_EQ(status, NPP_NO_ERROR);
    
    ASSERT_EQ(cudaMemcpy(hostDst.data(), deviceDst, dstWidth * dstHeight * channels * sizeof(Npp8u), cudaMemcpyDeviceToHost), cudaSuccess);
    
    for (int i = 0; i < dstWidth * dstHeight * channels; i++) {
        EXPECT_EQ(hostDst[i], hostRef[i]) << "Mismatch at index " << i;
    }
    
    cudaFree(deviceSrc);
    cudaFree(deviceDst);
}

TEST_F(NPPITransposeTest, Transpose_8u_C4R_Basic) {
    const int srcWidth = 16;
    const int srcHeight = 12;
    const int dstWidth = srcHeight;
    const int dstHeight = srcWidth;
    const int channels = 4;
    
    const int srcStep = srcWidth * channels * sizeof(Npp8u);
    const int dstStep = dstWidth * channels * sizeof(Npp8u);
    
    std::vector<Npp8u> hostSrc(srcWidth * srcHeight * channels);
    std::vector<Npp8u> hostDst(dstWidth * dstHeight * channels);
    std::vector<Npp8u> hostRef(dstWidth * dstHeight * channels);
    
    generateRandomData(hostSrc, Npp8u(0), Npp8u(255));
    cpuTranspose(hostSrc, hostRef, srcWidth, srcHeight, channels);
    
    Npp8u* deviceSrc = nullptr;
    Npp8u* deviceDst = nullptr;
    
    ASSERT_EQ(cudaMalloc(&deviceSrc, srcWidth * srcHeight * channels * sizeof(Npp8u)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceDst, dstWidth * dstHeight * channels * sizeof(Npp8u)), cudaSuccess);
    
    ASSERT_EQ(cudaMemcpy(deviceSrc, hostSrc.data(), srcWidth * srcHeight * channels * sizeof(Npp8u), cudaMemcpyHostToDevice), cudaSuccess);
    
    NppiSize oSrcROI = {srcWidth, srcHeight};
    NppStatus status = nppiTranspose_8u_C4R(deviceSrc, srcStep, deviceDst, dstStep, oSrcROI);
    ASSERT_EQ(status, NPP_NO_ERROR);
    
    ASSERT_EQ(cudaMemcpy(hostDst.data(), deviceDst, dstWidth * dstHeight * channels * sizeof(Npp8u), cudaMemcpyDeviceToHost), cudaSuccess);
    
    for (int i = 0; i < dstWidth * dstHeight * channels; i++) {
        EXPECT_EQ(hostDst[i], hostRef[i]) << "Mismatch at index " << i;
    }
    
    cudaFree(deviceSrc);
    cudaFree(deviceDst);
}

// ==================== 16-bit unsigned tests ====================

TEST_F(NPPITransposeTest, Transpose_16u_C1R_Basic) {
    const int srcWidth = 32;
    const int srcHeight = 24;
    const int dstWidth = srcHeight;
    const int dstHeight = srcWidth;
    
    const int srcStep = srcWidth * sizeof(Npp16u);
    const int dstStep = dstWidth * sizeof(Npp16u);
    
    std::vector<Npp16u> hostSrc(srcWidth * srcHeight);
    std::vector<Npp16u> hostDst(dstWidth * dstHeight);
    std::vector<Npp16u> hostRef(dstWidth * dstHeight);
    
    generateRandomData(hostSrc, Npp16u(0), Npp16u(65535));
    cpuTranspose(hostSrc, hostRef, srcWidth, srcHeight, 1);
    
    Npp16u* deviceSrc = nullptr;
    Npp16u* deviceDst = nullptr;
    
    ASSERT_EQ(cudaMalloc(&deviceSrc, srcWidth * srcHeight * sizeof(Npp16u)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceDst, dstWidth * dstHeight * sizeof(Npp16u)), cudaSuccess);
    
    ASSERT_EQ(cudaMemcpy(deviceSrc, hostSrc.data(), srcWidth * srcHeight * sizeof(Npp16u), cudaMemcpyHostToDevice), cudaSuccess);
    
    NppiSize oSrcROI = {srcWidth, srcHeight};
    NppStatus status = nppiTranspose_16u_C1R(deviceSrc, srcStep, deviceDst, dstStep, oSrcROI);
    ASSERT_EQ(status, NPP_NO_ERROR);
    
    ASSERT_EQ(cudaMemcpy(hostDst.data(), deviceDst, dstWidth * dstHeight * sizeof(Npp16u), cudaMemcpyDeviceToHost), cudaSuccess);
    
    for (int i = 0; i < dstWidth * dstHeight; i++) {
        EXPECT_EQ(hostDst[i], hostRef[i]) << "Mismatch at index " << i;
    }
    
    cudaFree(deviceSrc);
    cudaFree(deviceDst);
}

// ==================== 32-bit float tests ====================

TEST_F(NPPITransposeTest, Transpose_32f_C1R_Basic) {
    const int srcWidth = 32;
    const int srcHeight = 24;
    const int dstWidth = srcHeight;
    const int dstHeight = srcWidth;
    
    const int srcStep = srcWidth * sizeof(Npp32f);
    const int dstStep = dstWidth * sizeof(Npp32f);
    
    std::vector<Npp32f> hostSrc(srcWidth * srcHeight);
    std::vector<Npp32f> hostDst(dstWidth * dstHeight);
    std::vector<Npp32f> hostRef(dstWidth * dstHeight);
    
    generateRandomData(hostSrc, Npp32f(-1000.0f), Npp32f(1000.0f));
    cpuTranspose(hostSrc, hostRef, srcWidth, srcHeight, 1);
    
    Npp32f* deviceSrc = nullptr;
    Npp32f* deviceDst = nullptr;
    
    ASSERT_EQ(cudaMalloc(&deviceSrc, srcWidth * srcHeight * sizeof(Npp32f)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceDst, dstWidth * dstHeight * sizeof(Npp32f)), cudaSuccess);
    
    ASSERT_EQ(cudaMemcpy(deviceSrc, hostSrc.data(), srcWidth * srcHeight * sizeof(Npp32f), cudaMemcpyHostToDevice), cudaSuccess);
    
    NppiSize oSrcROI = {srcWidth, srcHeight};
    NppStatus status = nppiTranspose_32f_C1R(deviceSrc, srcStep, deviceDst, dstStep, oSrcROI);
    ASSERT_EQ(status, NPP_NO_ERROR);
    
    ASSERT_EQ(cudaMemcpy(hostDst.data(), deviceDst, dstWidth * dstHeight * sizeof(Npp32f), cudaMemcpyDeviceToHost), cudaSuccess);
    
    for (int i = 0; i < dstWidth * dstHeight; i++) {
        EXPECT_FLOAT_EQ(hostDst[i], hostRef[i]) << "Mismatch at index " << i;
    }
    
    cudaFree(deviceSrc);
    cudaFree(deviceDst);
}

// ==================== Error condition tests ====================

TEST_F(NPPITransposeTest, ErrorHandling_NullPointers) {
    NppiSize oSrcROI = {32, 24};
    const int srcStep = 32 * sizeof(Npp8u);
    const int dstStep = 24 * sizeof(Npp8u);
    
    Npp8u* deviceSrc = nullptr;
    Npp8u* deviceDst = nullptr;
    
    ASSERT_EQ(cudaMalloc(&deviceSrc, 32 * 24 * sizeof(Npp8u)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceDst, 24 * 32 * sizeof(Npp8u)), cudaSuccess);
    
    EXPECT_EQ(nppiTranspose_8u_C1R(nullptr, srcStep, deviceDst, dstStep, oSrcROI), NPP_NULL_POINTER_ERROR);
    EXPECT_EQ(nppiTranspose_8u_C1R(deviceSrc, srcStep, nullptr, dstStep, oSrcROI), NPP_NULL_POINTER_ERROR);
    
    cudaFree(deviceSrc);
    cudaFree(deviceDst);
}

TEST_F(NPPITransposeTest, ErrorHandling_InvalidSize) {
    const int srcStep = 32 * sizeof(Npp8u);
    const int dstStep = 24 * sizeof(Npp8u);
    
    Npp8u* deviceSrc = nullptr;
    Npp8u* deviceDst = nullptr;
    
    ASSERT_EQ(cudaMalloc(&deviceSrc, 32 * 24 * sizeof(Npp8u)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceDst, 24 * 32 * sizeof(Npp8u)), cudaSuccess);
    
    NppiSize invalidSize1 = {0, 24};
    NppiSize invalidSize2 = {32, 0};
    NppiSize invalidSize3 = {-32, 24};
    
    EXPECT_EQ(nppiTranspose_8u_C1R(deviceSrc, srcStep, deviceDst, dstStep, invalidSize1), NPP_SIZE_ERROR);
    EXPECT_EQ(nppiTranspose_8u_C1R(deviceSrc, srcStep, deviceDst, dstStep, invalidSize2), NPP_SIZE_ERROR);
    EXPECT_EQ(nppiTranspose_8u_C1R(deviceSrc, srcStep, deviceDst, dstStep, invalidSize3), NPP_SIZE_ERROR);
    
    cudaFree(deviceSrc);
    cudaFree(deviceDst);
}

TEST_F(NPPITransposeTest, ErrorHandling_InvalidStep) {
    NppiSize oSrcROI = {32, 24};
    
    Npp8u* deviceSrc = nullptr;
    Npp8u* deviceDst = nullptr;
    
    ASSERT_EQ(cudaMalloc(&deviceSrc, 32 * 24 * sizeof(Npp8u)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceDst, 24 * 32 * sizeof(Npp8u)), cudaSuccess);
    
    const int validSrcStep = 32 * sizeof(Npp8u);
    const int validDstStep = 24 * sizeof(Npp8u);
    const int invalidSrcStep = 16 * sizeof(Npp8u);  // Too small
    const int invalidDstStep = 12 * sizeof(Npp8u);  // Too small
    
    EXPECT_EQ(nppiTranspose_8u_C1R(deviceSrc, invalidSrcStep, deviceDst, validDstStep, oSrcROI), NPP_STRIDE_ERROR);
    EXPECT_EQ(nppiTranspose_8u_C1R(deviceSrc, validSrcStep, deviceDst, invalidDstStep, oSrcROI), NPP_STRIDE_ERROR);
    
    cudaFree(deviceSrc);
    cudaFree(deviceDst);
}

// ==================== Large image test ====================

TEST_F(NPPITransposeTest, Transpose_8u_C1R_LargeImage) {
    const int srcWidth = 512;
    const int srcHeight = 384;
    const int dstWidth = srcHeight;
    const int dstHeight = srcWidth;
    
    const int srcStep = srcWidth * sizeof(Npp8u);
    const int dstStep = dstWidth * sizeof(Npp8u);
    
    std::vector<Npp8u> hostSrc(srcWidth * srcHeight);
    std::vector<Npp8u> hostDst(dstWidth * dstHeight);
    std::vector<Npp8u> hostRef(dstWidth * dstHeight);
    
    generateRandomData(hostSrc, Npp8u(0), Npp8u(255));
    cpuTranspose(hostSrc, hostRef, srcWidth, srcHeight, 1);
    
    Npp8u* deviceSrc = nullptr;
    Npp8u* deviceDst = nullptr;
    
    ASSERT_EQ(cudaMalloc(&deviceSrc, srcWidth * srcHeight * sizeof(Npp8u)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceDst, dstWidth * dstHeight * sizeof(Npp8u)), cudaSuccess);
    
    ASSERT_EQ(cudaMemcpy(deviceSrc, hostSrc.data(), srcWidth * srcHeight * sizeof(Npp8u), cudaMemcpyHostToDevice), cudaSuccess);
    
    NppiSize oSrcROI = {srcWidth, srcHeight};
    NppStatus status = nppiTranspose_8u_C1R(deviceSrc, srcStep, deviceDst, dstStep, oSrcROI);
    ASSERT_EQ(status, NPP_NO_ERROR);
    
    ASSERT_EQ(cudaMemcpy(hostDst.data(), deviceDst, dstWidth * dstHeight * sizeof(Npp8u), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // Check random samples for large images to avoid excessive test time
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, dstWidth * dstHeight - 1);
    
    for (int test = 0; test < 1000; test++) {
        int idx = dis(gen);
        EXPECT_EQ(hostDst[idx], hostRef[idx]) << "Mismatch at index " << idx;
    }
    
    cudaFree(deviceSrc);
    cudaFree(deviceDst);
}