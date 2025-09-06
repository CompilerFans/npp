/**
 * @file test_nppi_magnitude.cpp
 * @brief NPP Image Magnitude Functions Unit Tests
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include "npp.h"

class NPPIMagnitudeTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess);
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    
    void generateRandomComplexData(std::vector<Npp32fc>& data, float minVal, float maxVal) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(minVal, maxVal);
        
        for (auto& val : data) {
            val.re = dis(gen);
            val.im = dis(gen);
        }
    }
    
    void cpuMagnitude(const std::vector<Npp32fc>& src, std::vector<Npp32f>& dst) {
        for (size_t i = 0; i < src.size(); i++) {
            float real = src[i].re;
            float imag = src[i].im;
            dst[i] = std::sqrt(real * real + imag * imag);
        }
    }
    
    void cpuMagnitudeSqr(const std::vector<Npp32fc>& src, std::vector<Npp32f>& dst) {
        for (size_t i = 0; i < src.size(); i++) {
            float real = src[i].re;
            float imag = src[i].im;
            dst[i] = real * real + imag * imag;
        }
    }
    
    bool compareFloatArrays(const std::vector<Npp32f>& a, const std::vector<Npp32f>& b, float tolerance = 1e-5f) {
        if (a.size() != b.size()) return false;
        
        for (size_t i = 0; i < a.size(); i++) {
            float diff = std::abs(a[i] - b[i]);
            float maxVal = std::max(std::abs(a[i]), std::abs(b[i]));
            if (maxVal > 0.0f) {
                float relativeError = diff / maxVal;
                if (relativeError > tolerance) {
                    printf("Mismatch at index %zu: expected %f, got %f (relative error: %f)\n", 
                           i, a[i], b[i], relativeError);
                    return false;
                }
            } else if (diff > tolerance) {
                printf("Mismatch at index %zu: expected %f, got %f (absolute error: %f)\n", 
                       i, a[i], b[i], diff);
                return false;
            }
        }
        return true;
    }
};

// ==================== Magnitude tests ====================

TEST_F(NPPIMagnitudeTest, Magnitude_32fc32f_C1R_Basic) {
    const int width = 64;
    const int height = 48;
    
    const int srcStep = width * sizeof(Npp32fc);
    const int dstStep = width * sizeof(Npp32f);
    
    std::vector<Npp32fc> hostSrc(width * height);
    std::vector<Npp32f> hostDst(width * height);
    std::vector<Npp32f> hostRef(width * height);
    
    generateRandomComplexData(hostSrc, -100.0f, 100.0f);
    cpuMagnitude(hostSrc, hostRef);
    
    Npp32fc* deviceSrc = nullptr;
    Npp32f* deviceDst = nullptr;
    
    ASSERT_EQ(cudaMalloc(&deviceSrc, width * height * sizeof(Npp32fc)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceDst, width * height * sizeof(Npp32f)), cudaSuccess);
    
    ASSERT_EQ(cudaMemcpy(deviceSrc, hostSrc.data(), width * height * sizeof(Npp32fc), cudaMemcpyHostToDevice), cudaSuccess);
    
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiMagnitude_32fc32f_C1R(deviceSrc, srcStep, deviceDst, dstStep, oSizeROI);
    ASSERT_EQ(status, NPP_NO_ERROR);
    
    ASSERT_EQ(cudaMemcpy(hostDst.data(), deviceDst, width * height * sizeof(Npp32f), cudaMemcpyDeviceToHost), cudaSuccess);
    
    EXPECT_TRUE(compareFloatArrays(hostDst, hostRef, 1e-5f));
    
    cudaFree(deviceSrc);
    cudaFree(deviceDst);
}

TEST_F(NPPIMagnitudeTest, Magnitude_32fc32f_C1R_SpecialValues) {
    const int width = 8;
    const int height = 1;
    
    const int srcStep = width * sizeof(Npp32fc);
    const int dstStep = width * sizeof(Npp32f);
    
    std::vector<Npp32fc> hostSrc = {
        {0.0f, 0.0f},     // magnitude = 0
        {3.0f, 4.0f},     // magnitude = 5
        {-3.0f, 4.0f},    // magnitude = 5
        {3.0f, -4.0f},    // magnitude = 5
        {-3.0f, -4.0f},   // magnitude = 5
        {1.0f, 0.0f},     // magnitude = 1
        {0.0f, 1.0f},     // magnitude = 1
        {5.0f, 12.0f}     // magnitude = 13
    };
    
    std::vector<Npp32f> hostDst(width * height);
    std::vector<Npp32f> hostRef = {0.0f, 5.0f, 5.0f, 5.0f, 5.0f, 1.0f, 1.0f, 13.0f};
    
    Npp32fc* deviceSrc = nullptr;
    Npp32f* deviceDst = nullptr;
    
    ASSERT_EQ(cudaMalloc(&deviceSrc, width * height * sizeof(Npp32fc)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceDst, width * height * sizeof(Npp32f)), cudaSuccess);
    
    ASSERT_EQ(cudaMemcpy(deviceSrc, hostSrc.data(), width * height * sizeof(Npp32fc), cudaMemcpyHostToDevice), cudaSuccess);
    
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiMagnitude_32fc32f_C1R(deviceSrc, srcStep, deviceDst, dstStep, oSizeROI);
    ASSERT_EQ(status, NPP_NO_ERROR);
    
    ASSERT_EQ(cudaMemcpy(hostDst.data(), deviceDst, width * height * sizeof(Npp32f), cudaMemcpyDeviceToHost), cudaSuccess);
    
    EXPECT_TRUE(compareFloatArrays(hostDst, hostRef, 1e-5f));
    
    cudaFree(deviceSrc);
    cudaFree(deviceDst);
}

// ==================== Magnitude squared tests ====================

TEST_F(NPPIMagnitudeTest, MagnitudeSqr_32fc32f_C1R_Basic) {
    const int width = 64;
    const int height = 48;
    
    const int srcStep = width * sizeof(Npp32fc);
    const int dstStep = width * sizeof(Npp32f);
    
    std::vector<Npp32fc> hostSrc(width * height);
    std::vector<Npp32f> hostDst(width * height);
    std::vector<Npp32f> hostRef(width * height);
    
    generateRandomComplexData(hostSrc, -100.0f, 100.0f);
    cpuMagnitudeSqr(hostSrc, hostRef);
    
    Npp32fc* deviceSrc = nullptr;
    Npp32f* deviceDst = nullptr;
    
    ASSERT_EQ(cudaMalloc(&deviceSrc, width * height * sizeof(Npp32fc)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceDst, width * height * sizeof(Npp32f)), cudaSuccess);
    
    ASSERT_EQ(cudaMemcpy(deviceSrc, hostSrc.data(), width * height * sizeof(Npp32fc), cudaMemcpyHostToDevice), cudaSuccess);
    
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiMagnitudeSqr_32fc32f_C1R(deviceSrc, srcStep, deviceDst, dstStep, oSizeROI);
    ASSERT_EQ(status, NPP_NO_ERROR);
    
    ASSERT_EQ(cudaMemcpy(hostDst.data(), deviceDst, width * height * sizeof(Npp32f), cudaMemcpyDeviceToHost), cudaSuccess);
    
    EXPECT_TRUE(compareFloatArrays(hostDst, hostRef, 1e-5f));
    
    cudaFree(deviceSrc);
    cudaFree(deviceDst);
}

TEST_F(NPPIMagnitudeTest, MagnitudeSqr_32fc32f_C1R_SpecialValues) {
    const int width = 8;
    const int height = 1;
    
    const int srcStep = width * sizeof(Npp32fc);
    const int dstStep = width * sizeof(Npp32f);
    
    std::vector<Npp32fc> hostSrc = {
        {0.0f, 0.0f},     // magnitude_sqr = 0
        {3.0f, 4.0f},     // magnitude_sqr = 25
        {-3.0f, 4.0f},    // magnitude_sqr = 25
        {3.0f, -4.0f},    // magnitude_sqr = 25
        {-3.0f, -4.0f},   // magnitude_sqr = 25
        {1.0f, 0.0f},     // magnitude_sqr = 1
        {0.0f, 1.0f},     // magnitude_sqr = 1
        {5.0f, 12.0f}     // magnitude_sqr = 169
    };
    
    std::vector<Npp32f> hostDst(width * height);
    std::vector<Npp32f> hostRef = {0.0f, 25.0f, 25.0f, 25.0f, 25.0f, 1.0f, 1.0f, 169.0f};
    
    Npp32fc* deviceSrc = nullptr;
    Npp32f* deviceDst = nullptr;
    
    ASSERT_EQ(cudaMalloc(&deviceSrc, width * height * sizeof(Npp32fc)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceDst, width * height * sizeof(Npp32f)), cudaSuccess);
    
    ASSERT_EQ(cudaMemcpy(deviceSrc, hostSrc.data(), width * height * sizeof(Npp32fc), cudaMemcpyHostToDevice), cudaSuccess);
    
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiMagnitudeSqr_32fc32f_C1R(deviceSrc, srcStep, deviceDst, dstStep, oSizeROI);
    ASSERT_EQ(status, NPP_NO_ERROR);
    
    ASSERT_EQ(cudaMemcpy(hostDst.data(), deviceDst, width * height * sizeof(Npp32f), cudaMemcpyDeviceToHost), cudaSuccess);
    
    EXPECT_TRUE(compareFloatArrays(hostDst, hostRef, 1e-5f));
    
    cudaFree(deviceSrc);
    cudaFree(deviceDst);
}

// ==================== Performance comparison test ====================

TEST_F(NPPIMagnitudeTest, MagnitudeVsMagnitudeSqr_Performance) {
    const int width = 512;
    const int height = 512;
    
    const int srcStep = width * sizeof(Npp32fc);
    const int dstStep = width * sizeof(Npp32f);
    
    std::vector<Npp32fc> hostSrc(width * height);
    std::vector<Npp32f> hostDstMag(width * height);
    std::vector<Npp32f> hostDstMagSqr(width * height);
    
    generateRandomComplexData(hostSrc, -1000.0f, 1000.0f);
    
    Npp32fc* deviceSrc = nullptr;
    Npp32f* deviceDstMag = nullptr;
    Npp32f* deviceDstMagSqr = nullptr;
    
    ASSERT_EQ(cudaMalloc(&deviceSrc, width * height * sizeof(Npp32fc)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceDstMag, width * height * sizeof(Npp32f)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceDstMagSqr, width * height * sizeof(Npp32f)), cudaSuccess);
    
    ASSERT_EQ(cudaMemcpy(deviceSrc, hostSrc.data(), width * height * sizeof(Npp32fc), cudaMemcpyHostToDevice), cudaSuccess);
    
    NppiSize oSizeROI = {width, height};
    
    // Test magnitude
    NppStatus statusMag = nppiMagnitude_32fc32f_C1R(deviceSrc, srcStep, deviceDstMag, dstStep, oSizeROI);
    ASSERT_EQ(statusMag, NPP_NO_ERROR);
    
    // Test magnitude squared
    NppStatus statusMagSqr = nppiMagnitudeSqr_32fc32f_C1R(deviceSrc, srcStep, deviceDstMagSqr, dstStep, oSizeROI);
    ASSERT_EQ(statusMagSqr, NPP_NO_ERROR);
    
    ASSERT_EQ(cudaMemcpy(hostDstMag.data(), deviceDstMag, width * height * sizeof(Npp32f), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(hostDstMagSqr.data(), deviceDstMagSqr, width * height * sizeof(Npp32f), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // Verify relationship: magnitude^2 = magnitude_sqr
    for (int i = 0; i < width * height; i++) {
        float mag = hostDstMag[i];
        float magSqr = hostDstMagSqr[i];
        float expectedMagSqr = mag * mag;
        
        // Use relative tolerance for large values, absolute tolerance for small values
        float tolerance = std::max(1e-3f, std::abs(expectedMagSqr) * 1e-4f);
        EXPECT_NEAR(magSqr, expectedMagSqr, tolerance) << "Mismatch at index " << i;
    }
    
    cudaFree(deviceSrc);
    cudaFree(deviceDstMag);
    cudaFree(deviceDstMagSqr);
}

// ==================== Error condition tests ====================

TEST_F(NPPIMagnitudeTest, ErrorHandling_NullPointers) {
    NppiSize oSizeROI = {32, 24};
    const int srcStep = 32 * sizeof(Npp32fc);
    const int dstStep = 32 * sizeof(Npp32f);
    
    Npp32fc* deviceSrc = nullptr;
    Npp32f* deviceDst = nullptr;
    
    ASSERT_EQ(cudaMalloc(&deviceSrc, 32 * 24 * sizeof(Npp32fc)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceDst, 32 * 24 * sizeof(Npp32f)), cudaSuccess);
    
    // Test null source pointer
    EXPECT_EQ(nppiMagnitude_32fc32f_C1R(nullptr, srcStep, deviceDst, dstStep, oSizeROI), NPP_NULL_POINTER_ERROR);
    EXPECT_EQ(nppiMagnitudeSqr_32fc32f_C1R(nullptr, srcStep, deviceDst, dstStep, oSizeROI), NPP_NULL_POINTER_ERROR);
    
    // Test null destination pointer
    EXPECT_EQ(nppiMagnitude_32fc32f_C1R(deviceSrc, srcStep, nullptr, dstStep, oSizeROI), NPP_NULL_POINTER_ERROR);
    EXPECT_EQ(nppiMagnitudeSqr_32fc32f_C1R(deviceSrc, srcStep, nullptr, dstStep, oSizeROI), NPP_NULL_POINTER_ERROR);
    
    cudaFree(deviceSrc);
    cudaFree(deviceDst);
}

TEST_F(NPPIMagnitudeTest, ErrorHandling_InvalidSize) {
    const int srcStep = 32 * sizeof(Npp32fc);
    const int dstStep = 32 * sizeof(Npp32f);
    
    Npp32fc* deviceSrc = nullptr;
    Npp32f* deviceDst = nullptr;
    
    ASSERT_EQ(cudaMalloc(&deviceSrc, 32 * 24 * sizeof(Npp32fc)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceDst, 32 * 24 * sizeof(Npp32f)), cudaSuccess);
    
    NppiSize invalidSize1 = {0, 24};
    NppiSize invalidSize2 = {32, 0};
    NppiSize invalidSize3 = {-32, 24};
    
    EXPECT_EQ(nppiMagnitude_32fc32f_C1R(deviceSrc, srcStep, deviceDst, dstStep, invalidSize1), NPP_SIZE_ERROR);
    EXPECT_EQ(nppiMagnitude_32fc32f_C1R(deviceSrc, srcStep, deviceDst, dstStep, invalidSize2), NPP_SIZE_ERROR);
    EXPECT_EQ(nppiMagnitude_32fc32f_C1R(deviceSrc, srcStep, deviceDst, dstStep, invalidSize3), NPP_SIZE_ERROR);
    
    EXPECT_EQ(nppiMagnitudeSqr_32fc32f_C1R(deviceSrc, srcStep, deviceDst, dstStep, invalidSize1), NPP_SIZE_ERROR);
    EXPECT_EQ(nppiMagnitudeSqr_32fc32f_C1R(deviceSrc, srcStep, deviceDst, dstStep, invalidSize2), NPP_SIZE_ERROR);
    EXPECT_EQ(nppiMagnitudeSqr_32fc32f_C1R(deviceSrc, srcStep, deviceDst, dstStep, invalidSize3), NPP_SIZE_ERROR);
    
    cudaFree(deviceSrc);
    cudaFree(deviceDst);
}

TEST_F(NPPIMagnitudeTest, ErrorHandling_InvalidStep) {
    NppiSize oSizeROI = {32, 24};
    
    Npp32fc* deviceSrc = nullptr;
    Npp32f* deviceDst = nullptr;
    
    ASSERT_EQ(cudaMalloc(&deviceSrc, 32 * 24 * sizeof(Npp32fc)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceDst, 32 * 24 * sizeof(Npp32f)), cudaSuccess);
    
    const int validSrcStep = 32 * sizeof(Npp32fc);
    const int validDstStep = 32 * sizeof(Npp32f);
    const int invalidSrcStep = 16 * sizeof(Npp32fc);  // Too small
    const int invalidDstStep = 16 * sizeof(Npp32f);   // Too small
    
    EXPECT_EQ(nppiMagnitude_32fc32f_C1R(deviceSrc, invalidSrcStep, deviceDst, validDstStep, oSizeROI), NPP_STRIDE_ERROR);
    EXPECT_EQ(nppiMagnitude_32fc32f_C1R(deviceSrc, validSrcStep, deviceDst, invalidDstStep, oSizeROI), NPP_STRIDE_ERROR);
    
    EXPECT_EQ(nppiMagnitudeSqr_32fc32f_C1R(deviceSrc, invalidSrcStep, deviceDst, validDstStep, oSizeROI), NPP_STRIDE_ERROR);
    EXPECT_EQ(nppiMagnitudeSqr_32fc32f_C1R(deviceSrc, validSrcStep, deviceDst, invalidDstStep, oSizeROI), NPP_STRIDE_ERROR);
    
    cudaFree(deviceSrc);
    cudaFree(deviceDst);
}

// ==================== Large image test ====================

TEST_F(NPPIMagnitudeTest, Magnitude_32fc32f_C1R_LargeImage) {
    const int width = 1024;
    const int height = 768;
    
    const int srcStep = width * sizeof(Npp32fc);
    const int dstStep = width * sizeof(Npp32f);
    
    std::vector<Npp32fc> hostSrc(width * height);
    std::vector<Npp32f> hostDst(width * height);
    std::vector<Npp32f> hostRef(width * height);
    
    generateRandomComplexData(hostSrc, -1000.0f, 1000.0f);
    cpuMagnitude(hostSrc, hostRef);
    
    Npp32fc* deviceSrc = nullptr;
    Npp32f* deviceDst = nullptr;
    
    ASSERT_EQ(cudaMalloc(&deviceSrc, width * height * sizeof(Npp32fc)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceDst, width * height * sizeof(Npp32f)), cudaSuccess);
    
    ASSERT_EQ(cudaMemcpy(deviceSrc, hostSrc.data(), width * height * sizeof(Npp32fc), cudaMemcpyHostToDevice), cudaSuccess);
    
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiMagnitude_32fc32f_C1R(deviceSrc, srcStep, deviceDst, dstStep, oSizeROI);
    ASSERT_EQ(status, NPP_NO_ERROR);
    
    ASSERT_EQ(cudaMemcpy(hostDst.data(), deviceDst, width * height * sizeof(Npp32f), cudaMemcpyDeviceToHost), cudaSuccess);
    
    // Check random samples for large images to avoid excessive test time
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, width * height - 1);
    
    for (int test = 0; test < 1000; test++) {
        int idx = dis(gen);
        float expected = hostRef[idx];
        float actual = hostDst[idx];
        float tolerance = std::max(1e-5f, expected * 1e-5f);
        EXPECT_NEAR(actual, expected, tolerance) << "Mismatch at index " << idx;
    }
    
    cudaFree(deviceSrc);
    cudaFree(deviceDst);
}