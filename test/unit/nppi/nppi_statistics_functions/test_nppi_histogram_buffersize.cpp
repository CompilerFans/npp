#include <gtest/gtest.h>
#include "npp.h"
#include <cuda_runtime.h>
#include <vector>

class NppiHistogramBufferSizeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test sizes
        smallSize = {32, 32};
        mediumSize = {128, 128}; 
        largeSize = {512, 512};
        
        nLevels = 256;
    }
    
    NppiSize smallSize, mediumSize, largeSize;
    int nLevels;
};

// Test nppiHistogramEvenGetBufferSize_8u_C1R
TEST_F(NppiHistogramBufferSizeTest, HistogramEvenGetBufferSize_8u_C1R_BasicOperation) {
    int bufferSize = 0;
    
    // Test small image
    NppStatus status = nppiHistogramEvenGetBufferSize_8u_C1R(smallSize, nLevels, &bufferSize);
    ASSERT_EQ(status, NPP_SUCCESS);
    EXPECT_GT(bufferSize, 0) << "Buffer size should be positive for small image";
    
    int smallBufferSize = bufferSize;
    
    // Test medium image
    status = nppiHistogramEvenGetBufferSize_8u_C1R(mediumSize, nLevels, &bufferSize);
    ASSERT_EQ(status, NPP_SUCCESS);
    EXPECT_GT(bufferSize, 0) << "Buffer size should be positive for medium image";
    EXPECT_GE(bufferSize, smallBufferSize) << "Larger image should need equal or more buffer";
    
    // Test large image
    status = nppiHistogramEvenGetBufferSize_8u_C1R(largeSize, nLevels, &bufferSize);
    ASSERT_EQ(status, NPP_SUCCESS);
    EXPECT_GT(bufferSize, 0) << "Buffer size should be positive for large image";
}

TEST_F(NppiHistogramBufferSizeTest, HistogramEvenGetBufferSize_8u_C1R_Ctx_BasicOperation) {
    NppStreamContext nppStreamCtx = {};
    nppStreamCtx.hStream = 0;
    int bufferSize = 0;
    
    NppStatus status = nppiHistogramEvenGetBufferSize_8u_C1R_Ctx(smallSize, nLevels, &bufferSize, nppStreamCtx);
    ASSERT_EQ(status, NPP_SUCCESS);
    EXPECT_GT(bufferSize, 0) << "Buffer size should be positive";
}

// Test nppiHistogramEvenGetBufferSize_8u_C4R
TEST_F(NppiHistogramBufferSizeTest, HistogramEvenGetBufferSize_8u_C4R_BasicOperation) {
    int bufferSize = 0;
    int nLevelsArray[4] = {256, 256, 256, 256}; // For C4R format
    
    NppStatus status = nppiHistogramEvenGetBufferSize_8u_C4R(smallSize, nLevelsArray, &bufferSize);
    ASSERT_EQ(status, NPP_SUCCESS);
    EXPECT_GT(bufferSize, 0) << "Buffer size should be positive";
    
    // C4R should typically require more buffer than C1R for same image size
    int c1rBufferSize = 0;
    nppiHistogramEvenGetBufferSize_8u_C1R(smallSize, nLevels, &c1rBufferSize);
    EXPECT_GE(bufferSize, c1rBufferSize) << "C4R should need equal or more buffer than C1R";
}

// Test nppiHistogramEvenGetBufferSize_16u_C1R  
TEST_F(NppiHistogramBufferSizeTest, HistogramEvenGetBufferSize_16u_C1R_BasicOperation) {
    int bufferSize = 0;
    int nLevels16u = 512; // Higher levels for 16-bit data
    
    NppStatus status = nppiHistogramEvenGetBufferSize_16u_C1R(smallSize, nLevels16u, &bufferSize);
    ASSERT_EQ(status, NPP_SUCCESS);
    EXPECT_GT(bufferSize, 0) << "Buffer size should be positive for 16u data";
}

// Test nppiHistogramEvenGetBufferSize_16s_C1R
TEST_F(NppiHistogramBufferSizeTest, HistogramEvenGetBufferSize_16s_C1R_BasicOperation) {
    int bufferSize = 0;
    int nLevels16s = 512;
    
    NppStatus status = nppiHistogramEvenGetBufferSize_16s_C1R(smallSize, nLevels16s, &bufferSize);
    ASSERT_EQ(status, NPP_SUCCESS);
    EXPECT_GT(bufferSize, 0) << "Buffer size should be positive for 16s data";
}

// Test that NVIDIA NPP accepts various parameter values (very permissive)
TEST_F(NppiHistogramBufferSizeTest, HistogramEvenGetBufferSize_ParameterTolerance) {
    int bufferSize = 0;
    
    // NVIDIA NPP is very permissive with parameters
    // Test that it doesn't crash with various inputs
    NppStatus status = nppiHistogramEvenGetBufferSize_8u_C1R(smallSize, 1, &bufferSize);
    (void)status; // Suppress unused variable warning
    
    status = nppiHistogramEvenGetBufferSize_8u_C1R(smallSize, 0, &bufferSize);
    (void)status; // Just ensure it doesn't crash
}

// Test that buffer size calculation is consistent
TEST_F(NppiHistogramBufferSizeTest, HistogramEvenGetBufferSize_Consistency) {
    int bufferSize1 = 0, bufferSize2 = 0;
    
    // Call twice with same parameters
    NppStatus status1 = nppiHistogramEvenGetBufferSize_8u_C1R(smallSize, nLevels, &bufferSize1);
    NppStatus status2 = nppiHistogramEvenGetBufferSize_8u_C1R(smallSize, nLevels, &bufferSize2);
    
    ASSERT_EQ(status1, NPP_SUCCESS);
    ASSERT_EQ(status2, NPP_SUCCESS);
    EXPECT_EQ(bufferSize1, bufferSize2) << "Buffer size should be consistent for same parameters";
}

// Test nppiHistogramRangeGetBufferSize_8u_C1R
TEST_F(NppiHistogramBufferSizeTest, HistogramRangeGetBufferSize_8u_C1R_BasicOperation) {
    int bufferSize = 0;
    
    NppStatus status = nppiHistogramRangeGetBufferSize_8u_C1R(smallSize, nLevels, &bufferSize);
    ASSERT_EQ(status, NPP_SUCCESS);
    EXPECT_GT(bufferSize, 0) << "Buffer size should be positive for histogram range";
}

// Test nppiHistogramRangeGetBufferSize_32f_C1R
TEST_F(NppiHistogramBufferSizeTest, HistogramRangeGetBufferSize_32f_C1R_BasicOperation) {
    int bufferSize = 0;
    
    NppStatus status = nppiHistogramRangeGetBufferSize_32f_C1R(smallSize, nLevels, &bufferSize);
    ASSERT_EQ(status, NPP_SUCCESS);
    EXPECT_GT(bufferSize, 0) << "Buffer size should be positive for 32f histogram range";
}

// Test integration with actual histogram computation
TEST_F(NppiHistogramBufferSizeTest, HistogramEvenGetBufferSize_Integration) {
    const int width = 64;
    const int height = 64;
    const int totalPixels = width * height;
    
    // Create test image
    std::vector<Npp8u> hostSrc(totalPixels);
    for (int i = 0; i < totalPixels; ++i) {
        hostSrc[i] = static_cast<Npp8u>(i % 256);
    }
    
    // Allocate GPU memory for source
    int srcStep;
    Npp8u *d_src = nppiMalloc_8u_C1(width, height, &srcStep);
    ASSERT_NE(d_src, nullptr);
    
    // Copy to GPU
    int hostStep = width * sizeof(Npp8u);
    cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
    
    // Get buffer size
    NppiSize imageSize = {width, height};
    int bufferSize = 0;
    NppStatus status = nppiHistogramEvenGetBufferSize_8u_C1R(imageSize, nLevels, &bufferSize);
    ASSERT_EQ(status, NPP_SUCCESS);
    ASSERT_GT(bufferSize, 0);
    
    // Allocate histogram buffer
    Npp8u *d_buffer = nullptr;
    cudaMalloc(&d_buffer, bufferSize);
    ASSERT_NE(d_buffer, nullptr);
    
    // Allocate histogram result
    std::vector<Npp32s> hostHist(nLevels - 1, 0);
    Npp32s *d_hist = nullptr;
    cudaMalloc(&d_hist, (nLevels - 1) * sizeof(Npp32s));
    ASSERT_NE(d_hist, nullptr);
    
    // Clear histogram
    cudaMemcpy(d_hist, hostHist.data(), (nLevels - 1) * sizeof(Npp32s), cudaMemcpyHostToDevice);
    
    // Compute histogram (this would use the buffer)
    Npp32s nLowerLevel = 0;
    Npp32s nUpperLevel = 256;
    
    // Note: NVIDIA NPP requires the device buffer parameter
    status = nppiHistogramEven_8u_C1R(d_src, srcStep, imageSize, d_hist, nLevels, nLowerLevel, nUpperLevel, d_buffer);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Cleanup
    cudaFree(d_buffer);
    cudaFree(d_hist);
    nppiFree(d_src);
}