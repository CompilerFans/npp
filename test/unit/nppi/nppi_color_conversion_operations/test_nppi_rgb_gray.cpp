#include <gtest/gtest.h>
#include "npp.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

class RGBToGrayFunctionalTest : public ::testing::Test {
protected:
    void SetUp() override {
        width = 16;
        height = 12;
        roi.width = width;
        roi.height = height;
    }

    int width, height;
    NppiSize roi;
    
    // Helper function to calculate expected grayscale value
    float calculateGrayValue(float r, float g, float b) {
        return 0.299f * r + 0.587f * g + 0.114f * b;
    }
    
    Npp8u calculateGrayValue8u(Npp8u r, Npp8u g, Npp8u b) {
        float gray = calculateGrayValue((float)r, (float)g, (float)b);
        return (Npp8u)(gray + 0.5f); // Round to nearest
    }
};

// Test 3-channel RGB to grayscale conversion (8-bit)
TEST_F(RGBToGrayFunctionalTest, RGBToGray_8u_C3C1R_BasicOperation) {
    std::vector<Npp8u> srcData(width * height * 3);
    std::vector<Npp8u> expectedData(width * height);
    
    // Create test RGB image with known values
    for (int i = 0; i < width * height; i++) {
        Npp8u r = (Npp8u)(i % 256);
        Npp8u g = (Npp8u)((i * 2) % 256);
        Npp8u b = (Npp8u)((i * 3) % 256);
        
        srcData[i * 3 + 0] = r;
        srcData[i * 3 + 1] = g;
        srcData[i * 3 + 2] = b;
        
        expectedData[i] = calculateGrayValue8u(r, g, b);
    }
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C3(width, height, &srcStep);
    Npp8u* d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU row by row to handle step properly
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep, 
                   srcData.data() + y * width * 3, 
                   width * 3 * sizeof(Npp8u), 
                   cudaMemcpyHostToDevice);
    }
    
    // Initialize destination memory to a pattern to detect if kernel runs
    cudaMemset(d_dst, 255, height * dstStep);
    
    // Execute NPP function
    NppStatus status = nppiRGBToGray_8u_C3C1R(d_src, srcStep, d_dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host
    std::vector<Npp8u> resultData(width * height);
    cudaMemcpy(resultData.data(), d_dst, width * height * sizeof(Npp8u), cudaMemcpyDeviceToHost);
    
    // Verify results (allow small rounding differences)
    for (int i = 0; i < width * height; i++) {
        EXPECT_NEAR(resultData[i], expectedData[i], 1) 
            << "Mismatch at pixel " << i << ": got " << (int)resultData[i] 
            << ", expected " << (int)expectedData[i];
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test 4-channel RGBA to grayscale conversion (8-bit)
TEST_F(RGBToGrayFunctionalTest, RGBToGray_8u_AC4C1R_BasicOperation) {
    std::vector<Npp8u> srcData(width * height * 4);
    std::vector<Npp8u> expectedData(width * height);
    
    // Create test RGBA image with known values
    for (int i = 0; i < width * height; i++) {
        Npp8u r = 100;
        Npp8u g = 150;
        Npp8u b = 200;
        Npp8u a = 255; // Alpha should be ignored
        
        srcData[i * 4 + 0] = r;
        srcData[i * 4 + 1] = g;
        srcData[i * 4 + 2] = b;
        srcData[i * 4 + 3] = a;
        
        expectedData[i] = calculateGrayValue8u(r, g, b);
    }
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C4(width, height, &srcStep);
    Npp8u* d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU row by row to handle step properly
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep, 
                   srcData.data() + y * width * 4, 
                   width * 4 * sizeof(Npp8u), 
                   cudaMemcpyHostToDevice);
    }
    
    // Execute NPP function
    NppStatus status = nppiRGBToGray_8u_AC4C1R(d_src, srcStep, d_dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host
    std::vector<Npp8u> resultData(width * height);
    cudaMemcpy(resultData.data(), d_dst, width * height * sizeof(Npp8u), cudaMemcpyDeviceToHost);
    
    // Verify results
    for (int i = 0; i < width * height; i++) {
        EXPECT_NEAR(resultData[i], expectedData[i], 1) 
            << "Mismatch at pixel " << i;
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test 3-channel RGB to grayscale conversion (32-bit float)
TEST_F(RGBToGrayFunctionalTest, RGBToGray_32f_C3C1R_BasicOperation) {
    std::vector<Npp32f> srcData(width * height * 3);
    std::vector<Npp32f> expectedData(width * height);
    
    // Create test RGB image with float values
    for (int i = 0; i < width * height; i++) {
        Npp32f r = 0.8f;
        Npp32f g = 0.6f;
        Npp32f b = 0.4f;
        
        srcData[i * 3 + 0] = r;
        srcData[i * 3 + 1] = g;
        srcData[i * 3 + 2] = b;
        
        expectedData[i] = calculateGrayValue(r, g, b);
    }
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C3(width, height, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C1(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU row by row to handle step properly
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep, 
                   srcData.data() + y * width * 3, 
                   width * 3 * sizeof(Npp32f), 
                   cudaMemcpyHostToDevice);
    }
    
    // Execute NPP function
    NppStatus status = nppiRGBToGray_32f_C3C1R(d_src, srcStep, d_dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host
    std::vector<Npp32f> resultData(width * height);
    cudaMemcpy(resultData.data(), d_dst, width * height * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    
    // Verify results with floating-point precision
    for (int i = 0; i < width * height; i++) {
        EXPECT_NEAR(resultData[i], expectedData[i], 0.001f) 
            << "Mismatch at pixel " << i;
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test primary colors conversion
TEST_F(RGBToGrayFunctionalTest, RGBToGray_PrimaryColors) {
    const int testSize = 3;
    NppiSize testRoi = {testSize, 1};
    
    // Test pure red, green, blue
    std::vector<Npp8u> srcData = {
        255, 0, 0,   // Pure red
        0, 255, 0,   // Pure green  
        0, 0, 255    // Pure blue
    };
    
    // Calculate expected values using standard weights
    std::vector<Npp8u> expectedData = {
        calculateGrayValue8u(255, 0, 0),   // Red -> ~76
        calculateGrayValue8u(0, 255, 0),   // Green -> ~150  
        calculateGrayValue8u(0, 0, 255)    // Blue -> ~29
    };
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C3(testSize, 1, &srcStep);
    Npp8u* d_dst = nppiMalloc_8u_C1(testSize, 1, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU
    cudaMemcpy(d_src, srcData.data(), testSize * 3 * sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    // Execute NPP function
    NppStatus status = nppiRGBToGray_8u_C3C1R(d_src, srcStep, d_dst, dstStep, testRoi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host
    std::vector<Npp8u> resultData(testSize);
    cudaMemcpy(resultData.data(), d_dst, testSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);
    
    // Verify primary color conversion
    for (int i = 0; i < testSize; i++) {
        EXPECT_NEAR(resultData[i], expectedData[i], 1) 
            << "Primary color " << i << " conversion failed";
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test error handling
TEST_F(RGBToGrayFunctionalTest, RGBToGray_ErrorHandling) {
    // Test null pointer
    NppStatus status = nppiRGBToGray_8u_C3C1R(nullptr, 32, nullptr, 16, roi);
    EXPECT_NE(status, NPP_SUCCESS);
    
    // Test invalid ROI
    NppiSize invalidRoi = {0, 0};
    status = nppiRGBToGray_8u_C3C1R(nullptr, 32, nullptr, 16, invalidRoi);
    EXPECT_NE(status, NPP_SUCCESS);
    
    // Test invalid step
    status = nppiRGBToGray_8u_C3C1R(nullptr, 0, nullptr, 16, roi);
    EXPECT_NE(status, NPP_SUCCESS);
}

// Test stream context version
TEST_F(RGBToGrayFunctionalTest, RGBToGray_StreamContext) {
    std::vector<Npp8u> srcData(width * height * 3, 128); // Gray color
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C3(width, height, &srcStep);
    Npp8u* d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU row by row to handle step properly
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep, 
                   srcData.data() + y * width * 3, 
                   width * 3 * sizeof(Npp8u), 
                   cudaMemcpyHostToDevice);
    }
    
    // Create stream context
    NppStreamContext nppStreamCtx = {0};
    
    // Execute NPP function with context
    NppStatus status = nppiRGBToGray_8u_C3C1R_Ctx(d_src, srcStep, d_dst, dstStep, roi, nppStreamCtx);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Verify result
    std::vector<Npp8u> resultData(width * height);
    cudaMemcpy(resultData.data(), d_dst, width * height * sizeof(Npp8u), cudaMemcpyDeviceToHost);
    
    Npp8u expectedGray = calculateGrayValue8u(128, 128, 128);
    for (int i = 0; i < width * height; i++) {
        EXPECT_NEAR(resultData[i], expectedGray, 1);
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}