#include <gtest/gtest.h>
#include "npp.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "../../../common/npp_test_utils.h"

class SqrFunctionalTest : public ::testing::Test {
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

// Test 8-bit unsigned square with scaling
TEST_F(SqrFunctionalTest, Sqr_8u_C1RSfs_BasicOperation) {
    std::vector<Npp8u> srcData(width * height);
    std::vector<Npp8u> expectedData(width * height);
    
    // Create test data: values 0-15 repeated
    for (int i = 0; i < width * height; i++) {
        Npp8u src_val = (Npp8u)(i % 16);
        srcData[i] = src_val;
        
        // Expected: square with scaling factor 0 (no scaling)
        expectedData[i] = (Npp8u)std::min((int)(src_val * src_val), 255);
    }
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C1(width, height, &srcStep);
    Npp8u* d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU row by row
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width,
                   width * sizeof(Npp8u),
                   cudaMemcpyHostToDevice);
    }
    
    // Execute NPP function
    int nScaleFactor = 0; // No scaling
    NppStatus status = nppiSqr_8u_C1RSfs(d_src, srcStep, d_dst, dstStep, roi, nScaleFactor);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host row by row
    std::vector<Npp8u> resultData(width * height);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width,
                   (char*)d_dst + y * dstStep,
                   width * sizeof(Npp8u),
                   cudaMemcpyDeviceToHost);
    }
    
    // Verify results using precision control system
    for (int i = 0; i < width * height; i++) {
        NPP_EXPECT_ARITHMETIC_EQUAL(resultData[i], expectedData[i], "nppiSqr_8u_C1RSfs")
            << "Mismatch at pixel " << i << ": got " << (int)resultData[i] 
            << ", expected " << (int)expectedData[i];
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test 8-bit unsigned square with scaling factor
TEST_F(SqrFunctionalTest, Sqr_8u_C1RSfs_WithScaling) {
    std::vector<Npp8u> srcData(width * height);
    std::vector<Npp8u> expectedData(width * height);
    
    // Create test data: values that will benefit from scaling
    for (int i = 0; i < width * height; i++) {
        Npp8u src_val = (Npp8u)((i % 32) + 8); // Values 8-39
        srcData[i] = src_val;
        
        // Expected: square with scaling factor 4 (divide by 16)
        expectedData[i] = (Npp8u)std::min((int)(src_val * src_val) >> 4, 255);
    }
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp8u* d_src = nppiMalloc_8u_C1(width, height, &srcStep);
    Npp8u* d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU row by row
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width,
                   width * sizeof(Npp8u),
                   cudaMemcpyHostToDevice);
    }
    
    // Execute NPP function with scaling
    int nScaleFactor = 4; // Divide result by 16
    NppStatus status = nppiSqr_8u_C1RSfs(d_src, srcStep, d_dst, dstStep, roi, nScaleFactor);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host row by row
    std::vector<Npp8u> resultData(width * height);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width,
                   (char*)d_dst + y * dstStep,
                   width * sizeof(Npp8u),
                   cudaMemcpyDeviceToHost);
    }
    
    // Verify results using precision control system
    for (int i = 0; i < width * height; i++) {
        NPP_EXPECT_ARITHMETIC_EQUAL(resultData[i], expectedData[i], "nppiSqr_8u_C1RSfs")
            << "Mismatch at pixel " << i << ": got " << (int)resultData[i] 
            << ", expected " << (int)expectedData[i];
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test 32-bit float square
TEST_F(SqrFunctionalTest, Sqr_32f_C1R_BasicOperation) {
    std::vector<Npp32f> srcData(width * height);
    std::vector<Npp32f> expectedData(width * height);
    
    // Create test data: various float values
    for (int i = 0; i < width * height; i++) {
        Npp32f src_val = 0.1f * (i % 20) - 1.0f; // Values from -1.0 to 0.9
        srcData[i] = src_val;
        expectedData[i] = src_val * src_val;
    }
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C1(width, height, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C1(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU row by row
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width,
                   width * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
    }
    
    // Execute NPP function
    NppStatus status = nppiSqr_32f_C1R(d_src, srcStep, d_dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host row by row
    std::vector<Npp32f> resultData(width * height);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width,
                   (char*)d_dst + y * dstStep,
                   width * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    // Verify results with floating-point precision
    for (int i = 0; i < width * height; i++) {
        EXPECT_NEAR(resultData[i], expectedData[i], 1e-6f) 
            << "Mismatch at pixel " << i << ": got " << resultData[i] 
            << ", expected " << expectedData[i];
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test 16-bit signed square with scaling
TEST_F(SqrFunctionalTest, Sqr_16s_C1RSfs_BasicOperation) {
    std::vector<Npp16s> srcData(width * height);
    std::vector<Npp16s> expectedData(width * height);
    
    // Create test data: positive and negative values
    for (int i = 0; i < width * height; i++) {
        Npp16s src_val = (Npp16s)((i % 21) - 10); // Values from -10 to 10
        srcData[i] = src_val;
        
        // Expected: square with scaling factor 0 (no scaling)
        int result = src_val * src_val;
        expectedData[i] = (Npp16s)std::max(std::min(result, 32767), -32768);
    }
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp16s* d_src = nppiMalloc_16s_C1(width, height, &srcStep);
    Npp16s* d_dst = nppiMalloc_16s_C1(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU row by row
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width,
                   width * sizeof(Npp16s),
                   cudaMemcpyHostToDevice);
    }
    
    // Execute NPP function
    int nScaleFactor = 0; // No scaling
    NppStatus status = nppiSqr_16s_C1RSfs(d_src, srcStep, d_dst, dstStep, roi, nScaleFactor);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host row by row
    std::vector<Npp16s> resultData(width * height);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width,
                   (char*)d_dst + y * dstStep,
                   width * sizeof(Npp16s),
                   cudaMemcpyDeviceToHost);
    }
    
    // Verify results
    for (int i = 0; i < width * height; i++) {
        EXPECT_EQ(resultData[i], expectedData[i]) 
            << "Mismatch at pixel " << i << ": got " << resultData[i] 
            << ", expected " << expectedData[i];
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test special values for 32-bit float
TEST_F(SqrFunctionalTest, Sqr_32f_C1R_SpecialValues) {
    const int testSize = 6;
    NppiSize testRoi = {testSize, 1};
    
    // Test special values: zero, positive, negative, small values
    std::vector<Npp32f> srcData = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f};
    std::vector<Npp32f> expectedData = {0.0f, 1.0f, 1.0f, 4.0f, 4.0f, 0.25f};
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C1(testSize, 1, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C1(testSize, 1, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU
    cudaMemcpy(d_src, srcData.data(), testSize * sizeof(Npp32f), cudaMemcpyHostToDevice);
    
    // Execute NPP function
    NppStatus status = nppiSqr_32f_C1R(d_src, srcStep, d_dst, dstStep, testRoi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host
    std::vector<Npp32f> resultData(testSize);
    cudaMemcpy(resultData.data(), d_dst, testSize * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    
    // Verify results
    for (int i = 0; i < testSize; i++) {
        EXPECT_NEAR(resultData[i], expectedData[i], 1e-6f) 
            << "Special value test failed at index " << i;
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test error handling
TEST_F(SqrFunctionalTest, Sqr_ErrorHandling) {
    // Test null pointer
    NppStatus status = nppiSqr_32f_C1R(nullptr, 32, nullptr, 16, roi);
    EXPECT_NE(status, NPP_SUCCESS);
    
    // Test invalid ROI
    NppiSize invalidRoi = {0, 0};
    status = nppiSqr_32f_C1R(nullptr, 32, nullptr, 16, invalidRoi);
    EXPECT_NE(status, NPP_SUCCESS);
    
    // Test invalid step
    status = nppiSqr_32f_C1R(nullptr, 0, nullptr, 16, roi);
    EXPECT_NE(status, NPP_SUCCESS);
    
    // Test negative scale factor
    status = nppiSqr_8u_C1RSfs(nullptr, 32, nullptr, 16, roi, -1);
    EXPECT_NE(status, NPP_SUCCESS);
}

// Test stream context version
TEST_F(SqrFunctionalTest, Sqr_StreamContext) {
    std::vector<Npp32f> srcData(width * height, 3.0f); // All pixels = 3.0
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C1(width, height, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C1(width, height, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU row by row
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width,
                   width * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
    }
    
    // Create stream context
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    
    // Execute NPP function with context
    NppStatus status = nppiSqr_32f_C1R_Ctx(d_src, srcStep, d_dst, dstStep, roi, nppStreamCtx);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Verify result - copy row by row
    std::vector<Npp32f> resultData(width * height);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width,
                   (char*)d_dst + y * dstStep,
                   width * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    // All results should be 9.0 (3.0 * 3.0)
    for (int i = 0; i < width * height; i++) {
        EXPECT_NEAR(resultData[i], 9.0f, 1e-6f);
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}