#include <gtest/gtest.h>
#include "npp.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>

class PowFunctionalTest : public ::testing::Test {
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

// Test 8-bit unsigned power with scaling
TEST_F(PowFunctionalTest, Pow_8u_C1RSfs_BasicOperation) {
    std::vector<Npp8u> srcData(width * height);
    std::vector<Npp8u> expectedData(width * height);
    
    // Create test data: small values that won't overflow when powered
    for (int i = 0; i < width * height; i++) {
        Npp8u src_val = (Npp8u)((i % 6) + 1); // Values 1-6
        srcData[i] = src_val;
        
        // Expected: square with no scaling
        int result = src_val * src_val; // pow(src, 2)
        expectedData[i] = (Npp8u)std::min(result, 255);
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
    
    // Execute NPP function (power of 2, no scaling)
    int nPower = 2;
    int nScaleFactor = 0;
    NppStatus status = nppiPow_8u_C1RSfs(d_src, srcStep, d_dst, dstStep, roi, nPower, nScaleFactor);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host row by row
    std::vector<Npp8u> resultData(width * height);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width,
                   (char*)d_dst + y * dstStep,
                   width * sizeof(Npp8u),
                   cudaMemcpyDeviceToHost);
    }
    
    // Verify results
    for (int i = 0; i < width * height; i++) {
        EXPECT_EQ(resultData[i], expectedData[i]) 
            << "Mismatch at pixel " << i << ": got " << (int)resultData[i] 
            << ", expected " << (int)expectedData[i]
            << ", src=" << (int)srcData[i];
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test 32-bit float power
TEST_F(PowFunctionalTest, Pow_32f_C1R_BasicOperation) {
    std::vector<Npp32f> srcData(width * height);
    std::vector<Npp32f> expectedData(width * height);
    
    // Create test data: various float values
    for (int i = 0; i < width * height; i++) {
        Npp32f src_val = 1.0f + 0.1f * (i % 20); // Values from 1.0 to 3.0
        srcData[i] = src_val;
        expectedData[i] = std::pow(src_val, 2.5f); // Power of 2.5
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
    Npp32f nPower = 2.5f;
    NppStatus status = nppiPow_32f_C1R(d_src, srcStep, d_dst, dstStep, roi, nPower);
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
        EXPECT_NEAR(resultData[i], expectedData[i], expectedData[i] * 1e-5f) 
            << "Mismatch at pixel " << i << ": got " << resultData[i] 
            << ", expected " << expectedData[i]
            << ", src=" << srcData[i];
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test special values for 32-bit float
TEST_F(PowFunctionalTest, Pow_32f_C1R_SpecialValues) {
    const int testSize = 8;
    NppiSize testRoi = {testSize, 1};
    
    // Test special values and powers
    std::vector<Npp32f> srcData = {1.0f, 2.0f, 4.0f, 0.0f, -2.0f, 0.5f, 10.0f, -1.0f};
    std::vector<Npp32f> expectedData(testSize);
    float nPower = 3.0f; // Power of 3
    
    // Calculate expected values
    expectedData[0] = 1.0f;          // 1^3 = 1
    expectedData[1] = 8.0f;          // 2^3 = 8
    expectedData[2] = 64.0f;         // 4^3 = 64
    expectedData[3] = 0.0f;          // 0^3 = 0
    expectedData[4] = -8.0f;         // (-2)^3 = -8
    expectedData[5] = 0.125f;        // 0.5^3 = 0.125
    expectedData[6] = 1000.0f;       // 10^3 = 1000
    expectedData[7] = -1.0f;         // (-1)^3 = -1
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C1(testSize, 1, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C1(testSize, 1, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU
    cudaMemcpy(d_src, srcData.data(), testSize * sizeof(Npp32f), cudaMemcpyHostToDevice);
    
    // Execute NPP function
    NppStatus status = nppiPow_32f_C1R(d_src, srcStep, d_dst, dstStep, testRoi, nPower);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host
    std::vector<Npp32f> resultData(testSize);
    cudaMemcpy(resultData.data(), d_dst, testSize * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    
    // Verify results
    for (int i = 0; i < testSize; i++) {
        EXPECT_NEAR(resultData[i], expectedData[i], std::abs(expectedData[i]) * 1e-5f + 1e-6f) 
            << "Special value test failed at index " << i
            << ", src=" << srcData[i];
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test power of 0 (should always be 1)
TEST_F(PowFunctionalTest, Pow_32f_C1R_PowerZero) {
    const int testSize = 5;
    NppiSize testRoi = {testSize, 1};
    
    // Test various values with power 0
    std::vector<Npp32f> srcData = {1.0f, 2.0f, -3.0f, 0.5f, 100.0f};
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C1(testSize, 1, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C1(testSize, 1, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU
    cudaMemcpy(d_src, srcData.data(), testSize * sizeof(Npp32f), cudaMemcpyHostToDevice);
    
    // Execute NPP function with power 0
    Npp32f nPower = 0.0f;
    NppStatus status = nppiPow_32f_C1R(d_src, srcStep, d_dst, dstStep, testRoi, nPower);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host
    std::vector<Npp32f> resultData(testSize);
    cudaMemcpy(resultData.data(), d_dst, testSize * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    
    // All results should be 1.0 (x^0 = 1)
    for (int i = 0; i < testSize; i++) {
        EXPECT_FLOAT_EQ(resultData[i], 1.0f)
            << "Power zero test failed at index " << i << ", src=" << srcData[i];
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test in-place operation
TEST_F(PowFunctionalTest, Pow_32f_C1IR_InPlace) {
    std::vector<Npp32f> data(width * height);
    std::vector<Npp32f> expectedData(width * height);
    
    // Create test data: simple square root operation
    for (int i = 0; i < width * height; i++) {
        Npp32f src_val = 1.0f + (i % 16); // Values from 1 to 16
        data[i] = src_val;
        expectedData[i] = std::sqrt(src_val); // Power of 0.5 (square root)
    }
    
    // Allocate GPU memory
    int step;
    Npp32f* d_data = nppiMalloc_32f_C1(width, height, &step);
    ASSERT_NE(d_data, nullptr);
    
    // Copy input data to GPU row by row
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_data + y * step,
                   data.data() + y * width,
                   width * sizeof(Npp32f),
                   cudaMemcpyHostToDevice);
    }
    
    // Execute NPP function (in-place, power 0.5)
    Npp32f nPower = 0.5f;
    NppStatus status = nppiPow_32f_C1IR(d_data, step, roi, nPower);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host row by row
    std::vector<Npp32f> resultData(width * height);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width,
                   (char*)d_data + y * step,
                   width * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    // Verify results
    for (int i = 0; i < width * height; i++) {
        EXPECT_NEAR(resultData[i], expectedData[i], expectedData[i] * 1e-5f + 1e-6f) 
            << "In-place operation failed at pixel " << i;
    }
    
    nppiFree(d_data);
}

// Test error handling
TEST_F(PowFunctionalTest, Pow_ErrorHandling) {
    // Test null pointer
    NppStatus status = nppiPow_32f_C1R(nullptr, 32, nullptr, 16, roi, 2.0f);
    EXPECT_NE(status, NPP_SUCCESS);
    
    // Test invalid ROI
    NppiSize invalidRoi = {0, 0};
    status = nppiPow_32f_C1R(nullptr, 32, nullptr, 16, invalidRoi, 2.0f);
    EXPECT_NE(status, NPP_SUCCESS);
    
    // Test invalid step
    status = nppiPow_32f_C1R(nullptr, 0, nullptr, 16, roi, 2.0f);
    EXPECT_NE(status, NPP_SUCCESS);
    
    // Test negative scale factor for integer version
    status = nppiPow_8u_C1RSfs(nullptr, 32, nullptr, 16, roi, 2, -1);
    EXPECT_NE(status, NPP_SUCCESS);
}

// Test stream context version
TEST_F(PowFunctionalTest, Pow_StreamContext) {
    std::vector<Npp32f> srcData(width * height, 2.0f); // All pixels = 2
    
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
    
    // Execute NPP function with context (power 3)
    Npp32f nPower = 3.0f;
    NppStatus status = nppiPow_32f_C1R_Ctx(d_src, srcStep, d_dst, dstStep, roi, nPower, nppStreamCtx);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Verify result - copy row by row
    std::vector<Npp32f> resultData(width * height);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width,
                   (char*)d_dst + y * dstStep,
                   width * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    // All results should be 8.0 (2^3 = 8)
    for (int i = 0; i < width * height; i++) {
        EXPECT_FLOAT_EQ(resultData[i], 8.0f)
            << "Stream context test failed at pixel " << i;
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test with scaling for 16-bit signed
TEST_F(PowFunctionalTest, Pow_16s_C1RSfs_WithScaling) {
    std::vector<Npp16s> srcData(width * height);
    std::vector<Npp16s> expectedData(width * height);
    
    // Create test data
    for (int i = 0; i < width * height; i++) {
        Npp16s src_val = (Npp16s)((i % 4) + 2); // Values 2-5
        srcData[i] = src_val;
        
        // Expected with scaling factor 1 (multiply by 2)
        float result = std::pow((float)src_val, 2.0f) * 2.0f;
        expectedData[i] = (Npp16s)std::min(std::max(result, -32768.0f), 32767.0f);
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
    
    // Execute NPP function with scaling
    int nPower = 2;
    int nScaleFactor = 1; // Multiply result by 2
    NppStatus status = nppiPow_16s_C1RSfs(d_src, srcStep, d_dst, dstStep, roi, nPower, nScaleFactor);
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
            << ", expected " << expectedData[i]
            << ", src=" << srcData[i];
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}