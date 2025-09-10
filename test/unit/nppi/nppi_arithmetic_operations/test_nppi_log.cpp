#include <gtest/gtest.h>
#include "npp.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>

class LogFunctionalTest : public ::testing::Test {
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

// Test basic 32-bit float common logarithm
TEST_F(LogFunctionalTest, Log_32f_C1R_BasicOperation) {
    std::vector<Npp32f> srcData(width * height);
    std::vector<Npp32f> expectedData(width * height);
    
    // Create test data: positive values for valid logarithm
    for (int i = 0; i < width * height; i++) {
        Npp32f src_val = 1.0f + 0.5f * (i % 20); // Values from 1.0 to 10.5
        srcData[i] = src_val;
        expectedData[i] = log10f(src_val); // Base-10 logarithm
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
    NppStatus status = nppiLog_32f_C1R(d_src, srcStep, d_dst, dstStep, roi);
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
        EXPECT_NEAR(resultData[i], expectedData[i], std::abs(expectedData[i]) * 1e-6f + 1e-7f) 
            << "Mismatch at pixel " << i << ": got " << resultData[i] 
            << ", expected " << expectedData[i]
            << ", src=" << srcData[i];
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test special values for 32-bit float logarithm
TEST_F(LogFunctionalTest, Log_32f_C1R_SpecialValues) {
    const int testSize = 8;
    NppiSize testRoi = {testSize, 1};
    
    // Test special values
    std::vector<Npp32f> srcData = {1.0f, 10.0f, 100.0f, 0.1f, 0.01f, 2.0f, 5.0f, 1000.0f};
    std::vector<Npp32f> expectedData(testSize);
    
    // Calculate expected values
    expectedData[0] = 0.0f;        // log10(1) = 0
    expectedData[1] = 1.0f;        // log10(10) = 1
    expectedData[2] = 2.0f;        // log10(100) = 2
    expectedData[3] = -1.0f;       // log10(0.1) = -1
    expectedData[4] = -2.0f;       // log10(0.01) = -2
    expectedData[5] = log10f(2.0f);     // log10(2) ≈ 0.30103
    expectedData[6] = log10f(5.0f);     // log10(5) ≈ 0.69897
    expectedData[7] = 3.0f;        // log10(1000) = 3
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C1(testSize, 1, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C1(testSize, 1, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU
    cudaMemcpy(d_src, srcData.data(), testSize * sizeof(Npp32f), cudaMemcpyHostToDevice);
    
    // Execute NPP function
    NppStatus status = nppiLog_32f_C1R(d_src, srcStep, d_dst, dstStep, testRoi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host
    std::vector<Npp32f> resultData(testSize);
    cudaMemcpy(resultData.data(), d_dst, testSize * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    
    // Verify results
    for (int i = 0; i < testSize; i++) {
        if (i < 5 || i == 7) {
            // Exact values for powers of 10
            EXPECT_FLOAT_EQ(resultData[i], expectedData[i]) 
                << "Special value test failed at index " << i
                << ", src=" << srcData[i];
        } else {
            // Approximate values for other numbers
            EXPECT_NEAR(resultData[i], expectedData[i], 1e-6f) 
                << "Special value test failed at index " << i
                << ", src=" << srcData[i];
        }
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test invalid values (negative and zero)
TEST_F(LogFunctionalTest, Log_32f_C1R_InvalidValues) {
    const int testSize = 5;
    NppiSize testRoi = {testSize, 1};
    
    // Test invalid values
    std::vector<Npp32f> srcData = {0.0f, -1.0f, -10.0f, -0.5f, -100.0f};
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C1(testSize, 1, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C1(testSize, 1, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU
    cudaMemcpy(d_src, srcData.data(), testSize * sizeof(Npp32f), cudaMemcpyHostToDevice);
    
    // Execute NPP function
    NppStatus status = nppiLog_32f_C1R(d_src, srcStep, d_dst, dstStep, testRoi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host
    std::vector<Npp32f> resultData(testSize);
    cudaMemcpy(resultData.data(), d_dst, testSize * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    
    // All results should be negative infinity for non-positive inputs
    for (int i = 0; i < testSize; i++) {
        EXPECT_TRUE(std::isinf(resultData[i]) && resultData[i] < 0.0f)
            << "Invalid value test failed at index " << i << ", src=" << srcData[i]
            << ", result=" << resultData[i];
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test in-place operation
TEST_F(LogFunctionalTest, Log_32f_C1IR_InPlace) {
    std::vector<Npp32f> data(width * height);
    std::vector<Npp32f> expectedData(width * height);
    
    // Create test data: powers of 10 and other values
    for (int i = 0; i < width * height; i++) {
        Npp32f src_val;
        if (i % 4 == 0) {
            src_val = powf(10.0f, (i % 8) - 2);  // 10^(-2) to 10^5
        } else {
            src_val = 1.0f + (i % 10);  // 1 to 10
        }
        data[i] = src_val;
        expectedData[i] = log10f(src_val);
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
    
    // Execute NPP function (in-place)
    NppStatus status = nppiLog_32f_C1IR(d_data, step, roi);
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
        EXPECT_NEAR(resultData[i], expectedData[i], std::abs(expectedData[i]) * 1e-6f + 1e-7f) 
            << "In-place operation failed at pixel " << i;
    }
    
    nppiFree(d_data);
}

// Test error handling
TEST_F(LogFunctionalTest, Log_ErrorHandling) {
    // Test null pointer
    NppStatus status = nppiLog_32f_C1R(nullptr, 32, nullptr, 16, roi);
    EXPECT_NE(status, NPP_SUCCESS);
    
    // Test invalid ROI
    NppiSize invalidRoi = {0, 0};
    status = nppiLog_32f_C1R(nullptr, 32, nullptr, 16, invalidRoi);
    EXPECT_NE(status, NPP_SUCCESS);
    
    // Test invalid step
    status = nppiLog_32f_C1R(nullptr, 0, nullptr, 16, roi);
    EXPECT_NE(status, NPP_SUCCESS);
}

// Test stream context version
TEST_F(LogFunctionalTest, Log_StreamContext) {
    const int testSize = 4;
    NppiSize testRoi = {testSize, 1};
    
    // Simple test data
    std::vector<Npp32f> srcData = {1.0f, 10.0f, 100.0f, 1000.0f};
    std::vector<Npp32f> expectedData = {0.0f, 1.0f, 2.0f, 3.0f};  // log10 values
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C1(testSize, 1, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C1(testSize, 1, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU
    cudaMemcpy(d_src, srcData.data(), testSize * sizeof(Npp32f), cudaMemcpyHostToDevice);
    
    // Create stream context
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    
    // Execute NPP function with context
    NppStatus status = nppiLog_32f_C1R_Ctx(d_src, srcStep, d_dst, dstStep, testRoi, nppStreamCtx);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host
    std::vector<Npp32f> resultData(testSize);
    cudaMemcpy(resultData.data(), d_dst, testSize * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    
    // Verify results
    for (int i = 0; i < testSize; i++) {
        EXPECT_FLOAT_EQ(resultData[i], expectedData[i])
            << "Stream context test failed at pixel " << i;
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test very small positive values
TEST_F(LogFunctionalTest, Log_32f_C1R_SmallValues) {
    const int testSize = 6;
    NppiSize testRoi = {testSize, 1};
    
    // Test very small positive values
    std::vector<Npp32f> srcData = {1e-6f, 1e-3f, 1e-1f, 1e-9f, 1e-12f, 1e-15f};
    std::vector<Npp32f> expectedData(testSize);
    
    // Calculate expected values
    for (int i = 0; i < testSize; i++) {
        expectedData[i] = log10f(srcData[i]);
    }
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C1(testSize, 1, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C1(testSize, 1, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU
    cudaMemcpy(d_src, srcData.data(), testSize * sizeof(Npp32f), cudaMemcpyHostToDevice);
    
    // Execute NPP function
    NppStatus status = nppiLog_32f_C1R(d_src, srcStep, d_dst, dstStep, testRoi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host
    std::vector<Npp32f> resultData(testSize);
    cudaMemcpy(resultData.data(), d_dst, testSize * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    
    // Verify results
    for (int i = 0; i < testSize; i++) {
        EXPECT_NEAR(resultData[i], expectedData[i], std::abs(expectedData[i]) * 1e-6f + 1e-6f) 
            << "Small value test failed at index " << i
            << ", src=" << srcData[i] << ", expected=" << expectedData[i];
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}