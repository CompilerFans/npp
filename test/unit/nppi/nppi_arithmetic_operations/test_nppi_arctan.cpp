#include <gtest/gtest.h>
#include "npp.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>

class ArcTanFunctionalTest : public ::testing::Test {
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

// Test basic 32-bit float arctangent
TEST_F(ArcTanFunctionalTest, ArcTan_32f_C1R_BasicOperation) {
    std::vector<Npp32f> srcData(width * height);
    std::vector<Npp32f> expectedData(width * height);
    
    // Create test data: various values for arctangent
    for (int i = 0; i < width * height; i++) {
        // Values from -5 to 5 with step 0.1
        Npp32f src_val = -5.0f + 10.0f * (i % 100) / 99.0f;
        srcData[i] = src_val;
        expectedData[i] = atanf(src_val); // Arctangent
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
    NppStatus status = nppiArcTan_32f_C1R(d_src, srcStep, d_dst, dstStep, roi);
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

// Test special values for 32-bit float arctangent
TEST_F(ArcTanFunctionalTest, ArcTan_32f_C1R_SpecialValues) {
    const int testSize = 10;
    NppiSize testRoi = {testSize, 1};
    
    // Test special values
    std::vector<Npp32f> srcData = {0.0f, 1.0f, -1.0f, 
                                   1000.0f, -1000.0f, // Large values
                                   0.577350269f, -0.577350269f, // tan(π/6) and tan(-π/6)
                                   1.732050808f, -1.732050808f, // tan(π/3) and tan(-π/3)  
                                   INFINITY}; // Positive infinity
    std::vector<Npp32f> expectedData(testSize);
    
    // Calculate expected values
    expectedData[0] = 0.0f;           // arctan(0) = 0
    expectedData[1] = M_PI / 4.0f;    // arctan(1) = π/4
    expectedData[2] = -M_PI / 4.0f;   // arctan(-1) = -π/4
    expectedData[3] = atanf(1000.0f); // arctan(large) ≈ π/2
    expectedData[4] = atanf(-1000.0f);// arctan(-large) ≈ -π/2
    expectedData[5] = atanf(0.577350269f); // arctan(tan(π/6)) ≈ π/6
    expectedData[6] = atanf(-0.577350269f);// arctan(tan(-π/6)) ≈ -π/6
    expectedData[7] = atanf(1.732050808f); // arctan(tan(π/3)) ≈ π/3
    expectedData[8] = atanf(-1.732050808f);// arctan(tan(-π/3)) ≈ -π/3
    expectedData[9] = M_PI / 2.0f;    // arctan(∞) = π/2
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C1(testSize, 1, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C1(testSize, 1, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU
    cudaMemcpy(d_src, srcData.data(), testSize * sizeof(Npp32f), cudaMemcpyHostToDevice);
    
    // Execute NPP function
    NppStatus status = nppiArcTan_32f_C1R(d_src, srcStep, d_dst, dstStep, testRoi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host
    std::vector<Npp32f> resultData(testSize);
    cudaMemcpy(resultData.data(), d_dst, testSize * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    
    // Verify results
    for (int i = 0; i < testSize; i++) {
        if (i == 0 || i == 1 || i == 2 || i == 9) {
            // Exact values for special cases
            EXPECT_NEAR(resultData[i], expectedData[i], 1e-6f) 
                << "Special value test failed at index " << i
                << ", src=" << srcData[i] << ", got=" << resultData[i] 
                << ", expected=" << expectedData[i];
        } else {
            // Approximate values for other numbers
            EXPECT_NEAR(resultData[i], expectedData[i], 1e-5f) 
                << "Special value test failed at index " << i
                << ", src=" << srcData[i];
        }
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test negative infinity
TEST_F(ArcTanFunctionalTest, ArcTan_32f_C1R_NegativeInfinity) {
    const int testSize = 1;
    NppiSize testRoi = {testSize, 1};
    
    // Test negative infinity
    std::vector<Npp32f> srcData = {-INFINITY};
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C1(testSize, 1, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C1(testSize, 1, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU
    cudaMemcpy(d_src, srcData.data(), testSize * sizeof(Npp32f), cudaMemcpyHostToDevice);
    
    // Execute NPP function
    NppStatus status = nppiArcTan_32f_C1R(d_src, srcStep, d_dst, dstStep, testRoi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host
    std::vector<Npp32f> resultData(testSize);
    cudaMemcpy(resultData.data(), d_dst, testSize * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    
    // Should be -π/2 for negative infinity
    EXPECT_NEAR(resultData[0], -M_PI / 2.0f, 1e-6f)
        << "Negative infinity test failed: got " << resultData[0] 
        << ", expected " << -M_PI / 2.0f;
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test in-place operation
TEST_F(ArcTanFunctionalTest, ArcTan_32f_C1IR_InPlace) {
    std::vector<Npp32f> data(width * height);
    std::vector<Npp32f> expectedData(width * height);
    
    // Create test data: values from -2 to 2
    for (int i = 0; i < width * height; i++) {
        Npp32f src_val = -2.0f + 4.0f * (i % 50) / 49.0f; // -2 to 2
        data[i] = src_val;
        expectedData[i] = atanf(src_val);
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
    NppStatus status = nppiArcTan_32f_C1IR(d_data, step, roi);
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
TEST_F(ArcTanFunctionalTest, ArcTan_ErrorHandling) {
    // Test null pointer
    NppStatus status = nppiArcTan_32f_C1R(nullptr, 32, nullptr, 16, roi);
    EXPECT_NE(status, NPP_SUCCESS);
    
    // Test invalid ROI
    NppiSize invalidRoi = {0, 0};
    status = nppiArcTan_32f_C1R(nullptr, 32, nullptr, 16, invalidRoi);
    EXPECT_NE(status, NPP_SUCCESS);
    
    // Test invalid step
    status = nppiArcTan_32f_C1R(nullptr, 0, nullptr, 16, roi);
    EXPECT_NE(status, NPP_SUCCESS);
}

// Test stream context version
TEST_F(ArcTanFunctionalTest, ArcTan_StreamContext) {
    const int testSize = 4;
    NppiSize testRoi = {testSize, 1};
    
    // Simple test data: special angles
    std::vector<Npp32f> srcData = {0.0f, 1.0f, -1.0f, 0.577350269f}; // 0, 1, -1, tan(π/6)
    std::vector<Npp32f> expectedData = {0.0f, M_PI/4.0f, -M_PI/4.0f, atanf(0.577350269f)};
    
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
    NppStatus status = nppiArcTan_32f_C1R_Ctx(d_src, srcStep, d_dst, dstStep, testRoi, nppStreamCtx);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host
    std::vector<Npp32f> resultData(testSize);
    cudaMemcpy(resultData.data(), d_dst, testSize * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    
    // Verify results
    for (int i = 0; i < testSize; i++) {
        EXPECT_NEAR(resultData[i], expectedData[i], 1e-6f)
            << "Stream context test failed at pixel " << i;
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test output range: arctangent is bounded to (-π/2, π/2)
TEST_F(ArcTanFunctionalTest, ArcTan_32f_C1R_OutputRange) {
    const int testSize = 8;
    NppiSize testRoi = {testSize, 1};
    
    // Test very large positive and negative values
    std::vector<Npp32f> srcData = {1e6f, -1e6f, 1e9f, -1e9f, 1e12f, -1e12f, INFINITY, -INFINITY};
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C1(testSize, 1, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C1(testSize, 1, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU
    cudaMemcpy(d_src, srcData.data(), testSize * sizeof(Npp32f), cudaMemcpyHostToDevice);
    
    // Execute NPP function
    NppStatus status = nppiArcTan_32f_C1R(d_src, srcStep, d_dst, dstStep, testRoi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host
    std::vector<Npp32f> resultData(testSize);
    cudaMemcpy(resultData.data(), d_dst, testSize * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    
    // Verify all results are in range (-π/2, π/2)
    for (int i = 0; i < testSize; i++) {
        EXPECT_GT(resultData[i], -M_PI / 2.0f - 1e-6f)
            << "Output range test failed at index " << i 
            << ": result " << resultData[i] << " is too small";
        EXPECT_LT(resultData[i], M_PI / 2.0f + 1e-6f)
            << "Output range test failed at index " << i 
            << ": result " << resultData[i] << " is too large";
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}