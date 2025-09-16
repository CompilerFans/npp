#include <gtest/gtest.h>
#include "npp.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "../../../common/npp_test_utils.h"

class ExpFunctionalTest : public ::testing::Test {
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

// Test 8-bit unsigned exponential with scaling
TEST_F(ExpFunctionalTest, Exp_8u_C1RSfs_BasicOperation) {
    std::vector<Npp8u> srcData(width * height);
    std::vector<Npp8u> expectedData(width * height);
    
    // Create test data: values from 0 to 5 (based on NVIDIA NPP testing)
    for (int i = 0; i < width * height; i++) {
        Npp8u src_val = (Npp8u)(i % 6); // Values 0-5
        srcData[i] = src_val;
        
        // Expected values based on NVIDIA NPP actual behavior (scale factor 0)
        // From our testing: Input [0,1,2,3,4,5] -> Output [1,3,7,20,55,148]
        if (src_val == 0) expectedData[i] = 1;
        else if (src_val == 1) expectedData[i] = 3;  // NVIDIA NPP gives 3, not 2
        else if (src_val == 2) expectedData[i] = 7;
        else if (src_val == 3) expectedData[i] = 20;
        else if (src_val == 4) expectedData[i] = 55;
        else if (src_val == 5) expectedData[i] = 148;
        else expectedData[i] = 255; // saturation
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
    int nScaleFactor = 0; // No additional scaling
    NppStatus status = nppiExp_8u_C1RSfs(d_src, srcStep, d_dst, dstStep, roi, nScaleFactor);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host row by row
    std::vector<Npp8u> resultData(width * height);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width,
                   (char*)d_dst + y * dstStep,
                   width * sizeof(Npp8u),
                   cudaMemcpyDeviceToHost);
    }
    
    // Verify results using new precision control system
    for (int i = 0; i < width * height; i++) {
        NPP_EXPECT_ARITHMETIC_EQUAL(resultData[i], expectedData[i], "nppiExp_8u_C1RSfs")
            << "Mismatch at pixel " << i << ": got " << (int)resultData[i] 
            << ", expected " << (int)expectedData[i]
            << ", src=" << (int)srcData[i];
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test 32-bit float exponential
TEST_F(ExpFunctionalTest, Exp_32f_C1R_BasicOperation) {
    std::vector<Npp32f> srcData(width * height);
    std::vector<Npp32f> expectedData(width * height);
    
    // Create test data: various float values
    for (int i = 0; i < width * height; i++) {
        Npp32f src_val = -2.0f + 0.1f * (i % 40); // Values from -2 to 2
        srcData[i] = src_val;
        expectedData[i] = std::exp(src_val);
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
    NppStatus status = nppiExp_32f_C1R(d_src, srcStep, d_dst, dstStep, roi);
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

// Test 16-bit signed exponential with scaling
// 修复：NVIDIA NPP直接计算exp(输入值)，不进行输入缩放
TEST_F(ExpFunctionalTest, Exp_16s_C1RSfs_BasicOperation) {
    std::vector<Npp16s> srcData(width * height);
    std::vector<Npp16s> expectedData(width * height);
    
    // Create test data: both positive and negative values
    for (int i = 0; i < width * height; i++) {
        Npp16s src_val = (Npp16s)((i % 21) - 10); // Values from -10 to 10
        srcData[i] = src_val;
        
        // Expected: NVIDIA NPP computes exp(src_val) directly
        // This matches observed behavior: input 2 -> output 7 (e^2 ≈ 7.39)
        float exp_val = std::exp((float)src_val);
        int result = (int)(exp_val + 0.5f);
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
    int nScaleFactor = 0; // No additional scaling
    NppStatus status = nppiExp_16s_C1RSfs(d_src, srcStep, d_dst, dstStep, roi, nScaleFactor);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host row by row
    std::vector<Npp16s> resultData(width * height);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width,
                   (char*)d_dst + y * dstStep,
                   width * sizeof(Npp16s),
                   cudaMemcpyDeviceToHost);
    }
    
    // Verify results (with tolerance)
    for (int i = 0; i < width * height; i++) {
        // Allow reasonable tolerance for exponential function
        int diff = std::abs((int)resultData[i] - (int)expectedData[i]);
        EXPECT_LE(diff, 5) 
            << "Mismatch at pixel " << i << ": got " << resultData[i] 
            << ", expected approximately " << expectedData[i]
            << ", src=" << srcData[i];
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test special values for 32-bit float
TEST_F(ExpFunctionalTest, Exp_32f_C1R_SpecialValues) {
    const int testSize = 8;
    NppiSize testRoi = {testSize, 1};
    
    // Test special values: 0, 1, -1, ln(2), large positive, large negative
    std::vector<Npp32f> srcData = {0.0f, 1.0f, -1.0f, (float)std::log(2.0), 10.0f, -10.0f, 88.0f, -88.0f};
    std::vector<Npp32f> expectedData(testSize);
    
    // Calculate expected values
    expectedData[0] = 1.0f;                // exp(0) = 1
    expectedData[1] = (float)M_E;          // exp(1) = e
    expectedData[2] = 1.0f/(float)M_E;     // exp(-1) = 1/e
    expectedData[3] = 2.0f;                // exp(ln(2)) = 2
    expectedData[4] = std::exp(10.0f);     // exp(10) ≈ 22026
    expectedData[5] = std::exp(-10.0f);    // exp(-10) ≈ 0.0000454
    expectedData[6] = std::exp(88.0f);     // Near float max
    expectedData[7] = std::exp(-88.0f);    // Very small positive
    
    // Allocate GPU memory
    int srcStep, dstStep;
    Npp32f* d_src = nppiMalloc_32f_C1(testSize, 1, &srcStep);
    Npp32f* d_dst = nppiMalloc_32f_C1(testSize, 1, &dstStep);
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU
    cudaMemcpy(d_src, srcData.data(), testSize * sizeof(Npp32f), cudaMemcpyHostToDevice);
    
    // Execute NPP function
    NppStatus status = nppiExp_32f_C1R(d_src, srcStep, d_dst, dstStep, testRoi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host
    std::vector<Npp32f> resultData(testSize);
    cudaMemcpy(resultData.data(), d_dst, testSize * sizeof(Npp32f), cudaMemcpyDeviceToHost);
    
    // Verify results
    for (int i = 0; i < testSize; i++) {
        if (expectedData[i] < 1e-6f) {
            // For very small values, use absolute error
            EXPECT_NEAR(resultData[i], expectedData[i], 1e-10f) 
                << "Special value test failed at index " << i
                << ", src=" << srcData[i];
        } else {
            // For larger values, use relative error
            EXPECT_NEAR(resultData[i], expectedData[i], expectedData[i] * 1e-5f) 
                << "Special value test failed at index " << i
                << ", src=" << srcData[i];
        }
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test in-place operation
TEST_F(ExpFunctionalTest, Exp_32f_C1IR_InPlace) {
    std::vector<Npp32f> data(width * height);
    std::vector<Npp32f> expectedData(width * height);
    
    // Create test data: values that result in reasonable exponentials
    for (int i = 0; i < width * height; i++) {
        Npp32f src_val = -1.0f + 0.1f * (i % 20); // Values from -1 to 1
        data[i] = src_val;
        expectedData[i] = std::exp(src_val);
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
    NppStatus status = nppiExp_32f_C1IR(d_data, step, roi);
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
        EXPECT_NEAR(resultData[i], expectedData[i], expectedData[i] * 1e-5f) 
            << "In-place operation failed at pixel " << i;
    }
    
    nppiFree(d_data);
}

// Test error handling
TEST_F(ExpFunctionalTest, Exp_ErrorHandling) {
    // Test null pointer
    NppStatus status = nppiExp_32f_C1R(nullptr, 32, nullptr, 16, roi);
    EXPECT_NE(status, NPP_SUCCESS);
    
    // Test invalid ROI
    NppiSize invalidRoi = {0, 0};
    status = nppiExp_32f_C1R(nullptr, 32, nullptr, 16, invalidRoi);
    EXPECT_NE(status, NPP_SUCCESS);
    
    // Test invalid step
    status = nppiExp_32f_C1R(nullptr, 0, nullptr, 16, roi);
    EXPECT_NE(status, NPP_SUCCESS);
    
    // Test negative scale factor
    status = nppiExp_8u_C1RSfs(nullptr, 32, nullptr, 16, roi, -1);
    EXPECT_NE(status, NPP_SUCCESS);
}

// Test stream context version
TEST_F(ExpFunctionalTest, Exp_StreamContext) {
    std::vector<Npp32f> srcData(width * height, 0.0f); // All pixels = 0
    
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
    NppStatus status = nppiExp_32f_C1R_Ctx(d_src, srcStep, d_dst, dstStep, roi, nppStreamCtx);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Verify result - copy row by row
    std::vector<Npp32f> resultData(width * height);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width,
                   (char*)d_dst + y * dstStep,
                   width * sizeof(Npp32f),
                   cudaMemcpyDeviceToHost);
    }
    
    // All results should be 1.0 (exp(0) = 1)
    for (int i = 0; i < width * height; i++) {
        EXPECT_NEAR(resultData[i], 1.0f, 1e-6f)
            << "Stream context test failed at pixel " << i;
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}

// Test with scaling factor for integer types
TEST_F(ExpFunctionalTest, Exp_8u_C1RSfs_WithScaling) {
    std::vector<Npp8u> srcData(width * height);
    std::vector<Npp8u> expectedData(width * height);
    
    // Create test data
    for (int i = 0; i < width * height; i++) {
        Npp8u src_val = (Npp8u)(i % 4); // Values 0-3
        srcData[i] = src_val;
        
        // Expected with scaling factor
        float scaled_input = (float)src_val / 255.0f * 5.0f;
        float exp_val = std::exp(scaled_input);
        int result = (int)(exp_val * 4.0f + 0.5f); // Scale factor 2 means multiply by 4
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
    
    // Execute NPP function with scaling
    int nScaleFactor = 2; // Multiply result by 4
    NppStatus status = nppiExp_8u_C1RSfs(d_src, srcStep, d_dst, dstStep, roi, nScaleFactor);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host row by row
    std::vector<Npp8u> resultData(width * height);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width,
                   (char*)d_dst + y * dstStep,
                   width * sizeof(Npp8u),
                   cudaMemcpyDeviceToHost);
    }
    
    // Verify results (with tolerance)
    for (int i = 0; i < width * height; i++) {
        int diff = std::abs((int)resultData[i] - (int)expectedData[i]);
        EXPECT_LE(diff, 10) 
            << "Mismatch at pixel " << i << ": got " << (int)resultData[i] 
            << ", expected approximately " << (int)expectedData[i]
            << ", src=" << (int)srcData[i];
    }
    
    nppiFree(d_src);
    nppiFree(d_dst);
}