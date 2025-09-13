#include <gtest/gtest.h>
#include "npp.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include "../../../common/npp_test_utils.h"

class MulScaleFunctionalTest : public ::testing::Test {
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

// Test 8-bit unsigned multiplication with scaling
TEST_F(MulScaleFunctionalTest, MulScale_8u_C1R_BasicOperation) {
    std::vector<Npp8u> src1Data(width * height);
    std::vector<Npp8u> src2Data(width * height);
    std::vector<Npp8u> expectedData(width * height);
    
    // Create test data: values that result in predictable scaled multiplication
    for (int i = 0; i < width * height; i++) {
        Npp8u src1_val = (Npp8u)((i % 16) + 1);     // Values 1-16
        Npp8u src2_val = (Npp8u)((i % 8) * 16);     // Values 0, 16, 32, 48, 64, 80, 96, 112
        
        src1Data[i] = src1_val;
        src2Data[i] = src2_val;
        
        // Expected: (src1 * src2) / 255
        int result = (src1_val * src2_val) / 255;
        expectedData[i] = (Npp8u)std::min(result, 255);
    }
    
    // Allocate GPU memory
    int src1Step, src2Step, dstStep;
    Npp8u* d_src1 = nppiMalloc_8u_C1(width, height, &src1Step);
    Npp8u* d_src2 = nppiMalloc_8u_C1(width, height, &src2Step);
    Npp8u* d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
    
    // Use RAII pattern to ensure cleanup even if test fails
    struct ResourceGuard {
        Npp8u* src1, *src2, *dst;
        ResourceGuard(Npp8u* s1, Npp8u* s2, Npp8u* d) : src1(s1), src2(s2), dst(d) {}
        ~ResourceGuard() {
            if (src1) nppiFree(src1);
            if (src2) nppiFree(src2);
            if (dst) nppiFree(dst);
        }
    } guard(d_src1, d_src2, d_dst);
    
    ASSERT_NE(d_src1, nullptr);
    ASSERT_NE(d_src2, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU row by row
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src1 + y * src1Step,
                   src1Data.data() + y * width,
                   width * sizeof(Npp8u),
                   cudaMemcpyHostToDevice);
        cudaMemcpy((char*)d_src2 + y * src2Step,
                   src2Data.data() + y * width,
                   width * sizeof(Npp8u),
                   cudaMemcpyHostToDevice);
    }
    
    // Execute NPP function
    NppStatus status = nppiMulScale_8u_C1R(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi);
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
        NPP_EXPECT_ARITHMETIC_EQUAL(resultData[i], expectedData[i], "nppiMulScale_8u_C1R")
            << "Mismatch at pixel " << i << ": got " << (int)resultData[i] 
            << ", expected " << (int)expectedData[i]
            << ", src1=" << (int)src1Data[i]
            << ", src2=" << (int)src2Data[i];
    }
    
    // Resources will be automatically cleaned up by ResourceGuard destructor
}

// Test 16-bit unsigned multiplication with scaling
TEST_F(MulScaleFunctionalTest, MulScale_16u_C1R_BasicOperation) {
    std::vector<Npp16u> src1Data(width * height);
    std::vector<Npp16u> src2Data(width * height);
    std::vector<Npp16u> expectedData(width * height);
    
    // Create test data: values that result in predictable scaled multiplication
    for (int i = 0; i < width * height; i++) {
        Npp16u src1_val = (Npp16u)((i % 256) + 1);        // Values 1-256
        Npp16u src2_val = (Npp16u)((i % 32) * 256);       // Values 0, 256, 512, ..., 7936
        
        src1Data[i] = src1_val;
        src2Data[i] = src2_val;
        
        // Expected: (src1 * src2) / 65535
        long long result = ((long long)src1_val * src2_val) / 65535;
        expectedData[i] = (Npp16u)std::min(result, 65535LL);
    }
    
    // Allocate GPU memory
    int src1Step, src2Step, dstStep;
    Npp16u* d_src1 = nppiMalloc_16u_C1(width, height, &src1Step);
    Npp16u* d_src2 = nppiMalloc_16u_C1(width, height, &src2Step);
    Npp16u* d_dst = nppiMalloc_16u_C1(width, height, &dstStep);
    
    // Use RAII pattern to ensure cleanup even if test fails
    struct ResourceGuard {
        Npp16u* src1, *src2, *dst;
        ResourceGuard(Npp16u* s1, Npp16u* s2, Npp16u* d) : src1(s1), src2(s2), dst(d) {}
        ~ResourceGuard() {
            if (src1) nppiFree(src1);
            if (src2) nppiFree(src2);
            if (dst) nppiFree(dst);
        }
    } guard(d_src1, d_src2, d_dst);
    
    ASSERT_NE(d_src1, nullptr);
    ASSERT_NE(d_src2, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU row by row
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src1 + y * src1Step,
                   src1Data.data() + y * width,
                   width * sizeof(Npp16u),
                   cudaMemcpyHostToDevice);
        cudaMemcpy((char*)d_src2 + y * src2Step,
                   src2Data.data() + y * width,
                   width * sizeof(Npp16u),
                   cudaMemcpyHostToDevice);
    }
    
    // Execute NPP function
    NppStatus status = nppiMulScale_16u_C1R(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host row by row
    std::vector<Npp16u> resultData(width * height);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width,
                   (char*)d_dst + y * dstStep,
                   width * sizeof(Npp16u),
                   cudaMemcpyDeviceToHost);
    }
    
    // Verify results
    for (int i = 0; i < width * height; i++) {
        EXPECT_EQ(resultData[i], expectedData[i]) 
            << "Mismatch at pixel " << i << ": got " << resultData[i] 
            << ", expected " << expectedData[i]
            << ", src1=" << src1Data[i]
            << ", src2=" << src2Data[i];
    }
    
    // Resources will be automatically cleaned up by ResourceGuard destructor
}

// Test extreme values for 8-bit
TEST_F(MulScaleFunctionalTest, MulScale_8u_C1R_ExtremeValues) {
    const int testSize = 6;
    NppiSize testRoi = {testSize, 1};
    
    // Test extreme combinations
    std::vector<Npp8u> src1Data = {255, 255, 128, 0, 1, 127};
    std::vector<Npp8u> src2Data = {255, 0, 128, 255, 255, 2};
    std::vector<Npp8u> expectedData(testSize);
    
    // Calculate expected values
    for (int i = 0; i < testSize; i++) {
        int result = (src1Data[i] * src2Data[i]) / 255;
        expectedData[i] = (Npp8u)std::min(result, 255);
    }
    
    // Allocate GPU memory
    int src1Step, src2Step, dstStep;
    Npp8u* d_src1 = nppiMalloc_8u_C1(testSize, 1, &src1Step);
    Npp8u* d_src2 = nppiMalloc_8u_C1(testSize, 1, &src2Step);
    Npp8u* d_dst = nppiMalloc_8u_C1(testSize, 1, &dstStep);
    
    // Use RAII pattern to ensure cleanup even if test fails
    struct ResourceGuard {
        Npp8u* src1, *src2, *dst;
        ResourceGuard(Npp8u* s1, Npp8u* s2, Npp8u* d) : src1(s1), src2(s2), dst(d) {}
        ~ResourceGuard() {
            if (src1) nppiFree(src1);
            if (src2) nppiFree(src2);
            if (dst) nppiFree(dst);
        }
    } guard(d_src1, d_src2, d_dst);
    
    ASSERT_NE(d_src1, nullptr);
    ASSERT_NE(d_src2, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU
    cudaMemcpy(d_src1, src1Data.data(), testSize * sizeof(Npp8u), cudaMemcpyHostToDevice);
    cudaMemcpy(d_src2, src2Data.data(), testSize * sizeof(Npp8u), cudaMemcpyHostToDevice);
    
    // Execute NPP function
    NppStatus status = nppiMulScale_8u_C1R(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, testRoi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host
    std::vector<Npp8u> resultData(testSize);
    cudaMemcpy(resultData.data(), d_dst, testSize * sizeof(Npp8u), cudaMemcpyDeviceToHost);
    
    // Verify results
    for (int i = 0; i < testSize; i++) {
        EXPECT_EQ(resultData[i], expectedData[i]) 
            << "Extreme value test failed at index " << i
            << ", src1=" << (int)src1Data[i]
            << ", src2=" << (int)src2Data[i];
    }
    
    // Resources will be automatically cleaned up by ResourceGuard destructor
}

// Test in-place operation for 8-bit
TEST_F(MulScaleFunctionalTest, MulScale_8u_C1IR_InPlace) {
    std::vector<Npp8u> srcData(width * height);
    std::vector<Npp8u> srcDstData(width * height);
    std::vector<Npp8u> expectedData(width * height);
    
    // Create test data
    for (int i = 0; i < width * height; i++) {
        Npp8u src_val = (Npp8u)((i % 10) + 1);      // Values 1-10
        Npp8u srcDst_val = (Npp8u)((i % 5) * 50);   // Values 0, 50, 100, 150, 200
        
        srcData[i] = src_val;
        srcDstData[i] = srcDst_val;
        
        // Expected: (src * srcDst) / 255
        int result = (src_val * srcDst_val) / 255;
        expectedData[i] = (Npp8u)std::min(result, 255);
    }
    
    // Allocate GPU memory
    int srcStep, srcDstStep;
    Npp8u* d_src = nppiMalloc_8u_C1(width, height, &srcStep);
    Npp8u* d_srcDst = nppiMalloc_8u_C1(width, height, &srcDstStep);
    
    // Use RAII pattern to ensure cleanup even if test fails
    struct ResourceGuard {
        Npp8u* src, *srcDst;
        ResourceGuard(Npp8u* s, Npp8u* sd) : src(s), srcDst(sd) {}
        ~ResourceGuard() {
            if (src) nppiFree(src);
            if (srcDst) nppiFree(srcDst);
        }
    } guard(d_src, d_srcDst);
    
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_srcDst, nullptr);
    
    // Copy input data to GPU row by row
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src + y * srcStep,
                   srcData.data() + y * width,
                   width * sizeof(Npp8u),
                   cudaMemcpyHostToDevice);
        cudaMemcpy((char*)d_srcDst + y * srcDstStep,
                   srcDstData.data() + y * width,
                   width * sizeof(Npp8u),
                   cudaMemcpyHostToDevice);
    }
    
    // Execute NPP function (in-place)
    NppStatus status = nppiMulScale_8u_C1IR(d_src, srcStep, d_srcDst, srcDstStep, roi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host row by row
    std::vector<Npp8u> resultData(width * height);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width,
                   (char*)d_srcDst + y * srcDstStep,
                   width * sizeof(Npp8u),
                   cudaMemcpyDeviceToHost);
    }
    
    // Verify results
    for (int i = 0; i < width * height; i++) {
        EXPECT_EQ(resultData[i], expectedData[i]) 
            << "In-place operation failed at pixel " << i;
    }
    
    // Resources will be automatically cleaned up by ResourceGuard destructor
}

// Test error handling
// NOTE: 测试已被禁用 - NVIDIA NPP对无效参数的错误检测行为与预期不符
TEST_F(MulScaleFunctionalTest, DISABLED_MulScale_ErrorHandling) {
    // Test null pointer
    NppStatus status = nppiMulScale_8u_C1R(nullptr, 32, nullptr, 32, nullptr, 16, roi);
    EXPECT_NE(status, NPP_SUCCESS);
    
    // Test invalid ROI
    NppiSize invalidRoi = {0, 0};
    status = nppiMulScale_8u_C1R(nullptr, 32, nullptr, 32, nullptr, 16, invalidRoi);
    EXPECT_NE(status, NPP_SUCCESS);
    
    // Test invalid step
    status = nppiMulScale_8u_C1R(nullptr, 0, nullptr, 32, nullptr, 16, roi);
    EXPECT_NE(status, NPP_SUCCESS);
}

// Test stream context version
TEST_F(MulScaleFunctionalTest, MulScale_StreamContext) {
    std::vector<Npp8u> src1Data(width * height, 255); // All pixels = 255
    std::vector<Npp8u> src2Data(width * height, 128); // All pixels = 128
    
    // Allocate GPU memory
    int src1Step, src2Step, dstStep;
    Npp8u* d_src1 = nppiMalloc_8u_C1(width, height, &src1Step);
    Npp8u* d_src2 = nppiMalloc_8u_C1(width, height, &src2Step);
    Npp8u* d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
    
    // Use RAII pattern to ensure cleanup even if test fails
    struct ResourceGuard {
        Npp8u* src1, *src2, *dst;
        ResourceGuard(Npp8u* s1, Npp8u* s2, Npp8u* d) : src1(s1), src2(s2), dst(d) {}
        ~ResourceGuard() {
            if (src1) nppiFree(src1);
            if (src2) nppiFree(src2);
            if (dst) nppiFree(dst);
        }
    } guard(d_src1, d_src2, d_dst);
    
    ASSERT_NE(d_src1, nullptr);
    ASSERT_NE(d_src2, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU row by row
    for (int y = 0; y < height; y++) {
        cudaMemcpy((char*)d_src1 + y * src1Step,
                   src1Data.data() + y * width,
                   width * sizeof(Npp8u),
                   cudaMemcpyHostToDevice);
        cudaMemcpy((char*)d_src2 + y * src2Step,
                   src2Data.data() + y * width,
                   width * sizeof(Npp8u),
                   cudaMemcpyHostToDevice);
    }
    
    // Create stream context
    NppStreamContext nppStreamCtx;
    nppStreamCtx.hStream = 0;
    
    // Execute NPP function with context
    NppStatus status = nppiMulScale_8u_C1R_Ctx(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, nppStreamCtx);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Verify result - copy row by row
    std::vector<Npp8u> resultData(width * height);
    for (int y = 0; y < height; y++) {
        cudaMemcpy(resultData.data() + y * width,
                   (char*)d_dst + y * dstStep,
                   width * sizeof(Npp8u),
                   cudaMemcpyDeviceToHost);
    }
    
    // All results should be 128 (255 * 128 / 255 = 128)
    for (int i = 0; i < width * height; i++) {
        EXPECT_EQ(resultData[i], 128)
            << "Stream context test failed at pixel " << i;
    }
    
    // Resources will be automatically cleaned up by ResourceGuard destructor
}

// Test overflow handling for 16-bit
TEST_F(MulScaleFunctionalTest, MulScale_16u_C1R_OverflowHandling) {
    const int testSize = 4;
    NppiSize testRoi = {testSize, 1};
    
    // Test values that would overflow without proper handling
    std::vector<Npp16u> src1Data = {65535, 32768, 16384, 1};
    std::vector<Npp16u> src2Data = {65535, 65535, 8192, 65535};
    std::vector<Npp16u> expectedData(testSize);
    
    // Calculate expected values with 64-bit intermediate
    for (int i = 0; i < testSize; i++) {
        long long result = ((long long)src1Data[i] * src2Data[i]) / 65535;
        expectedData[i] = (Npp16u)std::min(result, 65535LL);
    }
    
    // Allocate GPU memory
    int src1Step, src2Step, dstStep;
    Npp16u* d_src1 = nppiMalloc_16u_C1(testSize, 1, &src1Step);
    Npp16u* d_src2 = nppiMalloc_16u_C1(testSize, 1, &src2Step);
    Npp16u* d_dst = nppiMalloc_16u_C1(testSize, 1, &dstStep);
    
    // Use RAII pattern to ensure cleanup even if test fails
    struct ResourceGuard {
        Npp16u* src1, *src2, *dst;
        ResourceGuard(Npp16u* s1, Npp16u* s2, Npp16u* d) : src1(s1), src2(s2), dst(d) {}
        ~ResourceGuard() {
            if (src1) nppiFree(src1);
            if (src2) nppiFree(src2);
            if (dst) nppiFree(dst);
        }
    } guard(d_src1, d_src2, d_dst);
    
    ASSERT_NE(d_src1, nullptr);
    ASSERT_NE(d_src2, nullptr);
    ASSERT_NE(d_dst, nullptr);
    
    // Copy input data to GPU
    cudaMemcpy(d_src1, src1Data.data(), testSize * sizeof(Npp16u), cudaMemcpyHostToDevice);
    cudaMemcpy(d_src2, src2Data.data(), testSize * sizeof(Npp16u), cudaMemcpyHostToDevice);
    
    // Execute NPP function
    NppStatus status = nppiMulScale_16u_C1R(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, testRoi);
    EXPECT_EQ(status, NPP_SUCCESS);
    
    // Copy result back to host
    std::vector<Npp16u> resultData(testSize);
    cudaMemcpy(resultData.data(), d_dst, testSize * sizeof(Npp16u), cudaMemcpyDeviceToHost);
    
    // Verify results
    for (int i = 0; i < testSize; i++) {
        EXPECT_EQ(resultData[i], expectedData[i]) 
            << "Overflow test failed at index " << i
            << ", src1=" << src1Data[i]
            << ", src2=" << src2Data[i];
    }
    
    // Resources will be automatically cleaned up by ResourceGuard destructor
}