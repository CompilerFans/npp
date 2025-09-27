#include "npp.h"
#include "npp_test_base.h"
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>

class NppiMinEveryTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess);
    }

    void TearDown() override {
        cudaError_t err = cudaDeviceSynchronize();
        EXPECT_EQ(err, cudaSuccess);
    }
};

// ============================================================================
// MinEvery Tests
// ============================================================================

TEST_F(NppiMinEveryTest, MinEvery_8u_C1IR_BasicOperation) {
    const int width = 4;
    const int height = 3;
    const int totalPixels = width * height;

    std::vector<Npp8u> hostSrc = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
    std::vector<Npp8u> hostDst = {15, 15, 35, 35, 45, 65, 65, 85, 85, 95, 115, 115};

    // Expected result: min(src, dst)
    std::vector<Npp8u> expected = {10, 15, 30, 35, 45, 60, 65, 80, 85, 95, 110, 115};

    // Allocate GPU memory
    int srcStep, dstStep;
    Npp8u *d_src = nppiMalloc_8u_C1(width, height, &srcStep);
    Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);

    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    // Copy data to GPU
    int hostStep = width * sizeof(Npp8u);
    cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_dst, dstStep, hostDst.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

    // Execute operation
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiMinEvery_8u_C1IR(d_src, srcStep, d_dst, dstStep, oSizeROI);
    ASSERT_EQ(status, NPP_SUCCESS);

    // Copy result back
    std::vector<Npp8u> hostResult(totalPixels);
    cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < totalPixels; ++i) {
        EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
    }

    // Cleanup
    nppiFree(d_src);
    nppiFree(d_dst);
}

TEST_F(NppiMinEveryTest, MinEvery_32f_C1IR_FloatingPoint) {
    const int width = 3;
    const int height = 2;
    const int totalPixels = width * height;

    std::vector<Npp32f> hostSrc = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f};
    std::vector<Npp32f> hostDst = {2.0f, 2.0f, 3.0f, 5.0f, 5.0f, 6.0f};

    // Expected: min(src, dst)
    std::vector<Npp32f> expected = {1.5f, 2.0f, 3.0f, 4.5f, 5.0f, 6.0f};

    // Allocate GPU memory
    int srcStep, dstStep;
    Npp32f *d_src = nppiMalloc_32f_C1(width, height, &srcStep);
    Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);

    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    // Copy data to GPU
    int hostStep = width * sizeof(Npp32f);
    cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_dst, dstStep, hostDst.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

    // Execute operation
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiMinEvery_32f_C1IR(d_src, srcStep, d_dst, dstStep, oSizeROI);
    ASSERT_EQ(status, NPP_SUCCESS);

    // Copy result back
    std::vector<Npp32f> hostResult(totalPixels);
    cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < totalPixels; ++i) {
        EXPECT_NEAR(hostResult[i], expected[i], 1e-5f) << "Mismatch at index " << i;
    }

    // Cleanup
    nppiFree(d_src);
    nppiFree(d_dst);
}

TEST_F(NppiMinEveryTest, MinEvery_16s_C1IR_SignedValues) {
    const int width = 4;
    const int height = 2;
    const int totalPixels = width * height;

    std::vector<Npp16s> hostSrc = {-10, 20, -30, 40, -50, 60, -70, 80};
    std::vector<Npp16s> hostDst = {-5, 15, -35, 35, -45, 65, -75, 75};

    // Expected: min(src, dst) 
    std::vector<Npp16s> expected = {-10, 15, -35, 35, -50, 60, -75, 75};

    // Allocate GPU memory
    int srcStep, dstStep;
    Npp16s *d_src = nppiMalloc_16s_C1(width, height, &srcStep);
    Npp16s *d_dst = nppiMalloc_16s_C1(width, height, &dstStep);

    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    // Copy data to GPU
    int hostStep = width * sizeof(Npp16s);
    cudaMemcpy2D(d_src, srcStep, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_dst, dstStep, hostDst.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

    // Execute operation
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiMinEvery_16s_C1IR(d_src, srcStep, d_dst, dstStep, oSizeROI);
    ASSERT_EQ(status, NPP_SUCCESS);

    // Copy result back
    std::vector<Npp16s> hostResult(totalPixels);
    cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < totalPixels; ++i) {
        EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
    }

    // Cleanup
    nppiFree(d_src);
    nppiFree(d_dst);
}