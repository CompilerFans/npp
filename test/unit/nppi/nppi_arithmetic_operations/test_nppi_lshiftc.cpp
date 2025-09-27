#include "npp.h"
#include "npp_test_base.h"
#include <gtest/gtest.h>
#include <vector>

class NppiLShiftCTest : public ::testing::Test {
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
// LShiftC Tests
// ============================================================================

TEST_F(NppiLShiftCTest, LShiftC_8u_C1R_BasicOperation) {
    const int width = 4;
    const int height = 2;
    const int totalPixels = width * height;

    std::vector<Npp8u> hostSrc = {1, 2, 4, 8, 16, 32, 64, 128};
    Npp32u shiftCount = 2;
    std::vector<Npp8u> expected = {4, 8, 16, 32, 64, 128, 0, 0}; // src << 2 (with overflow saturation)

    // Allocate GPU memory
    int step;
    Npp8u *d_src = nppiMalloc_8u_C1(width, height, &step);
    Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &step);
    
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    // Copy data to GPU
    int hostStep = width * sizeof(Npp8u);
    cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

    // Execute operation
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiLShiftC_8u_C1R(d_src, step, shiftCount, d_dst, step, oSizeROI);
    ASSERT_EQ(status, NPP_SUCCESS);

    // Copy result back
    std::vector<Npp8u> hostResult(totalPixels);
    cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < totalPixels; ++i) {
        EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
    }

    // Cleanup
    nppiFree(d_src);
    nppiFree(d_dst);
}

TEST_F(NppiLShiftCTest, LShiftC_8u_C1IR_InPlace) {
    const int width = 4;
    const int height = 2;
    const int totalPixels = width * height;

    std::vector<Npp8u> hostSrc = {3, 5, 7, 9, 11, 13, 15, 17};
    Npp32u shiftCount = 1;
    std::vector<Npp8u> expected = {6, 10, 14, 18, 22, 26, 30, 34}; // src << 1

    // Allocate GPU memory
    int step;
    Npp8u *d_srcDst = nppiMalloc_8u_C1(width, height, &step);
    
    ASSERT_NE(d_srcDst, nullptr);

    // Copy data to GPU
    int hostStep = width * sizeof(Npp8u);
    cudaMemcpy2D(d_srcDst, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

    // Execute in-place operation
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiLShiftC_8u_C1IR(shiftCount, d_srcDst, step, oSizeROI);
    ASSERT_EQ(status, NPP_SUCCESS);

    // Copy result back
    std::vector<Npp8u> hostResult(totalPixels);
    cudaMemcpy2D(hostResult.data(), hostStep, d_srcDst, step, hostStep, height, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < totalPixels; ++i) {
        EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
    }

    // Cleanup
    nppiFree(d_srcDst);
}

TEST_F(NppiLShiftCTest, LShiftC_16u_C1R_DataTypes) {
    const int width = 3;
    const int height = 2;
    const int totalPixels = width * height;

    std::vector<Npp16u> hostSrc = {100, 200, 300, 400, 500, 600};
    Npp32u shiftCount = 3;
    std::vector<Npp16u> expected = {800, 1600, 2400, 3200, 4000, 4800}; // src << 3

    // Allocate GPU memory
    int step;
    Npp16u *d_src = nppiMalloc_16u_C1(width, height, &step);
    Npp16u *d_dst = nppiMalloc_16u_C1(width, height, &step);
    
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    // Copy data to GPU
    int hostStep = width * sizeof(Npp16u);
    cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

    // Execute operation
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiLShiftC_16u_C1R(d_src, step, shiftCount, d_dst, step, oSizeROI);
    ASSERT_EQ(status, NPP_SUCCESS);

    // Copy result back
    std::vector<Npp16u> hostResult(totalPixels);
    cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < totalPixels; ++i) {
        EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
    }

    // Cleanup
    nppiFree(d_src);
    nppiFree(d_dst);
}