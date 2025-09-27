#include "npp.h"
#include "npp_test_base.h"
#include <gtest/gtest.h>
#include <vector>

class NppiRShiftCTest : public ::testing::Test {
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
// RShiftC Tests
// ============================================================================

TEST_F(NppiRShiftCTest, RShiftC_8u_C1R_BasicOperation) {
    const int width = 4;
    const int height = 2;
    const int totalPixels = width * height;

    std::vector<Npp8u> hostSrc = {128, 64, 32, 16, 8, 4, 2, 1};
    Npp32u shiftCount = 2;
    std::vector<Npp8u> expected = {32, 16, 8, 4, 2, 1, 0, 0}; // src >> 2

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
    NppStatus status = nppiRShiftC_8u_C1R(d_src, step, shiftCount, d_dst, step, oSizeROI);
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

TEST_F(NppiRShiftCTest, RShiftC_16s_C1R_SignedDataType) {
    const int width = 3;
    const int height = 2;
    const int totalPixels = width * height;

    std::vector<Npp16s> hostSrc = {-128, -64, -32, 128, 64, 32};
    Npp32u shiftCount = 2;
    std::vector<Npp16s> expected = {-32, -16, -8, 32, 16, 8}; // src >> 2 (arithmetic shift for signed)

    // Allocate GPU memory
    int step;
    Npp16s *d_src = nppiMalloc_16s_C1(width, height, &step);
    Npp16s *d_dst = nppiMalloc_16s_C1(width, height, &step);
    
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    // Copy data to GPU
    int hostStep = width * sizeof(Npp16s);
    cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

    // Execute operation
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiRShiftC_16s_C1R(d_src, step, shiftCount, d_dst, step, oSizeROI);
    ASSERT_EQ(status, NPP_SUCCESS);

    // Copy result back
    std::vector<Npp16s> hostResult(totalPixels);
    cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < totalPixels; ++i) {
        EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
    }

    // Cleanup
    nppiFree(d_src);
    nppiFree(d_dst);
}