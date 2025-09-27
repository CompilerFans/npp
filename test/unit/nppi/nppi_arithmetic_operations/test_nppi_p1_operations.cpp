#include "npp.h"
#include "npp_test_base.h"
#include <gtest/gtest.h>
#include <vector>

class NppiP1OperationsTest : public ::testing::Test {
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
// AbsDiffC Tests
// ============================================================================

TEST_F(NppiP1OperationsTest, AbsDiffC_8u_C1R_BasicOperation) {
    const int width = 4;
    const int height = 2;
    const int totalPixels = width * height;

    std::vector<Npp8u> hostSrc = {10, 50, 100, 200, 5, 150, 75, 255};
    Npp8u constant = 100;
    std::vector<Npp8u> expected = {90, 50, 0, 100, 95, 50, 25, 155}; // |src - 100|

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
    NppStatus status = nppiAbsDiffC_8u_C1R(d_src, step, d_dst, step, oSizeROI, constant);
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

TEST_F(NppiP1OperationsTest, AbsDiffC_16u_C1R_DataTypes) {
    const int width = 3;
    const int height = 2;
    const int totalPixels = width * height;

    std::vector<Npp16u> hostSrc = {1000, 5000, 10000, 500, 15000, 2000};
    Npp16u constant = 5000;
    std::vector<Npp16u> expected = {4000, 0, 5000, 4500, 10000, 3000}; // |src - 5000|

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
    NppStatus status = nppiAbsDiffC_16u_C1R(d_src, step, d_dst, step, oSizeROI, constant);
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

TEST_F(NppiP1OperationsTest, AbsDiffC_32f_C1R_FloatingPoint) {
    const int width = 2;
    const int height = 2;
    const int totalPixels = width * height;

    std::vector<Npp32f> hostSrc = {1.5f, 3.7f, 2.1f, 4.9f};
    Npp32f constant = 3.0f;
    std::vector<Npp32f> expected = {1.5f, 0.7f, 0.9f, 1.9f}; // |src - 3.0|

    // Allocate GPU memory
    int step;
    Npp32f *d_src = nppiMalloc_32f_C1(width, height, &step);
    Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &step);
    
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);

    // Copy data to GPU
    int hostStep = width * sizeof(Npp32f);
    cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

    // Execute operation
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiAbsDiffC_32f_C1R(d_src, step, d_dst, step, oSizeROI, constant);
    ASSERT_EQ(status, NPP_SUCCESS);

    // Copy result back
    std::vector<Npp32f> hostResult(totalPixels);
    cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

    // Verify results with tolerance for floating point
    for (int i = 0; i < totalPixels; ++i) {
        EXPECT_NEAR(hostResult[i], expected[i], 1e-5f) << "Mismatch at index " << i;
    }

    // Cleanup
    nppiFree(d_src);
    nppiFree(d_dst);
}

// ============================================================================
// LShiftC Tests
// ============================================================================

TEST_F(NppiP1OperationsTest, LShiftC_8u_C1R_BasicOperation) {
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

TEST_F(NppiP1OperationsTest, LShiftC_8u_C1IR_InPlace) {
    const int width = 3;
    const int height = 2;
    const int totalPixels = width * height;

    std::vector<Npp8u> hostData = {1, 3, 7, 15, 31, 63};
    Npp32u shiftCount = 1;
    std::vector<Npp8u> expected = {2, 6, 14, 30, 62, 126}; // src << 1

    // Allocate GPU memory
    int step;
    Npp8u *d_data = nppiMalloc_8u_C1(width, height, &step);
    ASSERT_NE(d_data, nullptr);

    // Copy data to GPU
    int hostStep = width * sizeof(Npp8u);
    cudaMemcpy2D(d_data, step, hostData.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

    // Execute in-place operation
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiLShiftC_8u_C1IR(shiftCount, d_data, step, oSizeROI);
    ASSERT_EQ(status, NPP_SUCCESS);

    // Copy result back
    std::vector<Npp8u> hostResult(totalPixels);
    cudaMemcpy2D(hostResult.data(), hostStep, d_data, step, hostStep, height, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < totalPixels; ++i) {
        EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
    }

    // Cleanup
    nppiFree(d_data);
}

TEST_F(NppiP1OperationsTest, LShiftC_16u_C1R_DataTypes) {
    const int width = 2;
    const int height = 2;
    const int totalPixels = width * height;

    std::vector<Npp16u> hostSrc = {1, 2, 4, 8};
    Npp32u shiftCount = 4;
    std::vector<Npp16u> expected = {16, 32, 64, 128}; // src << 4

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

// ============================================================================
// RShiftC Tests
// ============================================================================

TEST_F(NppiP1OperationsTest, RShiftC_8u_C1R_BasicOperation) {
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

TEST_F(NppiP1OperationsTest, RShiftC_16s_C1R_SignedDataType) {
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

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(NppiP1OperationsTest, ShiftC_ErrorHandling) {
    int step;
    Npp8u *d_valid = nppiMalloc_8u_C1(2, 2, &step);
    ASSERT_NE(d_valid, nullptr);

    NppiSize oSizeROI = {2, 2};

    // Test invalid shift count
    EXPECT_EQ(nppiLShiftC_8u_C1R(d_valid, step, 32, d_valid, step, oSizeROI), NPP_BAD_ARGUMENT_ERROR);
    EXPECT_EQ(nppiRShiftC_8u_C1R(d_valid, step, 35, d_valid, step, oSizeROI), NPP_BAD_ARGUMENT_ERROR);

    // Test null pointer errors
    EXPECT_EQ(nppiLShiftC_8u_C1R(nullptr, step, 1, d_valid, step, oSizeROI), NPP_NULL_POINTER_ERROR);
    EXPECT_EQ(nppiRShiftC_8u_C1R(d_valid, step, 1, nullptr, step, oSizeROI), NPP_NULL_POINTER_ERROR);

    // Cleanup
    nppiFree(d_valid);
}