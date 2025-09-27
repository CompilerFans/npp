#include "npp.h"
#include "npp_test_base.h"
#include <gtest/gtest.h>
#include <vector>

class NppiMulDeviceCTest : public ::testing::Test {
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
// MulDeviceC 8u C1 Tests
// ============================================================================

TEST_F(NppiMulDeviceCTest, MulDeviceC_8u_C1RSfs_BasicOperation) {
    const int width = 4;
    const int height = 2;
    const int totalPixels = width * height;

    std::vector<Npp8u> hostSrc = {10, 20, 30, 40, 50, 60, 70, 80};
    Npp8u hostConstant = 3;
    std::vector<Npp8u> expected = {30, 60, 90, 120, 150, 180, 210, 240}; // src * 3

    // Allocate GPU memory for image data
    int step;
    Npp8u *d_src = nppiMalloc_8u_C1(width, height, &step);
    Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &step);
    
    // Allocate GPU memory for device constant
    Npp8u *d_constant;
    cudaMalloc(&d_constant, sizeof(Npp8u));
    
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    ASSERT_NE(d_constant, nullptr);

    // Copy data to GPU
    int hostStep = width * sizeof(Npp8u);
    cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_constant, &hostConstant, sizeof(Npp8u), cudaMemcpyHostToDevice);

    // Execute operation
    NppiSize oSizeROI = {width, height};
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    NppStatus status = nppiMulDeviceC_8u_C1RSfs_Ctx(d_src, step, d_constant, d_dst, step, oSizeROI, 0, nppStreamCtx);
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
    cudaFree(d_constant);
}

TEST_F(NppiMulDeviceCTest, MulDeviceC_8u_C1RSfs_WithScaling) {
    const int width = 3;
    const int height = 2;
    const int totalPixels = width * height;

    std::vector<Npp8u> hostSrc = {40, 80, 120, 160, 200, 240};
    Npp8u hostConstant = 4;
    int scaleFactor = 1; // Divide by 2 after multiplication
    std::vector<Npp8u> expected = {80, 160, 240, 255, 255, 255}; // (src * 4) / 2, with saturation

    // Allocate GPU memory
    int step;
    Npp8u *d_src = nppiMalloc_8u_C1(width, height, &step);
    Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &step);
    Npp8u *d_constant;
    cudaMalloc(&d_constant, sizeof(Npp8u));
    
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    ASSERT_NE(d_constant, nullptr);

    // Copy data to GPU
    int hostStep = width * sizeof(Npp8u);
    cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_constant, &hostConstant, sizeof(Npp8u), cudaMemcpyHostToDevice);

    // Execute operation
    NppiSize oSizeROI = {width, height};
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    NppStatus status = nppiMulDeviceC_8u_C1RSfs_Ctx(d_src, step, d_constant, d_dst, step, oSizeROI, scaleFactor, nppStreamCtx);
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
    cudaFree(d_constant);
}

// ============================================================================
// MulDeviceC 8u C3 Tests
// ============================================================================

TEST_F(NppiMulDeviceCTest, MulDeviceC_8u_C3RSfs_MultiChannel) {
    const int width = 2;
    const int height = 2;
    const int channels = 3;
    const int totalElements = width * height * channels;

    // Test data: 2x2 image with 3 channels (RGB)
    std::vector<Npp8u> hostSrc = {
        // Pixel (0,0): R=10, G=20, B=30
        10, 20, 30,
        // Pixel (1,0): R=40, G=50, B=60
        40, 50, 60,
        // Pixel (0,1): R=70, G=80, B=90
        70, 80, 90,
        // Pixel (1,1): R=100, G=110, B=120
        100, 110, 120
    };

    std::vector<Npp8u> hostConstants = {2, 3, 4}; // Multiply R*2, G*3, B*4
    int scaleFactor = 0; // No scaling

    // Expected results
    std::vector<Npp8u> expected = {
        // Pixel (0,0): R=10*2=20, G=20*3=60, B=30*4=120
        20, 60, 120,
        // Pixel (1,0): R=40*2=80, G=50*3=150, B=60*4=240
        80, 150, 240,
        // Pixel (0,1): R=70*2=140, G=80*3=240, B=90*4=255 (saturated)
        140, 240, 255,
        // Pixel (1,1): R=100*2=200, G=110*3=255 (saturated), B=120*4=255 (saturated)
        200, 255, 255
    };

    // Allocate GPU memory
    int step;
    Npp8u *d_src = nppiMalloc_8u_C3(width, height, &step);
    Npp8u *d_dst = nppiMalloc_8u_C3(width, height, &step);
    Npp8u *d_constants;
    cudaMalloc(&d_constants, channels * sizeof(Npp8u));
    
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    ASSERT_NE(d_constants, nullptr);

    // Copy data to GPU
    int hostStep = width * channels * sizeof(Npp8u);
    cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_constants, hostConstants.data(), channels * sizeof(Npp8u), cudaMemcpyHostToDevice);

    // Execute operation
    NppiSize oSizeROI = {width, height};
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    NppStatus status = nppiMulDeviceC_8u_C3RSfs_Ctx(d_src, step, d_constants, d_dst, step, oSizeROI, scaleFactor, nppStreamCtx);
    ASSERT_EQ(status, NPP_SUCCESS);

    // Copy result back
    std::vector<Npp8u> hostResult(totalElements);
    cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < totalElements; ++i) {
        EXPECT_EQ(hostResult[i], expected[i]) << "Mismatch at index " << i;
    }

    // Cleanup
    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_constants);
}

// ============================================================================
// MulDeviceC 32f Tests
// ============================================================================

TEST_F(NppiMulDeviceCTest, MulDeviceC_32f_C1R_FloatingPoint) {
    const int width = 2;
    const int height = 2;
    const int totalPixels = width * height;

    std::vector<Npp32f> hostSrc = {1.5f, 2.5f, 3.5f, 4.5f};
    Npp32f hostConstant = 2.25f;
    std::vector<Npp32f> expected = {3.375f, 5.625f, 7.875f, 10.125f}; // src * 2.25

    // Allocate GPU memory
    int step;
    Npp32f *d_src = nppiMalloc_32f_C1(width, height, &step);
    Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &step);
    Npp32f *d_constant;
    cudaMalloc(&d_constant, sizeof(Npp32f));
    
    ASSERT_NE(d_src, nullptr);
    ASSERT_NE(d_dst, nullptr);
    ASSERT_NE(d_constant, nullptr);

    // Copy data to GPU
    int hostStep = width * sizeof(Npp32f);
    cudaMemcpy2D(d_src, step, hostSrc.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_constant, &hostConstant, sizeof(Npp32f), cudaMemcpyHostToDevice);

    // Execute operation
    NppiSize oSizeROI = {width, height};
    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);
    NppStatus status = nppiMulDeviceC_32f_C1R_Ctx(d_src, step, d_constant, d_dst, step, oSizeROI, nppStreamCtx);
    ASSERT_EQ(status, NPP_SUCCESS);

    // Copy result back
    std::vector<Npp32f> hostResult(totalPixels);
    cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

    // Verify results with floating point tolerance
    for (int i = 0; i < totalPixels; ++i) {
        EXPECT_NEAR(hostResult[i], expected[i], 1e-5f) << "Mismatch at index " << i;
    }

    // Cleanup
    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_constant);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(NppiMulDeviceCTest, MulDeviceC_ErrorHandling) {
    int step;
    Npp8u *d_valid = nppiMalloc_8u_C1(2, 2, &step);
    Npp8u *d_constant;
    cudaMalloc(&d_constant, sizeof(Npp8u));
    
    ASSERT_NE(d_valid, nullptr);
    ASSERT_NE(d_constant, nullptr);

    NppiSize oSizeROI = {2, 2};

    NppStreamContext nppStreamCtx;
    nppGetStreamContext(&nppStreamCtx);

    // Test invalid scale factor
    EXPECT_EQ(nppiMulDeviceC_8u_C1RSfs_Ctx(d_valid, step, d_constant, d_valid, step, oSizeROI, -1, nppStreamCtx), NPP_BAD_ARGUMENT_ERROR);
    EXPECT_EQ(nppiMulDeviceC_8u_C1RSfs_Ctx(d_valid, step, d_constant, d_valid, step, oSizeROI, 32, nppStreamCtx), NPP_BAD_ARGUMENT_ERROR);

    // Test null pointer errors
    EXPECT_EQ(nppiMulDeviceC_8u_C1RSfs_Ctx(nullptr, step, d_constant, d_valid, step, oSizeROI, 0, nppStreamCtx), NPP_NULL_POINTER_ERROR);
    EXPECT_EQ(nppiMulDeviceC_8u_C1RSfs_Ctx(d_valid, step, nullptr, d_valid, step, oSizeROI, 0, nppStreamCtx), NPP_NULL_POINTER_ERROR);
    EXPECT_EQ(nppiMulDeviceC_8u_C1RSfs_Ctx(d_valid, step, d_constant, nullptr, step, oSizeROI, 0, nppStreamCtx), NPP_NULL_POINTER_ERROR);

    // Cleanup
    nppiFree(d_valid);
    cudaFree(d_constant);
}