#include "npp.h"
#include "npp_test_base.h"
#include <gtest/gtest.h>
#include <vector>

#ifndef USE_NVIDIA_NPP_TESTS
extern "C" {
    // DivDeviceC function declarations (MPP implementation only)
    NppStatus nppiDivDeviceC_8u_C1RSfs_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pConstant, 
                                           Npp8u *pDst, int nDstStep, NppiSize oSizeROI, 
                                           int nScaleFactor, NppStreamContext nppStreamCtx);
    
    NppStatus nppiDivDeviceC_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pConstant, 
                                         Npp32f *pDst, int nDstStep, NppiSize oSizeROI, 
                                         NppStreamContext nppStreamCtx);
}
#endif

#ifndef USE_NVIDIA_NPP_TESTS

class NppiDivDeviceCTest : public ::testing::Test {
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
// DivDeviceC 8u C1 Tests
// ============================================================================

TEST_F(NppiDivDeviceCTest, DivDeviceC_8u_C1RSfs_BasicOperation) {
    const int width = 4;
    const int height = 2;
    const int totalPixels = width * height;

    std::vector<Npp8u> hostSrc = {100, 200, 50, 150, 80, 160, 120, 240};
    Npp8u hostConstant = 5;
    std::vector<Npp8u> expected = {20, 40, 10, 30, 16, 32, 24, 48}; // src / 5

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
    NppStatus status = nppiDivDeviceC_8u_C1RSfs_Ctx(d_src, step, d_constant, d_dst, step, oSizeROI, 0, nppStreamCtx);
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

TEST_F(NppiDivDeviceCTest, DivDeviceC_8u_C1RSfs_DivisionByZero) {
    const int width = 2;
    const int height = 2;
    const int totalPixels = width * height;

    std::vector<Npp8u> hostSrc = {100, 200, 50, 150};
    Npp8u hostConstant = 0; // Division by zero
    std::vector<Npp8u> expected = {255, 255, 255, 255}; // Max value for unsigned

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
    NppStatus status = nppiDivDeviceC_8u_C1RSfs_Ctx(d_src, step, d_constant, d_dst, step, oSizeROI, 0, nppStreamCtx);
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
// DivDeviceC 32f C1 Tests
// ============================================================================

TEST_F(NppiDivDeviceCTest, DivDeviceC_32f_C1R_FloatingPoint) {
    const int width = 3;
    const int height = 2;
    const int totalPixels = width * height;

    std::vector<Npp32f> hostSrc = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    Npp32f hostConstant = 4.0f;
    std::vector<Npp32f> expected = {2.5f, 5.0f, 7.5f, 10.0f, 12.5f, 15.0f}; // src / 4.0

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
    NppStatus status = nppiDivDeviceC_32f_C1R_Ctx(d_src, step, d_constant, d_dst, step, oSizeROI, nppStreamCtx);
    ASSERT_EQ(status, NPP_SUCCESS);

    // Copy result back
    std::vector<Npp32f> hostResult(totalPixels);
    cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);

    // Verify results with tolerance
    for (int i = 0; i < totalPixels; ++i) {
        EXPECT_NEAR(hostResult[i], expected[i], 1e-6f) << "Mismatch at index " << i;
    }

    // Cleanup
    nppiFree(d_src);
    nppiFree(d_dst);
    cudaFree(d_constant);
}


#endif // USE_NVIDIA_NPP_TESTS