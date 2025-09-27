#include "npp.h"
#include "npp_test_base.h"
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>

class NppiAlphaCompTest : public ::testing::Test {
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
// AlphaCompC Tests - Alpha composition with constant alpha values
// ============================================================================

TEST_F(NppiAlphaCompTest, AlphaCompC_8u_C1R_AlphaOver) {
    const int width = 4;
    const int height = 2;
    const int totalPixels = width * height;

    std::vector<Npp8u> hostSrc1 = {100, 150, 200, 50, 75, 125, 175, 225};
    std::vector<Npp8u> hostSrc2 = {50, 75, 100, 200, 150, 125, 100, 75};
    
    Npp8u alpha1 = 128; // 0.5 in 8-bit
    Npp8u alpha2 = 192; // 0.75 in 8-bit

    // Expected result for ALPHA_OVER: src1*alpha1 + src2*alpha2*(1-alpha1)
    std::vector<Npp8u> expected(totalPixels);
    for (int i = 0; i < totalPixels; ++i) {
        double s1 = static_cast<double>(hostSrc1[i]) / 255.0;
        double s2 = static_cast<double>(hostSrc2[i]) / 255.0;
        double a1 = static_cast<double>(alpha1) / 255.0;
        double a2 = static_cast<double>(alpha2) / 255.0;
        double result = s1 * a1 + s2 * a2 * (1.0 - a1);
        expected[i] = static_cast<Npp8u>(std::min(255.0, std::max(0.0, result * 255.0)));
    }

    // Allocate GPU memory
    int src1Step, src2Step, dstStep;
    Npp8u *d_src1 = nppiMalloc_8u_C1(width, height, &src1Step);
    Npp8u *d_src2 = nppiMalloc_8u_C1(width, height, &src2Step);
    Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);

    ASSERT_NE(d_src1, nullptr);
    ASSERT_NE(d_src2, nullptr);
    ASSERT_NE(d_dst, nullptr);

    // Copy data to GPU
    int hostStep = width * sizeof(Npp8u);
    cudaMemcpy2D(d_src1, src1Step, hostSrc1.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_src2, src2Step, hostSrc2.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

    // Execute operation
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiAlphaCompC_8u_C1R(d_src1, src1Step, alpha1, 
                                             d_src2, src2Step, alpha2,
                                             d_dst, dstStep, oSizeROI, NPPI_OP_ALPHA_OVER);
    ASSERT_EQ(status, NPP_SUCCESS);

    // Copy result back
    std::vector<Npp8u> hostResult(totalPixels);
    cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);

    // Verify results (with tolerance for rounding)
    for (int i = 0; i < totalPixels; ++i) {
        EXPECT_NEAR(hostResult[i], expected[i], 2) << "Mismatch at index " << i 
                   << " got " << static_cast<int>(hostResult[i]) 
                   << " expected " << static_cast<int>(expected[i]);
    }

    // Cleanup
    nppiFree(d_src1);
    nppiFree(d_src2);
    nppiFree(d_dst);
}

TEST_F(NppiAlphaCompTest, AlphaCompC_8u_C1R_AlphaIn) {
    const int width = 3;
    const int height = 2;
    const int totalPixels = width * height;

    std::vector<Npp8u> hostSrc1 = {100, 150, 200, 50, 75, 125};
    std::vector<Npp8u> hostSrc2 = {50, 75, 100, 200, 150, 125};
    
    Npp8u alpha1 = 128; // 0.5 in 8-bit
    Npp8u alpha2 = 192; // 0.75 in 8-bit

    // Expected result for ALPHA_IN: src1 * alpha2
    std::vector<Npp8u> expected(totalPixels);
    for (int i = 0; i < totalPixels; ++i) {
        double s1 = static_cast<double>(hostSrc1[i]) / 255.0;
        double a2 = static_cast<double>(alpha2) / 255.0;
        double result = s1 * a2;
        expected[i] = static_cast<Npp8u>(std::min(255.0, std::max(0.0, result * 255.0)));
    }

    // Allocate GPU memory
    int src1Step, src2Step, dstStep;
    Npp8u *d_src1 = nppiMalloc_8u_C1(width, height, &src1Step);
    Npp8u *d_src2 = nppiMalloc_8u_C1(width, height, &src2Step);
    Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);

    // Copy data to GPU
    int hostStep = width * sizeof(Npp8u);
    cudaMemcpy2D(d_src1, src1Step, hostSrc1.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_src2, src2Step, hostSrc2.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

    // Execute operation
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiAlphaCompC_8u_C1R(d_src1, src1Step, alpha1, 
                                             d_src2, src2Step, alpha2,
                                             d_dst, dstStep, oSizeROI, NPPI_OP_ALPHA_IN);
    ASSERT_EQ(status, NPP_SUCCESS);

    // Copy result back
    std::vector<Npp8u> hostResult(totalPixels);
    cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < totalPixels; ++i) {
        EXPECT_NEAR(hostResult[i], expected[i], 2) << "Mismatch at index " << i 
                   << " got " << static_cast<int>(hostResult[i]) 
                   << " expected " << static_cast<int>(expected[i]);
    }

    // Cleanup
    nppiFree(d_src1);
    nppiFree(d_src2);
    nppiFree(d_dst);
}

TEST_F(NppiAlphaCompTest, AlphaCompC_32f_C1R_AlphaOver) {
    const int width = 3;
    const int height = 2;
    const int totalPixels = width * height;

    std::vector<Npp32f> hostSrc1 = {0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 0.0f};
    std::vector<Npp32f> hostSrc2 = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f, 0.2f};
    
    Npp32f alpha1 = 0.3f; // 30% opacity
    Npp32f alpha2 = 0.7f; // 70% opacity

    // Expected result for ALPHA_OVER: src1*alpha1 + src2*alpha2*(1-alpha1)
    std::vector<Npp32f> expected(totalPixels);
    for (int i = 0; i < totalPixels; ++i) {
        expected[i] = hostSrc1[i] * alpha1 + hostSrc2[i] * alpha2 * (1.0f - alpha1);
    }

    // Allocate GPU memory
    int src1Step, src2Step, dstStep;
    Npp32f *d_src1 = nppiMalloc_32f_C1(width, height, &src1Step);
    Npp32f *d_src2 = nppiMalloc_32f_C1(width, height, &src2Step);
    Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);

    // Copy data to GPU
    int hostStep = width * sizeof(Npp32f);
    cudaMemcpy2D(d_src1, src1Step, hostSrc1.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_src2, src2Step, hostSrc2.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

    // Execute operation
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiAlphaCompC_32f_C1R(d_src1, src1Step, alpha1, 
                                              d_src2, src2Step, alpha2,
                                              d_dst, dstStep, oSizeROI, NPPI_OP_ALPHA_OVER);
    ASSERT_EQ(status, NPP_SUCCESS);

    // Copy result back
    std::vector<Npp32f> hostResult(totalPixels);
    cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < totalPixels; ++i) {
        EXPECT_NEAR(hostResult[i], expected[i], 1e-6f) << "Mismatch at index " << i 
                   << " got " << hostResult[i] 
                   << " expected " << expected[i];
    }

    // Cleanup
    nppiFree(d_src1);
    nppiFree(d_src2);
    nppiFree(d_dst);
}

TEST_F(NppiAlphaCompTest, AlphaCompC_8u_C3R_MultiChannel) {
    const int width = 2;
    const int height = 2;
    const int channels = 3;
    const int totalPixels = width * height * channels;

    // RGB data: R,G,B,R,G,B,...
    std::vector<Npp8u> hostSrc1 = {255, 0, 0,    0, 255, 0,    // Red, Green
                                   0, 0, 255,   255, 255, 255}; // Blue, White
    std::vector<Npp8u> hostSrc2 = {0, 0, 0,     128, 128, 128, // Black, Gray
                                   255, 255, 0, 0, 255, 255};   // Yellow, Cyan
    
    Npp8u alpha1 = 102; // ~40% in 8-bit (102/255 ≈ 0.4)
    Npp8u alpha2 = 153; // ~60% in 8-bit (153/255 ≈ 0.6)

    // Expected result for ALPHA_OVER: src1*alpha1 + src2*alpha2*(1-alpha1)
    std::vector<Npp8u> expected(totalPixels);
    for (int i = 0; i < totalPixels; ++i) {
        double s1 = static_cast<double>(hostSrc1[i]) / 255.0;
        double s2 = static_cast<double>(hostSrc2[i]) / 255.0;
        double a1 = static_cast<double>(alpha1) / 255.0;
        double a2 = static_cast<double>(alpha2) / 255.0;
        double result = s1 * a1 + s2 * a2 * (1.0 - a1);
        expected[i] = static_cast<Npp8u>(std::min(255.0, std::max(0.0, result * 255.0)));
    }

    // Allocate GPU memory
    int src1Step, src2Step, dstStep;
    Npp8u *d_src1 = nppiMalloc_8u_C3(width, height, &src1Step);
    Npp8u *d_src2 = nppiMalloc_8u_C3(width, height, &src2Step);
    Npp8u *d_dst = nppiMalloc_8u_C3(width, height, &dstStep);

    // Copy data to GPU
    int hostStep = width * channels * sizeof(Npp8u);
    cudaMemcpy2D(d_src1, src1Step, hostSrc1.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_src2, src2Step, hostSrc2.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

    // Execute operation
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiAlphaCompC_8u_C3R(d_src1, src1Step, alpha1, 
                                             d_src2, src2Step, alpha2,
                                             d_dst, dstStep, oSizeROI, NPPI_OP_ALPHA_OVER);
    ASSERT_EQ(status, NPP_SUCCESS);

    // Copy result back
    std::vector<Npp8u> hostResult(totalPixels);
    cudaMemcpy2D(hostResult.data(), hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);

    // Verify results (with tolerance for rounding)
    for (int i = 0; i < totalPixels; ++i) {
        EXPECT_NEAR(hostResult[i], expected[i], 2) << "Mismatch at index " << i 
                   << " got " << static_cast<int>(hostResult[i]) 
                   << " expected " << static_cast<int>(expected[i]);
    }

    // Cleanup
    nppiFree(d_src1);
    nppiFree(d_src2);
    nppiFree(d_dst);
}

TEST_F(NppiAlphaCompTest, AlphaCompC_8u_C1R_UnsupportedOperation) {
    const int width = 2;
    const int height = 1;

    std::vector<Npp8u> hostSrc1 = {100, 200};
    std::vector<Npp8u> hostSrc2 = {50, 150};
    
    Npp8u alpha1 = 128;
    Npp8u alpha2 = 192;

    // Allocate GPU memory
    int src1Step, src2Step, dstStep;
    Npp8u *d_src1 = nppiMalloc_8u_C1(width, height, &src1Step);
    Npp8u *d_src2 = nppiMalloc_8u_C1(width, height, &src2Step);
    Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);

    // Copy data to GPU
    int hostStep = width * sizeof(Npp8u);
    cudaMemcpy2D(d_src1, src1Step, hostSrc1.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_src2, src2Step, hostSrc2.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);

    // Execute operation with unsupported mode (premul operations not implemented)
    NppiSize oSizeROI = {width, height};
    NppStatus status = nppiAlphaCompC_8u_C1R(d_src1, src1Step, alpha1, 
                                             d_src2, src2Step, alpha2,
                                             d_dst, dstStep, oSizeROI, NPPI_OP_ALPHA_OVER_PREMUL);
    EXPECT_EQ(status, NPP_NOT_SUPPORTED_MODE_ERROR);

    // Cleanup
    nppiFree(d_src1);
    nppiFree(d_src2);
    nppiFree(d_dst);
}