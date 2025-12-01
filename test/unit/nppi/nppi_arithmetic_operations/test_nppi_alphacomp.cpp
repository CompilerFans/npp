#include "npp.h"
#include <gtest/gtest.h>
#include <cmath>

class NppiAlphaCompPixelTest : public ::testing::Test {
protected:
  void SetUp() override {
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
      FAIL() << "Failed to set CUDA device";
    }
  }

  void TearDown() override {
    cudaDeviceSynchronize();
  }
};

TEST_F(NppiAlphaCompPixelTest, Basic_8u_AC4R_AlphaOver) {
  const int width = 2;
  const int height = 1;

  Npp8u src1[8] = {
    255, 0, 0, 128,    // Red with 50% alpha
    0, 255, 0, 255     // Green with 100% alpha
  };

  Npp8u src2[8] = {
    0, 0, 255, 64,     // Blue with 25% alpha
    255, 255, 255, 100 // White with ~39% alpha
  };

  // Allocate GPU memory
  int src1Step, src2Step, dstStep;
  Npp8u *d_src1 = nppiMalloc_8u_C4(width, height, &src1Step);
  Npp8u *d_src2 = nppiMalloc_8u_C4(width, height, &src2Step);
  Npp8u *d_dst = nppiMalloc_8u_C4(width, height, &dstStep);

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  // Copy data
  int hostStep = width * 4 * sizeof(Npp8u);
  cudaMemcpy2D(d_src1, src1Step, src1, hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, src2, hostStep, hostStep, height, cudaMemcpyHostToDevice);

  // Execute
  NppiSize roi = {width, height};
  NppStatus status = nppiAlphaComp_8u_AC4R(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, NPPI_OP_ALPHA_OVER);
  EXPECT_EQ(status, NPP_SUCCESS);

  if (status == NPP_SUCCESS) {
    Npp8u result[8];
    cudaMemcpy2D(result, hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);

    // Simple verification - just check that we got some output
    bool hasValidOutput = false;
    for (int i = 0; i < 8; i++) {
      if (result[i] != 0) {
        hasValidOutput = true;
        break;
      }
    }
    EXPECT_TRUE(hasValidOutput);
  }

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

TEST_F(NppiAlphaCompPixelTest, Ctx_8u_AC4R_AlphaOver) {
  const int width = 1;
  const int height = 1;

  Npp8u src1[4] = {100, 200, 50, 128};
  Npp8u src2[4] = {50, 150, 200, 64};

  int src1Step, src2Step, dstStep;
  Npp8u *d_src1 = nppiMalloc_8u_C4(width, height, &src1Step);
  Npp8u *d_src2 = nppiMalloc_8u_C4(width, height, &src2Step);
  Npp8u *d_dst = nppiMalloc_8u_C4(width, height, &dstStep);

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostStep = width * 4 * sizeof(Npp8u);
  cudaMemcpy2D(d_src1, src1Step, src1, hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, src2, hostStep, hostStep, height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  memset(&ctx, 0, sizeof(ctx));
  ctx.hStream = 0;
  NppStatus status = nppiAlphaComp_8u_AC4R_Ctx(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi,
                                              NPPI_OP_ALPHA_OVER, ctx);
  EXPECT_EQ(status, NPP_SUCCESS);

  if (status == NPP_SUCCESS) {
    Npp8u result[4];
    cudaMemcpy2D(result, hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);

    bool hasOutput = false;
    for (int i = 0; i < 4; i++) {
      if (result[i] != 0) {
        hasOutput = true;
        break;
      }
    }
    EXPECT_TRUE(hasOutput);
  }

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

TEST_F(NppiAlphaCompPixelTest, Basic_16u_AC4R_AlphaOver) {
  const int width = 1;
  const int height = 1;

  Npp16u src1[4] = {40000, 20000, 10000, 30000};
  Npp16u src2[4] = {10000, 30000, 40000, 20000};

  int src1Step, src2Step, dstStep;
  Npp16u *d_src1 = nppiMalloc_16u_C4(width, height, &src1Step);
  Npp16u *d_src2 = nppiMalloc_16u_C4(width, height, &src2Step);
  Npp16u *d_dst = nppiMalloc_16u_C4(width, height, &dstStep);

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostStep = width * 4 * sizeof(Npp16u);
  cudaMemcpy2D(d_src1, src1Step, src1, hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, src2, hostStep, hostStep, height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status = nppiAlphaComp_16u_AC4R(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, NPPI_OP_ALPHA_OVER);
  EXPECT_EQ(status, NPP_SUCCESS);

  if (status == NPP_SUCCESS) {
    Npp16u result[4];
    cudaMemcpy2D(result, hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);
    EXPECT_TRUE(result[0] != 0 || result[1] != 0 || result[2] != 0 || result[3] != 0);
  }

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

TEST_F(NppiAlphaCompPixelTest, Ctx_16u_AC4R_AlphaOver) {
  const int width = 1;
  const int height = 1;

  Npp16u src1[4] = {5000, 10000, 15000, 20000};
  Npp16u src2[4] = {10000, 5000, 20000, 10000};

  int src1Step, src2Step, dstStep;
  Npp16u *d_src1 = nppiMalloc_16u_C4(width, height, &src1Step);
  Npp16u *d_src2 = nppiMalloc_16u_C4(width, height, &src2Step);
  Npp16u *d_dst = nppiMalloc_16u_C4(width, height, &dstStep);

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostStep = width * 4 * sizeof(Npp16u);
  cudaMemcpy2D(d_src1, src1Step, src1, hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, src2, hostStep, hostStep, height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  memset(&ctx, 0, sizeof(ctx));
  ctx.hStream = 0;
  NppStatus status = nppiAlphaComp_16u_AC4R_Ctx(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi,
                                               NPPI_OP_ALPHA_OVER, ctx);
  EXPECT_EQ(status, NPP_SUCCESS);

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

TEST_F(NppiAlphaCompPixelTest, Basic_32f_AC4R_AlphaOver) {
  const int width = 1;
  const int height = 1;

  Npp32f src1[4] = {0.5f, 0.2f, 0.8f, 0.6f};
  Npp32f src2[4] = {0.1f, 0.7f, 0.3f, 0.4f};

  int src1Step, src2Step, dstStep;
  Npp32f *d_src1 = nppiMalloc_32f_C4(width, height, &src1Step);
  Npp32f *d_src2 = nppiMalloc_32f_C4(width, height, &src2Step);
  Npp32f *d_dst = nppiMalloc_32f_C4(width, height, &dstStep);

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostStep = width * 4 * sizeof(Npp32f);
  cudaMemcpy2D(d_src1, src1Step, src1, hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, src2, hostStep, hostStep, height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status = nppiAlphaComp_32f_AC4R(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, NPPI_OP_ALPHA_OVER);
  EXPECT_EQ(status, NPP_SUCCESS);

  if (status == NPP_SUCCESS) {
    Npp32f result[4];
    cudaMemcpy2D(result, hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);
    EXPECT_TRUE(result[0] != 0.0f || result[1] != 0.0f || result[2] != 0.0f || result[3] != 0.0f);
  }

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

TEST_F(NppiAlphaCompPixelTest, Ctx_32f_AC4R_AlphaOver) {
  const int width = 1;
  const int height = 1;

  Npp32f src1[4] = {0.3f, 0.6f, 0.9f, 0.7f};
  Npp32f src2[4] = {0.8f, 0.4f, 0.1f, 0.2f};

  int src1Step, src2Step, dstStep;
  Npp32f *d_src1 = nppiMalloc_32f_C4(width, height, &src1Step);
  Npp32f *d_src2 = nppiMalloc_32f_C4(width, height, &src2Step);
  Npp32f *d_dst = nppiMalloc_32f_C4(width, height, &dstStep);

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostStep = width * 4 * sizeof(Npp32f);
  cudaMemcpy2D(d_src1, src1Step, src1, hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, src2, hostStep, hostStep, height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  memset(&ctx, 0, sizeof(ctx));
  ctx.hStream = 0;
  NppStatus status = nppiAlphaComp_32f_AC4R_Ctx(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi,
                                               NPPI_OP_ALPHA_OVER, ctx);
  EXPECT_EQ(status, NPP_SUCCESS);

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

TEST_F(NppiAlphaCompPixelTest, DifferentAlphaOps) {
  const int width = 1;
  const int height = 1;

  Npp8u src1[4] = {200, 100, 50, 128};
  Npp8u src2[4] = {50, 150, 200, 64};

  int src1Step, src2Step, dstStep;
  Npp8u *d_src1 = nppiMalloc_8u_C4(width, height, &src1Step);
  Npp8u *d_src2 = nppiMalloc_8u_C4(width, height, &src2Step);
  Npp8u *d_dst = nppiMalloc_8u_C4(width, height, &dstStep);

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostStep = width * 4 * sizeof(Npp8u);
  cudaMemcpy2D(d_src1, src1Step, src1, hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, src2, hostStep, hostStep, height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};

  // Test different alpha operations
  NppiAlphaOp ops[] = {
    NPPI_OP_ALPHA_OVER,
    NPPI_OP_ALPHA_IN,
    NPPI_OP_ALPHA_OUT,
    NPPI_OP_ALPHA_ATOP,
    NPPI_OP_ALPHA_XOR,
    NPPI_OP_ALPHA_PLUS,
    NPPI_OP_ALPHA_OVER_PREMUL,
    NPPI_OP_ALPHA_IN_PREMUL,
    NPPI_OP_ALPHA_OUT_PREMUL,
    NPPI_OP_ALPHA_ATOP_PREMUL,
    NPPI_OP_ALPHA_XOR_PREMUL,
    NPPI_OP_ALPHA_PLUS_PREMUL
  };

  for (int i = 0; i < 12; i++) {
    NppStatus status = nppiAlphaComp_8u_AC4R(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, ops[i]);
    EXPECT_EQ(status, NPP_SUCCESS) << "Failed for operation index " << i;

    if (status == NPP_SUCCESS) {
      Npp8u result[4];
      cudaMemcpy2D(result, hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);
      // Just verify we got some output
      bool hasOutput = result[0] != 0 || result[1] != 0 || result[2] != 0 || result[3] != 0;
      EXPECT_TRUE(hasOutput) << "No output for operation index " << i;
    }
  }

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

TEST_F(NppiAlphaCompPixelTest, Ctx_32s_AC4R_AlphaOver) {
  const int width = 1;
  const int height = 1;

  Npp32s src1[4] = {75, 125, 175, 225};
  Npp32s src2[4] = {225, 175, 125, 75};

  int src1Step, src2Step, dstStep;
  Npp32s *d_src1 = nppiMalloc_32s_C4(width, height, &src1Step);
  Npp32s *d_src2 = nppiMalloc_32s_C4(width, height, &src2Step);
  Npp32s *d_dst = nppiMalloc_32s_C4(width, height, &dstStep);

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostStep = width * 4 * sizeof(Npp32s);
  cudaMemcpy2D(d_src1, src1Step, src1, hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, src2, hostStep, hostStep, height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStreamContext ctx;
  memset(&ctx, 0, sizeof(ctx));
  ctx.hStream = 0;
  NppStatus status = nppiAlphaComp_32s_AC4R_Ctx(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi,
                                               NPPI_OP_ALPHA_OVER, ctx);
  EXPECT_EQ(status, NPP_SUCCESS);

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}

TEST_F(NppiAlphaCompPixelTest, Basic_32s_AC4R_AlphaOver) {
  const int width = 1;
  const int height = 1;

  Npp32s src1[4] = {100, 200, 50, 128};
  Npp32s src2[4] = {50, 150, 200, 64};

  int src1Step, src2Step, dstStep;
  Npp32s *d_src1 = nppiMalloc_32s_C4(width, height, &src1Step);
  Npp32s *d_src2 = nppiMalloc_32s_C4(width, height, &src2Step);
  Npp32s *d_dst = nppiMalloc_32s_C4(width, height, &dstStep);

  ASSERT_NE(d_src1, nullptr);
  ASSERT_NE(d_src2, nullptr);
  ASSERT_NE(d_dst, nullptr);

  int hostStep = width * 4 * sizeof(Npp32s);
  cudaMemcpy2D(d_src1, src1Step, src1, hostStep, hostStep, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_src2, src2Step, src2, hostStep, hostStep, height, cudaMemcpyHostToDevice);

  NppiSize roi = {width, height};
  NppStatus status = nppiAlphaComp_32s_AC4R(d_src1, src1Step, d_src2, src2Step, d_dst, dstStep, roi, NPPI_OP_ALPHA_OVER);
  EXPECT_EQ(status, NPP_SUCCESS);

  if (status == NPP_SUCCESS) {
    Npp32s result[4];
    cudaMemcpy2D(result, hostStep, d_dst, dstStep, hostStep, height, cudaMemcpyDeviceToHost);
    EXPECT_TRUE(result[0] != 0 || result[1] != 0 || result[2] != 0 || result[3] != 0);
  }

  nppiFree(d_src1);
  nppiFree(d_src2);
  nppiFree(d_dst);
}
