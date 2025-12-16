#include "npp.h"
#include "framework/npp_test_base.h"
#include <vector>

using namespace npp_functional_test;

class NPPICompareCTest : public NppTestBase {
protected:
  void SetUp() override {
    NppTestBase::SetUp();
    width_ = 32;
    height_ = 24;
    roi_ = {width_, height_};
  }

  // Helper: verify compare result
  template <typename T>
  void verifyCompareC(const std::vector<T> &src, const std::vector<Npp8u> &result, T constant, NppCmpOp operation) {
    ASSERT_EQ(src.size(), result.size());
    for (size_t i = 0; i < result.size(); i++) {
      bool expected = false;
      switch (operation) {
      case NPP_CMP_LESS:
        expected = src[i] < constant;
        break;
      case NPP_CMP_LESS_EQ:
        expected = src[i] <= constant;
        break;
      case NPP_CMP_EQ:
        expected = src[i] == constant;
        break;
      case NPP_CMP_GREATER_EQ:
        expected = src[i] >= constant;
        break;
      case NPP_CMP_GREATER:
        expected = src[i] > constant;
        break;
      }
      Npp8u expectedValue = expected ? 255 : 0;
      EXPECT_EQ(result[i], expectedValue)
          << "Mismatch at index " << i << " src=" << static_cast<int>(src[i]) << " const=" << static_cast<int>(constant)
          << " op=" << operation;
    }
  }

  // Helper: create stream context
  NppStreamContext createStreamContext() {
    NppStreamContext ctx;
    ctx.hStream = 0;
    cudaGetDevice(&ctx.nCudaDeviceId);
    cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor,
                           ctx.nCudaDeviceId);
    cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor,
                           ctx.nCudaDeviceId);
    cudaStreamGetFlags(ctx.hStream, &ctx.nStreamFlags);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, ctx.nCudaDeviceId);
    ctx.nMultiProcessorCount = prop.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;
    return ctx;
  }

  int width_, height_;
  NppiSize roi_;
};

// Test nppiCompareC_8u_C1R and nppiCompareC_8u_C1R_Ctx
TEST_F(NPPICompareCTest, nppiCompareC_8u_C1R) {
  size_t dataSize = width_ * height_;
  std::vector<Npp8u> srcData(dataSize);
  std::vector<Npp8u> dstData(dataSize);

  TestDataGenerator::generateRandom(srcData, Npp8u(0), Npp8u(255), 42);
  Npp8u constant = 128;

  NppImageMemory<Npp8u> d_src(width_, height_);
  NppImageMemory<Npp8u> d_dst(width_, height_);
  d_src.copyFromHost(srcData);

  // Test all comparison operations with non-Ctx version
  NppCmpOp operations[] = {NPP_CMP_LESS, NPP_CMP_LESS_EQ, NPP_CMP_EQ, NPP_CMP_GREATER_EQ, NPP_CMP_GREATER};

  for (NppCmpOp op : operations) {
    NppStatus status = nppiCompareC_8u_C1R(d_src.get(), d_src.step(), constant, d_dst.get(), d_dst.step(), roi_, op);
    EXPECT_EQ(status, NPP_SUCCESS) << "Non-Ctx version failed for op=" << op;

    d_dst.copyToHost(dstData);
    verifyCompareC(srcData, dstData, constant, op);
  }

  // Test Ctx version
  NppStreamContext ctx = createStreamContext();
  for (NppCmpOp op : operations) {
    NppStatus status =
        nppiCompareC_8u_C1R_Ctx(d_src.get(), d_src.step(), constant, d_dst.get(), d_dst.step(), roi_, op, ctx);
    EXPECT_EQ(status, NPP_SUCCESS) << "Ctx version failed for op=" << op;

    d_dst.copyToHost(dstData);
    verifyCompareC(srcData, dstData, constant, op);
  }
}

// Test nppiCompareC_16s_C1R and nppiCompareC_16s_C1R_Ctx
TEST_F(NPPICompareCTest, nppiCompareC_16s_C1R) {
  size_t dataSize = width_ * height_;
  std::vector<Npp16s> srcData(dataSize);
  std::vector<Npp8u> dstData(dataSize);

  TestDataGenerator::generateRandom(srcData, Npp16s(-1000), Npp16s(1000), 42);
  Npp16s constant = 0;

  NppImageMemory<Npp16s> d_src(width_, height_);
  NppImageMemory<Npp8u> d_dst(width_, height_);
  d_src.copyFromHost(srcData);

  NppCmpOp operations[] = {NPP_CMP_LESS, NPP_CMP_LESS_EQ, NPP_CMP_EQ, NPP_CMP_GREATER_EQ, NPP_CMP_GREATER};

  // Test non-Ctx version
  for (NppCmpOp op : operations) {
    NppStatus status = nppiCompareC_16s_C1R(d_src.get(), d_src.step(), constant, d_dst.get(), d_dst.step(), roi_, op);
    EXPECT_EQ(status, NPP_SUCCESS) << "Non-Ctx version failed for op=" << op;

    d_dst.copyToHost(dstData);
    verifyCompareC(srcData, dstData, constant, op);
  }

  // Test Ctx version
  NppStreamContext ctx = createStreamContext();
  for (NppCmpOp op : operations) {
    NppStatus status =
        nppiCompareC_16s_C1R_Ctx(d_src.get(), d_src.step(), constant, d_dst.get(), d_dst.step(), roi_, op, ctx);
    EXPECT_EQ(status, NPP_SUCCESS) << "Ctx version failed for op=" << op;

    d_dst.copyToHost(dstData);
    verifyCompareC(srcData, dstData, constant, op);
  }
}

// Test nppiCompareC_16u_C1R and nppiCompareC_16u_C1R_Ctx
TEST_F(NPPICompareCTest, nppiCompareC_16u_C1R) {
  size_t dataSize = width_ * height_;
  std::vector<Npp16u> srcData(dataSize);
  std::vector<Npp8u> dstData(dataSize);

  TestDataGenerator::generateRandom(srcData, Npp16u(0), Npp16u(65535), 42);
  Npp16u constant = 32768;

  NppImageMemory<Npp16u> d_src(width_, height_);
  NppImageMemory<Npp8u> d_dst(width_, height_);
  d_src.copyFromHost(srcData);

  NppCmpOp operations[] = {NPP_CMP_LESS, NPP_CMP_LESS_EQ, NPP_CMP_EQ, NPP_CMP_GREATER_EQ, NPP_CMP_GREATER};

  // Test non-Ctx version
  for (NppCmpOp op : operations) {
    NppStatus status = nppiCompareC_16u_C1R(d_src.get(), d_src.step(), constant, d_dst.get(), d_dst.step(), roi_, op);
    EXPECT_EQ(status, NPP_SUCCESS) << "Non-Ctx version failed for op=" << op;

    d_dst.copyToHost(dstData);
    verifyCompareC(srcData, dstData, constant, op);
  }

  // Test Ctx version
  NppStreamContext ctx = createStreamContext();
  for (NppCmpOp op : operations) {
    NppStatus status =
        nppiCompareC_16u_C1R_Ctx(d_src.get(), d_src.step(), constant, d_dst.get(), d_dst.step(), roi_, op, ctx);
    EXPECT_EQ(status, NPP_SUCCESS) << "Ctx version failed for op=" << op;

    d_dst.copyToHost(dstData);
    verifyCompareC(srcData, dstData, constant, op);
  }
}

// Test nppiCompareC_32f_C1R and nppiCompareC_32f_C1R_Ctx
TEST_F(NPPICompareCTest, nppiCompareC_32f_C1R) {
  size_t dataSize = width_ * height_;
  std::vector<Npp32f> srcData(dataSize);
  std::vector<Npp8u> dstData(dataSize);

  TestDataGenerator::generateRandom(srcData, Npp32f(-1.0f), Npp32f(1.0f), 42);
  Npp32f constant = 0.0f;

  NppImageMemory<Npp32f> d_src(width_, height_);
  NppImageMemory<Npp8u> d_dst(width_, height_);
  d_src.copyFromHost(srcData);

  NppCmpOp operations[] = {NPP_CMP_LESS, NPP_CMP_LESS_EQ, NPP_CMP_EQ, NPP_CMP_GREATER_EQ, NPP_CMP_GREATER};

  // Test non-Ctx version
  for (NppCmpOp op : operations) {
    NppStatus status = nppiCompareC_32f_C1R(d_src.get(), d_src.step(), constant, d_dst.get(), d_dst.step(), roi_, op);
    EXPECT_EQ(status, NPP_SUCCESS) << "Non-Ctx version failed for op=" << op;

    d_dst.copyToHost(dstData);
    verifyCompareC(srcData, dstData, constant, op);
  }

  // Test Ctx version
  NppStreamContext ctx = createStreamContext();
  for (NppCmpOp op : operations) {
    NppStatus status =
        nppiCompareC_32f_C1R_Ctx(d_src.get(), d_src.step(), constant, d_dst.get(), d_dst.step(), roi_, op, ctx);
    EXPECT_EQ(status, NPP_SUCCESS) << "Ctx version failed for op=" << op;

    d_dst.copyToHost(dstData);
    verifyCompareC(srcData, dstData, constant, op);
  }
}

// Test with specific values to ensure correctness
TEST_F(NPPICompareCTest, nppiCompareC_8u_C1R_SpecificValues) {
  // Use small size for easy verification
  std::vector<Npp8u> srcData = {50, 100, 150, 200, 100};
  std::vector<Npp8u> dstData(5);
  Npp8u constant = 100;

  NppiSize smallRoi = {5, 1};

  NppImageMemory<Npp8u> d_src(5, 1);
  NppImageMemory<Npp8u> d_dst(5, 1);
  d_src.copyFromHost(srcData);

  // Test NPP_CMP_LESS: 50<100=true, 100<100=false, 150<100=false, 200<100=false, 100<100=false
  NppStatus status =
      nppiCompareC_8u_C1R(d_src.get(), d_src.step(), constant, d_dst.get(), d_dst.step(), smallRoi, NPP_CMP_LESS);
  EXPECT_EQ(status, NPP_SUCCESS);
  d_dst.copyToHost(dstData);
  EXPECT_EQ(dstData[0], 255); // 50 < 100
  EXPECT_EQ(dstData[1], 0);   // 100 < 100
  EXPECT_EQ(dstData[2], 0);   // 150 < 100
  EXPECT_EQ(dstData[3], 0);   // 200 < 100
  EXPECT_EQ(dstData[4], 0);   // 100 < 100

  // Test NPP_CMP_EQ: only index 1 and 4 should be true
  status =
      nppiCompareC_8u_C1R(d_src.get(), d_src.step(), constant, d_dst.get(), d_dst.step(), smallRoi, NPP_CMP_EQ);
  EXPECT_EQ(status, NPP_SUCCESS);
  d_dst.copyToHost(dstData);
  EXPECT_EQ(dstData[0], 0);   // 50 == 100
  EXPECT_EQ(dstData[1], 255); // 100 == 100
  EXPECT_EQ(dstData[2], 0);   // 150 == 100
  EXPECT_EQ(dstData[3], 0);   // 200 == 100
  EXPECT_EQ(dstData[4], 255); // 100 == 100

  // Test NPP_CMP_GREATER: 150>100=true, 200>100=true
  status =
      nppiCompareC_8u_C1R(d_src.get(), d_src.step(), constant, d_dst.get(), d_dst.step(), smallRoi, NPP_CMP_GREATER);
  EXPECT_EQ(status, NPP_SUCCESS);
  d_dst.copyToHost(dstData);
  EXPECT_EQ(dstData[0], 0);   // 50 > 100
  EXPECT_EQ(dstData[1], 0);   // 100 > 100
  EXPECT_EQ(dstData[2], 255); // 150 > 100
  EXPECT_EQ(dstData[3], 255); // 200 > 100
  EXPECT_EQ(dstData[4], 0);   // 100 > 100
}

// Test 16s with negative values
TEST_F(NPPICompareCTest, nppiCompareC_16s_C1R_NegativeValues) {
  std::vector<Npp16s> srcData = {-500, -100, 0, 100, 500};
  std::vector<Npp8u> dstData(5);
  Npp16s constant = 0;

  NppiSize smallRoi = {5, 1};

  NppImageMemory<Npp16s> d_src(5, 1);
  NppImageMemory<Npp8u> d_dst(5, 1);
  d_src.copyFromHost(srcData);

  // Test NPP_CMP_LESS: -500<0=true, -100<0=true, 0<0=false, 100<0=false, 500<0=false
  NppStatus status =
      nppiCompareC_16s_C1R(d_src.get(), d_src.step(), constant, d_dst.get(), d_dst.step(), smallRoi, NPP_CMP_LESS);
  EXPECT_EQ(status, NPP_SUCCESS);
  d_dst.copyToHost(dstData);
  EXPECT_EQ(dstData[0], 255); // -500 < 0
  EXPECT_EQ(dstData[1], 255); // -100 < 0
  EXPECT_EQ(dstData[2], 0);   // 0 < 0
  EXPECT_EQ(dstData[3], 0);   // 100 < 0
  EXPECT_EQ(dstData[4], 0);   // 500 < 0

  // Test NPP_CMP_GREATER_EQ
  status = nppiCompareC_16s_C1R(d_src.get(), d_src.step(), constant, d_dst.get(), d_dst.step(), smallRoi,
                                 NPP_CMP_GREATER_EQ);
  EXPECT_EQ(status, NPP_SUCCESS);
  d_dst.copyToHost(dstData);
  EXPECT_EQ(dstData[0], 0);   // -500 >= 0
  EXPECT_EQ(dstData[1], 0);   // -100 >= 0
  EXPECT_EQ(dstData[2], 255); // 0 >= 0
  EXPECT_EQ(dstData[3], 255); // 100 >= 0
  EXPECT_EQ(dstData[4], 255); // 500 >= 0
}
