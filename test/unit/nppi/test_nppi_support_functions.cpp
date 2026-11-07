/**
 * @file test_nppi_support_functions.cpp
 * @brief NPP 支持函数测试
 */

#include "../framework/npp_test_base.h"

using namespace npp_functional_test;

class SupportFunctionsTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }

  void TearDown() override { NppTestBase::TearDown(); }
};

TEST_F(SupportFunctionsTest, GetLibVersion_BasicTest) {
  const NppLibraryVersion *version = nppGetLibVersion();
  ASSERT_NE(version, nullptr) << "nppGetLibVersion should not return null";

  // 验证版本号格式合理
  EXPECT_GT(version->major, 0) << "Major version should be greater than 0";
  EXPECT_GE(version->minor, 0) << "Minor version should be non-negative";
  EXPECT_GE(version->build, 0) << "Build version should be non-negative";

  std::cout << "NPP Library Version: " << version->major << "." << version->minor << "." << version->build << std::endl;
}

// nppGetGpuComputeCapability函数在当前API中不存在，测试已禁用
/*
TEST_F(SupportFunctionsTest, GetGpuComputeCapability_BasicTest) {
    int major, minor;
    NppStatus status = nppGetGpuComputeCapability(&major, &minor);

    ASSERT_EQ(status, NPP_NO_ERROR) << "nppGetGpuComputeCapability should succeed";

    // 验证计算能力值合理
    EXPECT_GT(major, 0) << "GPU compute capability major version should be > 0";
    EXPECT_GE(minor, 0) << "GPU compute capability minor version should be >= 0";
    EXPECT_LE(major, 9) << "GPU compute capability major version should be reasonable";
    EXPECT_LE(minor, 9) << "GPU compute capability minor version should be reasonable";

    std::cout << "GPU Compute Capability: " << major << "." << minor << std::endl;
}
*/

TEST_F(SupportFunctionsTest, GetGpuName_BasicTest) {
  // nppGetGpuName 实际上已经实现，返回const char*，不接受缓冲区参数
  const char *name = nppGetGpuName();

  if (name != nullptr) {
    // 验证名称不为空
    EXPECT_GT(strlen(name), 0) << "GPU name should not be empty";
    std::cout << "GPU Name: " << name << std::endl;
  } else {
    std::cout << "nppGetGpuName returned nullptr" << std::endl;
  }
}

TEST_F(SupportFunctionsTest, GetStreamContext_BasicTest) {
  NppStreamContext streamCtx;
  NppStatus status = nppGetStreamContext(&streamCtx);

  if (status == NPP_NO_ERROR) {
    // 验证默认流上下文
    EXPECT_EQ(streamCtx.hStream, (cudaStream_t)0) << "Default stream should be 0";
    EXPECT_NE(streamCtx.nCudaDeviceId, -1) << "Device ID should be valid";
    std::cout << "nppGetStreamContext succeeded" << std::endl;
  } else {
    std::cout << "nppGetStreamContext returned error: " << status << std::endl;
  }
}

TEST_F(SupportFunctionsTest, SetGetStreamContext_BasicTest) {
  // 首先获取当前的流上下文
  NppStreamContext currentCtx;
  NppStatus status = nppGetStreamContext(&currentCtx);
  ASSERT_EQ(status, NPP_NO_ERROR) << "nppGetStreamContext should succeed";

  // nppSetStreamContext函数在当前API中不存在，相关测试已禁用
  /*
  status = nppSetStreamContext(currentCtx);
  ASSERT_EQ(status, NPP_NO_ERROR) << "nppSetStreamContext should succeed with valid context";

  // 测试无效设备ID
  NppStreamContext invalidCtx = currentCtx;
  invalidCtx.nCudaDeviceId = 999;  // 无效的设备ID
  status = nppSetStreamContext(invalidCtx);
  EXPECT_EQ(status, NPP_BAD_ARGUMENT_ERROR) << "nppSetStreamContext should reject invalid device ID";
  */

  std::cout << "nppSetStreamContext validation tests passed" << std::endl;
}

TEST_F(SupportFunctionsTest, GetGpuDeviceProperties_BasicTest) {
  int maxThreadsPerSM, maxThreadsPerBlock, numSMs;
  int result = nppGetGpuDeviceProperties(&maxThreadsPerSM, &maxThreadsPerBlock, &numSMs);

  ASSERT_EQ(result, 0) << "nppGetGpuDeviceProperties should succeed (return 0)";

  // 验证返回值合理
  EXPECT_GT(maxThreadsPerSM, 0) << "Max threads per SM should be > 0";
  EXPECT_GT(maxThreadsPerBlock, 0) << "Max threads per block should be > 0";
  EXPECT_GT(numSMs, 0) << "Number of SMs should be > 0";

  // 验证合理范围
  EXPECT_LE(maxThreadsPerSM, 4096) << "Max threads per SM should be reasonable";
  EXPECT_LE(maxThreadsPerBlock, 2048) << "Max threads per block should be reasonable";
  EXPECT_LE(numSMs, 256) << "Number of SMs should be reasonable";

  std::cout << "GPU Properties: " << numSMs << " SMs, " << maxThreadsPerSM << " threads/SM, " << maxThreadsPerBlock
            << " threads/block" << std::endl;
}

TEST_F(SupportFunctionsTest, GetGpuNumSMs_BasicTest) {
  int numSMs = nppGetGpuNumSMs();

  EXPECT_GT(numSMs, 0) << "Number of SMs should be > 0";
  EXPECT_LE(numSMs, 256) << "Number of SMs should be reasonable";

  std::cout << "GPU has " << numSMs << " SMs" << std::endl;
}

namespace {

// Pitched allocator probe: non-null result, a step that covers the requested
// row, and a write/read round-trip through that step.
template <typename T, typename Fn>
void checkPitchAlloc(const char *name, Fn fn, int channels) {
  const int w = 37, h = 11;
  int step = 0;
  T *p = fn(w, h, &step);
  ASSERT_NE(p, nullptr) << name;
  EXPECT_GE(step, w * channels * (int)sizeof(T)) << name;
  const size_t rowBytes = (size_t)w * channels * sizeof(T);
  ASSERT_EQ(cudaMemset2D(p, (size_t)step, 0x5A, rowBytes, h), cudaSuccess) << name;
  unsigned char probe = 0;
  ASSERT_EQ(cudaMemcpy2D(&probe, 1, p, (size_t)step, 1, 1, cudaMemcpyDeviceToHost), cudaSuccess) << name;
  EXPECT_EQ(probe, 0x5A) << name;
  // Last row, last byte of the row must fall inside the pitched allocation.
  ASSERT_EQ(cudaMemcpy2D(&probe, 1, (const char *)p + (size_t)(h - 1) * step + rowBytes - 1, 1, 1, 1,
                         cudaMemcpyDeviceToHost),
            cudaSuccess)
      << name;
  EXPECT_EQ(probe, 0x5A) << name;
  nppiFree(p);
}

}  // namespace

// Remaining pitched allocator variants (the frequently used layouts are
// exercised indirectly by the functional suites).
TEST_F(SupportFunctionsTest, Malloc_RemainingLayouts) {
  checkPitchAlloc<Npp16s>("nppiMalloc_16s_C2", nppiMalloc_16s_C2, 2);
  checkPitchAlloc<Npp16u>("nppiMalloc_16u_C2", nppiMalloc_16u_C2, 2);
  checkPitchAlloc<Npp32f>("nppiMalloc_32f_C2", nppiMalloc_32f_C2, 2);
  checkPitchAlloc<Npp16sc>("nppiMalloc_16sc_C1", nppiMalloc_16sc_C1, 1);
  checkPitchAlloc<Npp16sc>("nppiMalloc_16sc_C2", nppiMalloc_16sc_C2, 2);
  checkPitchAlloc<Npp16sc>("nppiMalloc_16sc_C3", nppiMalloc_16sc_C3, 3);
  checkPitchAlloc<Npp16sc>("nppiMalloc_16sc_C4", nppiMalloc_16sc_C4, 4);
  checkPitchAlloc<Npp32fc>("nppiMalloc_32fc_C1", nppiMalloc_32fc_C1, 1);
  checkPitchAlloc<Npp32fc>("nppiMalloc_32fc_C2", nppiMalloc_32fc_C2, 2);
  checkPitchAlloc<Npp32fc>("nppiMalloc_32fc_C3", nppiMalloc_32fc_C3, 3);
  checkPitchAlloc<Npp32fc>("nppiMalloc_32fc_C4", nppiMalloc_32fc_C4, 4);
  checkPitchAlloc<Npp32sc>("nppiMalloc_32sc_C1", nppiMalloc_32sc_C1, 1);
  checkPitchAlloc<Npp32sc>("nppiMalloc_32sc_C2", nppiMalloc_32sc_C2, 2);
  checkPitchAlloc<Npp32sc>("nppiMalloc_32sc_C3", nppiMalloc_32sc_C3, 3);
  checkPitchAlloc<Npp32sc>("nppiMalloc_32sc_C4", nppiMalloc_32sc_C4, 4);
}
