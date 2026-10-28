// Pilot differential test for nppsSum.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppsSumDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=... ./diff_tests --gtest_filter='NppsSumDiff.*'
//
// This file is the hand-written template for the generator
// (mpp_scripts/gen_family.py); it will be replaced by generated variants once
// the pipeline is validated.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;

namespace {

NppStreamContext current_context() {
  NppStreamContext ctx{};
  NppStatus status = nppGetStreamContext(&ctx);
  EXPECT_EQ(status, NPP_NO_ERROR) << "nppGetStreamContext failed";
  return ctx;
}

} // namespace

class NppsSumDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

TEST_F(NppsSumDiff, Sum_32f_Ctx) {
  const NppSignalLength n = 17;
  std::vector<Npp32f> src = make_uniform<Npp32f>(n, 0x32f);

  DeviceMemory<Npp32f> src_dev(n);
  DeviceMemory<Npp32f> sum_dev(1);
  src_dev.copyFromHost(src);

  NppStreamContext ctx = current_context();
  SIZE_TYPE buffer_size = 0;
  ASSERT_EQ(nppsSumGetBufferSize_32f_Ctx(n, &buffer_size, ctx), NPP_NO_ERROR);
  DeviceMemory<Npp8u> scratch(buffer_size > 0 ? static_cast<size_t>(buffer_size) : 1);

  clear_cuda_error();
  NppStatus status = nppsSum_32f_Ctx(src_dev.get(), n, sum_dev.get(), scratch.get(), ctx);

  Case c("nppsSum/32f/basic", "nppsSum_32f_Ctx");
  c.input(src.data(), src.size() * sizeof(Npp32f));
  c.status(status);
  c.device_output(sum_dev.get(), 1, ValueKind::F32, 1e-6, 1e-6);
  c.finish();
}

#if NPP_DIFF_HAS_NONCTX
TEST_F(NppsSumDiff, Sum_32f_NonCtx) {
  const NppSignalLength n = 17;
  std::vector<Npp32f> src = make_uniform<Npp32f>(n, 0x32f);

  DeviceMemory<Npp32f> src_dev(n);
  DeviceMemory<Npp32f> sum_dev(1);
  src_dev.copyFromHost(src);

  SIZE_TYPE buffer_size = 0;
  ASSERT_EQ(nppsSumGetBufferSize_32f(n, &buffer_size), NPP_NO_ERROR);
  DeviceMemory<Npp8u> scratch(buffer_size > 0 ? static_cast<size_t>(buffer_size) : 1);

  clear_cuda_error();
  NppStatus status = nppsSum_32f(src_dev.get(), n, sum_dev.get(), scratch.get());

  Case c("nppsSum/32f/basic", "nppsSum_32f");
  c.input(src.data(), src.size() * sizeof(Npp32f));
  c.status(status);
  c.device_output(sum_dev.get(), 1, ValueKind::F32, 1e-6, 1e-6);
  c.finish();
}
#endif

TEST_F(NppsSumDiff, Sum_32fc_Ctx) {
  const NppSignalLength n = 9;
  std::vector<Npp32fc> src(n);
  Rng rng(0x32fc);
  for (auto &v : src) {
    v.re = rng.unit_float() * 4.0f - 2.0f;
    v.im = rng.unit_float() * 4.0f - 2.0f;
  }

  DeviceMemory<Npp32fc> src_dev(n);
  DeviceMemory<Npp32fc> sum_dev(1);
  src_dev.copyFromHost(src);

  NppStreamContext ctx = current_context();
  SIZE_TYPE buffer_size = 0;
  ASSERT_EQ(nppsSumGetBufferSize_32fc_Ctx(n, &buffer_size, ctx), NPP_NO_ERROR);
  DeviceMemory<Npp8u> scratch(buffer_size > 0 ? static_cast<size_t>(buffer_size) : 1);

  clear_cuda_error();
  NppStatus status = nppsSum_32fc_Ctx(src_dev.get(), n, sum_dev.get(), scratch.get(), ctx);

  Case c("nppsSum/32fc/basic", "nppsSum_32fc_Ctx");
  c.input(src.data(), src.size() * sizeof(Npp32fc));
  c.status(status);
  c.device_output(sum_dev.get(), 1, ValueKind::CF32, 1e-5, 1e-6);
  c.finish();
}

TEST_F(NppsSumDiff, GetBufferSize_32f_Ctx) {
  const NppSignalLength n = 17;
  SIZE_TYPE buffer_size = 0;
  NppStatus status = nppsSumGetBufferSize_32f_Ctx(n, &buffer_size, current_context());

  // The exact scratch size is an implementation detail; the contract is the
  // status plus the fact that the returned size is sufficient for the call.
  EXPECT_GT(buffer_size, 0);
  Case c("nppsSum/GetBufferSize_32f/basic", "nppsSumGetBufferSize_32f_Ctx");
  c.input(&n, sizeof(n));
  c.status(status);
  c.finish();
}

TEST_F(NppsSumDiff, GetBufferSize_32f_NonCtx) {
#if NPP_DIFF_HAS_NONCTX
  const NppSignalLength n = 17;
  SIZE_TYPE buffer_size = 0;
  NppStatus status = nppsSumGetBufferSize_32f(n, &buffer_size);

  EXPECT_GT(buffer_size, 0);
  Case c("nppsSum/GetBufferSize_32f/basic", "nppsSumGetBufferSize_32f");
  c.input(&n, sizeof(n));
  c.status(status);
  c.finish();
#else
  GTEST_SKIP() << "NPP 13 removed the non-Ctx entry point";
#endif
}
