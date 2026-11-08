// Differential tests for nppiMSSSIM (deprecated MS-SSIM*).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   tools/remote5090.sh record 'NppiMsssimDiff.*'
//
// The reference combination of per-scale statistics is not the classic
// weighted MS-SSIM product (see nppi_msssim.cu).  These cases therefore pin
// only the behaviours this executor reproduces exactly: identical images
// (1.0), two different flat images (0.0), and the status contract (ROI >=
// 176x176, empty ROI no-op, source step >= width).
#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cstdint>
#include <string>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;

namespace {

NppStreamContext current_context() {
  NppStreamContext ctx{};
  NppStatus st = nppGetStreamContext(&ctx);
  EXPECT_EQ(st, NPP_NO_ERROR) << "nppGetStreamContext failed";
  return ctx;
}

void status_case(const std::string &id, const char *fname, NppStatus st) {
  Case c(id, fname);
  c.status(st);
  c.finish();
}

void value_case(const std::string &id, const char *fname, int w, int h, int kind) {
  std::vector<Npp8u> a((size_t)w * h), b((size_t)w * h);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const size_t i = (size_t)y * w + x;
      if (kind == 0) {  // identical smooth texture
        a[i] = b[i] = (Npp8u)((x * 7 + y * 13 + 5) & 0xff);
      } else if (kind == 1) {  // identical ramp
        a[i] = b[i] = (Npp8u)((x * 3 + y * 5) & 0xff);
      } else {  // two different flat images
        a[i] = 100;
        b[i] = 110;
      }
    }
  }
  DeviceMemory<Npp8u> src1(a.size()), src2(b.size());
  src1.copyFromHost(a);
  src2.copyFromHost(b);
#ifdef USE_NVIDIA_NPP_TESTS
  size_t bufferSize = 0;
#else
  NPP_BUFFER_SIZE_TYPE bufferSize = 0;
#endif
  nppiMSSSIMGetBufferHostSize_8u_C1R_Ctx({w, h}, &bufferSize, current_context());
  DeviceMemory<Npp8u> buffer(bufferSize > 0 ? bufferSize : 1);
  DeviceMemory<Npp32f> result(1);
  std::vector<Npp32f> init(1, -1.0e30f);
  result.copyFromHost(init);

  NppStreamContext ctx = current_context();
  NppStatus st = nppiMSSSIM_8u_C1R_Ctx(src1.get(), w, src2.get(), w, {w, h}, result.get(), buffer.get(), ctx);
  clear_cuda_error();

  Case c(id, fname);
  c.status(st);
  c.input(a.data(), a.size());
  c.input(b.data(), b.size());
  if (st == NPP_NO_ERROR) {
    std::vector<Npp32f> host(1);
    result.copyToHost(host);
    c.host_output(host.data(), 1, ValueKind::F32, 1e-5, 1e-5);
  }
  c.finish();
}

}  // namespace

TEST(NppiMsssimDiff, Values) {
  const char *fname = "nppiMSSSIM_8u_C1R_Ctx";
  value_case("s2/MSSSIM/identical_smooth_176", fname, 176, 176, 0);
  value_case("s2/MSSSIM/identical_ramp_176", fname, 176, 176, 1);
  value_case("s2/MSSSIM/flat_100_110_176", fname, 176, 176, 2);
  value_case("s2/MSSSIM/identical_smooth_200x176", fname, 200, 176, 0);
}

TEST(NppiMsssimDiff, Contract) {
  const int w = 176, h = 176;
  std::vector<Npp8u> img((size_t)w * h, 7);
  DeviceMemory<Npp8u> src(img.size()), src2(img.size());
  src.copyFromHost(img);
  src2.copyFromHost(img);
#ifdef USE_NVIDIA_NPP_TESTS
  size_t bufferSize = 0;
#else
  NPP_BUFFER_SIZE_TYPE bufferSize = 0;
#endif
  nppiMSSSIMGetBufferHostSize_8u_C1R_Ctx({w, h}, &bufferSize, current_context());
  DeviceMemory<Npp8u> buffer(bufferSize > 0 ? bufferSize : 1);
  DeviceMemory<Npp32f> result(1);
  NppStreamContext ctx = current_context();
  const char *fname = "nppiMSSSIM_8u_C1R_Ctx";

  status_case("s2/MSSSIM/null_src1", fname,
              nppiMSSSIM_8u_C1R_Ctx(nullptr, w, src2.get(), w, {w, h}, result.get(), buffer.get(), ctx));
  status_case("s2/MSSSIM/null_src2", fname,
              nppiMSSSIM_8u_C1R_Ctx(src.get(), w, nullptr, w, {w, h}, result.get(), buffer.get(), ctx));
  status_case("s2/MSSSIM/null_out", fname,
              nppiMSSSIM_8u_C1R_Ctx(src.get(), w, src2.get(), w, {w, h}, nullptr, buffer.get(), ctx));
  status_case("s2/MSSSIM/null_buffer", fname,
              nppiMSSSIM_8u_C1R_Ctx(src.get(), w, src2.get(), w, {w, h}, result.get(), nullptr, ctx));
  status_case("s2/MSSSIM/roi_175x176", fname,
              nppiMSSSIM_8u_C1R_Ctx(src.get(), w, src2.get(), w, {175, 176}, result.get(), buffer.get(), ctx));
  status_case("s2/MSSSIM/roi_176x175", fname,
              nppiMSSSIM_8u_C1R_Ctx(src.get(), w, src2.get(), w, {176, 175}, result.get(), buffer.get(), ctx));
  status_case("s2/MSSSIM/roi_zero_width", fname,
              nppiMSSSIM_8u_C1R_Ctx(src.get(), w, src2.get(), w, {0, 176}, result.get(), buffer.get(), ctx));
  status_case("s2/MSSSIM/roi_negative", fname,
              nppiMSSSIM_8u_C1R_Ctx(src.get(), w, src2.get(), w, {-1, 176}, result.get(), buffer.get(), ctx));
  status_case("s2/MSSSIM/step_short", fname,
              nppiMSSSIM_8u_C1R_Ctx(src.get(), 175, src2.get(), w, {w, h}, result.get(), buffer.get(), ctx));
  status_case("s2/MSSSIM/valid", fname,
              nppiMSSSIM_8u_C1R_Ctx(src.get(), w, src2.get(), w, {w, h}, result.get(), buffer.get(), ctx));
  clear_cuda_error();
}
