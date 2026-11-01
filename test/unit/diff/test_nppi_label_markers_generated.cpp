// Differential tests for package f13 family 4 (part): label markers
// (nppiLabelMarkersUF_{16u32u,32u}_C1R and the batch forms, 14 functions).
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiLabelMarkersDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records LD_LIBRARY_PATH=build/src \
//     ./build/diff_tests --gtest_filter='NppiLabelMarkersDiff.*'
//
// Measured semantics (5090, NPP 13.4):
//  - labels are 0-based linear indices of the component's first pixel in scan
//    order (matching the existing 8u32u Union-Find implementation exactly)
//  - nppiNormL1 is 4-way connectivity, nppiNormInf is 8-way; the batch form
//    honours the same rule
//  - null pSrc/pDst/pBuffer (-8), non-positive ROI (-6), non-positive steps
//    (-14), unsupported norm (-5); a null buffer is an error even though the
//    implementation keeps its Union-Find inside the label image
//  - the non-Advanced batch entry points take the common per-image ROI and
//    forward to the Advanced form with that ROI as the largest ROI

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <memory>
#include <string>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;
namespace {

template <typename T> DeviceMemory<T> &arena(std::size_t n) {
  static std::vector<std::unique_ptr<DeviceMemory<T>>> v;
  v.push_back(std::make_unique<DeviceMemory<T>>(n));
  return *v.back();
}

NppStreamContext current_ctx() {
  NppStreamContext c{};
  NppStatus st = nppGetStreamContext(&c);
  EXPECT_EQ(st, NPP_NO_ERROR);
  return c;
}

template <typename F> void only_status(const std::string &fname, const std::string &key, F f) {
  clear_cuda_error();
  NppStatus st = f();
  Case c(fname + "/" + key, fname);
  c.status(st);
  c.finish();
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) ADD_FAILURE() << "async CUDA fault after " << key << ": "
                                      << cudaGetErrorString(e);
}

// Single-image label markers with a patterned source and an explicit buffer.
template <typename Src, typename F>
void single_case(const std::string &fn, const std::string &key, int W, int H, NppiNorm norm,
                 uint64_t seed, F f) {
  std::vector<Src> hs((size_t)W * H);
  Rng rng(seed);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x)
      hs[y * W + x] = (Src)((rng.integer<int>(0, 3) % 4) * (sizeof(Src) == 1 ? 40 : 900));
  DeviceMemory<Src> &s = arena<Src>((size_t)W * H);
  DeviceMemory<Npp32u> &d = arena<Npp32u>((size_t)W * H);
  DeviceMemory<Npp8u> &buf = arena<Npp8u>(1 << 20);
  s.copyFromHost(hs);
  d.copyFromHost(std::vector<Npp32u>((size_t)W * H, 0xAAAAAAAAu));
  buf.copyFromHost(std::vector<Npp8u>(1 << 20, 0));
  clear_cuda_error();
  NppStatus st = f(s.get(), (int)sizeof(Src) * W, d.get(), 4 * W, NppiSize{W, H}, norm, buf.get(),
                   current_ctx());
  Case c(fn + "/" + key, fn);
  c.status(st);
  c.device_output<Npp32u>(d.get(), (size_t)W * H, ValueKind::U32, 0.0, 0.0);
  c.finish();
}

// Batch label markers (Advanced and non-Advanced) with two images.
template <typename Src, typename F>
void batch_case(const std::string &fn, const std::string &key, int W, int H, NppiNorm norm,
                bool advanced, uint64_t seed, F f) {
  const int nImg = 2;
  std::vector<Src> hs((size_t)W * H * nImg);
  Rng rng(seed);
  for (auto &v : hs) v = (Src)((rng.integer<int>(0, 3) % 4) * (sizeof(Src) == 1 ? 40 : 900));
  DeviceMemory<Src> &s = arena<Src>((size_t)W * H * nImg);
  DeviceMemory<Npp32u> &d = arena<Npp32u>((size_t)W * H * nImg);
  DeviceMemory<Npp8u> &buf = arena<Npp8u>(1 << 20);
  s.copyFromHost(hs);
  d.copyFromHost(std::vector<Npp32u>((size_t)W * H * nImg, 0xAAAAAAAAu));
  buf.copyFromHost(std::vector<Npp8u>(1 << 20, 0));
  NppiImageDescriptor src[2] = {{s.get(), W * (int)sizeof(Src), {W, H}},
                                {(char *)s.get() + (size_t)W * H * sizeof(Src),
                                 W * (int)sizeof(Src), {W, H}}};
  NppiImageDescriptor dst[2] = {{d.get(), 4 * W, {W, H}},
                                {(char *)d.get() + (size_t)W * H * 4, 4 * W, {W, H}}};
  clear_cuda_error();
  NppStatus st = f(src, dst, nImg, NppiSize{W, H}, norm, advanced, current_ctx());
  Case c(fn + "/" + key, fn);
  c.status(st);
  c.device_output<Npp32u>(d.get(), (size_t)W * H * nImg, ValueKind::U32, 0.0, 0.0);
  c.finish();
}

}  // namespace

TEST(NppiLabelMarkersDiff, Single) {
  const int W = 14, H = 10;
  for (int norm = 0; norm < 2; ++norm) {
    const NppiNorm eNorm = norm ? nppiNormInf : nppiNormL1;
    const std::string n = norm ? "Inf" : "L1";
    single_case<Npp16u>("nppiLabelMarkersUF_16u32u_C1R", n + "/16u32u", W, H, eNorm, 0x11 + norm,
                        [eNorm](Npp16u *s, int ss, Npp32u *d, int ds, NppiSize roi, NppiNorm,
                                Npp8u *buf, NppStreamContext ctx) {
                          return nppiLabelMarkersUF_16u32u_C1R_Ctx(s, ss, d, ds, roi, eNorm, buf,
                                                                   ctx);
                        });
    single_case<Npp32u>("nppiLabelMarkersUF_32u_C1R", n + "/32u", W, H, eNorm, 0x21 + norm,
                        [eNorm](Npp32u *s, int ss, Npp32u *d, int ds, NppiSize roi, NppiNorm,
                                Npp8u *buf, NppStreamContext ctx) {
                          return nppiLabelMarkersUF_32u_C1R_Ctx(s, ss, d, ds, roi, eNorm, buf, ctx);
                        });
    single_case<Npp8u>("nppiLabelMarkersUF_8u32u_C1R", n + "/8u32u", W, H, eNorm, 0x31 + norm,
                       [eNorm](Npp8u *s, int ss, Npp32u *d, int ds, NppiSize roi, NppiNorm,
                               Npp8u *buf, NppStreamContext ctx) {
                         return nppiLabelMarkersUF_8u32u_C1R_Ctx(s, ss, d, ds, roi, eNorm, buf, ctx);
                       });
  }
}

TEST(NppiLabelMarkersDiff, Batch) {
  const int W = 12, H = 9;
  for (int norm = 0; norm < 2; ++norm) {
    const NppiNorm eNorm = norm ? nppiNormInf : nppiNormL1;
    const std::string n = norm ? "Inf" : "L1";
    batch_case<Npp8u>("nppiLabelMarkersUFBatch_8u32u_C1R", n + "/8u32u/plain", W, H, eNorm, false,
                      0x41 + norm,
                      [eNorm](const NppiImageDescriptor *s, NppiImageDescriptor *d, int nb,
                              NppiSize roi, NppiNorm, bool, NppStreamContext ctx) {
                        return nppiLabelMarkersUFBatch_8u32u_C1R_Ctx(s, d, nb, roi, eNorm, ctx);
                      });
    batch_case<Npp8u>("nppiLabelMarkersUFBatch_8u32u_C1R_Advanced", n + "/8u32u/adv", W, H, eNorm,
                      true, 0x53 + norm,
                      [eNorm](const NppiImageDescriptor *s, NppiImageDescriptor *d, int nb,
                              NppiSize roi, NppiNorm, bool, NppStreamContext ctx) {
                        return nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx(s, d, nb, roi, eNorm,
                                                                               ctx);
                      });
    batch_case<Npp16u>("nppiLabelMarkersUFBatch_16u32u_C1R", n + "/16u32u/plain", W, H, eNorm,
                       false, 0x61 + norm,
                       [eNorm](const NppiImageDescriptor *s, NppiImageDescriptor *d, int nb,
                               NppiSize roi, NppiNorm, bool, NppStreamContext ctx) {
                         return nppiLabelMarkersUFBatch_16u32u_C1R_Ctx(s, d, nb, roi, eNorm, ctx);
                       });
    batch_case<Npp16u>("nppiLabelMarkersUFBatch_16u32u_C1R_Advanced", n + "/16u32u/adv", W, H,
                       eNorm, true, 0x71 + norm,
                       [eNorm](const NppiImageDescriptor *s, NppiImageDescriptor *d, int nb,
                               NppiSize roi, NppiNorm, bool, NppStreamContext ctx) {
                         return nppiLabelMarkersUFBatch_16u32u_C1R_Advanced_Ctx(s, d, nb, roi,
                                                                                 eNorm, ctx);
                       });
    batch_case<Npp32u>("nppiLabelMarkersUFBatch_32u_C1R", n + "/32u/plain", W, H, eNorm, false,
                       0x81 + norm,
                       [eNorm](const NppiImageDescriptor *s, NppiImageDescriptor *d, int nb,
                               NppiSize roi, NppiNorm, bool, NppStreamContext ctx) {
                         return nppiLabelMarkersUFBatch_32u_C1R_Ctx(s, d, nb, roi, eNorm, ctx);
                       });
    batch_case<Npp32u>("nppiLabelMarkersUFBatch_32u_C1R_Advanced", n + "/32u/adv", W, H, eNorm,
                       true, 0x91 + norm,
                       [eNorm](const NppiImageDescriptor *s, NppiImageDescriptor *d, int nb,
                               NppiSize roi, NppiNorm, bool, NppStreamContext ctx) {
                         return nppiLabelMarkersUFBatch_32u_C1R_Advanced_Ctx(s, d, nb, roi, eNorm,
                                                                              ctx);
                       });
  }
}

TEST(NppiLabelMarkersDiff, Errors) {
  const int W = 10, H = 8;
  std::vector<Npp8u> hs((size_t)W * H, 0);
  for (size_t i = 0; i < hs.size(); ++i) hs[i] = (Npp8u)((i % 3) * 60);
  DeviceMemory<Npp8u> &s = arena<Npp8u>((size_t)W * H);
  DeviceMemory<Npp32u> &d = arena<Npp32u>((size_t)W * H);
  DeviceMemory<Npp8u> &buf = arena<Npp8u>(1 << 20);
  s.copyFromHost(hs);
  d.copyFromHost(std::vector<Npp32u>((size_t)W * H, 0));
  buf.copyFromHost(std::vector<Npp8u>(1 << 20, 0));
  const std::string fn = "nppiLabelMarkersUF_8u32u_C1R";
  auto run = [&](Npp8u *src, int ss, Npp32u *dst, int ds, NppiSize roi, NppiNorm norm,
                 Npp8u *b) {
    return nppiLabelMarkersUF_8u32u_C1R_Ctx(src, ss, dst, ds, roi, norm, b, current_ctx());
  };
  only_status(fn, "ok", [&]() { return run(s.get(), W, d.get(), W * 4, {W, H}, nppiNormL1, buf.get()); });
  only_status(fn, "null_src", [&]() { return run(nullptr, W, d.get(), W * 4, {W, H}, nppiNormL1, buf.get()); });
  only_status(fn, "null_dst", [&]() { return run(s.get(), W, nullptr, W * 4, {W, H}, nppiNormL1, buf.get()); });
  only_status(fn, "null_buf", [&]() { return run(s.get(), W, d.get(), W * 4, {W, H}, nppiNormL1, nullptr); });
  only_status(fn, "roi_zero", [&]() { return run(s.get(), W, d.get(), W * 4, {0, H}, nppiNormL1, buf.get()); });
  only_status(fn, "roi_neg", [&]() { return run(s.get(), W, d.get(), W * 4, {-1, H}, nppiNormL1, buf.get()); });
  // NPP does not validate the source step (a zero step reads row 0 and returns
  // success) and reports a zero destination step as a kernel execution error,
  // so those two inputs are not part of the differential status matrix.
  only_status(fn, "bad_norm", [&]() {
    return run(s.get(), W, d.get(), W * 4, {W, H}, (NppiNorm)7, buf.get());
  });
  const std::string f16 = "nppiLabelMarkersUF_32u_C1R";
  only_status(f16, "ok", [&]() {
    return nppiLabelMarkersUF_32u_C1R_Ctx((Npp32u *)s.get(), W * 4, d.get(), W * 4, {W, H},
                                          nppiNormInf, buf.get(), current_ctx());
  });
  only_status(f16, "null_src", [&]() {
    return nppiLabelMarkersUF_32u_C1R_Ctx(nullptr, W * 4, d.get(), W * 4, {W, H}, nppiNormInf,
                                          buf.get(), current_ctx());
  });
  // batch status matrix (size 1 is rejected by the Advanced form)
  const std::string fb = "nppiLabelMarkersUFBatch_8u32u_C1R_Advanced";
  NppiImageDescriptor src1[2] = {{s.get(), W, {W, H}}, {s.get(), W, {W, H}}};
  NppiImageDescriptor dst1[2] = {{d.get(), W * 4, {W, H}}, {d.get(), W * 4, {W, H}}};
  only_status(fb, "ok", [&]() {
    return nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx(src1, dst1, 2, {W, H}, nppiNormL1,
                                                          current_ctx());
  });
  only_status(fb, "null_src", [&]() {
    return nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx(nullptr, dst1, 2, {W, H}, nppiNormL1,
                                                          current_ctx());
  });
  only_status(fb, "batch0", [&]() {
    return nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx(src1, dst1, 0, {W, H}, nppiNormL1,
                                                          current_ctx());
  });
  only_status(fb, "roi_zero", [&]() {
    return nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx(src1, dst1, 2, {0, H}, nppiNormL1,
                                                          current_ctx());
  });
  only_status(fb, "bad_norm", [&]() {
    return nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx(src1, dst1, 2, {W, H}, (NppiNorm)7,
                                                          current_ctx());
  });
}
