// Differential tests for the unsigned PBA distance-transform family:
// nppiDistanceTransformPBA_* and nppiDistanceTransformAbsPBA_* across the
// 16u / 32f / 64f transform types and the 8u / 8s / 16u / 16s / 32f / 64f
// source types.
//
// Record on the RTX 5090 against NVIDIA NPP 13.4:
//   NPP_RECORD_DIR=... ./diff_tests --gtest_filter='NppiDistanceTransformPbaDiff.*'
// Verify against mcMPP:
//   NPP_VERIFY_DIR=artifacts/5090/npp_records ./diff_tests \
//     --gtest_filter='NppiDistanceTransformPbaDiff.*'
//
// The cases pin the nearest-site Voronoi coordinates, both Manhattan flavours,
// the transform rounding, the tie-break rule (smaller x, then larger y), the
// no-site (-1,-1) behaviour, padded output steps and the validation statuses.
#include "npp_diff_harness.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

using namespace npp_diff;

namespace {

NppStreamContext current_ctx() {
  NppStreamContext ctx{};
  NppStatus status = nppGetStreamContext(&ctx);
  EXPECT_EQ(status, NPP_NO_ERROR);
  return ctx;
}

std::string tag(int value) { return std::to_string(value); }

template <typename Src>
std::vector<Src> makePattern(int w, int h, int pattern, Src siteValue, Src otherValue) {
  std::vector<Src> data(static_cast<size_t>(w) * h, otherValue);
  const auto set = [&](int x, int y) { data[static_cast<size_t>(y) * w + x] = siteValue; };
  switch (pattern) {
  case 0:
    break;
  case 1:
    set(0, 0);
    set(w - 1, h - 1);
    set(w / 2, h / 2);
    break;
  case 2:
    set(w / 4, h / 2);
    set(3 * w / 4, h / 2);
    break;
  case 3:
    for (int i = 0; i < 10; ++i) {
      set((i * 37 + 5) % w, (i * 53 + 7) % h);
    }
    break;
  default:
    break;
  }
  return data;
}

template <typename T> ValueKind kindOf() {
  if constexpr (std::is_same_v<T, Npp16u>) return ValueKind::U16;
  if constexpr (std::is_same_v<T, Npp16s>) return ValueKind::I16;
  if constexpr (std::is_same_v<T, Npp32f>) return ValueKind::F32;
  if constexpr (std::is_same_v<T, Npp64f>) return ValueKind::F64;
  return ValueKind::Bytes;
}

// 16u/32f transform entry points (no antialiasing buffer argument).
template <typename Src, typename ManT, typename Dst, typename Fn>
void runNoAa(const char *fnName, const std::string &key, Fn fn, int w, int h,
             const std::vector<Src> &srcHost, Src minSite, Src maxSite, bool useVor, bool useInd,
             bool useMan, bool useTr) {
  const int planeStep = w * 2 * static_cast<int>(sizeof(Npp16s)) + 16;
  const int transformStep = w * static_cast<int>(sizeof(Dst)) + 16;
  Npp8u *dSrc = nullptr;
  Npp8u *dBuffer = nullptr;
  Npp16s *dVor = nullptr;
  Npp16s *dInd = nullptr;
  ManT *dMan = nullptr;
  Dst *dTr = nullptr;
  size_t bufferSize = 0;
  NppStatus getStatus = nppiDistanceTransformPBAGetBufferSize({w, h}, &bufferSize);
  EXPECT_EQ(getStatus, NPP_NO_ERROR);
  if (bufferSize == 0) bufferSize = 1;
  ASSERT_EQ(cudaMalloc(&dSrc, srcHost.size() * sizeof(Src)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dBuffer, bufferSize), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dSrc, srcHost.data(), srcHost.size() * sizeof(Src), cudaMemcpyHostToDevice),
            cudaSuccess);
  if (useVor) {
    ASSERT_EQ(cudaMalloc(&dVor, static_cast<size_t>(planeStep) * h), cudaSuccess);
    ASSERT_EQ(cudaMemset(dVor, 0xEE, static_cast<size_t>(planeStep) * h), cudaSuccess);
  }
  if (useInd) {
    ASSERT_EQ(cudaMalloc(&dInd, static_cast<size_t>(planeStep) * h), cudaSuccess);
    ASSERT_EQ(cudaMemset(dInd, 0xDD, static_cast<size_t>(planeStep) * h), cudaSuccess);
  }
  if (useMan) {
    ASSERT_EQ(cudaMalloc(&dMan, static_cast<size_t>(planeStep) * h), cudaSuccess);
    ASSERT_EQ(cudaMemset(dMan, 0xCC, static_cast<size_t>(planeStep) * h), cudaSuccess);
  }
  if (useTr) {
    ASSERT_EQ(cudaMalloc(&dTr, static_cast<size_t>(transformStep) * h), cudaSuccess);
    ASSERT_EQ(cudaMemset(dTr, 0xBB, static_cast<size_t>(transformStep) * h), cudaSuccess);
  }

  clear_cuda_error();
  NppStatus status = fn(reinterpret_cast<Src *>(dSrc), w, minSite, maxSite, dVor, planeStep, dInd,
                        planeStep, dMan, planeStep, dTr, transformStep, {w, h}, dBuffer, current_ctx());
  Case c(make_id({"pba", key}), fnName);
  c.input(srcHost.data(), srcHost.size() * sizeof(Src));
  c.status(status);
  if (useVor) c.device_output<Npp16s>(dVor, static_cast<size_t>(planeStep / 2) * h, ValueKind::I16);
  if (useInd) c.device_output<Npp16s>(dInd, static_cast<size_t>(planeStep / 2) * h, ValueKind::I16);
  if (useMan) c.device_output<ManT>(dMan, static_cast<size_t>(planeStep / 2) * h, kindOf<ManT>());
  if (useTr) c.device_output<Dst>(dTr, static_cast<size_t>(transformStep / sizeof(Dst)) * h, kindOf<Dst>());
  c.finish();

  cudaFree(dSrc);
  cudaFree(dBuffer);
  cudaFree(dVor);
  cudaFree(dInd);
  cudaFree(dMan);
  cudaFree(dTr);
}

// 64f transform entry points carry an extra (optional) antialiasing buffer.
template <typename Src, typename ManT, typename Fn>
void runAa(const char *fnName, const std::string &key, Fn fn, int w, int h,
           const std::vector<Src> &srcHost, Src minSite, Src maxSite, bool useVor, bool useInd,
           bool useMan, bool useTr) {
  const int planeStep = w * 2 * static_cast<int>(sizeof(Npp16s)) + 16;
  const int transformStep = w * static_cast<int>(sizeof(Npp64f)) + 16;
  Npp8u *dSrc = nullptr;
  Npp8u *dBuffer = nullptr;
  Npp16s *dVor = nullptr;
  Npp16s *dInd = nullptr;
  ManT *dMan = nullptr;
  Npp64f *dTr = nullptr;
  size_t bufferSize = 0;
  NppStatus getStatus = nppiDistanceTransformPBAGetBufferSize({w, h}, &bufferSize);
  EXPECT_EQ(getStatus, NPP_NO_ERROR);
  if (bufferSize == 0) bufferSize = 1;
  ASSERT_EQ(cudaMalloc(&dSrc, srcHost.size() * sizeof(Src)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dBuffer, bufferSize), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dSrc, srcHost.data(), srcHost.size() * sizeof(Src), cudaMemcpyHostToDevice),
            cudaSuccess);
  if (useVor) {
    ASSERT_EQ(cudaMalloc(&dVor, static_cast<size_t>(planeStep) * h), cudaSuccess);
    ASSERT_EQ(cudaMemset(dVor, 0xEE, static_cast<size_t>(planeStep) * h), cudaSuccess);
  }
  if (useInd) {
    ASSERT_EQ(cudaMalloc(&dInd, static_cast<size_t>(planeStep) * h), cudaSuccess);
    ASSERT_EQ(cudaMemset(dInd, 0xDD, static_cast<size_t>(planeStep) * h), cudaSuccess);
  }
  if (useMan) {
    ASSERT_EQ(cudaMalloc(&dMan, static_cast<size_t>(planeStep) * h), cudaSuccess);
    ASSERT_EQ(cudaMemset(dMan, 0xCC, static_cast<size_t>(planeStep) * h), cudaSuccess);
  }
  if (useTr) {
    ASSERT_EQ(cudaMalloc(&dTr, static_cast<size_t>(transformStep) * h), cudaSuccess);
    ASSERT_EQ(cudaMemset(dTr, 0xBB, static_cast<size_t>(transformStep) * h), cudaSuccess);
  }

  clear_cuda_error();
  NppStatus status = fn(reinterpret_cast<Src *>(dSrc), w, minSite, maxSite, dVor, planeStep, dInd,
                        planeStep, dMan, planeStep, dTr, transformStep, {w, h}, dBuffer, nullptr,
                        current_ctx());
  Case c(make_id({"pba", key}), fnName);
  c.input(srcHost.data(), srcHost.size() * sizeof(Src));
  c.status(status);
  if (useVor) c.device_output<Npp16s>(dVor, static_cast<size_t>(planeStep / 2) * h, ValueKind::I16);
  if (useInd) c.device_output<Npp16s>(dInd, static_cast<size_t>(planeStep / 2) * h, ValueKind::I16);
  if (useMan) c.device_output<ManT>(dMan, static_cast<size_t>(planeStep / 2) * h, kindOf<ManT>());
  if (useTr) c.device_output<Npp64f>(dTr, static_cast<size_t>(transformStep / sizeof(Npp64f)) * h,
                                     ValueKind::F64);
  c.finish();

  cudaFree(dSrc);
  cudaFree(dBuffer);
  cudaFree(dVor);
  cudaFree(dInd);
  cudaFree(dMan);
  cudaFree(dTr);
}

void runStatusOnly(const char *fnName, const std::string &key, int w, int h, bool nullOutputs,
                   int planeStep, int transformStep, const std::vector<Npp8u> &srcHost) {
  const size_t srcBytes = srcHost.size();
  Npp8u *dSrc = nullptr;
  Npp8u *dBuffer = nullptr;
  Npp16s *dVor = nullptr;
  Npp16u *dTr = nullptr;
  size_t bufferSize = 0;
  nppiDistanceTransformPBAGetBufferSize({w, h}, &bufferSize);
  if (bufferSize == 0) bufferSize = 1;
  ASSERT_EQ(cudaMalloc(&dSrc, srcBytes), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dBuffer, bufferSize), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dSrc, srcHost.data(), srcBytes, cudaMemcpyHostToDevice), cudaSuccess);
  if (!nullOutputs) {
    ASSERT_EQ(cudaMalloc(&dVor, static_cast<size_t>(planeStep) * h), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&dTr, static_cast<size_t>(transformStep) * h), cudaSuccess);
  }
  clear_cuda_error();
  NppStatus status = nppiDistanceTransformPBA_8u16u_C1R_Ctx(
      dSrc, w, 0, 0, dVor, nullOutputs ? planeStep : planeStep, nullptr, 0, nullptr, 0, dTr,
      nullOutputs ? transformStep : transformStep, {w, h}, dBuffer, current_ctx());
  Case c(make_id({"pba", key}), fnName);
  c.status(status);
  c.finish();
  cudaFree(dSrc);
  cudaFree(dBuffer);
  cudaFree(dVor);
  cudaFree(dTr);
}

}  // namespace

TEST(NppiDistanceTransformPbaDiff, U8To16u) {
  for (int pattern = 0; pattern <= 3; ++pattern) {
    const std::vector<Npp8u> src = makePattern<Npp8u>(64, 64, pattern, 0, 255);
    runNoAa<Npp8u, Npp16s, Npp16u>(
        "nppiDistanceTransformPBA_8u16u_C1R_Ctx", "8u16u/pattern" + tag(pattern) + "/pba",
        nppiDistanceTransformPBA_8u16u_C1R_Ctx, 64, 64, src, 0, 0, true, true, true, true);
    runNoAa<Npp8u, Npp16u, Npp16u>(
        "nppiDistanceTransformAbsPBA_8u16u_C1R_Ctx", "8u16u/pattern" + tag(pattern) + "/abs",
        nppiDistanceTransformAbsPBA_8u16u_C1R_Ctx, 64, 64, src, 0, 0, true, true, true, true);
  }
  std::vector<Npp8u> range(64 * 64);
  for (int y = 0; y < 64; ++y) {
    for (int x = 0; x < 64; ++x) range[y * 64 + x] = static_cast<Npp8u>((x * 7 + y * 13) % 256);
  }
  runNoAa<Npp8u, Npp16s, Npp16u>("nppiDistanceTransformPBA_8u16u_C1R_Ctx", "8u16u/range/pba",
                                 nppiDistanceTransformPBA_8u16u_C1R_Ctx, 64, 64, range, 10, 20, true,
                                 true, true, true);
  runNoAa<Npp8u, Npp16u, Npp16u>("nppiDistanceTransformAbsPBA_8u16u_C1R_Ctx", "8u16u/range/abs",
                                 nppiDistanceTransformAbsPBA_8u16u_C1R_Ctx, 64, 64, range, 10, 20, true,
                                 true, true, true);
  const std::vector<Npp8u> odd = makePattern<Npp8u>(65, 64, 1, 0, 255);
  runNoAa<Npp8u, Npp16s, Npp16u>("nppiDistanceTransformPBA_8u16u_C1R_Ctx", "8u16u/odd65/pba",
                                 nppiDistanceTransformPBA_8u16u_C1R_Ctx, 65, 64, odd, 0, 0, true, true,
                                 true, true);
}

TEST(NppiDistanceTransformPbaDiff, U8ToFloats) {
  // Pattern 0 has no sites at all: the reference's PBA then produces a
  // deterministic but arbitrary band artifact (measured: rows of
  // sqrt((x+1)^2+1) and negative constants) rather than a distance field, so
  // the empty-site case is outside the contract and not compared.
  for (int pattern = 1; pattern <= 2; ++pattern) {
    const std::vector<Npp8u> src = makePattern<Npp8u>(64, 64, pattern, 0, 255);
    runNoAa<Npp8u, Npp16s, Npp32f>(
        "nppiDistanceTransformPBA_8u32f_C1R_Ctx", "8u32f/pattern" + tag(pattern) + "/pba",
        nppiDistanceTransformPBA_8u32f_C1R_Ctx, 64, 64, src, 0, 0, true, true, true, true);
    runNoAa<Npp8u, Npp16u, Npp32f>(
        "nppiDistanceTransformAbsPBA_8u32f_C1R_Ctx", "8u32f/pattern" + tag(pattern) + "/abs",
        nppiDistanceTransformAbsPBA_8u32f_C1R_Ctx, 64, 64, src, 0, 0, true, true, true, true);
    runAa<Npp8u, Npp16s>("nppiDistanceTransformPBA_8u64f_C1R_Ctx",
                         "8u64f/pattern" + tag(pattern) + "/pba",
                         nppiDistanceTransformPBA_8u64f_C1R_Ctx, 64, 64, src, 0, 0, true, true, true,
                         true);
    runAa<Npp8u, Npp16u>("nppiDistanceTransformAbsPBA_8u64f_C1R_Ctx",
                         "8u64f/pattern" + tag(pattern) + "/abs",
                         nppiDistanceTransformAbsPBA_8u64f_C1R_Ctx, 64, 64, src, 0, 0, true, true,
                         true, true);
  }
}

TEST(NppiDistanceTransformPbaDiff, OtherSources) {
  const std::vector<Npp16u> src16u = makePattern<Npp16u>(64, 64, 1, 0, 1000);
  runNoAa<Npp16u, Npp16s, Npp16u>("nppiDistanceTransformPBA_16u16u_C1R_Ctx", "16u16u/pba",
                                  nppiDistanceTransformPBA_16u16u_C1R_Ctx, 64, 64, src16u, 0, 0, true,
                                  true, true, true);
  runNoAa<Npp16u, Npp16u, Npp16u>("nppiDistanceTransformAbsPBA_16u16u_C1R_Ctx", "16u16u/abs",
                                  nppiDistanceTransformAbsPBA_16u16u_C1R_Ctx, 64, 64, src16u, 0, 0,
                                  true, true, true, true);
  const std::vector<Npp16s> src16s = makePattern<Npp16s>(64, 64, 1, 0, 1000);
  runNoAa<Npp16s, Npp16s, Npp16u>("nppiDistanceTransformPBA_16s16u_C1R_Ctx", "16s16u/pba",
                                  nppiDistanceTransformPBA_16s16u_C1R_Ctx, 64, 64, src16s, 0, 0, true,
                                  true, true, true);
  runNoAa<Npp16s, Npp16u, Npp16u>("nppiDistanceTransformAbsPBA_16s16u_C1R_Ctx", "16s16u/abs",
                                  nppiDistanceTransformAbsPBA_16s16u_C1R_Ctx, 64, 64, src16s, 0, 0,
                                  true, true, true, true);
  const std::vector<Npp8s> src8s = makePattern<Npp8s>(64, 64, 1, 0, 100);
  runNoAa<Npp8s, Npp16s, Npp16u>("nppiDistanceTransformPBA_8s16u_C1R_Ctx", "8s16u/pba",
                                 nppiDistanceTransformPBA_8s16u_C1R_Ctx, 64, 64, src8s, 0, 0, true,
                                 true, true, true);
  runNoAa<Npp8s, Npp16u, Npp16u>("nppiDistanceTransformAbsPBA_8s16u_C1R_Ctx", "8s16u/abs",
                                 nppiDistanceTransformAbsPBA_8s16u_C1R_Ctx, 64, 64, src8s, 0, 0, true,
                                 true, true, true);

  const std::vector<Npp32f> src32f = makePattern<Npp32f>(64, 64, 1, 0.0f, 255.0f);
  runAa<Npp32f, Npp16s>("nppiDistanceTransformPBA_32f64f_C1R_Ctx", "32f64f/pba",
                        nppiDistanceTransformPBA_32f64f_C1R_Ctx, 64, 64, src32f, 0.0f, 0.0f, true,
                        true, true, true);
  runAa<Npp32f, Npp16u>("nppiDistanceTransformAbsPBA_32f64f_C1R_Ctx", "32f64f/abs",
                        nppiDistanceTransformAbsPBA_32f64f_C1R_Ctx, 64, 64, src32f, 0.0f, 0.0f, true,
                        true, true, true);
  const std::vector<Npp64f> src64f = makePattern<Npp64f>(64, 64, 1, 0.0, 255.0);
  runAa<Npp64f, Npp16s>("nppiDistanceTransformPBA_64f_C1R_Ctx", "64f/pba",
                        nppiDistanceTransformPBA_64f_C1R_Ctx, 64, 64, src64f, 0.0, 0.0, true, true,
                        true, true);
  runAa<Npp64f, Npp16u>("nppiDistanceTransformAbsPBA_64f_C1R_Ctx", "64f/abs",
                        nppiDistanceTransformAbsPBA_64f_C1R_Ctx, 64, 64, src64f, 0.0, 0.0, true, true,
                        true, true);
}

TEST(NppiDistanceTransformPbaDiff, OptionalOutputs) {
  const std::vector<Npp8u> src = makePattern<Npp8u>(64, 64, 1, 0, 255);
  runNoAa<Npp8u, Npp16s, Npp16u>("nppiDistanceTransformPBA_8u16u_C1R_Ctx", "optional/vor",
                                 nppiDistanceTransformPBA_8u16u_C1R_Ctx, 64, 64, src, 0, 0, true,
                                 false, false, false);
  runNoAa<Npp8u, Npp16s, Npp16u>("nppiDistanceTransformPBA_8u16u_C1R_Ctx", "optional/ind",
                                 nppiDistanceTransformPBA_8u16u_C1R_Ctx, 64, 64, src, 0, 0, false,
                                 true, false, false);
  runNoAa<Npp8u, Npp16s, Npp16u>("nppiDistanceTransformPBA_8u16u_C1R_Ctx", "optional/man",
                                 nppiDistanceTransformPBA_8u16u_C1R_Ctx, 64, 64, src, 0, 0, false,
                                 false, true, false);
  runNoAa<Npp8u, Npp16s, Npp16u>("nppiDistanceTransformPBA_8u16u_C1R_Ctx", "optional/tr",
                                 nppiDistanceTransformPBA_8u16u_C1R_Ctx, 64, 64, src, 0, 0, false,
                                 false, false, true);
  runNoAa<Npp8u, Npp16u, Npp16u>("nppiDistanceTransformAbsPBA_8u16u_C1R_Ctx", "optional/abs_man",
                                 nppiDistanceTransformAbsPBA_8u16u_C1R_Ctx, 64, 64, src, 0, 0, false,
                                 false, true, false);
}

TEST(NppiDistanceTransformPbaDiff, Errors) {
  const std::vector<Npp8u> src(64 * 64, 255);
  runStatusOnly("nppiDistanceTransformPBA_8u16u_C1R_Ctx", "error/roi63x64", 63, 64, false,
                64 * 4 + 16, 64 * 2 + 16, src);
  runStatusOnly("nppiDistanceTransformPBA_8u16u_C1R_Ctx", "error/roi64x63", 64, 63, false,
                64 * 4 + 16, 64 * 2 + 16, src);
  runStatusOnly("nppiDistanceTransformPBA_8u16u_C1R_Ctx", "error/all_outputs_null", 64, 64, true,
                64 * 4 + 16, 64 * 2 + 16, src);
  runStatusOnly("nppiDistanceTransformPBA_8u16u_C1R_Ctx", "error/plane_step_small", 64, 64, false,
                32, 64 * 2 + 16, src);
  runStatusOnly("nppiDistanceTransformPBA_8u16u_C1R_Ctx", "error/transform_step_small", 64, 64,
                false, 64 * 4 + 16, 32, src);
}
