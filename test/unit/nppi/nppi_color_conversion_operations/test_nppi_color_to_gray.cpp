#include "npp.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

class ColorToGrayTest : public ::testing::Test {
protected:
  void SetUp() override {
    width = 12;
    height = 8;
    roi.width = width;
    roi.height = height;
  }

  int width = 0;
  int height = 0;
  NppiSize roi{};
};

namespace {

struct GrayCtxCase {
  int width;
  int height;
  unsigned int seed;
};

template <typename T> struct GrayCtxTraits;

template <> struct GrayCtxTraits<Npp8u> {
  using Type = Npp8u;
  static Npp8u *mallocC3(int w, int h, int *step) { return nppiMalloc_8u_C3(w, h, step); }
  static Npp8u *mallocC4(int w, int h, int *step) { return nppiMalloc_8u_C4(w, h, step); }
  static Npp8u *mallocC1(int w, int h, int *step) { return nppiMalloc_8u_C1(w, h, step); }
  static void freeMem(Npp8u *ptr) { nppiFree(ptr); }
  static NppStatus c3c1(const Npp8u *src, int srcStep, Npp8u *dst, int dstStep, NppiSize roi,
                        const Npp32f *coeffs) {
    return nppiColorToGray_8u_C3C1R(src, srcStep, dst, dstStep, roi, coeffs);
  }
  static NppStatus c3c1Ctx(const Npp8u *src, int srcStep, Npp8u *dst, int dstStep, NppiSize roi,
                           const Npp32f *coeffs, NppStreamContext ctx) {
    return nppiColorToGray_8u_C3C1R_Ctx(src, srcStep, dst, dstStep, roi, coeffs, ctx);
  }
  static NppStatus ac4c1(const Npp8u *src, int srcStep, Npp8u *dst, int dstStep, NppiSize roi,
                         const Npp32f *coeffs) {
    return nppiColorToGray_8u_AC4C1R(src, srcStep, dst, dstStep, roi, coeffs);
  }
  static NppStatus ac4c1Ctx(const Npp8u *src, int srcStep, Npp8u *dst, int dstStep, NppiSize roi,
                            const Npp32f *coeffs, NppStreamContext ctx) {
    return nppiColorToGray_8u_AC4C1R_Ctx(src, srcStep, dst, dstStep, roi, coeffs, ctx);
  }
  static NppStatus c4c1(const Npp8u *src, int srcStep, Npp8u *dst, int dstStep, NppiSize roi,
                        const Npp32f *coeffs) {
    return nppiColorToGray_8u_C4C1R(src, srcStep, dst, dstStep, roi, coeffs);
  }
  static NppStatus c4c1Ctx(const Npp8u *src, int srcStep, Npp8u *dst, int dstStep, NppiSize roi,
                           const Npp32f *coeffs, NppStreamContext ctx) {
    return nppiColorToGray_8u_C4C1R_Ctx(src, srcStep, dst, dstStep, roi, coeffs, ctx);
  }
};

template <> struct GrayCtxTraits<Npp16u> {
  using Type = Npp16u;
  static Npp16u *mallocC3(int w, int h, int *step) { return nppiMalloc_16u_C3(w, h, step); }
  static Npp16u *mallocC4(int w, int h, int *step) { return nppiMalloc_16u_C4(w, h, step); }
  static Npp16u *mallocC1(int w, int h, int *step) { return nppiMalloc_16u_C1(w, h, step); }
  static void freeMem(Npp16u *ptr) { nppiFree(ptr); }
  static NppStatus c3c1(const Npp16u *src, int srcStep, Npp16u *dst, int dstStep, NppiSize roi,
                        const Npp32f *coeffs) {
    return nppiColorToGray_16u_C3C1R(src, srcStep, dst, dstStep, roi, coeffs);
  }
  static NppStatus c3c1Ctx(const Npp16u *src, int srcStep, Npp16u *dst, int dstStep, NppiSize roi,
                           const Npp32f *coeffs, NppStreamContext ctx) {
    return nppiColorToGray_16u_C3C1R_Ctx(src, srcStep, dst, dstStep, roi, coeffs, ctx);
  }
  static NppStatus ac4c1(const Npp16u *src, int srcStep, Npp16u *dst, int dstStep, NppiSize roi,
                         const Npp32f *coeffs) {
    return nppiColorToGray_16u_AC4C1R(src, srcStep, dst, dstStep, roi, coeffs);
  }
  static NppStatus ac4c1Ctx(const Npp16u *src, int srcStep, Npp16u *dst, int dstStep, NppiSize roi,
                            const Npp32f *coeffs, NppStreamContext ctx) {
    return nppiColorToGray_16u_AC4C1R_Ctx(src, srcStep, dst, dstStep, roi, coeffs, ctx);
  }
  static NppStatus c4c1(const Npp16u *src, int srcStep, Npp16u *dst, int dstStep, NppiSize roi,
                        const Npp32f *coeffs) {
    return nppiColorToGray_16u_C4C1R(src, srcStep, dst, dstStep, roi, coeffs);
  }
  static NppStatus c4c1Ctx(const Npp16u *src, int srcStep, Npp16u *dst, int dstStep, NppiSize roi,
                           const Npp32f *coeffs, NppStreamContext ctx) {
    return nppiColorToGray_16u_C4C1R_Ctx(src, srcStep, dst, dstStep, roi, coeffs, ctx);
  }
};

template <> struct GrayCtxTraits<Npp16s> {
  using Type = Npp16s;
  static Npp16s *mallocC3(int w, int h, int *step) {
    Npp16s *ptr = nullptr;
    size_t pitch = 0;
    cudaMallocPitch(reinterpret_cast<void **>(&ptr), &pitch, w * sizeof(Npp16s) * 3, h);
    *step = static_cast<int>(pitch);
    return ptr;
  }
  static Npp16s *mallocC4(int w, int h, int *step) {
    Npp16s *ptr = nullptr;
    size_t pitch = 0;
    cudaMallocPitch(reinterpret_cast<void **>(&ptr), &pitch, w * sizeof(Npp16s) * 4, h);
    *step = static_cast<int>(pitch);
    return ptr;
  }
  static Npp16s *mallocC1(int w, int h, int *step) { return nppiMalloc_16s_C1(w, h, step); }
  static void freeMem(Npp16s *ptr) { cudaFree(ptr); }
  static NppStatus c3c1(const Npp16s *src, int srcStep, Npp16s *dst, int dstStep, NppiSize roi,
                        const Npp32f *coeffs) {
    return nppiColorToGray_16s_C3C1R(src, srcStep, dst, dstStep, roi, coeffs);
  }
  static NppStatus c3c1Ctx(const Npp16s *src, int srcStep, Npp16s *dst, int dstStep, NppiSize roi,
                           const Npp32f *coeffs, NppStreamContext ctx) {
    return nppiColorToGray_16s_C3C1R_Ctx(src, srcStep, dst, dstStep, roi, coeffs, ctx);
  }
  static NppStatus ac4c1(const Npp16s *src, int srcStep, Npp16s *dst, int dstStep, NppiSize roi,
                         const Npp32f *coeffs) {
    return nppiColorToGray_16s_AC4C1R(src, srcStep, dst, dstStep, roi, coeffs);
  }
  static NppStatus ac4c1Ctx(const Npp16s *src, int srcStep, Npp16s *dst, int dstStep, NppiSize roi,
                            const Npp32f *coeffs, NppStreamContext ctx) {
    return nppiColorToGray_16s_AC4C1R_Ctx(src, srcStep, dst, dstStep, roi, coeffs, ctx);
  }
  static NppStatus c4c1(const Npp16s *src, int srcStep, Npp16s *dst, int dstStep, NppiSize roi,
                        const Npp32f *coeffs) {
    return nppiColorToGray_16s_C4C1R(src, srcStep, dst, dstStep, roi, coeffs);
  }
  static NppStatus c4c1Ctx(const Npp16s *src, int srcStep, Npp16s *dst, int dstStep, NppiSize roi,
                           const Npp32f *coeffs, NppStreamContext ctx) {
    return nppiColorToGray_16s_C4C1R_Ctx(src, srcStep, dst, dstStep, roi, coeffs, ctx);
  }
};

template <> struct GrayCtxTraits<Npp32f> {
  using Type = Npp32f;
  static Npp32f *mallocC3(int w, int h, int *step) { return nppiMalloc_32f_C3(w, h, step); }
  static Npp32f *mallocC4(int w, int h, int *step) { return nppiMalloc_32f_C4(w, h, step); }
  static Npp32f *mallocC1(int w, int h, int *step) { return nppiMalloc_32f_C1(w, h, step); }
  static void freeMem(Npp32f *ptr) { nppiFree(ptr); }
  static NppStatus c3c1(const Npp32f *src, int srcStep, Npp32f *dst, int dstStep, NppiSize roi,
                        const Npp32f *coeffs) {
    return nppiColorToGray_32f_C3C1R(src, srcStep, dst, dstStep, roi, coeffs);
  }
  static NppStatus c3c1Ctx(const Npp32f *src, int srcStep, Npp32f *dst, int dstStep, NppiSize roi,
                           const Npp32f *coeffs, NppStreamContext ctx) {
    return nppiColorToGray_32f_C3C1R_Ctx(src, srcStep, dst, dstStep, roi, coeffs, ctx);
  }
  static NppStatus ac4c1(const Npp32f *src, int srcStep, Npp32f *dst, int dstStep, NppiSize roi,
                         const Npp32f *coeffs) {
    return nppiColorToGray_32f_AC4C1R(src, srcStep, dst, dstStep, roi, coeffs);
  }
  static NppStatus ac4c1Ctx(const Npp32f *src, int srcStep, Npp32f *dst, int dstStep, NppiSize roi,
                            const Npp32f *coeffs, NppStreamContext ctx) {
    return nppiColorToGray_32f_AC4C1R_Ctx(src, srcStep, dst, dstStep, roi, coeffs, ctx);
  }
  static NppStatus c4c1(const Npp32f *src, int srcStep, Npp32f *dst, int dstStep, NppiSize roi,
                        const Npp32f *coeffs) {
    return nppiColorToGray_32f_C4C1R(src, srcStep, dst, dstStep, roi, coeffs);
  }
  static NppStatus c4c1Ctx(const Npp32f *src, int srcStep, Npp32f *dst, int dstStep, NppiSize roi,
                           const Npp32f *coeffs, NppStreamContext ctx) {
    return nppiColorToGray_32f_C4C1R_Ctx(src, srcStep, dst, dstStep, roi, coeffs, ctx);
  }
};

template <typename T>
void fill_random(std::vector<T> &data, std::mt19937 &rng, int width, int height, int channels) {
  data.resize(static_cast<size_t>(width) * height * channels);
  if constexpr (std::is_same_v<T, Npp32f>) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : data) {
      v = static_cast<T>(dist(rng));
    }
  } else if constexpr (std::is_same_v<T, Npp16s>) {
    std::uniform_int_distribution<int> dist(-2000, 2000);
    for (auto &v : data) {
      v = static_cast<T>(dist(rng));
    }
  } else if constexpr (std::is_same_v<T, Npp16u>) {
    std::uniform_int_distribution<int> dist(0, 65535);
    for (auto &v : data) {
      v = static_cast<T>(dist(rng));
    }
  } else {
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto &v : data) {
      v = static_cast<T>(dist(rng));
    }
  }
}

template <typename T>
void expect_equal(const std::vector<T> &a, const std::vector<T> &b) {
  ASSERT_EQ(a.size(), b.size());
  if constexpr (std::is_same_v<T, Npp32f>) {
    for (size_t i = 0; i < a.size(); ++i) {
      EXPECT_NEAR(a[i], b[i], 1e-5f) << "Mismatch at " << i;
    }
  } else {
    for (size_t i = 0; i < a.size(); ++i) {
      EXPECT_EQ(a[i], b[i]) << "Mismatch at " << i;
    }
  }
}

template <typename T, typename Traits>
void run_gray_ctx_case(const GrayCtxCase &param, int channels, bool use_ac4) {
  constexpr Npp32f kCoeff3[3] = {0.3f, 0.59f, 0.11f};
  constexpr Npp32f kCoeff4[4] = {0.25f, 0.25f, 0.25f, 0.25f};

  std::mt19937 rng(param.seed);
  std::vector<T> hostSrc;
  fill_random(hostSrc, rng, param.width, param.height, channels);

  int srcStep = 0;
  int dstStep = 0;
  int ctxStep = 0;

  T *d_src = nullptr;
  T *d_dst = nullptr;
  T *d_ctx = nullptr;
  if (channels == 3) {
    d_src = Traits::mallocC3(param.width, param.height, &srcStep);
  } else {
    d_src = Traits::mallocC4(param.width, param.height, &srcStep);
  }
  d_dst = Traits::mallocC1(param.width, param.height, &dstStep);
  d_ctx = Traits::mallocC1(param.width, param.height, &ctxStep);

  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);
  ASSERT_NE(d_ctx, nullptr);

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), param.width * sizeof(T) * channels,
               param.width * sizeof(T) * channels, param.height, cudaMemcpyHostToDevice);

  NppiSize roi{param.width, param.height};
  NppStatus status = NPP_SUCCESS;
  NppStatus status_ctx = NPP_SUCCESS;

  if (channels == 3) {
    status = Traits::c3c1(d_src, srcStep, d_dst, dstStep, roi, kCoeff3);
    NppStreamContext ctx{};
    nppGetStreamContext(&ctx);
    ctx.hStream = 0;
    status_ctx = Traits::c3c1Ctx(d_src, srcStep, d_ctx, ctxStep, roi, kCoeff3, ctx);
  } else if (use_ac4) {
    status = Traits::ac4c1(d_src, srcStep, d_dst, dstStep, roi, kCoeff3);
    NppStreamContext ctx{};
    nppGetStreamContext(&ctx);
    ctx.hStream = 0;
    status_ctx = Traits::ac4c1Ctx(d_src, srcStep, d_ctx, ctxStep, roi, kCoeff3, ctx);
  } else {
    status = Traits::c4c1(d_src, srcStep, d_dst, dstStep, roi, kCoeff4);
    NppStreamContext ctx{};
    nppGetStreamContext(&ctx);
    ctx.hStream = 0;
    status_ctx = Traits::c4c1Ctx(d_src, srcStep, d_ctx, ctxStep, roi, kCoeff4, ctx);
  }

  ASSERT_EQ(status, NPP_SUCCESS);
  ASSERT_EQ(status_ctx, NPP_SUCCESS);

  std::vector<T> hostDst(param.width * param.height);
  std::vector<T> hostCtx(param.width * param.height);
  cudaMemcpy2D(hostDst.data(), param.width * sizeof(T), d_dst, dstStep, param.width * sizeof(T), param.height,
               cudaMemcpyDeviceToHost);
  cudaMemcpy2D(hostCtx.data(), param.width * sizeof(T), d_ctx, ctxStep, param.width * sizeof(T), param.height,
               cudaMemcpyDeviceToHost);

  expect_equal(hostDst, hostCtx);

  Traits::freeMem(d_src);
  Traits::freeMem(d_dst);
  Traits::freeMem(d_ctx);
}

} // namespace

template <typename T>
class ColorToGrayCtxParamTest : public ::testing::TestWithParam<GrayCtxCase> {};

using ColorToGrayCtxParamTest8u = ColorToGrayCtxParamTest<Npp8u>;
using ColorToGrayCtxParamTest16u = ColorToGrayCtxParamTest<Npp16u>;
using ColorToGrayCtxParamTest16s = ColorToGrayCtxParamTest<Npp16s>;
using ColorToGrayCtxParamTest32f = ColorToGrayCtxParamTest<Npp32f>;

TEST_F(ColorToGrayTest, ColorToGray_8u_C3C1R_IdentityCoeff) {
  const Npp32f coeffs[3] = {1.0f, 0.0f, 0.0f};
  std::vector<Npp8u> srcData(width * height * 3);
  std::vector<Npp8u> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 3;
      Npp8u r = static_cast<Npp8u>(((x + y) * 4) & 0xFF);
      Npp8u g = static_cast<Npp8u>(((x * 3 + y) * 7) & 0xFF);
      Npp8u b = static_cast<Npp8u>(((x * 5 + y * 2) * 3) & 0xFF);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      expected[y * width + x] = r;
    }
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_8u_C3C1R(d_src, srcStep, d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> result(width * height);
  cudaMemcpy2D(result.data(), width, d_dst, dstStep, width, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(ColorToGrayTest, ColorToGray_8u_AC4C1R_WeightedNoAlpha) {
  const Npp32f coeffs[3] = {0.25f, 0.5f, 0.25f};
  std::vector<Npp8u> srcData(width * height * 4);
  std::vector<Npp8u> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 4;
      Npp8u r = static_cast<Npp8u>(((x + y) * 4) & 0xFF);
      Npp8u g = static_cast<Npp8u>(((x * 2 + y) * 4) & 0xFF);
      Npp8u b = static_cast<Npp8u>(((x + y * 2) * 4) & 0xFF);
      Npp8u a = static_cast<Npp8u>((x * 17 + y * 13) & 0xFF);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      srcData[idx + 3] = a;
      expected[y * width + x] = static_cast<Npp8u>((r + 2 * g + b) / 4);
    }
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp8u *d_src = nppiMalloc_8u_C4(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 4, width * 4, height, cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_8u_AC4C1R(d_src, srcStep, d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> result(width * height);
  cudaMemcpy2D(result.data(), width, d_dst, dstStep, width, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(ColorToGrayTest, ColorToGray_8u_C4C1R_UsesAlphaCoeff) {
  const Npp32f coeffs[4] = {0.25f, 0.25f, 0.25f, 0.25f};
  std::vector<Npp8u> srcData(width * height * 4);
  std::vector<Npp8u> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 4;
      Npp8u r = static_cast<Npp8u>(((x + y) * 4) & 0xFF);
      Npp8u g = static_cast<Npp8u>(((x * 3 + y) * 4) & 0xFF);
      Npp8u b = static_cast<Npp8u>(((x + y * 2) * 4) & 0xFF);
      Npp8u a = static_cast<Npp8u>(((x * 2 + y * 3) * 4) & 0xFF);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      srcData[idx + 3] = a;
      expected[y * width + x] = static_cast<Npp8u>((r + g + b + a) / 4);
    }
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp8u *d_src = nppiMalloc_8u_C4(width, height, &srcStep);
  Npp8u *d_dst = nppiMalloc_8u_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 4, width * 4, height, cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_8u_C4C1R(d_src, srcStep, d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> result(width * height);
  cudaMemcpy2D(result.data(), width, d_dst, dstStep, width, height, cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(ColorToGrayTest, ColorToGray_16u_C3C1R_IdentityCoeff) {
  const Npp32f coeffs[3] = {1.0f, 0.0f, 0.0f};
  std::vector<Npp16u> srcData(width * height * 3);
  std::vector<Npp16u> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 3;
      Npp16u r = static_cast<Npp16u>(((x + y) * 64) & 0xFFFF);
      Npp16u g = static_cast<Npp16u>(((x * 3 + y) * 32) & 0xFFFF);
      Npp16u b = static_cast<Npp16u>(((x + y * 2) * 48) & 0xFFFF);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      expected[y * width + x] = r;
    }
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp16u *d_src = nppiMalloc_16u_C3(width, height, &srcStep);
  Npp16u *d_dst = nppiMalloc_16u_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp16u) * 3, width * sizeof(Npp16u) * 3, height,
               cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_16u_C3C1R(d_src, srcStep, d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> result(width * height);
  cudaMemcpy2D(result.data(), width * sizeof(Npp16u), d_dst, dstStep, width * sizeof(Npp16u), height,
               cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(ColorToGrayTest, ColorToGray_16u_AC4C1R_WeightedNoAlpha) {
  const Npp32f coeffs[3] = {0.25f, 0.5f, 0.25f};
  std::vector<Npp16u> srcData(width * height * 4);
  std::vector<Npp16u> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 4;
      Npp16u r = static_cast<Npp16u>(((x + y) * 64) & 0xFFFF);
      Npp16u g = static_cast<Npp16u>(((x * 2 + y) * 64) & 0xFFFF);
      Npp16u b = static_cast<Npp16u>(((x + y * 2) * 64) & 0xFFFF);
      Npp16u a = static_cast<Npp16u>(((x * 5 + y * 7) * 64) & 0xFFFF);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      srcData[idx + 3] = a;
      expected[y * width + x] = static_cast<Npp16u>((r + 2 * g + b) / 4);
    }
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp16u *d_src = nppiMalloc_16u_C4(width, height, &srcStep);
  Npp16u *d_dst = nppiMalloc_16u_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp16u) * 4, width * sizeof(Npp16u) * 4, height,
               cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_16u_AC4C1R(d_src, srcStep, d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> result(width * height);
  cudaMemcpy2D(result.data(), width * sizeof(Npp16u), d_dst, dstStep, width * sizeof(Npp16u), height,
               cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(ColorToGrayTest, ColorToGray_16u_C4C1R_UsesAlphaCoeff) {
  const Npp32f coeffs[4] = {0.25f, 0.25f, 0.25f, 0.25f};
  std::vector<Npp16u> srcData(width * height * 4);
  std::vector<Npp16u> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 4;
      Npp16u r = static_cast<Npp16u>(((x + y) * 64) & 0xFFFF);
      Npp16u g = static_cast<Npp16u>(((x * 3 + y) * 64) & 0xFFFF);
      Npp16u b = static_cast<Npp16u>(((x + y * 2) * 64) & 0xFFFF);
      Npp16u a = static_cast<Npp16u>(((x * 2 + y * 3) * 64) & 0xFFFF);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      srcData[idx + 3] = a;
      expected[y * width + x] = static_cast<Npp16u>((r + g + b + a) / 4);
    }
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp16u *d_src = nppiMalloc_16u_C4(width, height, &srcStep);
  Npp16u *d_dst = nppiMalloc_16u_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp16u) * 4, width * sizeof(Npp16u) * 4, height,
               cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_16u_C4C1R(d_src, srcStep, d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16u> result(width * height);
  cudaMemcpy2D(result.data(), width * sizeof(Npp16u), d_dst, dstStep, width * sizeof(Npp16u), height,
               cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(ColorToGrayTest, ColorToGray_32f_C3C1R_IdentityCoeff) {
  const Npp32f coeffs[3] = {1.0f, 0.0f, 0.0f};
  std::vector<Npp32f> srcData(width * height * 3);
  std::vector<Npp32f> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 3;
      float r = static_cast<float>((x + y) * 1.25f);
      float g = static_cast<float>((x * 3 + y) * 0.5f);
      float b = static_cast<float>((x + y * 2) * 0.75f);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      expected[y * width + x] = r;
    }
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp32f *d_src = nppiMalloc_32f_C3(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp32f) * 3, width * sizeof(Npp32f) * 3, height,
               cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_32f_C3C1R(d_src, srcStep, d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> result(width * height);
  cudaMemcpy2D(result.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_NEAR(result[i], expected[i], 1e-5f) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(ColorToGrayTest, ColorToGray_32f_AC4C1R_WeightedNoAlpha) {
  const Npp32f coeffs[3] = {0.25f, 0.5f, 0.25f};
  std::vector<Npp32f> srcData(width * height * 4);
  std::vector<Npp32f> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 4;
      float r = static_cast<float>((x + y) * 1.0f);
      float g = static_cast<float>((x * 2 + y) * 1.0f);
      float b = static_cast<float>((x + y * 2) * 1.0f);
      float a = static_cast<float>((x * 3 + y * 4) * 1.0f);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      srcData[idx + 3] = a;
      expected[y * width + x] = 0.25f * r + 0.5f * g + 0.25f * b;
    }
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp32f *d_src = nppiMalloc_32f_C4(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp32f) * 4, width * sizeof(Npp32f) * 4, height,
               cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_32f_AC4C1R(d_src, srcStep, d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> result(width * height);
  cudaMemcpy2D(result.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_NEAR(result[i], expected[i], 1e-5f) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(ColorToGrayTest, ColorToGray_32f_C4C1R_UsesAlphaCoeff) {
  const Npp32f coeffs[4] = {0.25f, 0.25f, 0.25f, 0.25f};
  std::vector<Npp32f> srcData(width * height * 4);
  std::vector<Npp32f> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 4;
      float r = static_cast<float>((x + y) * 1.25f);
      float g = static_cast<float>((x * 3 + y) * 0.5f);
      float b = static_cast<float>((x + y * 2) * 0.75f);
      float a = static_cast<float>((x * 2 + y * 3) * 0.25f);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      srcData[idx + 3] = a;
      expected[y * width + x] = 0.25f * (r + g + b + a);
    }
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp32f *d_src = nppiMalloc_32f_C4(width, height, &srcStep);
  Npp32f *d_dst = nppiMalloc_32f_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp32f) * 4, width * sizeof(Npp32f) * 4, height,
               cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_32f_C4C1R(d_src, srcStep, d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp32f> result(width * height);
  cudaMemcpy2D(result.data(), width * sizeof(Npp32f), d_dst, dstStep, width * sizeof(Npp32f), height,
               cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_NEAR(result[i], expected[i], 1e-5f) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(ColorToGrayTest, ColorToGray_16s_C3C1R_IdentityCoeff) {
  const Npp32f coeffs[3] = {1.0f, 0.0f, 0.0f};
  std::vector<Npp16s> srcData(width * height * 3);
  std::vector<Npp16s> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 3;
      Npp16s r = static_cast<Npp16s>((x + y) * 64 - 512);
      Npp16s g = static_cast<Npp16s>((x * 3 + y) * 32 - 256);
      Npp16s b = static_cast<Npp16s>((x + y * 2) * 48 - 384);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      expected[y * width + x] = r;
    }
  }

  size_t srcStepBytes = 0;
  int dstStep = 0;
  Npp16s *d_src = nullptr;
  cudaError_t cudaStatus = cudaMallocPitch(reinterpret_cast<void **>(&d_src), &srcStepBytes,
                                           width * sizeof(Npp16s) * 3, height);
  ASSERT_EQ(cudaStatus, cudaSuccess);
  Npp16s *d_dst = nppiMalloc_16s_C1(width, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStepBytes, srcData.data(), width * sizeof(Npp16s) * 3, width * sizeof(Npp16s) * 3, height,
               cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_16s_C3C1R(d_src, static_cast<int>(srcStepBytes), d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16s> result(width * height);
  cudaMemcpy2D(result.data(), width * sizeof(Npp16s), d_dst, dstStep, width * sizeof(Npp16s), height,
               cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at pixel " << i;
  }

  cudaFree(d_src);
  nppiFree(d_dst);
}

TEST_F(ColorToGrayTest, ColorToGray_16s_AC4C1R_WeightedNoAlpha) {
  const Npp32f coeffs[3] = {0.25f, 0.5f, 0.25f};
  std::vector<Npp16s> srcData(width * height * 4);
  std::vector<Npp16s> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 4;
      Npp16s r = static_cast<Npp16s>((x + y) * 64 - 512);
      Npp16s g = static_cast<Npp16s>((x * 2 + y) * 64 - 512);
      Npp16s b = static_cast<Npp16s>((x + y * 2) * 64 - 512);
      Npp16s a = static_cast<Npp16s>((x * 5 + y * 7) * 64 - 512);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      srcData[idx + 3] = a;
      expected[y * width + x] = static_cast<Npp16s>((r + 2 * g + b) / 4);
    }
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp16s *d_src = nppiMalloc_16s_C4(width, height, &srcStep);
  Npp16s *d_dst = nppiMalloc_16s_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp16s) * 4, width * sizeof(Npp16s) * 4, height,
               cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_16s_AC4C1R(d_src, srcStep, d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16s> result(width * height);
  cudaMemcpy2D(result.data(), width * sizeof(Npp16s), d_dst, dstStep, width * sizeof(Npp16s), height,
               cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_F(ColorToGrayTest, ColorToGray_16s_C4C1R_UsesAlphaCoeff) {
  const Npp32f coeffs[4] = {0.25f, 0.25f, 0.25f, 0.25f};
  std::vector<Npp16s> srcData(width * height * 4);
  std::vector<Npp16s> expected(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 4;
      Npp16s r = static_cast<Npp16s>((x + y) * 64 - 512);
      Npp16s g = static_cast<Npp16s>((x * 3 + y) * 64 - 512);
      Npp16s b = static_cast<Npp16s>((x + y * 2) * 64 - 512);
      Npp16s a = static_cast<Npp16s>((x * 2 + y * 3) * 64 - 512);
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
      srcData[idx + 3] = a;
      expected[y * width + x] = static_cast<Npp16s>((r + g + b + a) / 4);
    }
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp16s *d_src = nppiMalloc_16s_C4(width, height, &srcStep);
  Npp16s *d_dst = nppiMalloc_16s_C1(width, height, &dstStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_NE(d_dst, nullptr);

  cudaMemcpy2D(d_src, srcStep, srcData.data(), width * sizeof(Npp16s) * 4, width * sizeof(Npp16s) * 4, height,
               cudaMemcpyHostToDevice);

  NppStatus status = nppiColorToGray_16s_C4C1R(d_src, srcStep, d_dst, dstStep, roi, coeffs);
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<Npp16s> result(width * height);
  cudaMemcpy2D(result.data(), width * sizeof(Npp16s), d_dst, dstStep, width * sizeof(Npp16s), height,
               cudaMemcpyDeviceToHost);

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(result[i], expected[i]) << "Mismatch at pixel " << i;
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}

TEST_P(ColorToGrayCtxParamTest8u, C3C1R_CtxMatches) {
  run_gray_ctx_case<Npp8u, GrayCtxTraits<Npp8u>>(GetParam(), 3, false);
}

TEST_P(ColorToGrayCtxParamTest8u, AC4C1R_CtxMatches) {
  run_gray_ctx_case<Npp8u, GrayCtxTraits<Npp8u>>(GetParam(), 4, true);
}

TEST_P(ColorToGrayCtxParamTest8u, C4C1R_CtxMatches) {
  run_gray_ctx_case<Npp8u, GrayCtxTraits<Npp8u>>(GetParam(), 4, false);
}

TEST_P(ColorToGrayCtxParamTest16u, C3C1R_CtxMatches) {
  run_gray_ctx_case<Npp16u, GrayCtxTraits<Npp16u>>(GetParam(), 3, false);
}

TEST_P(ColorToGrayCtxParamTest16u, AC4C1R_CtxMatches) {
  run_gray_ctx_case<Npp16u, GrayCtxTraits<Npp16u>>(GetParam(), 4, true);
}

TEST_P(ColorToGrayCtxParamTest16u, C4C1R_CtxMatches) {
  run_gray_ctx_case<Npp16u, GrayCtxTraits<Npp16u>>(GetParam(), 4, false);
}

TEST_P(ColorToGrayCtxParamTest16s, C3C1R_CtxMatches) {
  run_gray_ctx_case<Npp16s, GrayCtxTraits<Npp16s>>(GetParam(), 3, false);
}

TEST_P(ColorToGrayCtxParamTest16s, AC4C1R_CtxMatches) {
  run_gray_ctx_case<Npp16s, GrayCtxTraits<Npp16s>>(GetParam(), 4, true);
}

TEST_P(ColorToGrayCtxParamTest16s, C4C1R_CtxMatches) {
  run_gray_ctx_case<Npp16s, GrayCtxTraits<Npp16s>>(GetParam(), 4, false);
}

TEST_P(ColorToGrayCtxParamTest32f, C3C1R_CtxMatches) {
  run_gray_ctx_case<Npp32f, GrayCtxTraits<Npp32f>>(GetParam(), 3, false);
}

TEST_P(ColorToGrayCtxParamTest32f, AC4C1R_CtxMatches) {
  run_gray_ctx_case<Npp32f, GrayCtxTraits<Npp32f>>(GetParam(), 4, true);
}

TEST_P(ColorToGrayCtxParamTest32f, C4C1R_CtxMatches) {
  run_gray_ctx_case<Npp32f, GrayCtxTraits<Npp32f>>(GetParam(), 4, false);
}

INSTANTIATE_TEST_SUITE_P(FunctionalCases, ColorToGrayCtxParamTest8u,
                         ::testing::Values(GrayCtxCase{8, 6, 11}, GrayCtxCase{13, 7, 22}));
INSTANTIATE_TEST_SUITE_P(PrecisionCases, ColorToGrayCtxParamTest8u,
                         ::testing::Values(GrayCtxCase{64, 32, 33}, GrayCtxCase{128, 16, 44}));

INSTANTIATE_TEST_SUITE_P(FunctionalCases, ColorToGrayCtxParamTest16u,
                         ::testing::Values(GrayCtxCase{8, 6, 55}, GrayCtxCase{13, 7, 66}));
INSTANTIATE_TEST_SUITE_P(PrecisionCases, ColorToGrayCtxParamTest16u,
                         ::testing::Values(GrayCtxCase{64, 32, 77}, GrayCtxCase{128, 16, 88}));

INSTANTIATE_TEST_SUITE_P(FunctionalCases, ColorToGrayCtxParamTest16s,
                         ::testing::Values(GrayCtxCase{8, 6, 99}, GrayCtxCase{13, 7, 111}));
INSTANTIATE_TEST_SUITE_P(PrecisionCases, ColorToGrayCtxParamTest16s,
                         ::testing::Values(GrayCtxCase{64, 32, 222}, GrayCtxCase{128, 16, 333}));

INSTANTIATE_TEST_SUITE_P(FunctionalCases, ColorToGrayCtxParamTest32f,
                         ::testing::Values(GrayCtxCase{8, 6, 444}, GrayCtxCase{13, 7, 555}));
INSTANTIATE_TEST_SUITE_P(PrecisionCases, ColorToGrayCtxParamTest32f,
                         ::testing::Values(GrayCtxCase{64, 32, 666}, GrayCtxCase{128, 16, 777}));
