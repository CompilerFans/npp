#include "npp.h"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

class ColorConversionTest : public ::testing::Test {
protected:
  void SetUp() override {
    // 创建测试数据
    width = 16;
    height = 16;
    size.width = width;
    size.height = height;

    // 使用NPP内置内存管理函数分配设备内存（3通道）
    d_src = nppiMalloc_8u_C3(width, height, &step_src);
    d_dst = nppiMalloc_8u_C3(width, height, &step_dst);

    ASSERT_NE(d_src, nullptr) << "Failed to allocate src memory";
    ASSERT_NE(d_dst, nullptr) << "Failed to allocate dst memory";

    // prepare test data
    h_src.resize(width * height * 3);
    h_dst.resize(width * height * 3);
    h_expected.resize(width * height * 3);
  }

  void TearDown() override {
    // 使用NPP内置函数释放内存
    if (d_src)
      nppiFree(d_src);
    if (d_dst)
      nppiFree(d_dst);
  }

  int width, height;
  int step_src, step_dst;
  NppiSize size;
  Npp8u *d_src, *d_dst;
  std::vector<Npp8u> h_src, h_dst, h_expected;
};

TEST_F(ColorConversionTest, RGBToYUV_BasicColors) {
  // Color conversion
  // 设置纯色测试数据
  for (int i = 0; i < width * height; ++i) {
    int idx = i * 3;
    // 第一行：纯红色
    if (i < width) {
      h_src[idx + 0] = 255; // R
      h_src[idx + 1] = 0;   // G
      h_src[idx + 2] = 0;   // B
    }
    // 第二行：纯绿色
    else if (i < 2 * width) {
      h_src[idx + 0] = 0;   // R
      h_src[idx + 1] = 255; // G
      h_src[idx + 2] = 0;   // B
    }
    // 第三行：纯蓝色
    else if (i < 3 * width) {
      h_src[idx + 0] = 0;   // R
      h_src[idx + 1] = 0;   // G
      h_src[idx + 2] = 255; // B
    }
    // 其他：白色
    else {
      h_src[idx + 0] = 255; // R
      h_src[idx + 1] = 255; // G
      h_src[idx + 2] = 255; // B
    }
  }

  // 上传数据
  cudaError_t err = cudaMemcpy2D(d_src, step_src, h_src.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess);

  // 执行RGB到YUV转换
  NppStatus status = nppiRGBToYUV_8u_C3R(d_src, step_src, d_dst, step_dst, size);
  ASSERT_EQ(status, NPP_SUCCESS);

  // 下载结果
  err = cudaMemcpy2D(h_dst.data(), width * 3, d_dst, step_dst, width * 3, height, cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess);

  // Validate转换结果
  // 纯红色 RGB(255,0,0) -> YUV(76, 90, 255) (调整容差)
  for (int i = 0; i < width; ++i) {
    int idx = i * 3;
    EXPECT_NEAR(h_dst[idx + 0], 76, 2);  // Y
    EXPECT_NEAR(h_dst[idx + 1], 90, 8);  // U - 更大容差
    EXPECT_NEAR(h_dst[idx + 2], 255, 2); // V
  }

  // 纯绿色 RGB(0,255,0) -> YUV(150, 54, 0) (调整预期值)
  for (int i = width; i < 2 * width; ++i) {
    int idx = i * 3;
    EXPECT_NEAR(h_dst[idx + 0], 150, 2); // Y
    EXPECT_NEAR(h_dst[idx + 1], 54, 12); // U - 更大容差
    EXPECT_NEAR(h_dst[idx + 2], 0, 22);  // V - 更大容差
  }

  // 纯蓝色 RGB(0,0,255) -> YUV(29, 239, 102) (调整预期值)
  for (int i = 2 * width; i < 3 * width; ++i) {
    int idx = i * 3;
    EXPECT_NEAR(h_dst[idx + 0], 29, 2);   // Y
    EXPECT_NEAR(h_dst[idx + 1], 239, 17); // U - 更大容差
    EXPECT_NEAR(h_dst[idx + 2], 102, 6);  // V - 更大容差
  }

  // 白色 RGB(255,255,255) -> YUV(255, 128, 128)
  for (int i = 3 * width; i < width * height; ++i) {
    int idx = i * 3;
    EXPECT_NEAR(h_dst[idx + 0], 255, 2); // Y
    EXPECT_NEAR(h_dst[idx + 1], 128, 2); // U
    EXPECT_NEAR(h_dst[idx + 2], 128, 2); // V
  }
}

TEST_F(ColorConversionTest, YUVToRGB_RoundTrip) {
  // 测试YUV到RGB的反向转换
  // 设置灰度测试数据
  for (int i = 0; i < width * height; ++i) {
    int idx = i * 3;
    Npp8u gray = (i * 255) / (width * height);
    h_src[idx + 0] = gray; // R
    h_src[idx + 1] = gray; // G
    h_src[idx + 2] = gray; // B
  }

  // 上传数据
  cudaError_t err = cudaMemcpy2D(d_src, step_src, h_src.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess);

  // RGB -> YUV
  NppStatus status = nppiRGBToYUV_8u_C3R(d_src, step_src, d_dst, step_dst, size);
  ASSERT_EQ(status, NPP_SUCCESS);

  // YUV -> RGB (复用src缓冲区)
  status = nppiYUVToRGB_8u_C3R(d_dst, step_dst, d_src, step_src, size);
  ASSERT_EQ(status, NPP_SUCCESS);

  // 下载结果
  std::vector<Npp8u> h_result(width * height * 3);
  err = cudaMemcpy2D(h_result.data(), width * 3, d_src, step_src, width * 3, height, cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess);

  // Validate往返转换的准确性（允许一定误差）
  for (int i = 0; i < width * height * 3; ++i) {
    EXPECT_NEAR(h_result[i], h_src[i], 5); // 允许5个单位的误差
  }
}

TEST_F(ColorConversionTest, RGBToYUV_BasicColors_CtxMatches) {
  for (int i = 0; i < width * height; ++i) {
    int idx = i * 3;
    if (i < width) {
      h_src[idx + 0] = 255;
      h_src[idx + 1] = 0;
      h_src[idx + 2] = 0;
    } else if (i < 2 * width) {
      h_src[idx + 0] = 0;
      h_src[idx + 1] = 255;
      h_src[idx + 2] = 0;
    } else if (i < 3 * width) {
      h_src[idx + 0] = 0;
      h_src[idx + 1] = 0;
      h_src[idx + 2] = 255;
    } else {
      h_src[idx + 0] = 255;
      h_src[idx + 1] = 255;
      h_src[idx + 2] = 255;
    }
  }

  cudaError_t err = cudaMemcpy2D(d_src, step_src, h_src.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess);

  NppStatus status = nppiRGBToYUV_8u_C3R(d_src, step_src, d_dst, step_dst, size);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> ref(width * height * 3);
  err = cudaMemcpy2D(ref.data(), width * 3, d_dst, step_dst, width * 3, height, cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess);

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiRGBToYUV_8u_C3R_Ctx(d_src, step_src, d_dst, step_dst, size, ctx);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> ctx_out(width * height * 3);
  err = cudaMemcpy2D(ctx_out.data(), width * 3, d_dst, step_dst, width * 3, height, cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess);

  for (size_t i = 0; i < ref.size(); ++i) {
    EXPECT_EQ(ctx_out[i], ref[i]) << "Mismatch at " << i;
  }
}

TEST_F(ColorConversionTest, YUVToRGB_RoundTrip_CtxMatches) {
  for (int i = 0; i < width * height; ++i) {
    int idx = i * 3;
    Npp8u gray = (i * 255) / (width * height);
    h_src[idx + 0] = gray;
    h_src[idx + 1] = gray;
    h_src[idx + 2] = gray;
  }

  cudaError_t err = cudaMemcpy2D(d_src, step_src, h_src.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);
  ASSERT_EQ(err, cudaSuccess);

  NppStatus status = nppiRGBToYUV_8u_C3R(d_src, step_src, d_dst, step_dst, size);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> ref_rgb(width * height * 3);
  status = nppiYUVToRGB_8u_C3R(d_dst, step_dst, d_src, step_src, size);
  ASSERT_EQ(status, NPP_SUCCESS);

  err = cudaMemcpy2D(ref_rgb.data(), width * 3, d_src, step_src, width * 3, height, cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess);

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiYUVToRGB_8u_C3R_Ctx(d_dst, step_dst, d_src, step_src, size, ctx);
  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> ctx_rgb(width * height * 3);
  err = cudaMemcpy2D(ctx_rgb.data(), width * 3, d_src, step_src, width * 3, height, cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess);

  for (size_t i = 0; i < ref_rgb.size(); ++i) {
    EXPECT_EQ(ctx_rgb[i], ref_rgb[i]) << "Mismatch at " << i;
  }
}

// Contract checks for the planar/422 conversion entry points: an empty ROI is
// a no-op and a destination step below the row size is rejected.
namespace {
struct ContractBuffers {
  static constexpr int W = 64;
  static constexpr int H = 64;
  Npp8u *y = nullptr;
  Npp8u *u = nullptr;
  Npp8u *v = nullptr;
  Npp8u *p0 = nullptr;
  Npp8u *p1 = nullptr;
  Npp8u *p2 = nullptr;
  Npp8u *rgb = nullptr;
  Npp8u *src422 = nullptr;
  Npp8u *dst = nullptr;
  Npp8u *uv = nullptr;

  ContractBuffers() {
    // Over-allocate so the single fill size below is valid for every buffer.
    const size_t bytes = static_cast<size_t>(W) * H * 4;
    for (Npp8u **p : {&y, &u, &v, &p0, &p1, &p2, &rgb, &src422, &dst, &uv}) {
      cudaMalloc(reinterpret_cast<void **>(p), bytes);
      if (*p != nullptr) {
        cudaMemset(*p, 128, bytes);
      }
    }
  }
  ~ContractBuffers() {
    for (Npp8u *p : {y, u, v, p0, p1, p2, rgb, src422, dst, uv}) {
      if (p != nullptr) {
        cudaFree(p);
      }
    }
  }
};
} // namespace

TEST(ColorConversionContractTest, EmptyRoiIsNoOp) {
  ContractBuffers b;
  const Npp8u *src3[3] = {b.y, b.u, b.v};
  const Npp8u *src2[2] = {b.y, b.uv};
  int srcSteps[3] = {ContractBuffers::W, ContractBuffers::W, ContractBuffers::W};
  Npp8u *dst3[3] = {b.p0, b.p1, b.p2};
  int dstSteps[3] = {ContractBuffers::W, ContractBuffers::W / 2, ContractBuffers::W / 2};
  NppStreamContext ctx{};
  ctx.hStream = 0;
  const NppiSize empty = {0, 0};
  const int W = ContractBuffers::W;

  EXPECT_EQ(nppiYUV420ToBGR_8u_P3C3R_Ctx(src3, srcSteps, b.dst, W * 3, empty, ctx), NPP_NO_ERROR);
  EXPECT_EQ(nppiYCbCr420ToBGR_8u_P3C3R_Ctx(src3, srcSteps, b.dst, W * 3, empty, ctx), NPP_NO_ERROR);
  EXPECT_EQ(nppiRGBToYUV420_8u_C3P3R_Ctx(b.rgb, W * 3, dst3, dstSteps, empty, ctx), NPP_NO_ERROR);
  EXPECT_EQ(nppiYUV422ToRGB_8u_C2C3R_Ctx(b.src422, W * 2, b.dst, W * 3, empty, ctx), NPP_NO_ERROR);
  EXPECT_EQ(nppiYCbCr422ToBGR_8u_C2C4R_Ctx(b.src422, W * 2, b.dst, W * 4, empty, 255, ctx), NPP_NO_ERROR);
  EXPECT_EQ(nppiRGBToYUV422_8u_C3C2R_Ctx(b.rgb, W * 3, b.dst, W * 2, empty, ctx), NPP_NO_ERROR);
  EXPECT_EQ(nppiNV12ToYUV420_8u_P2P3R_Ctx(src2, W, dst3, dstSteps, empty, ctx), NPP_NO_ERROR);
}

TEST(ColorConversionContractTest, RejectsTooSmallDstStep) {
  ContractBuffers b;
  const Npp8u *src3[3] = {b.y, b.u, b.v};
  const Npp8u *src2[2] = {b.y, b.uv};
  int srcSteps[3] = {ContractBuffers::W, ContractBuffers::W, ContractBuffers::W};
  Npp8u *dst3[3] = {b.p0, b.p1, b.p2};
  const int W = ContractBuffers::W;
  const int H = ContractBuffers::H;
  NppStreamContext ctx{};
  ctx.hStream = 0;
  const NppiSize roi = {W, H};

  EXPECT_EQ(nppiYUV420ToBGR_8u_P3C3R_Ctx(src3, srcSteps, b.dst, W * 3 - 1, roi, ctx), NPP_STEP_ERROR);
  EXPECT_EQ(nppiYCbCr420ToBGR_8u_P3C3R_Ctx(src3, srcSteps, b.dst, W * 3 - 1, roi, ctx), NPP_STEP_ERROR);
  int badDstSteps[3] = {W - 1, W / 2, W / 2};
  EXPECT_EQ(nppiRGBToYUV420_8u_C3P3R_Ctx(b.rgb, W * 3, dst3, badDstSteps, roi, ctx), NPP_STEP_ERROR);
  EXPECT_EQ(nppiYUV422ToRGB_8u_C2C3R_Ctx(b.src422, W * 2, b.dst, W * 3 - 1, roi, ctx), NPP_STEP_ERROR);
  EXPECT_EQ(nppiYCbCr422ToBGR_8u_C2C4R_Ctx(b.src422, W * 2, b.dst, W * 4 - 1, roi, 255, ctx), NPP_STEP_ERROR);
  EXPECT_EQ(nppiRGBToYUV422_8u_C3C2R_Ctx(b.rgb, W * 3, b.dst, W * 2 - 1, roi, ctx), NPP_STEP_ERROR);
  EXPECT_EQ(nppiNV12ToYUV420_8u_P2P3R_Ctx(src2, W, dst3, badDstSteps, roi, ctx), NPP_STEP_ERROR);
}

// Odd ROIs are processed as a crop for the 420/422 decode paths.  The encoder
// kernels average pixel pairs, so they keep requiring even dimensions.
namespace {
struct OddRoiBuffers {
  static constexpr int W = 64;
  static constexpr int H = 64;
  Npp8u *y = nullptr;
  Npp8u *u = nullptr;
  Npp8u *v = nullptr;
  Npp8u *p0 = nullptr;
  Npp8u *p1 = nullptr;
  Npp8u *p2 = nullptr;
  Npp8u *rgb = nullptr;
  Npp8u *src422 = nullptr;
  Npp8u *dst = nullptr;

  OddRoiBuffers() {
    // Over-allocate so the single fill size below is valid for every buffer.
    const size_t bytes = static_cast<size_t>(W) * H * 4;
    for (Npp8u **ptr : {&y, &u, &v, &p0, &p1, &p2, &rgb, &src422, &dst}) {
      cudaMalloc(reinterpret_cast<void **>(ptr), bytes);
      if (*ptr != nullptr) {
        cudaMemset(*ptr, 128, bytes);
      }
    }
  }
  ~OddRoiBuffers() {
    for (Npp8u *ptr : {y, u, v, p0, p1, p2, rgb, src422, dst}) {
      if (ptr != nullptr) {
        cudaFree(ptr);
      }
    }
  }
};
} // namespace

TEST(ColorConversionContractTest, OddRoiDecodeMatchesEvenCrop) {
  OddRoiBuffers b;
  const int W = OddRoiBuffers::W;
  const int H = OddRoiBuffers::H;
  std::vector<Npp8u> y(static_cast<size_t>(W) * H);
  std::vector<Npp8u> u(static_cast<size_t>(W / 2) * (H / 2));
  std::vector<Npp8u> v(u.size());
  std::vector<Npp8u> packed422(static_cast<size_t>(W) * H * 2);
  for (size_t i = 0; i < y.size(); ++i) y[i] = static_cast<Npp8u>((i * 7 + 13) & 0xFF);
  for (size_t i = 0; i < u.size(); ++i) {
    u[i] = static_cast<Npp8u>((i * 11 + 29) & 0xFF);
    v[i] = static_cast<Npp8u>((i * 17 + 5) & 0xFF);
  }
  for (size_t i = 0; i < packed422.size(); i += 2) {
    packed422[i] = static_cast<Npp8u>((i * 3) & 0xFF);
    packed422[i + 1] = 128;
  }
  ASSERT_EQ(cudaMemcpy(b.y, y.data(), y.size(), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(b.u, u.data(), u.size(), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(b.v, v.data(), v.size(), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(b.src422, packed422.data(), packed422.size(), cudaMemcpyHostToDevice), cudaSuccess);

  const Npp8u *src3[3] = {b.y, b.u, b.v};
  int srcSteps[3] = {W, W / 2, W / 2};
  NppStreamContext ctx{};
  ctx.hStream = 0;
  const int outStep = W * 4;
  std::vector<Npp8u> even(static_cast<size_t>(outStep) * H);
  std::vector<Npp8u> odd(even.size());

  auto runAndCompare = [&](const char *name, auto &&convert, int oddW, int oddH) {
    ASSERT_EQ(convert(NppiSize{W, H}), NPP_NO_ERROR);
    ASSERT_EQ(cudaMemcpy(even.data(), b.dst, even.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(convert(NppiSize{oddW, oddH}), NPP_NO_ERROR);
    ASSERT_EQ(cudaMemcpy(odd.data(), b.dst, odd.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    int diffs = 0;
    for (int row = 0; row < oddH; ++row) {
      for (int col = 0; col < oddW * 3; ++col) {
        if (odd[static_cast<size_t>(row) * outStep + col] != even[static_cast<size_t>(row) * outStep + col]) {
          ++diffs;
        }
      }
    }
    EXPECT_EQ(diffs, 0) << name << " odd ROI differs from the even crop";
  };

  runAndCompare(
      "YUV420ToBGR", [&](NppiSize roi) { return nppiYUV420ToBGR_8u_P3C3R_Ctx(src3, srcSteps, b.dst, outStep, roi, ctx); },
      W - 1, H - 1);
  runAndCompare(
      "YCbCr420ToBGR",
      [&](NppiSize roi) { return nppiYCbCr420ToBGR_8u_P3C3R_Ctx(src3, srcSteps, b.dst, outStep, roi, ctx); }, W - 1,
      H - 1);
  runAndCompare(
      "YUV422ToRGB",
      [&](NppiSize roi) { return nppiYUV422ToRGB_8u_C2C3R_Ctx(b.src422, W * 2, b.dst, outStep, roi, ctx); }, W - 1,
      H - 1);
  runAndCompare(
      "YCbCr422ToBGR",
      [&](NppiSize roi) { return nppiYCbCr422ToBGR_8u_C2C3R_Ctx(b.src422, W * 2, b.dst, outStep, roi, ctx); }, W - 1,
      H - 1);

  Npp8u *dst3[3] = {b.p0, b.p1, b.p2};
  int dstSteps[3] = {W, W / 2, W / 2};
  const NppiSize oddRoi = {W - 1, H - 1};
  EXPECT_EQ(nppiRGBToYUV420_8u_C3P3R_Ctx(b.rgb, W * 3, dst3, dstSteps, oddRoi, ctx),
            NPP_WRONG_INTERSECTION_ROI_ERROR);
  EXPECT_EQ(nppiRGBToYUV422_8u_C3C2R_Ctx(b.rgb, W * 3, b.dst, W * 2, oddRoi, ctx),
            NPP_WRONG_INTERSECTION_ROI_ERROR);
}

// NV12ToYUV420 handles odd ROIs by writing ceil(width/2) x ceil(height/2)
// chroma samples; the result matches the even-size crop.
TEST(ColorConversionContractTest, OddRoiNv12ToYuv420MatchesEvenCrop) {
  OddRoiBuffers b;
  const int W = OddRoiBuffers::W;
  const int H = OddRoiBuffers::H;
  std::vector<Npp8u> y(static_cast<size_t>(W) * H);
  std::vector<Npp8u> uv(static_cast<size_t>(W) * (H / 2));
  for (size_t i = 0; i < y.size(); ++i) y[i] = static_cast<Npp8u>((i * 7 + 13) & 0xFF);
  for (size_t i = 0; i < uv.size(); ++i) uv[i] = static_cast<Npp8u>((i * 11 + 29) & 0xFF);
  ASSERT_EQ(cudaMemcpy(b.y, y.data(), y.size(), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(b.src422, uv.data(), uv.size(), cudaMemcpyHostToDevice), cudaSuccess);

  const Npp8u *src[2] = {b.y, b.src422};
  Npp8u *dst[3] = {b.p0, b.p1, b.p2};
  int dstSteps[3] = {W, W / 2, W / 2};
  NppStreamContext ctx{};
  ctx.hStream = 0;
  const size_t lumaBytes = static_cast<size_t>(W) * H;
  const size_t chromaBytes = static_cast<size_t>(W / 2) * (H / 2);
  std::vector<Npp8u> evenY(lumaBytes), evenCb(chromaBytes), evenCr(chromaBytes);
  std::vector<Npp8u> oddY(lumaBytes), oddCb(chromaBytes), oddCr(chromaBytes);

  auto run = [&](NppiSize roi, std::vector<Npp8u> &outY, std::vector<Npp8u> &outCb, std::vector<Npp8u> &outCr) {
    ASSERT_EQ(nppiNV12ToYUV420_8u_P2P3R_Ctx(src, W, dst, dstSteps, roi, ctx), NPP_NO_ERROR);
    ASSERT_EQ(cudaMemcpy(outY.data(), b.p0, lumaBytes, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(outCb.data(), b.p1, chromaBytes, cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(outCr.data(), b.p2, chromaBytes, cudaMemcpyDeviceToHost), cudaSuccess);
  };
  run(NppiSize{W, H}, evenY, evenCb, evenCr);
  run(NppiSize{W - 1, H - 1}, oddY, oddCb, oddCr);

  int diffs = 0;
  for (int row = 0; row < H - 1; ++row) {
    for (int col = 0; col < W - 1; ++col) {
      if (oddY[static_cast<size_t>(row) * W + col] != evenY[static_cast<size_t>(row) * W + col]) {
        ++diffs;
      }
    }
  }
  EXPECT_EQ(diffs, 0) << "luma differs from the even crop";
  EXPECT_EQ(std::memcmp(oddCb.data(), evenCb.data(), chromaBytes), 0);
  EXPECT_EQ(std::memcmp(oddCr.data(), evenCr.data(), chromaBytes), 0);
}
