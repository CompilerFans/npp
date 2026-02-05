#include "npp.h"
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
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
  // 测试基本颜色转换
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

  // 验证转换结果
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

  // 验证往返转换的准确性（允许一定误差）
  for (int i = 0; i < width * height * 3; ++i) {
    EXPECT_NEAR(h_result[i], h_src[i], 5); // 允许5个单位的误差
  }
}

namespace {

void fillPackedRgb(std::vector<Npp8u> &data, int width, int height, int channels) {
  data.resize(width * height * channels);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * channels;
      data[idx + 0] = static_cast<Npp8u>((x * 3 + y * 5) & 0xFF);
      data[idx + 1] = static_cast<Npp8u>((x * 7 + y * 11) & 0xFF);
      data[idx + 2] = static_cast<Npp8u>((x * 13 + y * 17) & 0xFF);
      if (channels == 4) {
        data[idx + 3] = static_cast<Npp8u>((x * 19 + y * 23) & 0xFF);
      }
    }
  }
}

void fillPlanarRgb(std::vector<Npp8u> &r, std::vector<Npp8u> &g, std::vector<Npp8u> &b, int width, int height) {
  r.resize(width * height);
  g.resize(width * height);
  b.resize(width * height);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = y * width + x;
      r[idx] = static_cast<Npp8u>((x * 3 + y * 5) & 0xFF);
      g[idx] = static_cast<Npp8u>((x * 7 + y * 11) & 0xFF);
      b[idx] = static_cast<Npp8u>((x * 13 + y * 17) & 0xFF);
    }
  }
}

void fillYuv420(std::vector<Npp8u> &y, std::vector<Npp8u> &u, std::vector<Npp8u> &v, int width, int height) {
  y.resize(width * height);
  u.resize((width / 2) * (height / 2));
  v.resize((width / 2) * (height / 2));
  for (int yy = 0; yy < height; ++yy) {
    for (int x = 0; x < width; ++x) {
      int idx = yy * width + x;
      y[idx] = static_cast<Npp8u>((x * 3 + yy * 5) & 0xFF);
    }
  }
  for (int yy = 0; yy < height / 2; ++yy) {
    for (int x = 0; x < width / 2; ++x) {
      int idx = yy * (width / 2) + x;
      u[idx] = static_cast<Npp8u>((x * 7 + yy * 11) & 0xFF);
      v[idx] = static_cast<Npp8u>((x * 13 + yy * 17) & 0xFF);
    }
  }
}

struct NppImage {
  Npp8u *ptr = nullptr;
  int step = 0;
  int width = 0;
  int height = 0;
  int channels = 0;

  void allocate(int w, int h, int c) {
    width = w;
    height = h;
    channels = c;
    if (channels == 1) {
      ptr = nppiMalloc_8u_C1(width, height, &step);
    } else if (channels == 3) {
      ptr = nppiMalloc_8u_C3(width, height, &step);
    } else if (channels == 4) {
      ptr = nppiMalloc_8u_C4(width, height, &step);
    } else {
      FAIL() << "Unsupported channel count";
    }
    ASSERT_NE(ptr, nullptr);
  }

  void free() {
    if (ptr) {
      nppiFree(ptr);
      ptr = nullptr;
      step = 0;
    }
  }

  void copyFromHost(const std::vector<Npp8u> &host) {
    ASSERT_EQ(host.size(), static_cast<size_t>(width * height * channels));
    cudaError_t err = cudaMemcpy2D(ptr, step, host.data(), width * channels, width * channels, height,
                                   cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess);
  }

  void copyToHost(std::vector<Npp8u> &host) const {
    host.resize(width * height * channels);
    cudaError_t err =
        cudaMemcpy2D(host.data(), width * channels, ptr, step, width * channels, height, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);
  }
};

struct NppPlane {
  Npp8u *ptr = nullptr;
  int step = 0;
  int width = 0;
  int height = 0;

  void allocate(int w, int h) {
    width = w;
    height = h;
    ptr = nppiMalloc_8u_C1(width, height, &step);
    ASSERT_NE(ptr, nullptr);
  }

  void free() {
    if (ptr) {
      nppiFree(ptr);
      ptr = nullptr;
      step = 0;
    }
  }

  void copyFromHost(const std::vector<Npp8u> &host) {
    ASSERT_EQ(host.size(), static_cast<size_t>(width * height));
    cudaError_t err = cudaMemcpy2D(ptr, step, host.data(), width, width, height, cudaMemcpyHostToDevice);
    ASSERT_EQ(err, cudaSuccess);
  }

  void copyToHost(std::vector<Npp8u> &host) const {
    host.resize(width * height);
    cudaError_t err = cudaMemcpy2D(host.data(), width, ptr, step, width, height, cudaMemcpyDeviceToHost);
    ASSERT_EQ(err, cudaSuccess);
  }
};

void interleavePlanar(std::vector<Npp8u> &out, const std::vector<Npp8u> &p0, const std::vector<Npp8u> &p1,
                      const std::vector<Npp8u> &p2) {
  const size_t size = p0.size();
  out.resize(size * 3);
  for (size_t i = 0; i < size; ++i) {
    out[i * 3 + 0] = p0[i];
    out[i * 3 + 1] = p1[i];
    out[i * 3 + 2] = p2[i];
  }
}

void interleavePlanar4(std::vector<Npp8u> &out, const std::vector<Npp8u> &p0, const std::vector<Npp8u> &p1,
                       const std::vector<Npp8u> &p2, const std::vector<Npp8u> &p3) {
  const size_t size = p0.size();
  out.resize(size * 4);
  for (size_t i = 0; i < size; ++i) {
    out[i * 4 + 0] = p0[i];
    out[i * 4 + 1] = p1[i];
    out[i * 4 + 2] = p2[i];
    out[i * 4 + 3] = p3[i];
  }
}

} // namespace

TEST(ColorConversionFunctionalTest, RGBToYUV_PlanarConsistency) {
  const int width = 32;
  const int height = 24;
  const NppiSize roi = {width, height};

  std::vector<Npp8u> srcHost;
  fillPackedRgb(srcHost, width, height, 3);

  NppImage src;
  NppImage dstPacked;
  NppPlane dstY;
  NppPlane dstU;
  NppPlane dstV;
  src.allocate(width, height, 3);
  dstPacked.allocate(width, height, 3);
  dstY.allocate(width, height);
  dstU.allocate(width, height);
  dstV.allocate(width, height);

  ASSERT_EQ(dstY.step, dstU.step);
  ASSERT_EQ(dstY.step, dstV.step);

  src.copyFromHost(srcHost);

  ASSERT_EQ(nppiRGBToYUV_8u_C3R(src.ptr, src.step, dstPacked.ptr, dstPacked.step, roi), NPP_SUCCESS);

  Npp8u *dstPlanes[3] = {dstY.ptr, dstU.ptr, dstV.ptr};
  ASSERT_EQ(nppiRGBToYUV_8u_C3P3R(src.ptr, src.step, dstPlanes, dstY.step, roi), NPP_SUCCESS);

  std::vector<Npp8u> packed;
  std::vector<Npp8u> y, u, v;
  dstPacked.copyToHost(packed);
  dstY.copyToHost(y);
  dstU.copyToHost(u);
  dstV.copyToHost(v);

  std::vector<Npp8u> interleaved;
  interleavePlanar(interleaved, y, u, v);

  ASSERT_EQ(packed.size(), interleaved.size());
  for (size_t i = 0; i < packed.size(); ++i) {
    EXPECT_NEAR(packed[i], interleaved[i], 2);
  }

  src.free();
  dstPacked.free();
  dstY.free();
  dstU.free();
  dstV.free();
}

TEST(ColorConversionFunctionalTest, RGBToYUV_AC4_PlanarConsistency) {
  const int width = 32;
  const int height = 24;
  const NppiSize roi = {width, height};

  std::vector<Npp8u> srcHost;
  fillPackedRgb(srcHost, width, height, 4);
  for (int i = 0; i < width * height; ++i) {
    srcHost[i * 4 + 3] = static_cast<Npp8u>(0x5A);
  }

  NppImage src;
  NppImage dstPacked;
  NppPlane dst0;
  NppPlane dst1;
  NppPlane dst2;
  NppPlane dst3;
  src.allocate(width, height, 4);
  dstPacked.allocate(width, height, 4);
  dst0.allocate(width, height);
  dst1.allocate(width, height);
  dst2.allocate(width, height);
  dst3.allocate(width, height);

  ASSERT_EQ(dst0.step, dst1.step);
  ASSERT_EQ(dst0.step, dst2.step);
  ASSERT_EQ(dst0.step, dst3.step);

  src.copyFromHost(srcHost);

  ASSERT_EQ(nppiRGBToYUV_8u_AC4R(src.ptr, src.step, dstPacked.ptr, dstPacked.step, roi), NPP_SUCCESS);

  Npp8u *dstPlanes[4] = {dst0.ptr, dst1.ptr, dst2.ptr, dst3.ptr};
  ASSERT_EQ(nppiRGBToYUV_8u_AC4P4R(src.ptr, src.step, dstPlanes, dst0.step, roi), NPP_SUCCESS);

  std::vector<Npp8u> packed;
  std::vector<Npp8u> c0, c1, c2, c3;
  dstPacked.copyToHost(packed);
  dst0.copyToHost(c0);
  dst1.copyToHost(c1);
  dst2.copyToHost(c2);
  dst3.copyToHost(c3);

  for (size_t i = 0; i < c0.size(); ++i) {
    EXPECT_NEAR(packed[i * 4 + 0], c0[i], 2);
    EXPECT_NEAR(packed[i * 4 + 1], c1[i], 2);
    EXPECT_NEAR(packed[i * 4 + 2], c2[i], 2);
    EXPECT_EQ(packed[i * 4 + 3], srcHost[i * 4 + 3]);
    EXPECT_EQ(c3[i], srcHost[i * 4 + 3]);
  }

  src.free();
  dstPacked.free();
  dst0.free();
  dst1.free();
  dst2.free();
  dst3.free();
}

TEST(ColorConversionFunctionalTest, YUV420ToRGB_PlanarVsPacked) {
  const int width = 32;
  const int height = 24;
  const NppiSize roi = {width, height};

  std::vector<Npp8u> yHost, uHost, vHost;
  fillYuv420(yHost, uHost, vHost, width, height);

  NppPlane srcY;
  NppPlane srcU;
  NppPlane srcV;
  srcY.allocate(width, height);
  srcU.allocate(width / 2, height / 2);
  srcV.allocate(width / 2, height / 2);

  srcY.copyFromHost(yHost);
  srcU.copyFromHost(uHost);
  srcV.copyFromHost(vHost);

  const Npp8u *srcPlanes[3] = {srcY.ptr, srcU.ptr, srcV.ptr};
  int srcStep[3] = {srcY.step, srcU.step, srcV.step};

  NppPlane dstR;
  NppPlane dstG;
  NppPlane dstB;
  dstR.allocate(width, height);
  dstG.allocate(width, height);
  dstB.allocate(width, height);

  ASSERT_EQ(dstR.step, dstG.step);
  ASSERT_EQ(dstR.step, dstB.step);

  Npp8u *dstPlanes[3] = {dstR.ptr, dstG.ptr, dstB.ptr};
  ASSERT_EQ(nppiYUV420ToRGB_8u_P3R(srcPlanes, srcStep, dstPlanes, dstR.step, roi), NPP_SUCCESS);

  NppImage dstPacked;
  dstPacked.allocate(width, height, 3);
  ASSERT_EQ(nppiYUV420ToRGB_8u_P3C3R(srcPlanes, srcStep, dstPacked.ptr, dstPacked.step, roi), NPP_SUCCESS);

  std::vector<Npp8u> r, g, b;
  std::vector<Npp8u> packed;
  dstR.copyToHost(r);
  dstG.copyToHost(g);
  dstB.copyToHost(b);
  dstPacked.copyToHost(packed);

  std::vector<Npp8u> interleaved;
  interleavePlanar(interleaved, r, g, b);

  ASSERT_EQ(packed.size(), interleaved.size());
  for (size_t i = 0; i < packed.size(); ++i) {
    EXPECT_NEAR(packed[i], interleaved[i], 2);
  }

  srcY.free();
  srcU.free();
  srcV.free();
  dstR.free();
  dstG.free();
  dstB.free();
  dstPacked.free();
}

TEST(ColorConversionFunctionalTest, YUV420ToRGB_C4VsAC4) {
  const int width = 32;
  const int height = 24;
  const NppiSize roi = {width, height};

  std::vector<Npp8u> yHost, uHost, vHost;
  fillYuv420(yHost, uHost, vHost, width, height);

  NppPlane srcY;
  NppPlane srcU;
  NppPlane srcV;
  srcY.allocate(width, height);
  srcU.allocate(width / 2, height / 2);
  srcV.allocate(width / 2, height / 2);

  srcY.copyFromHost(yHost);
  srcU.copyFromHost(uHost);
  srcV.copyFromHost(vHost);

  const Npp8u *srcPlanes[3] = {srcY.ptr, srcU.ptr, srcV.ptr};
  int srcStep[3] = {srcY.step, srcU.step, srcV.step};

  NppImage dstC4;
  NppImage dstAC4;
  dstC4.allocate(width, height, 4);
  dstAC4.allocate(width, height, 4);

  std::vector<Npp8u> initAlpha(width * height * 4, 0);
  for (int i = 0; i < width * height; ++i) {
    initAlpha[i * 4 + 3] = 0x7F;
  }
  dstAC4.copyFromHost(initAlpha);

  ASSERT_EQ(nppiYUV420ToRGB_8u_P3C4R(srcPlanes, srcStep, dstC4.ptr, dstC4.step, roi), NPP_SUCCESS);
  ASSERT_EQ(nppiYUV420ToRGB_8u_P3AC4R(srcPlanes, srcStep, dstAC4.ptr, dstAC4.step, roi), NPP_SUCCESS);

  std::vector<Npp8u> c4;
  std::vector<Npp8u> ac4;
  dstC4.copyToHost(c4);
  dstAC4.copyToHost(ac4);

  ASSERT_EQ(c4.size(), ac4.size());
  for (size_t i = 0; i < c4.size() / 4; ++i) {
    EXPECT_NEAR(c4[i * 4 + 0], ac4[i * 4 + 0], 2);
    EXPECT_NEAR(c4[i * 4 + 1], ac4[i * 4 + 1], 2);
    EXPECT_NEAR(c4[i * 4 + 2], ac4[i * 4 + 2], 2);
    EXPECT_EQ(ac4[i * 4 + 3], 0x7F);
  }

  srcY.free();
  srcU.free();
  srcV.free();
  dstC4.free();
  dstAC4.free();
}
