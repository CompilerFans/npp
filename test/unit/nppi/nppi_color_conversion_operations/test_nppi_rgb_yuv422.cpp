#include "npp.h"
#include "npp_test_base.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

namespace {

class RGBToYUV422Test : public ::testing::Test {
protected:
  void SetUp() override {
    width = 6;
    height = 2;
    ASSERT_EQ(width % 2, 0);
  }

  void createPackedRGB(std::vector<Npp8u> &rgb) const {
    rgb.resize(width * height * 3);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int idx = (y * width + x) * 3;
        rgb[idx + 0] = static_cast<Npp8u>(10 + x * 20 + y * 15);
        rgb[idx + 1] = static_cast<Npp8u>(30 + x * 10 + y * 20);
        rgb[idx + 2] = static_cast<Npp8u>(60 + x * 5 + y * 10);
      }
    }
  }

  void createPlanarRGB(std::vector<Npp8u> &r, std::vector<Npp8u> &g, std::vector<Npp8u> &b) const {
    r.resize(width * height);
    g.resize(width * height);
    b.resize(width * height);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int idx = y * width + x;
        r[idx] = static_cast<Npp8u>(10 + x * 20 + y * 15);
        g[idx] = static_cast<Npp8u>(30 + x * 10 + y * 20);
        b[idx] = static_cast<Npp8u>(60 + x * 5 + y * 10);
      }
    }
  }

  void rgbToYuvPixel(Npp8u r, Npp8u g, Npp8u b, Npp8u &y, Npp8u &u, Npp8u &v) const {
    const float R = static_cast<float>(r);
    const float G = static_cast<float>(g);
    const float B = static_cast<float>(b);

    const float Y = 0.299f * R + 0.587f * G + 0.114f * B;
    const float U = -0.147f * R - 0.289f * G + 0.436f * B + 128.0f;
    const float V = 0.615f * R - 0.515f * G - 0.100f * B + 128.0f;

    auto clamp = [](float val) -> Npp8u {
      if (val < 0.0f) {
        return 0;
      }
      if (val > 255.0f) {
        return 255;
      }
      return static_cast<Npp8u>(val);
    };

    y = clamp(Y);
    u = clamp(U);
    v = clamp(V);
  }

  void computeExpectedPlanar(const std::vector<Npp8u> &rgb, std::vector<Npp8u> &expY, std::vector<Npp8u> &expU,
                             std::vector<Npp8u> &expV) const {
    expY.resize(width * height);
    expU.resize((width / 2) * height);
    expV.resize((width / 2) * height);

    std::vector<Npp8u> uTemp(width * height);
    std::vector<Npp8u> vTemp(width * height);

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int idx = (y * width + x) * 3;
        Npp8u yv, uv, vv;
        rgbToYuvPixel(rgb[idx + 0], rgb[idx + 1], rgb[idx + 2], yv, uv, vv);
        expY[y * width + x] = yv;
        uTemp[y * width + x] = uv;
        vTemp[y * width + x] = vv;
      }
    }

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; x += 2) {
        int dstIdx = y * (width / 2) + (x / 2);
        int sumU = uTemp[y * width + x] + uTemp[y * width + x + 1];
        int sumV = vTemp[y * width + x] + vTemp[y * width + x + 1];
        expU[dstIdx] = static_cast<Npp8u>(sumU / 2);
        expV[dstIdx] = static_cast<Npp8u>(sumV / 2);
      }
    }
  }

  int width = 0;
  int height = 0;
};

} // namespace

TEST_F(RGBToYUV422Test, RGBToYUV422_8u_C3P3R) {
  std::vector<Npp8u> hostRGB;
  createPackedRGB(hostRGB);

  int srcStep = 0;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);
  cudaMemcpy2D(d_src, srcStep, hostRGB.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);

  int yStep = 0;
  int uStep = 0;
  int vStep = 0;
  Npp8u *d_y = nppiMalloc_8u_C1(width, height, &yStep);
  Npp8u *d_u = nppiMalloc_8u_C1(width / 2, height, &uStep);
  Npp8u *d_v = nppiMalloc_8u_C1(width / 2, height, &vStep);
  ASSERT_NE(d_y, nullptr);
  ASSERT_NE(d_u, nullptr);
  ASSERT_NE(d_v, nullptr);

  Npp8u *pDst[3] = {d_y, d_u, d_v};
  int dstSteps[3] = {yStep, uStep, vStep};
  NppiSize roi = {width, height};

  NppStatus status = nppiRGBToYUV422_8u_C3P3R(d_src, srcStep, pDst, dstSteps, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> outY(yStep * height);
  std::vector<Npp8u> outU(uStep * height);
  std::vector<Npp8u> outV(vStep * height);
  cudaMemcpy(outY.data(), d_y, outY.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(outU.data(), d_u, outU.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(outV.data(), d_v, outV.size(), cudaMemcpyDeviceToHost);

  std::vector<Npp8u> expY, expU, expV;
  computeExpectedPlanar(hostRGB, expY, expU, expV);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      EXPECT_EQ(outY[y * yStep + x], expY[y * width + x]);
    }
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width / 2; ++x) {
      int idx = y * uStep + x;
      int expIdx = y * (width / 2) + x;
      EXPECT_EQ(outU[idx], expU[expIdx]);
      EXPECT_EQ(outV[idx], expV[expIdx]);
    }
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiRGBToYUV422_8u_C3P3R_Ctx(d_src, srcStep, pDst, dstSteps, roi, ctx);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctxY(yStep * height);
  std::vector<Npp8u> ctxU(uStep * height);
  std::vector<Npp8u> ctxV(vStep * height);
  cudaMemcpy(ctxY.data(), d_y, ctxY.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(ctxU.data(), d_u, ctxU.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(ctxV.data(), d_v, ctxV.size(), cudaMemcpyDeviceToHost);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      EXPECT_EQ(ctxY[y * yStep + x], expY[y * width + x]);
    }
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width / 2; ++x) {
      int idx = y * uStep + x;
      int expIdx = y * (width / 2) + x;
      EXPECT_EQ(ctxU[idx], expU[expIdx]);
      EXPECT_EQ(ctxV[idx], expV[expIdx]);
    }
  }

  nppiFree(d_src);
  nppiFree(d_y);
  nppiFree(d_u);
  nppiFree(d_v);
}

TEST_F(RGBToYUV422Test, RGBToYUV422_8u_P3R) {
  std::vector<Npp8u> hostR, hostG, hostB;
  createPlanarRGB(hostR, hostG, hostB);

  int rStep = 0;
  int gStep = 0;
  int bStep = 0;
  Npp8u *d_r = nppiMalloc_8u_C1(width, height, &rStep);
  Npp8u *d_g = nppiMalloc_8u_C1(width, height, &gStep);
  Npp8u *d_b = nppiMalloc_8u_C1(width, height, &bStep);
  ASSERT_NE(d_r, nullptr);
  ASSERT_NE(d_g, nullptr);
  ASSERT_NE(d_b, nullptr);
  cudaMemcpy2D(d_r, rStep, hostR.data(), width, width, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_g, gStep, hostG.data(), width, width, height, cudaMemcpyHostToDevice);
  cudaMemcpy2D(d_b, bStep, hostB.data(), width, width, height, cudaMemcpyHostToDevice);

  int yStep = 0;
  int uStep = 0;
  int vStep = 0;
  Npp8u *d_y = nppiMalloc_8u_C1(width, height, &yStep);
  Npp8u *d_u = nppiMalloc_8u_C1(width / 2, height, &uStep);
  Npp8u *d_v = nppiMalloc_8u_C1(width / 2, height, &vStep);
  ASSERT_NE(d_y, nullptr);
  ASSERT_NE(d_u, nullptr);
  ASSERT_NE(d_v, nullptr);

  const Npp8u *pSrc[3] = {d_r, d_g, d_b};
  Npp8u *pDst[3] = {d_y, d_u, d_v};
  int dstSteps[3] = {yStep, uStep, vStep};
  NppiSize roi = {width, height};

  NppStatus status = nppiRGBToYUV422_8u_P3R(pSrc, rStep, pDst, dstSteps, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> outY(yStep * height);
  std::vector<Npp8u> outU(uStep * height);
  std::vector<Npp8u> outV(vStep * height);
  cudaMemcpy(outY.data(), d_y, outY.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(outU.data(), d_u, outU.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(outV.data(), d_v, outV.size(), cudaMemcpyDeviceToHost);

  std::vector<Npp8u> packedRGB(width * height * 3);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 3;
      int p = y * width + x;
      packedRGB[idx + 0] = hostR[p];
      packedRGB[idx + 1] = hostG[p];
      packedRGB[idx + 2] = hostB[p];
    }
  }

  std::vector<Npp8u> expY, expU, expV;
  computeExpectedPlanar(packedRGB, expY, expU, expV);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      EXPECT_EQ(outY[y * yStep + x], expY[y * width + x]);
    }
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width / 2; ++x) {
      int idx = y * uStep + x;
      int expIdx = y * (width / 2) + x;
      EXPECT_EQ(outU[idx], expU[expIdx]);
      EXPECT_EQ(outV[idx], expV[expIdx]);
    }
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiRGBToYUV422_8u_P3R_Ctx(pSrc, rStep, pDst, dstSteps, roi, ctx);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctxY(yStep * height);
  std::vector<Npp8u> ctxU(uStep * height);
  std::vector<Npp8u> ctxV(vStep * height);
  cudaMemcpy(ctxY.data(), d_y, ctxY.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(ctxU.data(), d_u, ctxU.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(ctxV.data(), d_v, ctxV.size(), cudaMemcpyDeviceToHost);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      EXPECT_EQ(ctxY[y * yStep + x], expY[y * width + x]);
    }
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width / 2; ++x) {
      int idx = y * uStep + x;
      int expIdx = y * (width / 2) + x;
      EXPECT_EQ(ctxU[idx], expU[expIdx]);
      EXPECT_EQ(ctxV[idx], expV[expIdx]);
    }
  }

  nppiFree(d_r);
  nppiFree(d_g);
  nppiFree(d_b);
  nppiFree(d_y);
  nppiFree(d_u);
  nppiFree(d_v);
}

TEST_F(RGBToYUV422Test, RGBToYUV422_8u_C3C2R_YUYV) {
  std::vector<Npp8u> hostRGB;
  createPackedRGB(hostRGB);

  int srcStep = 0;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);
  cudaMemcpy2D(d_src, srcStep, hostRGB.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);

  int dstStep = 0;
  Npp8u *d_dst = nppiMalloc_8u_C2(width, height, &dstStep);
  ASSERT_NE(d_dst, nullptr);

  NppiSize roi = {width, height};
  NppStatus status = nppiRGBToYUV422_8u_C3C2R(d_src, srcStep, d_dst, dstStep, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> out(dstStep * height);
  cudaMemcpy(out.data(), d_dst, out.size(), cudaMemcpyDeviceToHost);

  std::vector<Npp8u> expY, expU, expV;
  computeExpectedPlanar(hostRGB, expY, expU, expV);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; x += 2) {
      int outIdx = y * dstStep + x * 2;
      int y0 = expY[y * width + x];
      int y1 = expY[y * width + x + 1];
      int u = expU[y * (width / 2) + (x / 2)];
      int v = expV[y * (width / 2) + (x / 2)];

      EXPECT_EQ(out[outIdx + 0], y0);
      EXPECT_EQ(out[outIdx + 1], u);
      EXPECT_EQ(out[outIdx + 2], y1);
      EXPECT_EQ(out[outIdx + 3], v);
    }
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiRGBToYUV422_8u_C3C2R_Ctx(d_src, srcStep, d_dst, dstStep, roi, ctx);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctxOut(dstStep * height);
  cudaMemcpy(ctxOut.data(), d_dst, ctxOut.size(), cudaMemcpyDeviceToHost);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; x += 2) {
      int outIdx = y * dstStep + x * 2;
      int y0 = expY[y * width + x];
      int y1 = expY[y * width + x + 1];
      int u = expU[y * (width / 2) + (x / 2)];
      int v = expV[y * (width / 2) + (x / 2)];

      EXPECT_EQ(ctxOut[outIdx + 0], y0);
      EXPECT_EQ(ctxOut[outIdx + 1], u);
      EXPECT_EQ(ctxOut[outIdx + 2], y1);
      EXPECT_EQ(ctxOut[outIdx + 3], v);
    }
  }

  nppiFree(d_src);
  nppiFree(d_dst);
}
