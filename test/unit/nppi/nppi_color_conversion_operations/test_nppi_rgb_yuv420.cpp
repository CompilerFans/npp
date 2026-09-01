#include "npp.h"
#include "npp_test_base.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

namespace {

class RGBToYUV420Test : public ::testing::Test {
protected:
  void SetUp() override {
    width = 4;
    height = 4;
    ASSERT_EQ(width % 2, 0);
    ASSERT_EQ(height % 2, 0);
  }

  void createPackedRGB(std::vector<Npp8u> &rgb) const {
    rgb.resize(width * height * 3);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int idx = (y * width + x) * 3;
        rgb[idx + 0] = static_cast<Npp8u>(16 + x * 40 + y * 10);
        rgb[idx + 1] = static_cast<Npp8u>(32 + x * 20 + y * 15);
        rgb[idx + 2] = static_cast<Npp8u>(64 + x * 10 + y * 20);
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
        r[idx] = static_cast<Npp8u>(16 + x * 40 + y * 10);
        g[idx] = static_cast<Npp8u>(32 + x * 20 + y * 15);
        b[idx] = static_cast<Npp8u>(64 + x * 10 + y * 20);
      }
    }
  }

  void createPackedBGRAC4(std::vector<Npp8u> &bgra) const {
    bgra.resize(width * height * 4);
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int idx = (y * width + x) * 4;
        Npp8u r = static_cast<Npp8u>(16 + x * 40 + y * 10);
        Npp8u g = static_cast<Npp8u>(32 + x * 20 + y * 15);
        Npp8u b = static_cast<Npp8u>(64 + x * 10 + y * 20);
        bgra[idx + 0] = b;
        bgra[idx + 1] = g;
        bgra[idx + 2] = r;
        bgra[idx + 3] = 123;
      }
    }
  }

  void rgbToYuvPixel(Npp8u r, Npp8u g, Npp8u b, Npp8u &y, Npp8u &u, Npp8u &v) const {
    const float R = static_cast<float>(r);
    const float G = static_cast<float>(g);
    const float B = static_cast<float>(b);

    const float Y = 0.299f * R + 0.587f * G + 0.114f * B;
    const float U = 0.492f * (B - Y) + 128.0f;
    const float V = 0.877f * (R - Y) + 128.0f;

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

  void computeYUV420ExpectedFromPacked(const std::vector<Npp8u> &rgb, std::vector<Npp8u> &yPlane,
                                       std::vector<Npp8u> &uPlane, std::vector<Npp8u> &vPlane) const {
    yPlane.resize(width * height);
    uPlane.resize((width / 2) * (height / 2));
    vPlane.resize((width / 2) * (height / 2));

    std::vector<Npp8u> uTemp(width * height);
    std::vector<Npp8u> vTemp(width * height);

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int idx = (y * width + x) * 3;
        Npp8u r = rgb[idx + 0];
        Npp8u g = rgb[idx + 1];
        Npp8u b = rgb[idx + 2];
        Npp8u yv, uv, vv;
        rgbToYuvPixel(r, g, b, yv, uv, vv);
        yPlane[y * width + x] = yv;
        uTemp[y * width + x] = uv;
        vTemp[y * width + x] = vv;
      }
    }

    for (int y = 0; y < height; y += 2) {
      for (int x = 0; x < width; x += 2) {
        int idx = (y / 2) * (width / 2) + (x / 2);
        int sumU = uTemp[y * width + x] + uTemp[y * width + x + 1] + uTemp[(y + 1) * width + x] +
                   uTemp[(y + 1) * width + x + 1];
        int sumV = vTemp[y * width + x] + vTemp[y * width + x + 1] + vTemp[(y + 1) * width + x] +
                   vTemp[(y + 1) * width + x + 1];

        uPlane[idx] = static_cast<Npp8u>(sumU / 4);
        vPlane[idx] = static_cast<Npp8u>(sumV / 4);
      }
    }
  }

  int width = 0;
  int height = 0;
};

} // namespace

TEST_F(RGBToYUV420Test, RGBToYUV420_8u_C3P3R) {
  std::vector<Npp8u> hostRGB;
  createPackedRGB(hostRGB);

  int srcStep = 0;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);

  int yStep = 0;
  int uStep = 0;
  int vStep = 0;
  Npp8u *d_y = nppiMalloc_8u_C1(width, height, &yStep);
  Npp8u *d_u = nppiMalloc_8u_C1(width / 2, height / 2, &uStep);
  Npp8u *d_v = nppiMalloc_8u_C1(width / 2, height / 2, &vStep);

  ASSERT_NE(d_y, nullptr);
  ASSERT_NE(d_u, nullptr);
  ASSERT_NE(d_v, nullptr);

  cudaMemcpy2D(d_src, srcStep, hostRGB.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice);

  Npp8u *pDst[3] = {d_y, d_u, d_v};
  int dstSteps[3] = {yStep, uStep, vStep};
  NppiSize roi = {width, height};

  NppStatus status = nppiRGBToYUV420_8u_C3P3R(d_src, srcStep, pDst, dstSteps, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> hostY(yStep * height);
  std::vector<Npp8u> hostU(uStep * (height / 2));
  std::vector<Npp8u> hostV(vStep * (height / 2));

  cudaMemcpy(hostY.data(), d_y, hostY.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostU.data(), d_u, hostU.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostV.data(), d_v, hostV.size(), cudaMemcpyDeviceToHost);

  std::vector<Npp8u> expY, expU, expV;
  computeYUV420ExpectedFromPacked(hostRGB, expY, expU, expV);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      EXPECT_EQ(hostY[y * yStep + x], expY[y * width + x]);
    }
  }

  for (int y = 0; y < height / 2; ++y) {
    for (int x = 0; x < width / 2; ++x) {
      int idx = y * uStep + x;
      int expIdx = y * (width / 2) + x;
      EXPECT_EQ(hostU[idx], expU[expIdx]);
      EXPECT_EQ(hostV[idx], expV[expIdx]);
    }
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiRGBToYUV420_8u_C3P3R_Ctx(d_src, srcStep, pDst, dstSteps, roi, ctx);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctxY(yStep * height);
  std::vector<Npp8u> ctxU(uStep * (height / 2));
  std::vector<Npp8u> ctxV(vStep * (height / 2));
  cudaMemcpy(ctxY.data(), d_y, ctxY.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(ctxU.data(), d_u, ctxU.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(ctxV.data(), d_v, ctxV.size(), cudaMemcpyDeviceToHost);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      EXPECT_EQ(ctxY[y * yStep + x], expY[y * width + x]);
    }
  }
  for (int y = 0; y < height / 2; ++y) {
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

TEST_F(RGBToYUV420Test, RGBToYUV420_8u_P3R) {
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
  Npp8u *d_u = nppiMalloc_8u_C1(width / 2, height / 2, &uStep);
  Npp8u *d_v = nppiMalloc_8u_C1(width / 2, height / 2, &vStep);

  ASSERT_NE(d_y, nullptr);
  ASSERT_NE(d_u, nullptr);
  ASSERT_NE(d_v, nullptr);

  const Npp8u *pSrc[3] = {d_r, d_g, d_b};
  Npp8u *pDst[3] = {d_y, d_u, d_v};
  int dstSteps[3] = {yStep, uStep, vStep};
  NppiSize roi = {width, height};

  NppStatus status = nppiRGBToYUV420_8u_P3R(pSrc, rStep, pDst, dstSteps, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> hostY(yStep * height);
  std::vector<Npp8u> hostU(uStep * (height / 2));
  std::vector<Npp8u> hostV(vStep * (height / 2));

  cudaMemcpy(hostY.data(), d_y, hostY.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostU.data(), d_u, hostU.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostV.data(), d_v, hostV.size(), cudaMemcpyDeviceToHost);

  std::vector<Npp8u> packedRGB;
  packedRGB.resize(width * height * 3);
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
  computeYUV420ExpectedFromPacked(packedRGB, expY, expU, expV);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      EXPECT_EQ(hostY[y * yStep + x], expY[y * width + x]);
    }
  }

  for (int y = 0; y < height / 2; ++y) {
    for (int x = 0; x < width / 2; ++x) {
      int idx = y * uStep + x;
      int expIdx = y * (width / 2) + x;
      EXPECT_EQ(hostU[idx], expU[expIdx]);
      EXPECT_EQ(hostV[idx], expV[expIdx]);
    }
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiRGBToYUV420_8u_P3R_Ctx(pSrc, rStep, pDst, dstSteps, roi, ctx);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctxY(yStep * height);
  std::vector<Npp8u> ctxU(uStep * (height / 2));
  std::vector<Npp8u> ctxV(vStep * (height / 2));
  cudaMemcpy(ctxY.data(), d_y, ctxY.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(ctxU.data(), d_u, ctxU.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(ctxV.data(), d_v, ctxV.size(), cudaMemcpyDeviceToHost);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      EXPECT_EQ(ctxY[y * yStep + x], expY[y * width + x]);
    }
  }

  for (int y = 0; y < height / 2; ++y) {
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

TEST_F(RGBToYUV420Test, BGRToYUV420_8u_AC4P3R) {
  std::vector<Npp8u> hostBGRA;
  createPackedBGRAC4(hostBGRA);

  int srcStep = 0;
  Npp8u *d_src = nppiMalloc_8u_C4(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);

  int yStep = 0;
  int uStep = 0;
  int vStep = 0;
  Npp8u *d_y = nppiMalloc_8u_C1(width, height, &yStep);
  Npp8u *d_u = nppiMalloc_8u_C1(width / 2, height / 2, &uStep);
  Npp8u *d_v = nppiMalloc_8u_C1(width / 2, height / 2, &vStep);

  ASSERT_NE(d_y, nullptr);
  ASSERT_NE(d_u, nullptr);
  ASSERT_NE(d_v, nullptr);

  cudaMemcpy2D(d_src, srcStep, hostBGRA.data(), width * 4, width * 4, height, cudaMemcpyHostToDevice);

  Npp8u *pDst[3] = {d_y, d_u, d_v};
  int dstSteps[3] = {yStep, uStep, vStep};
  NppiSize roi = {width, height};

  NppStatus status = nppiBGRToYUV420_8u_AC4P3R(d_src, srcStep, pDst, dstSteps, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> hostY(yStep * height);
  std::vector<Npp8u> hostU(uStep * (height / 2));
  std::vector<Npp8u> hostV(vStep * (height / 2));

  cudaMemcpy(hostY.data(), d_y, hostY.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostU.data(), d_u, hostU.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(hostV.data(), d_v, hostV.size(), cudaMemcpyDeviceToHost);

  std::vector<Npp8u> packedRGB;
  packedRGB.resize(width * height * 3);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 4;
      int rgbIdx = (y * width + x) * 3;
      packedRGB[rgbIdx + 0] = hostBGRA[idx + 2];
      packedRGB[rgbIdx + 1] = hostBGRA[idx + 1];
      packedRGB[rgbIdx + 2] = hostBGRA[idx + 0];
    }
  }

  std::vector<Npp8u> expY, expU, expV;
  computeYUV420ExpectedFromPacked(packedRGB, expY, expU, expV);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      EXPECT_EQ(hostY[y * yStep + x], expY[y * width + x]);
    }
  }

  for (int y = 0; y < height / 2; ++y) {
    for (int x = 0; x < width / 2; ++x) {
      int idx = y * uStep + x;
      int expIdx = y * (width / 2) + x;
      EXPECT_EQ(hostU[idx], expU[expIdx]);
      EXPECT_EQ(hostV[idx], expV[expIdx]);
    }
  }

  NppStreamContext ctx{};
  nppGetStreamContext(&ctx);
  ctx.hStream = 0;
  status = nppiBGRToYUV420_8u_AC4P3R_Ctx(d_src, srcStep, pDst, dstSteps, roi, ctx);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> ctxY(yStep * height);
  std::vector<Npp8u> ctxU(uStep * (height / 2));
  std::vector<Npp8u> ctxV(vStep * (height / 2));
  cudaMemcpy(ctxY.data(), d_y, ctxY.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(ctxU.data(), d_u, ctxU.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(ctxV.data(), d_v, ctxV.size(), cudaMemcpyDeviceToHost);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      EXPECT_EQ(ctxY[y * yStep + x], expY[y * width + x]);
    }
  }

  for (int y = 0; y < height / 2; ++y) {
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

TEST_F(RGBToYUV420Test, RGBToYUV420_8u_C3P3R_CameraSize_640x480) {
  const int camWidth = 640;
  const int camHeight = 480;
  std::vector<Npp8u> srcData(static_cast<size_t>(camWidth) * camHeight * 3);
  for (size_t i = 0; i < srcData.size(); ++i) {
    srcData[i] = static_cast<Npp8u>((i * 11 + 37) % 256);
  }

  int srcStep = 0;
  Npp8u *d_src = nppiMalloc_8u_C3(camWidth, camHeight, &srcStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_EQ(cudaMemcpy2D(d_src, srcStep, srcData.data(), camWidth * 3, camWidth * 3, camHeight, cudaMemcpyHostToDevice),
            cudaSuccess);

  int yStep = 0;
  int uStep = 0;
  int vStep = 0;
  Npp8u *d_y = nppiMalloc_8u_C1(camWidth, camHeight, &yStep);
  Npp8u *d_u = nppiMalloc_8u_C1(camWidth / 2, camHeight / 2, &uStep);
  Npp8u *d_v = nppiMalloc_8u_C1(camWidth / 2, camHeight / 2, &vStep);
  ASSERT_NE(d_y, nullptr);
  ASSERT_NE(d_u, nullptr);
  ASSERT_NE(d_v, nullptr);

  Npp8u *pDst[3] = {d_y, d_u, d_v};
  int dstSteps[3] = {yStep, uStep, vStep};
  const NppiSize roi{camWidth, camHeight};
  ASSERT_EQ(nppiRGBToYUV420_8u_C3P3R(d_src, srcStep, pDst, dstSteps, roi), NPP_NO_ERROR);

  std::vector<Npp8u> hostY(static_cast<size_t>(yStep) * camHeight);
  std::vector<Npp8u> hostU(static_cast<size_t>(uStep) * camHeight / 2);
  std::vector<Npp8u> hostV(static_cast<size_t>(vStep) * camHeight / 2);
  ASSERT_EQ(cudaMemcpy(hostY.data(), d_y, hostY.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(hostU.data(), d_u, hostU.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(hostV.data(), d_v, hostV.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  for (size_t i = 0; i < hostY.size(); ++i) {
    EXPECT_GE(hostY[i], 0) << "Y out of range at " << i;
    EXPECT_LE(hostY[i], 255) << "Y out of range at " << i;
  }

  // Ctx variant must match the default-stream result exactly.
  NppStreamContext ctx{};
  ASSERT_EQ(nppGetStreamContext(&ctx), NPP_SUCCESS);
  ASSERT_EQ(nppiRGBToYUV420_8u_C3P3R_Ctx(d_src, srcStep, pDst, dstSteps, roi, ctx), NPP_NO_ERROR);
  std::vector<Npp8u> ctxY(static_cast<size_t>(yStep) * camHeight);
  ASSERT_EQ(cudaMemcpy(ctxY.data(), d_y, ctxY.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  for (size_t i = 0; i < ctxY.size(); ++i) {
    EXPECT_EQ(ctxY[i], hostY[i]) << "Ctx Y mismatch at " << i;
  }

  nppiFree(d_src);
  nppiFree(d_y);
  nppiFree(d_u);
  nppiFree(d_v);
}

TEST_F(RGBToYUV420Test, RGBToYUV420_8u_C3P3R_PartialEvenROI) {
  const int width = 128;
  const int height = 128;
  const int roiX = 32;
  const int roiY = 16;
  const int roiW = 64;
  const int roiH = 48;
  std::vector<Npp8u> srcData(static_cast<size_t>(width) * height * 3);
  for (size_t i = 0; i < srcData.size(); ++i) {
    srcData[i] = static_cast<Npp8u>((i * 17 + 61) % 256);
  }

  int srcStep = 0;
  Npp8u *d_src = nppiMalloc_8u_C3(width, height, &srcStep);
  ASSERT_NE(d_src, nullptr);
  ASSERT_EQ(cudaMemcpy2D(d_src, srcStep, srcData.data(), width * 3, width * 3, height, cudaMemcpyHostToDevice),
            cudaSuccess);

  int yStep = 0;
  int uStep = 0;
  int vStep = 0;
  Npp8u *d_y = nppiMalloc_8u_C1(width, height, &yStep);
  Npp8u *d_u = nppiMalloc_8u_C1(width / 2, height / 2, &uStep);
  Npp8u *d_v = nppiMalloc_8u_C1(width / 2, height / 2, &vStep);
  Npp8u *d_roiY = nppiMalloc_8u_C1(roiW, roiH, &yStep);
  Npp8u *d_roiU = nppiMalloc_8u_C1(roiW / 2, roiH / 2, &uStep);
  Npp8u *d_roiV = nppiMalloc_8u_C1(roiW / 2, roiH / 2, &vStep);
  ASSERT_NE(d_y, nullptr);
  ASSERT_NE(d_u, nullptr);
  ASSERT_NE(d_v, nullptr);
  ASSERT_NE(d_roiY, nullptr);
  ASSERT_NE(d_roiU, nullptr);
  ASSERT_NE(d_roiV, nullptr);

  // Reference: full-image conversion, then take the same ROI window.
  Npp8u *pDst[3] = {d_y, d_u, d_v};
  int dstSteps[3] = {yStep, uStep, vStep};
  const NppiSize fullRoi{width, height};
  ASSERT_EQ(nppiRGBToYUV420_8u_C3P3R(d_src, srcStep, pDst, dstSteps, fullRoi), NPP_NO_ERROR);

  Npp8u *d_roiSrc = d_src + static_cast<size_t>(roiY) * srcStep + roiX * 3;
  Npp8u *roiDst[3] = {d_roiY, d_roiU, d_roiV};
  const NppiSize roi{roiW, roiH};
  ASSERT_EQ(nppiRGBToYUV420_8u_C3P3R(d_roiSrc, srcStep, roiDst, dstSteps, roi), NPP_NO_ERROR);

  std::vector<Npp8u> hostY(static_cast<size_t>(yStep) * height);
  std::vector<Npp8u> roiYv(static_cast<size_t>(yStep) * roiH);
  ASSERT_EQ(cudaMemcpy(hostY.data(), d_y, hostY.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(roiYv.data(), d_roiY, roiYv.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  for (int i = 0; i < roiW * roiH; ++i) {
    const int fullRow = roiY + i / roiW;
    const int fullCol = roiX + i % roiW;
    EXPECT_EQ(roiYv[static_cast<size_t>(i / roiW) * yStep + i % roiW],
              hostY[static_cast<size_t>(fullRow) * yStep + fullCol])
        << "Y ROI mismatch at " << i;
  }

  nppiFree(d_src);
  nppiFree(d_y);
  nppiFree(d_u);
  nppiFree(d_v);
  nppiFree(d_roiY);
  nppiFree(d_roiU);
  nppiFree(d_roiV);
}
