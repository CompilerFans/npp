#include "npp.h"
#include "npp_test_base.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

namespace {

class NV12ToYUV420Test : public ::testing::Test {
protected:
  void SetUp() override {
    width = 8;
    height = 4;
    ASSERT_EQ(width % 2, 0);
    ASSERT_EQ(height % 2, 0);
  }

  void createNV12(std::vector<Npp8u> &yPlane, std::vector<Npp8u> &uvPlane) const {
    yPlane.resize(width * height);
    uvPlane.resize(width * height / 2);

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        yPlane[y * width + x] = static_cast<Npp8u>(16 + x + y * 10);
      }
    }

    for (int y = 0; y < height / 2; ++y) {
      for (int x = 0; x < width; x += 2) {
        int idx = y * width + x;
        uvPlane[idx] = static_cast<Npp8u>(64 + x + y * 3);
        uvPlane[idx + 1] = static_cast<Npp8u>(96 + x + y * 5);
      }
    }
  }

  void computeExpected(const std::vector<Npp8u> &yPlane, const std::vector<Npp8u> &uvPlane,
                       std::vector<Npp8u> &expY, std::vector<Npp8u> &expU, std::vector<Npp8u> &expV) const {
    expY = yPlane;
    expU.resize((width / 2) * (height / 2));
    expV.resize((width / 2) * (height / 2));

    for (int y = 0; y < height / 2; ++y) {
      for (int x = 0; x < width; x += 2) {
        int uvIdx = y * width + x;
        int dstIdx = y * (width / 2) + (x / 2);
        expU[dstIdx] = uvPlane[uvIdx];
        expV[dstIdx] = uvPlane[uvIdx + 1];
      }
    }
  }

  int width = 0;
  int height = 0;
};

} // namespace

TEST_F(NV12ToYUV420Test, NV12ToYUV420_8u_P2P3R) {
  std::vector<Npp8u> hostY, hostUV;
  createNV12(hostY, hostUV);

  int srcStep = width;
  Npp8u *d_srcY = nppsMalloc_8u(width * height);
  Npp8u *d_srcUV = nppsMalloc_8u(width * height / 2);
  ASSERT_NE(d_srcY, nullptr);
  ASSERT_NE(d_srcUV, nullptr);

  cudaMemcpy(d_srcY, hostY.data(), hostY.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_srcUV, hostUV.data(), hostUV.size(), cudaMemcpyHostToDevice);

  int yStep = 0;
  int uStep = 0;
  int vStep = 0;
  Npp8u *d_dstY = nppiMalloc_8u_C1(width, height, &yStep);
  Npp8u *d_dstU = nppiMalloc_8u_C1(width / 2, height / 2, &uStep);
  Npp8u *d_dstV = nppiMalloc_8u_C1(width / 2, height / 2, &vStep);
  ASSERT_NE(d_dstY, nullptr);
  ASSERT_NE(d_dstU, nullptr);
  ASSERT_NE(d_dstV, nullptr);

  const Npp8u *pSrc[2] = {d_srcY, d_srcUV};
  Npp8u *pDst[3] = {d_dstY, d_dstU, d_dstV};
  int dstSteps[3] = {yStep, uStep, vStep};
  NppiSize roi = {width, height};

  NppStatus status = nppiNV12ToYUV420_8u_P2P3R(pSrc, srcStep, pDst, dstSteps, roi);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> outY(yStep * height);
  std::vector<Npp8u> outU(uStep * (height / 2));
  std::vector<Npp8u> outV(vStep * (height / 2));
  cudaMemcpy(outY.data(), d_dstY, outY.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(outU.data(), d_dstU, outU.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(outV.data(), d_dstV, outV.size(), cudaMemcpyDeviceToHost);

  std::vector<Npp8u> expY, expU, expV;
  computeExpected(hostY, hostUV, expY, expU, expV);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      EXPECT_EQ(outY[y * yStep + x], expY[y * width + x]);
    }
  }

  for (int y = 0; y < height / 2; ++y) {
    for (int x = 0; x < width / 2; ++x) {
      int idx = y * uStep + x;
      int expIdx = y * (width / 2) + x;
      EXPECT_EQ(outU[idx], expU[expIdx]);
      EXPECT_EQ(outV[idx], expV[expIdx]);
    }
  }

  nppsFree(d_srcY);
  nppsFree(d_srcUV);
  nppiFree(d_dstY);
  nppiFree(d_dstU);
  nppiFree(d_dstV);
}

TEST_F(NV12ToYUV420Test, NV12ToYUV420_8u_P2P3R_Ctx) {
  std::vector<Npp8u> hostY, hostUV;
  createNV12(hostY, hostUV);

  int srcStep = width;
  Npp8u *d_srcY = nppsMalloc_8u(width * height);
  Npp8u *d_srcUV = nppsMalloc_8u(width * height / 2);
  ASSERT_NE(d_srcY, nullptr);
  ASSERT_NE(d_srcUV, nullptr);

  cudaMemcpy(d_srcY, hostY.data(), hostY.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_srcUV, hostUV.data(), hostUV.size(), cudaMemcpyHostToDevice);

  int yStep = 0;
  int uStep = 0;
  int vStep = 0;
  Npp8u *d_dstY = nppiMalloc_8u_C1(width, height, &yStep);
  Npp8u *d_dstU = nppiMalloc_8u_C1(width / 2, height / 2, &uStep);
  Npp8u *d_dstV = nppiMalloc_8u_C1(width / 2, height / 2, &vStep);
  ASSERT_NE(d_dstY, nullptr);
  ASSERT_NE(d_dstU, nullptr);
  ASSERT_NE(d_dstV, nullptr);

  const Npp8u *pSrc[2] = {d_srcY, d_srcUV};
  Npp8u *pDst[3] = {d_dstY, d_dstU, d_dstV};
  int dstSteps[3] = {yStep, uStep, vStep};
  NppiSize roi = {width, height};

  NppStreamContext ctx;
  ctx.hStream = 0;

  NppStatus status = nppiNV12ToYUV420_8u_P2P3R_Ctx(pSrc, srcStep, pDst, dstSteps, roi, ctx);
  EXPECT_EQ(status, NPP_NO_ERROR);

  std::vector<Npp8u> outY(yStep * height);
  std::vector<Npp8u> outU(uStep * (height / 2));
  std::vector<Npp8u> outV(vStep * (height / 2));
  cudaMemcpy(outY.data(), d_dstY, outY.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(outU.data(), d_dstU, outU.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(outV.data(), d_dstV, outV.size(), cudaMemcpyDeviceToHost);

  std::vector<Npp8u> expY, expU, expV;
  computeExpected(hostY, hostUV, expY, expU, expV);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      EXPECT_EQ(outY[y * yStep + x], expY[y * width + x]);
    }
  }

  for (int y = 0; y < height / 2; ++y) {
    for (int x = 0; x < width / 2; ++x) {
      int idx = y * uStep + x;
      int expIdx = y * (width / 2) + x;
      EXPECT_EQ(outU[idx], expU[expIdx]);
      EXPECT_EQ(outV[idx], expV[expIdx]);
    }
  }

  nppsFree(d_srcY);
  nppsFree(d_srcUV);
  nppiFree(d_dstY);
  nppiFree(d_dstU);
  nppiFree(d_dstV);
}
