#include "npp.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

TEST(NppiConvertC4Test, Convert_8u32f_C4R_And_Ctx) {
  const int width = 13;
  const int height = 7;
  const NppiSize roi{width, height};
  std::vector<Npp8u> source(static_cast<size_t>(width) * height * 4);
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<Npp8u>((i * 17 + 9) % 256);
  }

  int srcStep = 0;
  int dstStep = 0;
  Npp8u *dSource = nppiMalloc_8u_C4(width, height, &srcStep);
  Npp32f *dDestination = nppiMalloc_32f_C4(width, height, &dstStep);
  ASSERT_NE(dSource, nullptr);
  ASSERT_NE(dDestination, nullptr);
  ASSERT_EQ(cudaMemcpy2D(dSource, srcStep, source.data(), width * 4, width * 4, height, cudaMemcpyHostToDevice),
            cudaSuccess);

  ASSERT_EQ(nppiConvert_8u32f_C4R(dSource, srcStep, dDestination, dstStep, roi), NPP_SUCCESS);
  std::vector<Npp32f> output(static_cast<size_t>(dstStep) * height / sizeof(Npp32f));
  ASSERT_EQ(cudaMemcpy(output.data(), dDestination, static_cast<size_t>(dstStep) * height, cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width * 4; ++x) {
      EXPECT_FLOAT_EQ(output[static_cast<size_t>(y) * dstStep / sizeof(Npp32f) + x],
                      static_cast<Npp32f>(source[static_cast<size_t>(y) * width * 4 + x]));
    }
  }

  NppStreamContext context{};
  ASSERT_EQ(nppGetStreamContext(&context), NPP_SUCCESS);
  EXPECT_EQ(nppiConvert_8u32f_C4R_Ctx(dSource, srcStep, dDestination, dstStep, roi, context), NPP_SUCCESS);
  EXPECT_EQ(nppiConvert_8u32f_C4R(nullptr, srcStep, dDestination, dstStep, roi), NPP_NULL_POINTER_ERROR);
  EXPECT_EQ(nppiConvert_8u32f_C4R(dSource, width * 4 - 1, dDestination, dstStep, roi), NPP_STEP_ERROR);
  EXPECT_EQ(nppiConvert_8u32f_C4R(dSource, srcStep, dDestination,
                                 static_cast<int>(width * 4 * sizeof(Npp32f)) - 1, roi),
            NPP_STEP_ERROR);

  nppiFree(dSource);
  nppiFree(dDestination);
}
