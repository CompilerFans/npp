#include "npp.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

template <typename T> struct ExpectedOutputs;

template <> struct ExpectedOutputs<Npp8u> {
  static constexpr int kSize = 16;
  static const Npp8u kL1[kSize];
  static const Npp8u kL2[kSize];
  static const Npp8u kInf[kSize];
};

template <> struct ExpectedOutputs<Npp16u> {
  static constexpr int kSize = 16;
  static const Npp16u kL1[kSize];
  static const Npp16u kL2[kSize];
  static const Npp16u kInf[kSize];
};

template <> struct ExpectedOutputs<Npp16s> {
  static constexpr int kSize = 16;
  static const Npp16s kL1[kSize];
  static const Npp16s kL2[kSize];
  static const Npp16s kInf[kSize];
};

template <> struct ExpectedOutputs<Npp32f> {
  static constexpr int kSize = 16;
  static const Npp32f kL1[kSize];
  static const Npp32f kL2[kSize];
  static const Npp32f kInf[kSize];
};

template <typename T> struct NppTypeTraits {};
template <> struct NppTypeTraits<Npp8u> {
  using DstType = Npp8u;
};
template <> struct NppTypeTraits<Npp16u> {
  using DstType = Npp16u;
};
template <> struct NppTypeTraits<Npp16s> {
  using DstType = Npp16s;
};
template <> struct NppTypeTraits<Npp32f> {
  using DstType = Npp32f;
};

template <typename T>
void fillInput(std::vector<T> &src, int width, int height) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 3;
      float r = x * 10.0f + y * 3.0f;
      float g = x * 5.0f + y * 7.0f;
      float b = x * 2.0f + y * 11.0f;
      if constexpr (std::is_same_v<T, Npp16s>) {
        r -= 20.0f;
        g -= 10.0f;
        b -= 5.0f;
      }
      src[idx + 0] = static_cast<T>(r);
      src[idx + 1] = static_cast<T>(g);
      src[idx + 2] = static_cast<T>(b);
    }
  }
}

template <typename T>
std::vector<typename NppTypeTraits<T>::DstType> runGradientToGray(NppiNorm norm) {
  constexpr int width = 4;
  constexpr int height = 4;
  NppiSize roi{width, height};

  std::vector<T> hostSrc(width * height * 3);
  fillInput(hostSrc, width, height);

  int srcStep = 0;
  int dstStep = 0;
  T *d_src = nullptr;
  typename NppTypeTraits<T>::DstType *d_dst = nullptr;

  if constexpr (std::is_same_v<T, Npp8u>) {
    d_src = reinterpret_cast<T *>(nppiMalloc_8u_C3(width, height, &srcStep));
    d_dst = reinterpret_cast<typename NppTypeTraits<T>::DstType *>(nppiMalloc_8u_C1(width, height, &dstStep));
  } else if constexpr (std::is_same_v<T, Npp16u>) {
    d_src = reinterpret_cast<T *>(nppiMalloc_16u_C3(width, height, &srcStep));
    d_dst = reinterpret_cast<typename NppTypeTraits<T>::DstType *>(nppiMalloc_16u_C1(width, height, &dstStep));
  } else if constexpr (std::is_same_v<T, Npp32f>) {
    d_src = reinterpret_cast<T *>(nppiMalloc_32f_C3(width, height, &srcStep));
    d_dst = reinterpret_cast<typename NppTypeTraits<T>::DstType *>(nppiMalloc_32f_C1(width, height, &dstStep));
  } else {
    size_t srcStepBytes = 0;
    size_t dstStepBytes = 0;
    cudaMallocPitch(reinterpret_cast<void **>(&d_src), &srcStepBytes, width * sizeof(T) * 3, height);
    cudaMallocPitch(reinterpret_cast<void **>(&d_dst), &dstStepBytes, width * sizeof(T), height);
    srcStep = static_cast<int>(srcStepBytes);
    dstStep = static_cast<int>(dstStepBytes);
  }

  cudaMemcpy2D(d_src, srcStep, hostSrc.data(), width * sizeof(T) * 3, width * sizeof(T) * 3, height,
               cudaMemcpyHostToDevice);

  NppStatus status = NPP_SUCCESS;
  if constexpr (std::is_same_v<T, Npp8u>) {
    status = nppiGradientColorToGray_8u_C3C1R(d_src, srcStep, d_dst, dstStep, roi, norm);
  } else if constexpr (std::is_same_v<T, Npp16u>) {
    status = nppiGradientColorToGray_16u_C3C1R(d_src, srcStep, d_dst, dstStep, roi, norm);
  } else if constexpr (std::is_same_v<T, Npp16s>) {
    status = nppiGradientColorToGray_16s_C3C1R(reinterpret_cast<const Npp16s *>(d_src), srcStep,
                                               reinterpret_cast<Npp16s *>(d_dst), dstStep, roi, norm);
  } else {
    status = nppiGradientColorToGray_32f_C3C1R(reinterpret_cast<const Npp32f *>(d_src), srcStep,
                                               reinterpret_cast<Npp32f *>(d_dst), dstStep, roi, norm);
  }
  EXPECT_EQ(status, NPP_SUCCESS);

  std::vector<typename NppTypeTraits<T>::DstType> hostDst(width * height);
  cudaMemcpy2D(hostDst.data(), width * sizeof(typename NppTypeTraits<T>::DstType), d_dst, dstStep,
               width * sizeof(typename NppTypeTraits<T>::DstType), height, cudaMemcpyDeviceToHost);

  if constexpr (std::is_same_v<T, Npp8u>) {
    nppiFree(d_src);
    nppiFree(d_dst);
  } else if constexpr (std::is_same_v<T, Npp16u>) {
    nppiFree(d_src);
    nppiFree(d_dst);
  } else if constexpr (std::is_same_v<T, Npp32f>) {
    nppiFree(d_src);
    nppiFree(d_dst);
  } else {
    cudaFree(d_src);
    cudaFree(d_dst);
  }

  return hostDst;
}

template <typename T>
void dumpOutputs(const char *label, const std::vector<typename NppTypeTraits<T>::DstType> &vals) {
  std::cout << label << " = {";
  for (size_t i = 0; i < vals.size(); ++i) {
    if (i) {
      std::cout << ", ";
    }
    if constexpr (std::is_same_v<T, Npp32f>) {
      std::cout << std::fixed << std::setprecision(6) << vals[i];
    } else {
      std::cout << static_cast<long long>(vals[i]);
    }
  }
  std::cout << "};\n";
}

template <typename T>
void checkExpected(const std::vector<typename NppTypeTraits<T>::DstType> &vals,
                   const typename NppTypeTraits<T>::DstType *expected) {
  ASSERT_EQ(vals.size(), static_cast<size_t>(ExpectedOutputs<T>::kSize));
  for (size_t i = 0; i < vals.size(); ++i) {
    if constexpr (std::is_same_v<T, Npp32f>) {
      EXPECT_NEAR(vals[i], expected[i], 1e-5f) << "Mismatch at " << i;
    } else {
      EXPECT_EQ(vals[i], expected[i]) << "Mismatch at " << i;
    }
  }
}

} // namespace

TEST(GradientColorToGrayTest, GradientColorToGray_8u_C3C1R_AllNorms) {
  auto l1 = runGradientToGray<Npp8u>(nppiNormL1);
  auto l2 = runGradientToGray<Npp8u>(nppiNormL2);
  auto linf = runGradientToGray<Npp8u>(nppiNormInf);

  if (std::getenv("NPP_DUMP_EXPECTED")) {
    dumpOutputs<Npp8u>("kExpected8uL1", l1);
    dumpOutputs<Npp8u>("kExpected8uL2", l2);
    dumpOutputs<Npp8u>("kExpected8uInf", linf);
    GTEST_SKIP();
  }

  checkExpected<Npp8u>(l1, ExpectedOutputs<Npp8u>::kL1);
  checkExpected<Npp8u>(l2, ExpectedOutputs<Npp8u>::kL2);
  checkExpected<Npp8u>(linf, ExpectedOutputs<Npp8u>::kInf);
}

TEST(GradientColorToGrayTest, GradientColorToGray_16u_C3C1R_AllNorms) {
  auto l1 = runGradientToGray<Npp16u>(nppiNormL1);
  auto l2 = runGradientToGray<Npp16u>(nppiNormL2);
  auto linf = runGradientToGray<Npp16u>(nppiNormInf);

  if (std::getenv("NPP_DUMP_EXPECTED")) {
    dumpOutputs<Npp16u>("kExpected16uL1", l1);
    dumpOutputs<Npp16u>("kExpected16uL2", l2);
    dumpOutputs<Npp16u>("kExpected16uInf", linf);
    GTEST_SKIP();
  }

  checkExpected<Npp16u>(l1, ExpectedOutputs<Npp16u>::kL1);
  checkExpected<Npp16u>(l2, ExpectedOutputs<Npp16u>::kL2);
  checkExpected<Npp16u>(linf, ExpectedOutputs<Npp16u>::kInf);
}

TEST(GradientColorToGrayTest, GradientColorToGray_16s_C3C1R_AllNorms) {
  auto l1 = runGradientToGray<Npp16s>(nppiNormL1);
  auto l2 = runGradientToGray<Npp16s>(nppiNormL2);
  auto linf = runGradientToGray<Npp16s>(nppiNormInf);

  if (std::getenv("NPP_DUMP_EXPECTED")) {
    dumpOutputs<Npp16s>("kExpected16sL1", l1);
    dumpOutputs<Npp16s>("kExpected16sL2", l2);
    dumpOutputs<Npp16s>("kExpected16sInf", linf);
    GTEST_SKIP();
  }

  checkExpected<Npp16s>(l1, ExpectedOutputs<Npp16s>::kL1);
  checkExpected<Npp16s>(l2, ExpectedOutputs<Npp16s>::kL2);
  checkExpected<Npp16s>(linf, ExpectedOutputs<Npp16s>::kInf);
}

TEST(GradientColorToGrayTest, GradientColorToGray_32f_C3C1R_AllNorms) {
  auto l1 = runGradientToGray<Npp32f>(nppiNormL1);
  auto l2 = runGradientToGray<Npp32f>(nppiNormL2);
  auto linf = runGradientToGray<Npp32f>(nppiNormInf);

  if (std::getenv("NPP_DUMP_EXPECTED")) {
    dumpOutputs<Npp32f>("kExpected32fL1", l1);
    dumpOutputs<Npp32f>("kExpected32fL2", l2);
    dumpOutputs<Npp32f>("kExpected32fInf", linf);
    GTEST_SKIP();
  }

  checkExpected<Npp32f>(l1, ExpectedOutputs<Npp32f>::kL1);
  checkExpected<Npp32f>(l2, ExpectedOutputs<Npp32f>::kL2);
  checkExpected<Npp32f>(linf, ExpectedOutputs<Npp32f>::kInf);
}

const Npp8u ExpectedOutputs<Npp8u>::kL1[kSize] = {0, 5, 11, 17, 7, 12, 18, 24, 14, 19, 25, 31, 21, 26, 32, 38};
const Npp8u ExpectedOutputs<Npp8u>::kL2[kSize] = {0, 6, 13, 19, 7, 12, 18, 24, 15, 19, 25, 31, 23, 27, 32, 38};
const Npp8u ExpectedOutputs<Npp8u>::kInf[kSize] = {0, 10, 20, 30, 11, 13, 23, 33, 22, 24, 26, 36, 33, 35, 37, 39};
const Npp16u ExpectedOutputs<Npp16u>::kL1[kSize] = {0, 5, 11, 17, 7, 12, 18, 24, 14, 19, 25, 31, 21, 26, 32, 38};
const Npp16u ExpectedOutputs<Npp16u>::kL2[kSize] = {0, 6, 13, 19, 7, 12, 18, 24, 15, 19, 25, 31, 23, 27, 32, 38};
const Npp16u ExpectedOutputs<Npp16u>::kInf[kSize] = {0, 10, 20, 30, 11, 13, 23, 33, 22, 24, 26, 36, 33, 35, 37, 39};
const Npp16s ExpectedOutputs<Npp16s>::kL1[kSize] = {-11, -6, 0, 5, -4, 1, 6, 12, 2, 8, 13, 19, 9, 15, 20, 26};
const Npp16s ExpectedOutputs<Npp16s>::kL2[kSize] = {13, 6, 0, 6, 10, 6, 7, 12, 12, 12, 14, 19, 18, 19, 22, 27};
const Npp16s ExpectedOutputs<Npp16s>::kInf[kSize] = {-5, -3, 0, 10, 6, 8, 10, 13, 17, 19, 21, 23, 28, 30, 32, 34};
const Npp32f ExpectedOutputs<Npp32f>::kL1[kSize] = {0.0f, 17.0f, 34.0f, 51.0f, 21.0f, 38.0f, 55.0f, 72.0f,
                                                    42.0f, 59.0f, 76.0f, 93.0f, 63.0f, 80.0f, 97.0f, 114.0f};
const Npp32f ExpectedOutputs<Npp32f>::kL2[kSize] = {0.0f, 11.357817f, 22.715633f, 34.073448f, 13.379088f, 21.954498f,
                                                    32.295509f, 43.150898f, 26.758177f, 34.539833f, 43.908997f,
                                                    54.046276f, 40.137264f, 47.560490f, 56.311634f, 65.863495f};
const Npp32f ExpectedOutputs<Npp32f>::kInf[kSize] = {0.0f, 10.0f, 20.0f, 30.0f, 11.0f, 13.0f, 23.0f, 33.0f,
                                                     22.0f, 24.0f, 26.0f, 36.0f, 33.0f, 35.0f, 37.0f, 39.0f};
