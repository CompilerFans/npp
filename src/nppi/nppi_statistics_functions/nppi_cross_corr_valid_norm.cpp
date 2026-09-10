#include "npp.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>

extern "C" {
cudaError_t mppCrossCorrValidNorm8u8uLaunch(const Npp8u *, int, NppiSize, const Npp8u *, int, NppiSize, Npp8u *,
                                            int, int, bool, int, cudaStream_t);
cudaError_t mppCrossCorrValidNorm8u32fLaunch(const Npp8u *, int, NppiSize, const Npp8u *, int, NppiSize, Npp32f *,
                                             int, int, bool, int, cudaStream_t);
cudaError_t mppCrossCorrValidNorm8s32fLaunch(const Npp8s *, int, NppiSize, const Npp8s *, int, NppiSize, Npp32f *,
                                             int, int, bool, int, cudaStream_t);
cudaError_t mppCrossCorrValidNorm16u32fLaunch(const Npp16u *, int, NppiSize, const Npp16u *, int, NppiSize, Npp32f *,
                                              int, int, bool, int, cudaStream_t);
cudaError_t mppCrossCorrValidNorm32f32fLaunch(const Npp32f *, int, NppiSize, const Npp32f *, int, NppiSize, Npp32f *,
                                              int, int, bool, int, cudaStream_t);
cudaError_t mppCrossCorrValidNorm64f64fLaunch(const Npp64f *, int, NppiSize, const Npp64f *, int, NppiSize, Npp64f *,
                                              int, int, bool, int, cudaStream_t);
}

namespace {

template <typename Source, typename Destination> struct KernelLauncher;

template <> struct KernelLauncher<Npp8u, Npp8u> {
  static cudaError_t launch(const Npp8u *source, int sourceStep, NppiSize sourceSize, const Npp8u *tpl,
                            int templateStep, NppiSize templateSize, Npp8u *destination, int destinationStep,
                            int channels, bool preserveAlpha, int scaleFactor, cudaStream_t stream) {
    return mppCrossCorrValidNorm8u8uLaunch(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                           destination, destinationStep, channels, preserveAlpha, scaleFactor, stream);
  }
};

template <> struct KernelLauncher<Npp8u, Npp32f> {
  static cudaError_t launch(const Npp8u *source, int sourceStep, NppiSize sourceSize, const Npp8u *tpl,
                            int templateStep, NppiSize templateSize, Npp32f *destination, int destinationStep,
                            int channels, bool preserveAlpha, int scaleFactor, cudaStream_t stream) {
    return mppCrossCorrValidNorm8u32fLaunch(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                            destination, destinationStep, channels, preserveAlpha, scaleFactor, stream);
  }
};

template <> struct KernelLauncher<Npp8s, Npp32f> {
  static cudaError_t launch(const Npp8s *source, int sourceStep, NppiSize sourceSize, const Npp8s *tpl,
                            int templateStep, NppiSize templateSize, Npp32f *destination, int destinationStep,
                            int channels, bool preserveAlpha, int scaleFactor, cudaStream_t stream) {
    return mppCrossCorrValidNorm8s32fLaunch(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                            destination, destinationStep, channels, preserveAlpha, scaleFactor, stream);
  }
};

template <> struct KernelLauncher<Npp16u, Npp32f> {
  static cudaError_t launch(const Npp16u *source, int sourceStep, NppiSize sourceSize, const Npp16u *tpl,
                            int templateStep, NppiSize templateSize, Npp32f *destination, int destinationStep,
                            int channels, bool preserveAlpha, int scaleFactor, cudaStream_t stream) {
    return mppCrossCorrValidNorm16u32fLaunch(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                             destination, destinationStep, channels, preserveAlpha, scaleFactor,
                                             stream);
  }
};

template <> struct KernelLauncher<Npp32f, Npp32f> {
  static cudaError_t launch(const Npp32f *source, int sourceStep, NppiSize sourceSize, const Npp32f *tpl,
                            int templateStep, NppiSize templateSize, Npp32f *destination, int destinationStep,
                            int channels, bool preserveAlpha, int scaleFactor, cudaStream_t stream) {
    return mppCrossCorrValidNorm32f32fLaunch(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                             destination, destinationStep, channels, preserveAlpha, scaleFactor,
                                             stream);
  }
};

template <> struct KernelLauncher<Npp64f, Npp64f> {
  static cudaError_t launch(const Npp64f *source, int sourceStep, NppiSize sourceSize, const Npp64f *tpl,
                            int templateStep, NppiSize templateSize, Npp64f *destination, int destinationStep,
                            int channels, bool preserveAlpha, int scaleFactor, cudaStream_t stream) {
    return mppCrossCorrValidNorm64f64fLaunch(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                             destination, destinationStep, channels, preserveAlpha, scaleFactor,
                                             stream);
  }
};

template <typename T> bool stepIsTooSmall(int step, int width, int channels) {
  if (step <= 0) {
    return true;
  }
  const long long required = static_cast<long long>(width) * channels * sizeof(T);
  return required > std::numeric_limits<int>::max() || step < required;
}

template <typename Source, typename Destination>
NppStatus crossCorrValidNorm(const Source *source, int sourceStep, NppiSize sourceSize, const Source *tpl,
                             int templateStep, NppiSize templateSize, Destination *destination, int destinationStep,
                             int channels, bool preserveAlpha, int scaleFactor, NppStreamContext context) {
  if (!source || !tpl || !destination) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (sourceSize.width <= 0 || sourceSize.height <= 0 || templateSize.width <= 0 || templateSize.height <= 0 ||
      templateSize.width > sourceSize.width || templateSize.height > sourceSize.height) {
    return NPP_SIZE_ERROR;
  }
  const int destinationWidth = sourceSize.width - templateSize.width + 1;
  if (stepIsTooSmall<Source>(sourceStep, sourceSize.width, channels) ||
      stepIsTooSmall<Source>(templateStep, templateSize.width, channels) ||
      stepIsTooSmall<Destination>(destinationStep, destinationWidth, channels)) {
    return NPP_STEP_ERROR;
  }

  const cudaError_t error = KernelLauncher<Source, Destination>::launch(
      source, sourceStep, sourceSize, tpl, templateStep, templateSize, destination, destinationStep, channels,
      preserveAlpha, scaleFactor, context.hStream);
  return error == cudaSuccess ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStreamContext defaultStreamContext() {
  NppStreamContext context{};
  nppGetStreamContext(&context);
  return context;
}

} // namespace

NppStatus nppiCrossCorrValid_Norm_8u_C1RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                                 const Npp8u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp8u *pDst,
                                                 int nDstStep, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 1, false,
                            nScaleFactor, nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_8u_C1RSfs(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                             const Npp8u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp8u *pDst,
                                             int nDstStep, int nScaleFactor) {
  return nppiCrossCorrValid_Norm_8u_C1RSfs_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                               nDstStep, nScaleFactor, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_8u_C3RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                                 const Npp8u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp8u *pDst,
                                                 int nDstStep, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 3, false,
                            nScaleFactor, nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_8u_C3RSfs(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                             const Npp8u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp8u *pDst,
                                             int nDstStep, int nScaleFactor) {
  return nppiCrossCorrValid_Norm_8u_C3RSfs_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                               nDstStep, nScaleFactor, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_8u_C4RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                                 const Npp8u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp8u *pDst,
                                                 int nDstStep, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 4, false,
                            nScaleFactor, nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_8u_C4RSfs(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                             const Npp8u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp8u *pDst,
                                             int nDstStep, int nScaleFactor) {
  return nppiCrossCorrValid_Norm_8u_C4RSfs_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                               nDstStep, nScaleFactor, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_8u_AC4RSfs_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                                  const Npp8u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp8u *pDst,
                                                  int nDstStep, int nScaleFactor, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 4, true,
                            nScaleFactor, nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_8u_AC4RSfs(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                              const Npp8u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp8u *pDst,
                                              int nDstStep, int nScaleFactor) {
  return nppiCrossCorrValid_Norm_8u_AC4RSfs_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                                nDstStep, nScaleFactor, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_8u32f_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                                const Npp8u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                                int nDstStep, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 1, false, 0,
                            nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_8u32f_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                            const Npp8u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                            int nDstStep) {
  return nppiCrossCorrValid_Norm_8u32f_C1R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                               nDstStep, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_8u32f_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                                const Npp8u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                                int nDstStep, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 3, false, 0,
                            nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_8u32f_C3R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                            const Npp8u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                            int nDstStep) {
  return nppiCrossCorrValid_Norm_8u32f_C3R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                               nDstStep, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_8u32f_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                                const Npp8u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                                int nDstStep, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 4, false, 0,
                            nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_8u32f_C4R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                            const Npp8u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                            int nDstStep) {
  return nppiCrossCorrValid_Norm_8u32f_C4R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                               nDstStep, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_8u32f_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                                 const Npp8u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                                 int nDstStep, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 4, true, 0,
                            nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_8u32f_AC4R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                             const Npp8u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                             int nDstStep) {
  return nppiCrossCorrValid_Norm_8u32f_AC4R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                                nDstStep, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_8s32f_C1R_Ctx(const Npp8s *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                                const Npp8s *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                                int nDstStep, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 1, false, 0,
                            nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_8s32f_C1R(const Npp8s *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                            const Npp8s *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                            int nDstStep) {
  return nppiCrossCorrValid_Norm_8s32f_C1R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                               nDstStep, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_8s32f_C3R_Ctx(const Npp8s *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                                const Npp8s *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                                int nDstStep, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 3, false, 0,
                            nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_8s32f_C3R(const Npp8s *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                            const Npp8s *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                            int nDstStep) {
  return nppiCrossCorrValid_Norm_8s32f_C3R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                               nDstStep, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_8s32f_C4R_Ctx(const Npp8s *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                                const Npp8s *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                                int nDstStep, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 4, false, 0,
                            nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_8s32f_C4R(const Npp8s *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                            const Npp8s *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                            int nDstStep) {
  return nppiCrossCorrValid_Norm_8s32f_C4R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                               nDstStep, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_8s32f_AC4R_Ctx(const Npp8s *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                                 const Npp8s *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                                 int nDstStep, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 4, true, 0,
                            nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_8s32f_AC4R(const Npp8s *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                             const Npp8s *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                             int nDstStep) {
  return nppiCrossCorrValid_Norm_8s32f_AC4R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                                nDstStep, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_16u32f_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                                 const Npp16u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                                 int nDstStep, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 1, false, 0,
                            nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_16u32f_C1R(const Npp16u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                             const Npp16u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                             int nDstStep) {
  return nppiCrossCorrValid_Norm_16u32f_C1R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                                nDstStep, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_16u32f_C3R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                                 const Npp16u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                                 int nDstStep, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 3, false, 0,
                            nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_16u32f_C3R(const Npp16u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                             const Npp16u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                             int nDstStep) {
  return nppiCrossCorrValid_Norm_16u32f_C3R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                                nDstStep, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_16u32f_C4R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                                 const Npp16u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                                 int nDstStep, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 4, false, 0,
                            nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_16u32f_C4R(const Npp16u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                             const Npp16u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                             int nDstStep) {
  return nppiCrossCorrValid_Norm_16u32f_C4R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                                nDstStep, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_16u32f_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                                  const Npp16u *pTpl, int nTplStep, NppiSize oTplRoiSize,
                                                  Npp32f *pDst, int nDstStep, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 4, true, 0,
                            nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_16u32f_AC4R(const Npp16u *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                              const Npp16u *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                              int nDstStep) {
  return nppiCrossCorrValid_Norm_16u32f_AC4R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                                 nDstStep, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                              const Npp32f *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                              int nDstStep, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 1, false, 0,
                            nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_32f_C1R(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                          const Npp32f *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                          int nDstStep) {
  return nppiCrossCorrValid_Norm_32f_C1R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                             nDstStep, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                              const Npp32f *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                              int nDstStep, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 3, false, 0,
                            nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_32f_C3R(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                          const Npp32f *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                          int nDstStep) {
  return nppiCrossCorrValid_Norm_32f_C3R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                             nDstStep, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                              const Npp32f *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                              int nDstStep, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 4, false, 0,
                            nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_32f_C4R(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                          const Npp32f *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                          int nDstStep) {
  return nppiCrossCorrValid_Norm_32f_C4R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                             nDstStep, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                               const Npp32f *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                               int nDstStep, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 4, true, 0,
                            nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_32f_AC4R(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                           const Npp32f *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp32f *pDst,
                                           int nDstStep) {
  return nppiCrossCorrValid_Norm_32f_AC4R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                              nDstStep, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_64f_C1R_Ctx(const Npp64f *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                              const Npp64f *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp64f *pDst,
                                              int nDstStep, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 1, false, 0,
                            nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_64f_C1R(const Npp64f *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                          const Npp64f *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp64f *pDst,
                                          int nDstStep) {
  return nppiCrossCorrValid_Norm_64f_C1R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                             nDstStep, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_64f_C3R_Ctx(const Npp64f *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                              const Npp64f *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp64f *pDst,
                                              int nDstStep, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 3, false, 0,
                            nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_64f_C3R(const Npp64f *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                          const Npp64f *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp64f *pDst,
                                          int nDstStep) {
  return nppiCrossCorrValid_Norm_64f_C3R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                             nDstStep, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_64f_C4R_Ctx(const Npp64f *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                              const Npp64f *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp64f *pDst,
                                              int nDstStep, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 4, false, 0,
                            nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_64f_C4R(const Npp64f *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                          const Npp64f *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp64f *pDst,
                                          int nDstStep) {
  return nppiCrossCorrValid_Norm_64f_C4R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                             nDstStep, defaultStreamContext());
}

NppStatus nppiCrossCorrValid_Norm_64f_AC4R_Ctx(const Npp64f *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                               const Npp64f *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp64f *pDst,
                                               int nDstStep, NppStreamContext nppStreamCtx) {
  return crossCorrValidNorm(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep, 4, true, 0,
                            nppStreamCtx);
}

NppStatus nppiCrossCorrValid_Norm_64f_AC4R(const Npp64f *pSrc, int nSrcStep, NppiSize oSrcRoiSize,
                                           const Npp64f *pTpl, int nTplStep, NppiSize oTplRoiSize, Npp64f *pDst,
                                           int nDstStep) {
  return nppiCrossCorrValid_Norm_64f_AC4R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,
                                              nDstStep, defaultStreamContext());
}
