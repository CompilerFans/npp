#include "npp.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>

extern "C" {
cudaError_t mppCrossCorrFullSameNorm8u8uLaunch(const Npp8u *, int, NppiSize, const Npp8u *, int, NppiSize, Npp8u *,
                                               int, int, bool, int, bool, cudaStream_t);
cudaError_t mppCrossCorrFullSameNorm8u32fLaunch(const Npp8u *, int, NppiSize, const Npp8u *, int, NppiSize, Npp32f *,
                                                int, int, bool, int, bool, cudaStream_t);
cudaError_t mppCrossCorrFullSameNorm8s32fLaunch(const Npp8s *, int, NppiSize, const Npp8s *, int, NppiSize, Npp32f *,
                                                int, int, bool, int, bool, cudaStream_t);
cudaError_t mppCrossCorrFullSameNorm16u32fLaunch(const Npp16u *, int, NppiSize, const Npp16u *, int, NppiSize,
                                                 Npp32f *, int, int, bool, int, bool, cudaStream_t);
cudaError_t mppCrossCorrFullSameNorm32f32fLaunch(const Npp32f *, int, NppiSize, const Npp32f *, int, NppiSize,
                                                 Npp32f *, int, int, bool, int, bool, cudaStream_t);
cudaError_t mppCrossCorrFullSameNorm64f64fLaunch(const Npp64f *, int, NppiSize, const Npp64f *, int, NppiSize,
                                                 Npp64f *, int, int, bool, int, bool, cudaStream_t);
}

namespace {

template <typename Source, typename Destination> struct KernelLauncher;

template <> struct KernelLauncher<Npp8u, Npp8u> {
  static cudaError_t launch(const Npp8u *source, int sourceStep, NppiSize sourceSize, const Npp8u *tpl,
                            int templateStep, NppiSize templateSize, Npp8u *destination, int destinationStep,
                            int channels, bool preserveAlpha, int scaleFactor, bool sameMode, cudaStream_t stream) {
    return mppCrossCorrFullSameNorm8u8uLaunch(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                              destination, destinationStep, channels, preserveAlpha, scaleFactor,
                                              sameMode, stream);
  }
};

template <> struct KernelLauncher<Npp8u, Npp32f> {
  static cudaError_t launch(const Npp8u *source, int sourceStep, NppiSize sourceSize, const Npp8u *tpl,
                            int templateStep, NppiSize templateSize, Npp32f *destination, int destinationStep,
                            int channels, bool preserveAlpha, int scaleFactor, bool sameMode, cudaStream_t stream) {
    return mppCrossCorrFullSameNorm8u32fLaunch(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                               destination, destinationStep, channels, preserveAlpha, scaleFactor,
                                               sameMode, stream);
  }
};

template <> struct KernelLauncher<Npp8s, Npp32f> {
  static cudaError_t launch(const Npp8s *source, int sourceStep, NppiSize sourceSize, const Npp8s *tpl,
                            int templateStep, NppiSize templateSize, Npp32f *destination, int destinationStep,
                            int channels, bool preserveAlpha, int scaleFactor, bool sameMode, cudaStream_t stream) {
    return mppCrossCorrFullSameNorm8s32fLaunch(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                               destination, destinationStep, channels, preserveAlpha, scaleFactor,
                                               sameMode, stream);
  }
};

template <> struct KernelLauncher<Npp16u, Npp32f> {
  static cudaError_t launch(const Npp16u *source, int sourceStep, NppiSize sourceSize, const Npp16u *tpl,
                            int templateStep, NppiSize templateSize, Npp32f *destination, int destinationStep,
                            int channels, bool preserveAlpha, int scaleFactor, bool sameMode, cudaStream_t stream) {
    return mppCrossCorrFullSameNorm16u32fLaunch(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                                destination, destinationStep, channels, preserveAlpha, scaleFactor,
                                                sameMode, stream);
  }
};

template <> struct KernelLauncher<Npp32f, Npp32f> {
  static cudaError_t launch(const Npp32f *source, int sourceStep, NppiSize sourceSize, const Npp32f *tpl,
                            int templateStep, NppiSize templateSize, Npp32f *destination, int destinationStep,
                            int channels, bool preserveAlpha, int scaleFactor, bool sameMode, cudaStream_t stream) {
    return mppCrossCorrFullSameNorm32f32fLaunch(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                                destination, destinationStep, channels, preserveAlpha, scaleFactor,
                                                sameMode, stream);
  }
};

template <> struct KernelLauncher<Npp64f, Npp64f> {
  static cudaError_t launch(const Npp64f *source, int sourceStep, NppiSize sourceSize, const Npp64f *tpl,
                            int templateStep, NppiSize templateSize, Npp64f *destination, int destinationStep,
                            int channels, bool preserveAlpha, int scaleFactor, bool sameMode, cudaStream_t stream) {
    return mppCrossCorrFullSameNorm64f64fLaunch(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                                destination, destinationStep, channels, preserveAlpha, scaleFactor,
                                                sameMode, stream);
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
NppStatus crossCorrFullSameNorm(const Source *source, int sourceStep, NppiSize sourceSize, const Source *tpl,
                                int templateStep, NppiSize templateSize, Destination *destination,
                                int destinationStep, int channels, bool preserveAlpha, int scaleFactor, bool sameMode,
                                NppStreamContext context) {
  if (!source || !tpl || !destination) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (sourceSize.width <= 0 || sourceSize.height <= 0 || templateSize.width <= 0 || templateSize.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  const int destinationWidth = sameMode ? sourceSize.width : sourceSize.width + templateSize.width - 1;
  if (stepIsTooSmall<Source>(sourceStep, sourceSize.width, channels) ||
      stepIsTooSmall<Source>(templateStep, templateSize.width, channels) ||
      stepIsTooSmall<Destination>(destinationStep, destinationWidth, channels)) {
    return NPP_STEP_ERROR;
  }

  const cudaError_t error = KernelLauncher<Source, Destination>::launch(
      source, sourceStep, sourceSize, tpl, templateStep, templateSize, destination, destinationStep, channels,
      preserveAlpha, scaleFactor, sameMode, context.hStream);
  return error == cudaSuccess ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStreamContext defaultStreamContext() {
  NppStreamContext context{};
  nppGetStreamContext(&context);
  return context;
}

template <typename Source, typename Destination>
NppStatus fullDispatch(const Source *src, int srcStep, NppiSize srcSize, const Source *tpl, int tplStep,
                       NppiSize tplSize, Destination *dst, int dstStep, int scaleFactor, NppStreamContext ctx) {
  return crossCorrFullSameNorm(src, srcStep, srcSize, tpl, tplStep, tplSize, dst, dstStep, 1, false, scaleFactor,
                               false, ctx);
}

template <typename Source, typename Destination>
NppStatus fullDispatch3(const Source *src, int srcStep, NppiSize srcSize, const Source *tpl, int tplStep,
                        NppiSize tplSize, Destination *dst, int dstStep, int scaleFactor, NppStreamContext ctx) {
  return crossCorrFullSameNorm(src, srcStep, srcSize, tpl, tplStep, tplSize, dst, dstStep, 3, false, scaleFactor,
                               false, ctx);
}

template <typename Source, typename Destination>
NppStatus fullDispatch4(const Source *src, int srcStep, NppiSize srcSize, const Source *tpl, int tplStep,
                        NppiSize tplSize, Destination *dst, int dstStep, int scaleFactor, NppStreamContext ctx) {
  return crossCorrFullSameNorm(src, srcStep, srcSize, tpl, tplStep, tplSize, dst, dstStep, 4, false, scaleFactor,
                               false, ctx);
}

template <typename Source, typename Destination>
NppStatus fullDispatchAC4(const Source *src, int srcStep, NppiSize srcSize, const Source *tpl, int tplStep,
                          NppiSize tplSize, Destination *dst, int dstStep, int scaleFactor, NppStreamContext ctx) {
  return crossCorrFullSameNorm(src, srcStep, srcSize, tpl, tplStep, tplSize, dst, dstStep, 4, true, scaleFactor, false,
                               ctx);
}

template <typename Source, typename Destination>
NppStatus sameDispatch(const Source *src, int srcStep, NppiSize srcSize, const Source *tpl, int tplStep,
                       NppiSize tplSize, Destination *dst, int dstStep, int scaleFactor, NppStreamContext ctx) {
  return crossCorrFullSameNorm(src, srcStep, srcSize, tpl, tplStep, tplSize, dst, dstStep, 1, false, scaleFactor, true,
                               ctx);
}

template <typename Source, typename Destination>
NppStatus sameDispatch3(const Source *src, int srcStep, NppiSize srcSize, const Source *tpl, int tplStep,
                        NppiSize tplSize, Destination *dst, int dstStep, int scaleFactor, NppStreamContext ctx) {
  return crossCorrFullSameNorm(src, srcStep, srcSize, tpl, tplStep, tplSize, dst, dstStep, 3, false, scaleFactor, true,
                               ctx);
}

template <typename Source, typename Destination>
NppStatus sameDispatch4(const Source *src, int srcStep, NppiSize srcSize, const Source *tpl, int tplStep,
                        NppiSize tplSize, Destination *dst, int dstStep, int scaleFactor, NppStreamContext ctx) {
  return crossCorrFullSameNorm(src, srcStep, srcSize, tpl, tplStep, tplSize, dst, dstStep, 4, false, scaleFactor, true,
                               ctx);
}

template <typename Source, typename Destination>
NppStatus sameDispatchAC4(const Source *src, int srcStep, NppiSize srcSize, const Source *tpl, int tplStep,
                          NppiSize tplSize, Destination *dst, int dstStep, int scaleFactor, NppStreamContext ctx) {
  return crossCorrFullSameNorm(src, srcStep, srcSize, tpl, tplStep, tplSize, dst, dstStep, 4, true, scaleFactor, true,
                               ctx);
}

} // namespace

// ============================================================================
// 8u scaled (Sfs) variants
// ============================================================================

#define MPP_CROSS_CORR_SFS_ENTRY_POINTS(prefix, TYPENAME, TYPE)                                                  \
  NppStatus prefix##Full_Norm_##TYPENAME##_C1RSfs_Ctx(const TYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,       \
                                                  const TYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,           \
                                                  TYPE *pDst, int nDstStep, int nScaleFactor,                     \
                                                  NppStreamContext nppStreamCtx) {                                \
    return fullDispatch<TYPE, TYPE>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep,     \
                                    nScaleFactor, nppStreamCtx);                                                  \
  }                                                                                                               \
  NppStatus prefix##Full_Norm_##TYPENAME##_C1RSfs(const TYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,               \
                                              const TYPE *pTpl, int nTplStep, NppiSize oTplRoiSize, TYPE *pDst,   \
                                              int nDstStep, int nScaleFactor) {                                   \
    return prefix##Full_Norm_##TYPENAME##_C1RSfs_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,  \
                                                 nDstStep, nScaleFactor, defaultStreamContext());                 \
  }                                                                                                               \
  NppStatus prefix##Full_Norm_##TYPENAME##_C3RSfs_Ctx(const TYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,           \
                                                  const TYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,           \
                                                  TYPE *pDst, int nDstStep, int nScaleFactor,                     \
                                                  NppStreamContext nppStreamCtx) {                                \
    return fullDispatch3<TYPE, TYPE>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep,    \
                                     nScaleFactor, nppStreamCtx);                                                 \
  }                                                                                                               \
  NppStatus prefix##Full_Norm_##TYPENAME##_C3RSfs(const TYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,               \
                                              const TYPE *pTpl, int nTplStep, NppiSize oTplRoiSize, TYPE *pDst,   \
                                              int nDstStep, int nScaleFactor) {                                   \
    return prefix##Full_Norm_##TYPENAME##_C3RSfs_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,  \
                                                 nDstStep, nScaleFactor, defaultStreamContext());                 \
  }                                                                                                               \
  NppStatus prefix##Full_Norm_##TYPENAME##_C4RSfs_Ctx(const TYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,           \
                                                  const TYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,           \
                                                  TYPE *pDst, int nDstStep, int nScaleFactor,                     \
                                                  NppStreamContext nppStreamCtx) {                                \
    return fullDispatch4<TYPE, TYPE>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep,    \
                                     nScaleFactor, nppStreamCtx);                                                 \
  }                                                                                                               \
  NppStatus prefix##Full_Norm_##TYPENAME##_C4RSfs(const TYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,               \
                                              const TYPE *pTpl, int nTplStep, NppiSize oTplRoiSize, TYPE *pDst,   \
                                              int nDstStep, int nScaleFactor) {                                   \
    return prefix##Full_Norm_##TYPENAME##_C4RSfs_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,  \
                                                 nDstStep, nScaleFactor, defaultStreamContext());                 \
  }                                                                                                               \
  NppStatus prefix##Full_Norm_##TYPENAME##_AC4RSfs_Ctx(const TYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,          \
                                                   const TYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,          \
                                                   TYPE *pDst, int nDstStep, int nScaleFactor,                    \
                                                   NppStreamContext nppStreamCtx) {                               \
    return fullDispatchAC4<TYPE, TYPE>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep,  \
                                       nScaleFactor, nppStreamCtx);                                               \
  }                                                                                                               \
  NppStatus prefix##Full_Norm_##TYPENAME##_AC4RSfs(const TYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,              \
                                               const TYPE *pTpl, int nTplStep, NppiSize oTplRoiSize, TYPE *pDst,  \
                                               int nDstStep, int nScaleFactor) {                                  \
    return prefix##Full_Norm_##TYPENAME##_AC4RSfs_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, \
                                                  nDstStep, nScaleFactor, defaultStreamContext());                \
  }                                                                                                               \
  NppStatus prefix##Same_Norm_##TYPENAME##_C1RSfs_Ctx(const TYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,           \
                                                  const TYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,           \
                                                  TYPE *pDst, int nDstStep, int nScaleFactor,                     \
                                                  NppStreamContext nppStreamCtx) {                                \
    return sameDispatch<TYPE, TYPE>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep,     \
                                    nScaleFactor, nppStreamCtx);                                                  \
  }                                                                                                               \
  NppStatus prefix##Same_Norm_##TYPENAME##_C1RSfs(const TYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,               \
                                              const TYPE *pTpl, int nTplStep, NppiSize oTplRoiSize, TYPE *pDst,   \
                                              int nDstStep, int nScaleFactor) {                                   \
    return prefix##Same_Norm_##TYPENAME##_C1RSfs_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,  \
                                                 nDstStep, nScaleFactor, defaultStreamContext());                 \
  }                                                                                                               \
  NppStatus prefix##Same_Norm_##TYPENAME##_C3RSfs_Ctx(const TYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,           \
                                                  const TYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,           \
                                                  TYPE *pDst, int nDstStep, int nScaleFactor,                     \
                                                  NppStreamContext nppStreamCtx) {                                \
    return sameDispatch3<TYPE, TYPE>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep,    \
                                     nScaleFactor, nppStreamCtx);                                                 \
  }                                                                                                               \
  NppStatus prefix##Same_Norm_##TYPENAME##_C3RSfs(const TYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,               \
                                              const TYPE *pTpl, int nTplStep, NppiSize oTplRoiSize, TYPE *pDst,   \
                                              int nDstStep, int nScaleFactor) {                                   \
    return prefix##Same_Norm_##TYPENAME##_C3RSfs_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,  \
                                                 nDstStep, nScaleFactor, defaultStreamContext());                 \
  }                                                                                                               \
  NppStatus prefix##Same_Norm_##TYPENAME##_C4RSfs_Ctx(const TYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,           \
                                                  const TYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,           \
                                                  TYPE *pDst, int nDstStep, int nScaleFactor,                     \
                                                  NppStreamContext nppStreamCtx) {                                \
    return sameDispatch4<TYPE, TYPE>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep,    \
                                     nScaleFactor, nppStreamCtx);                                                 \
  }                                                                                                               \
  NppStatus prefix##Same_Norm_##TYPENAME##_C4RSfs(const TYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,               \
                                              const TYPE *pTpl, int nTplStep, NppiSize oTplRoiSize, TYPE *pDst,   \
                                              int nDstStep, int nScaleFactor) {                                   \
    return prefix##Same_Norm_##TYPENAME##_C4RSfs_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,  \
                                                 nDstStep, nScaleFactor, defaultStreamContext());                 \
  }                                                                                                               \
  NppStatus prefix##Same_Norm_##TYPENAME##_AC4RSfs_Ctx(const TYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,          \
                                                   const TYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,          \
                                                   TYPE *pDst, int nDstStep, int nScaleFactor,                    \
                                                   NppStreamContext nppStreamCtx) {                               \
    return sameDispatchAC4<TYPE, TYPE>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, nDstStep,  \
                                       nScaleFactor, nppStreamCtx);                                               \
  }                                                                                                               \
  NppStatus prefix##Same_Norm_##TYPENAME##_AC4RSfs(const TYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,              \
                                               const TYPE *pTpl, int nTplStep, NppiSize oTplRoiSize, TYPE *pDst,  \
                                               int nDstStep, int nScaleFactor) {                                  \
    return prefix##Same_Norm_##TYPENAME##_AC4RSfs_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst, \
                                                  nDstStep, nScaleFactor, defaultStreamContext());                \
  }

MPP_CROSS_CORR_SFS_ENTRY_POINTS(nppiCrossCorr, 8u, Npp8u)

// ============================================================================
// 8u32f / 8s32f / 16u32f / 32f / 64f unscaled variants
// ============================================================================

#define MPP_CROSS_CORR_ENTRY_POINTS(prefix, SRCTYPE, DSTTYPE, TYPENAME)                                          \
  NppStatus prefix##Full_Norm_##TYPENAME##_C1R_Ctx(const SRCTYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,       \
                                                   const SRCTYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,       \
                                                   DSTTYPE *pDst, int nDstStep, NppStreamContext nppStreamCtx) {  \
    return fullDispatch<SRCTYPE, DSTTYPE>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,         \
                                          nDstStep, 0, nppStreamCtx);                                             \
  }                                                                                                               \
  NppStatus prefix##Full_Norm_##TYPENAME##_C1R(const SRCTYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,           \
                                               const SRCTYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,           \
                                               DSTTYPE *pDst, int nDstStep) {                                     \
    return prefix##Full_Norm_##TYPENAME##_C1R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize,       \
                                                  pDst, nDstStep, defaultStreamContext());                        \
  }                                                                                                               \
  NppStatus prefix##Full_Norm_##TYPENAME##_C3R_Ctx(const SRCTYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,       \
                                                   const SRCTYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,       \
                                                   DSTTYPE *pDst, int nDstStep, NppStreamContext nppStreamCtx) {  \
    return fullDispatch3<SRCTYPE, DSTTYPE>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,        \
                                           nDstStep, 0, nppStreamCtx);                                            \
  }                                                                                                               \
  NppStatus prefix##Full_Norm_##TYPENAME##_C3R(const SRCTYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,           \
                                               const SRCTYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,           \
                                               DSTTYPE *pDst, int nDstStep) {                                     \
    return prefix##Full_Norm_##TYPENAME##_C3R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize,       \
                                                  pDst, nDstStep, defaultStreamContext());                        \
  }                                                                                                               \
  NppStatus prefix##Full_Norm_##TYPENAME##_C4R_Ctx(const SRCTYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,       \
                                                   const SRCTYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,       \
                                                   DSTTYPE *pDst, int nDstStep, NppStreamContext nppStreamCtx) {  \
    return fullDispatch4<SRCTYPE, DSTTYPE>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,         \
                                           nDstStep, 0, nppStreamCtx);                                            \
  }                                                                                                               \
  NppStatus prefix##Full_Norm_##TYPENAME##_C4R(const SRCTYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,           \
                                               const SRCTYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,           \
                                               DSTTYPE *pDst, int nDstStep) {                                     \
    return prefix##Full_Norm_##TYPENAME##_C4R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize,       \
                                                  pDst, nDstStep, defaultStreamContext());                        \
  }                                                                                                               \
  NppStatus prefix##Full_Norm_##TYPENAME##_AC4R_Ctx(const SRCTYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,      \
                                                    const SRCTYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,      \
                                                    DSTTYPE *pDst, int nDstStep, NppStreamContext nppStreamCtx) { \
    return fullDispatchAC4<SRCTYPE, DSTTYPE>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,      \
                                             nDstStep, 0, nppStreamCtx);                                          \
  }                                                                                                               \
  NppStatus prefix##Full_Norm_##TYPENAME##_AC4R(const SRCTYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,          \
                                                const SRCTYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,          \
                                                DSTTYPE *pDst, int nDstStep) {                                    \
    return prefix##Full_Norm_##TYPENAME##_AC4R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize,      \
                                                   pDst, nDstStep, defaultStreamContext());                       \
  }                                                                                                               \
  NppStatus prefix##Same_Norm_##TYPENAME##_C1R_Ctx(const SRCTYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,       \
                                                   const SRCTYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,       \
                                                   DSTTYPE *pDst, int nDstStep, NppStreamContext nppStreamCtx) {  \
    return sameDispatch<SRCTYPE, DSTTYPE>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,         \
                                          nDstStep, 0, nppStreamCtx);                                             \
  }                                                                                                               \
  NppStatus prefix##Same_Norm_##TYPENAME##_C1R(const SRCTYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,           \
                                               const SRCTYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,           \
                                               DSTTYPE *pDst, int nDstStep) {                                     \
    return prefix##Same_Norm_##TYPENAME##_C1R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize,       \
                                                  pDst, nDstStep, defaultStreamContext());                        \
  }                                                                                                               \
  NppStatus prefix##Same_Norm_##TYPENAME##_C3R_Ctx(const SRCTYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,       \
                                                   const SRCTYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,       \
                                                   DSTTYPE *pDst, int nDstStep, NppStreamContext nppStreamCtx) {  \
    return sameDispatch3<SRCTYPE, DSTTYPE>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,        \
                                           nDstStep, 0, nppStreamCtx);                                            \
  }                                                                                                               \
  NppStatus prefix##Same_Norm_##TYPENAME##_C3R(const SRCTYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,           \
                                               const SRCTYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,           \
                                               DSTTYPE *pDst, int nDstStep) {                                     \
    return prefix##Same_Norm_##TYPENAME##_C3R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize,       \
                                                  pDst, nDstStep, defaultStreamContext());                        \
  }                                                                                                               \
  NppStatus prefix##Same_Norm_##TYPENAME##_C4R_Ctx(const SRCTYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,       \
                                                   const SRCTYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,       \
                                                   DSTTYPE *pDst, int nDstStep, NppStreamContext nppStreamCtx) {  \
    return sameDispatch4<SRCTYPE, DSTTYPE>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,        \
                                           nDstStep, 0, nppStreamCtx);                                            \
  }                                                                                                               \
  NppStatus prefix##Same_Norm_##TYPENAME##_C4R(const SRCTYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,           \
                                               const SRCTYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,           \
                                               DSTTYPE *pDst, int nDstStep) {                                     \
    return prefix##Same_Norm_##TYPENAME##_C4R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize,       \
                                                  pDst, nDstStep, defaultStreamContext());                        \
  }                                                                                                               \
  NppStatus prefix##Same_Norm_##TYPENAME##_AC4R_Ctx(const SRCTYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,      \
                                                    const SRCTYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,      \
                                                    DSTTYPE *pDst, int nDstStep, NppStreamContext nppStreamCtx) { \
    return sameDispatchAC4<SRCTYPE, DSTTYPE>(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize, pDst,      \
                                             nDstStep, 0, nppStreamCtx);                                          \
  }                                                                                                               \
  NppStatus prefix##Same_Norm_##TYPENAME##_AC4R(const SRCTYPE *pSrc, int nSrcStep, NppiSize oSrcRoiSize,          \
                                                const SRCTYPE *pTpl, int nTplStep, NppiSize oTplRoiSize,          \
                                                DSTTYPE *pDst, int nDstStep) {                                    \
    return prefix##Same_Norm_##TYPENAME##_AC4R_Ctx(pSrc, nSrcStep, oSrcRoiSize, pTpl, nTplStep, oTplRoiSize,      \
                                                   pDst, nDstStep, defaultStreamContext());                       \
  }

MPP_CROSS_CORR_ENTRY_POINTS(nppiCrossCorr, Npp8u, Npp32f, 8u32f)
MPP_CROSS_CORR_ENTRY_POINTS(nppiCrossCorr, Npp8s, Npp32f, 8s32f)
MPP_CROSS_CORR_ENTRY_POINTS(nppiCrossCorr, Npp16u, Npp32f, 16u32f)
MPP_CROSS_CORR_ENTRY_POINTS(nppiCrossCorr, Npp32f, Npp32f, 32f)
MPP_CROSS_CORR_ENTRY_POINTS(nppiCrossCorr, Npp64f, Npp64f, 64f)
