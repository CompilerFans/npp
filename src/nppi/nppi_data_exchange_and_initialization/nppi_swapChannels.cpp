#include "npp.h"
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for CUDA implementation functions
template<typename T, int SRC_CHANNELS, int DST_CHANNELS>
NppStatus nppiSwapChannels_impl(const T *pSrc, int nSrcStep,
                                 T *pDst, int nDstStep,
                                 NppiSize oSizeROI,
                                 const int *aDstOrder,
                                 T fillValue,
                                 NppStreamContext nppStreamCtx);

template<typename T>
NppStatus nppiSwapChannels_AC4R_impl(const T *pSrc, int nSrcStep,
                                      T *pDst, int nDstStep,
                                      NppiSize oSizeROI,
                                      const int aDstOrder[3],
                                      NppStreamContext nppStreamCtx);

// Input validation helpers
template<int CHANNELS>
static inline NppStatus validateSwapChannelsInputs(const void *pPtr, int nStep,
                                                    NppiSize oSizeROI,
                                                    const int aDstOrder[]) {
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (!pPtr || !aDstOrder) {
    return NPP_NULL_POINTER_ERROR;
  }

  for (int i = 0; i < CHANNELS; i++) {
    if (aDstOrder[i] < 0 || aDstOrder[i] >= CHANNELS) {
      return NPP_CHANNEL_ORDER_ERROR;
    }
  }

  return NPP_SUCCESS;
}

// Macro to generate in-place functions
#define GENERATE_INPLACE_FUNC(TYPE, CHANNELS, SUFFIX) \
NppStatus nppiSwapChannels_##TYPE##_##SUFFIX##IR_Ctx( \
    Npp##TYPE *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, \
    const int aDstOrder[CHANNELS], NppStreamContext nppStreamCtx) { \
  NppStatus status = validateSwapChannelsInputs<CHANNELS>(pSrcDst, nSrcDstStep, oSizeROI, aDstOrder); \
  if (status != NPP_SUCCESS) { \
    return status; \
  } \
  \
  int step = oSizeROI.width * CHANNELS * sizeof(Npp##TYPE); \
  Npp##TYPE *pTemp; \
  cudaError_t err = cudaMalloc(&pTemp, step * oSizeROI.height); \
  if (err != cudaSuccess) { \
    return NPP_MEMORY_ALLOCATION_ERR; \
  } \
  \
  err = cudaMemcpy2DAsync(pTemp, step, pSrcDst, nSrcDstStep, \
                          oSizeROI.width * CHANNELS * sizeof(Npp##TYPE), oSizeROI.height, \
                          cudaMemcpyDeviceToDevice, nppStreamCtx.hStream); \
  if (err != cudaSuccess) { \
    cudaFree(pTemp); \
    return NPP_MEMORY_ALLOCATION_ERR; \
  } \
  \
  status = nppiSwapChannels_impl<Npp##TYPE, CHANNELS, CHANNELS>( \
      pTemp, step, pSrcDst, nSrcDstStep, oSizeROI, aDstOrder, 0, nppStreamCtx); \
  \
  cudaStreamSynchronize(nppStreamCtx.hStream); \
  cudaFree(pTemp); \
  return status; \
} \
\
NppStatus nppiSwapChannels_##TYPE##_##SUFFIX##IR( \
    Npp##TYPE *pSrcDst, int nSrcDstStep, NppiSize oSizeROI, \
    const int aDstOrder[CHANNELS]) { \
  NppStreamContext nppStreamCtx; \
  nppGetStreamContext(&nppStreamCtx); \
  return nppiSwapChannels_##TYPE##_##SUFFIX##IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aDstOrder, nppStreamCtx); \
}

// Macro to generate C3R and C4R functions
#define GENERATE_CNR_FUNC(TYPE, CHANNELS, SUFFIX) \
NppStatus nppiSwapChannels_##TYPE##_##SUFFIX##R_Ctx( \
    const Npp##TYPE *pSrc, int nSrcStep, \
    Npp##TYPE *pDst, int nDstStep, NppiSize oSizeROI, \
    const int aDstOrder[CHANNELS], NppStreamContext nppStreamCtx) { \
  NppStatus status = validateSwapChannelsInputs<CHANNELS>(pSrc, nSrcStep, oSizeROI, aDstOrder); \
  if (status != NPP_SUCCESS) { \
    return status; \
  } \
  if (!pDst || nDstStep <= 0) { \
    return NPP_NULL_POINTER_ERROR; \
  } \
  return nppiSwapChannels_impl<Npp##TYPE, CHANNELS, CHANNELS>( \
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aDstOrder, 0, nppStreamCtx); \
} \
\
NppStatus nppiSwapChannels_##TYPE##_##SUFFIX##R( \
    const Npp##TYPE *pSrc, int nSrcStep, \
    Npp##TYPE *pDst, int nDstStep, NppiSize oSizeROI, \
    const int aDstOrder[CHANNELS]) { \
  NppStreamContext nppStreamCtx; \
  nppGetStreamContext(&nppStreamCtx); \
  return nppiSwapChannels_##TYPE##_##SUFFIX##R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aDstOrder, nppStreamCtx); \
}

// Macro to generate C4C3R functions
#define GENERATE_C4C3R_FUNC(TYPE) \
NppStatus nppiSwapChannels_##TYPE##_C4C3R_Ctx( \
    const Npp##TYPE *pSrc, int nSrcStep, \
    Npp##TYPE *pDst, int nDstStep, NppiSize oSizeROI, \
    const int aDstOrder[3], NppStreamContext nppStreamCtx) { \
  NppStatus status = validateSwapChannelsInputs<3>(pSrc, nSrcStep, oSizeROI, aDstOrder); \
  if (status != NPP_SUCCESS) { \
    return status; \
  } \
  if (!pDst || nDstStep <= 0) { \
    return NPP_NULL_POINTER_ERROR; \
  } \
  return nppiSwapChannels_impl<Npp##TYPE, 4, 3>( \
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aDstOrder, 0, nppStreamCtx); \
} \
\
NppStatus nppiSwapChannels_##TYPE##_C4C3R( \
    const Npp##TYPE *pSrc, int nSrcStep, \
    Npp##TYPE *pDst, int nDstStep, NppiSize oSizeROI, \
    const int aDstOrder[3]) { \
  NppStreamContext nppStreamCtx; \
  nppGetStreamContext(&nppStreamCtx); \
  return nppiSwapChannels_##TYPE##_C4C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aDstOrder, nppStreamCtx); \
}

// Macro to generate C3C4R functions with fill value
#define GENERATE_C3C4R_FUNC(TYPE) \
NppStatus nppiSwapChannels_##TYPE##_C3C4R_Ctx( \
    const Npp##TYPE *pSrc, int nSrcStep, \
    Npp##TYPE *pDst, int nDstStep, NppiSize oSizeROI, \
    const int aDstOrder[4], const Npp##TYPE nValue, NppStreamContext nppStreamCtx) { \
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) { \
    return NPP_SIZE_ERROR; \
  } \
  if (nSrcStep <= 0 || nDstStep <= 0) { \
    return NPP_STEP_ERROR; \
  } \
  if (!pSrc || !pDst || !aDstOrder) { \
    return NPP_NULL_POINTER_ERROR; \
  } \
  return nppiSwapChannels_impl<Npp##TYPE, 3, 4>( \
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aDstOrder, nValue, nppStreamCtx); \
} \
\
NppStatus nppiSwapChannels_##TYPE##_C3C4R( \
    const Npp##TYPE *pSrc, int nSrcStep, \
    Npp##TYPE *pDst, int nDstStep, NppiSize oSizeROI, \
    const int aDstOrder[4], const Npp##TYPE nValue) { \
  NppStreamContext nppStreamCtx; \
  nppGetStreamContext(&nppStreamCtx); \
  return nppiSwapChannels_##TYPE##_C3C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aDstOrder, nValue, nppStreamCtx); \
}

// Special validation for AC4R (validates channel order for 4-channel data but only 3 order values)
static inline NppStatus validateAC4RInputs(const void *pSrc, const void *pDst,
                                            int nSrcStep, int nDstStep,
                                            NppiSize oSizeROI,
                                            const int aDstOrder[3]) {
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (!pSrc || !pDst || !aDstOrder) {
    return NPP_NULL_POINTER_ERROR;
  }

  // AC4R: aDstOrder values should be in [0, 3) for first 3 channels
  for (int i = 0; i < 3; i++) {
    if (aDstOrder[i] < 0 || aDstOrder[i] >= 3) {
      return NPP_CHANNEL_ORDER_ERROR;
    }
  }

  return NPP_SUCCESS;
}

// Macro to generate AC4R functions
#define GENERATE_AC4R_FUNC(TYPE) \
NppStatus nppiSwapChannels_##TYPE##_AC4R_Ctx( \
    const Npp##TYPE *pSrc, int nSrcStep, \
    Npp##TYPE *pDst, int nDstStep, NppiSize oSizeROI, \
    const int aDstOrder[3], NppStreamContext nppStreamCtx) { \
  NppStatus status = validateAC4RInputs(pSrc, pDst, nSrcStep, nDstStep, oSizeROI, aDstOrder); \
  if (status != NPP_SUCCESS) { \
    return status; \
  } \
  return nppiSwapChannels_AC4R_impl<Npp##TYPE>( \
      pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aDstOrder, nppStreamCtx); \
} \
\
NppStatus nppiSwapChannels_##TYPE##_AC4R( \
    const Npp##TYPE *pSrc, int nSrcStep, \
    Npp##TYPE *pDst, int nDstStep, NppiSize oSizeROI, \
    const int aDstOrder[3]) { \
  NppStreamContext nppStreamCtx; \
  nppGetStreamContext(&nppStreamCtx); \
  return nppiSwapChannels_##TYPE##_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aDstOrder, nppStreamCtx); \
}

// Generate all function variants for each data type
#define GENERATE_ALL_SWAP_CHANNELS(TYPE) \
  GENERATE_CNR_FUNC(TYPE, 3, C3) \
  GENERATE_CNR_FUNC(TYPE, 4, C4) \
  GENERATE_INPLACE_FUNC(TYPE, 3, C3) \
  GENERATE_INPLACE_FUNC(TYPE, 4, C4) \
  GENERATE_C4C3R_FUNC(TYPE) \
  GENERATE_C3C4R_FUNC(TYPE) \
  GENERATE_AC4R_FUNC(TYPE)

GENERATE_ALL_SWAP_CHANNELS(8u)
GENERATE_ALL_SWAP_CHANNELS(16u)
GENERATE_ALL_SWAP_CHANNELS(16s)
GENERATE_ALL_SWAP_CHANNELS(32s)
GENERATE_ALL_SWAP_CHANNELS(32f)

#undef GENERATE_ALL_SWAP_CHANNELS
#undef GENERATE_AC4R_FUNC
#undef GENERATE_C3C4R_FUNC
#undef GENERATE_C4C3R_FUNC
#undef GENERATE_CNR_FUNC
#undef GENERATE_INPLACE_FUNC
