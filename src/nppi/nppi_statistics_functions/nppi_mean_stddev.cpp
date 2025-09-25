#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// Buffer size calculation for 8u single channel
NppStatus nppiMeanStdDevGetBufferHostSize_8u_C1R_Ctx(NppiSize oSizeROI, size_t *hpBufferSize,
                                                     NppStreamContext /*nppStreamCtx*/) {
  printf("not implement\n");
  return NPP_ERROR;
}

NppStatus nppiMeanStdDevGetBufferHostSize_8u_C1R(NppiSize oSizeROI, size_t *hpBufferSize) {
  printf("not implement\n");
  return NPP_ERROR;
}

// Buffer size calculation for 32-bit float single channel
NppStatus nppiMeanStdDevGetBufferHostSize_32f_C1R_Ctx(NppiSize oSizeROI, size_t *hpBufferSize,
                                                      NppStreamContext /*nppStreamCtx*/) {
  printf("not implement\n");
  return NPP_ERROR;
}

NppStatus nppiMeanStdDevGetBufferHostSize_32f_C1R(NppiSize oSizeROI, size_t *hpBufferSize) {
  printf("not implement\n");
  return NPP_ERROR;
}

// Mean and standard deviation calculation for 8-bit unsigned single channel
NppStatus nppiMean_StdDev_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
                                     Npp64f *pMean, Npp64f *pStdDev, NppStreamContext nppStreamCtx) {
  printf("not implement\n");
  return NPP_ERROR;
}

NppStatus nppiMean_StdDev_8u_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
                                 Npp64f *pMean, Npp64f *pStdDev) {
  printf("not implement\n");
  return NPP_ERROR;
}

// Mean and standard deviation calculation for 32-bit float single channel
NppStatus nppiMean_StdDev_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
                                      Npp64f *pMean, Npp64f *pStdDev, NppStreamContext nppStreamCtx) {
  printf("not implement\n");
  return NPP_ERROR;
}

NppStatus nppiMean_StdDev_32f_C1R(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u *pDeviceBuffer,
                                  Npp64f *pMean, Npp64f *pStdDev) {
  printf("not implement\n");
  return NPP_ERROR;
}