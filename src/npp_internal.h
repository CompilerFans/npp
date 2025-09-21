#ifndef NPP_INTERNAL_H
#define NPP_INTERNAL_H

#include "npp.h"
#include <cstring>
#include <cuda_runtime.h>

// Implementation file

#ifdef __cplusplus
extern "C" {
#endif

// Implementation file
static inline NppStreamContext nppCreateDefaultStreamContext(void) {
  NppStreamContext ctx;

  // Get current device ID
  int deviceId;
  if (cudaGetDevice(&deviceId) != cudaSuccess) {
    deviceId = 0; // fallback
  }

  // Get device properties
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess) {
    // Set safe defaults if we can't get properties
    memset(&ctx, 0, sizeof(NppStreamContext));
    ctx.hStream = (cudaStream_t)0;
    return ctx;
  }

  // Initialize all fields
  ctx.hStream = (cudaStream_t)0; // Default stream
  ctx.nCudaDeviceId = deviceId;
  ctx.nMultiProcessorCount = prop.multiProcessorCount;
  ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
  ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;
  ctx.nCudaDevAttrComputeCapabilityMajor = prop.major;
  ctx.nCudaDevAttrComputeCapabilityMinor = prop.minor;
  ctx.nStreamFlags = cudaStreamDefault;
  ctx.nReserved0 = 0;

  return ctx;
}

#ifdef __cplusplus
}
#endif

#endif // NPP_INTERNAL_H