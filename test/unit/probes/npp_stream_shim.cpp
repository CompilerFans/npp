// Minimal nppGetStreamContext for the CUDA-side verification build of the
// WarpAffine family (see verify_warp_affine_cuda.sh).
//
// NPP 13 removed the managed-stream API from the public headers, but libnppc
// still exports it.  When the differential tests are linked against the mcMPP
// implementation (instead of NVIDIA NPP) the reference symbol is missing, so
// this shim fills the context from the CUDA runtime.  It is host-only and does
// not affect the shipped library.

#include <cuda_runtime.h>
#include <npp.h>

extern "C" NppStatus nppGetStreamContext(NppStreamContext *c) {
  if (c == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  int dev = 0;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
  cudaDeviceProp prop{};
  err = cudaGetDeviceProperties(&prop, dev);
  if (err != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }
  c->hStream = nullptr;
  c->nCudaDeviceId = dev;
  c->nMultiProcessorCount = prop.multiProcessorCount;
  c->nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  c->nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
  c->nSharedMemPerBlock = static_cast<size_t>(prop.sharedMemPerBlock);
  c->nCudaDevAttrComputeCapabilityMajor = prop.major;
  c->nCudaDevAttrComputeCapabilityMinor = prop.minor;
  c->nStreamFlags = 0;
  c->nReserved0 = 0;
  return NPP_NO_ERROR;
}
