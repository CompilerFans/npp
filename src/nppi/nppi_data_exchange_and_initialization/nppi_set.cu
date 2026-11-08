#include "npp.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

template <typename T>
__global__ void nppiSet_C1R_kernel(T nValue, T *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  T *dstRow = reinterpret_cast<T *>(reinterpret_cast<char *>(pDst) + y * nDstStep);
  dstRow[x] = nValue;
}

template <typename T, int CHANNELS>
struct NppiSetValue {
  T v[CHANNELS];
};

template <typename T, int CHANNELS>
__global__ void nppiSet_CxR_kernel(NppiSetValue<T, CHANNELS> value, T *pDst, int nDstStep, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  T *dstRow = reinterpret_cast<T *>(reinterpret_cast<char *>(pDst) + y * nDstStep);
  int idx = x * CHANNELS;

#pragma unroll
  for (int c = 0; c < CHANNELS; ++c) {
    dstRow[idx + c] = value.v[c];
  }
}

// Row-wise 16-byte fill for C1 when the destination base and row stride are
// 16-byte aligned.  Threads with px < packetsPerRow write one uint4; the
// single px == packetsPerRow thread fills the sub-packet tail.
template <typename T>
__global__ void nppiSet_C1R_vec_kernel(uint4 value, T scalar, T *pDst, int nDstStep, int width, int height,
                                       int packetsPerRow, int tailStart) {
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (y >= height || px > packetsPerRow) return;
  T *dstRow = reinterpret_cast<T *>(reinterpret_cast<char *>(pDst) + y * nDstStep);
  if (px < packetsPerRow) {
    reinterpret_cast<uint4 *>(dstRow)[px] = value;
  } else {
    for (int x = tailStart; x < width; ++x) dstRow[x] = scalar;
  }
}

template <typename T>
inline bool make_set_packet(T value, uint4 &packet, int &elemsPerPacket) {
  constexpr size_t kSize = sizeof(T);
  if constexpr ((kSize & (kSize - 1)) != 0 || kSize > 16) {
    (void)value;
    (void)packet;
    (void)elemsPerPacket;
    return false;
  } else {
    alignas(16) unsigned char bytes[16];
    const unsigned char *v = reinterpret_cast<const unsigned char *>(&value);
    for (size_t i = 0; i < 16; ++i) bytes[i] = v[i % kSize];
    std::memcpy(&packet, bytes, 16);
    elemsPerPacket = (int)(16 / kSize);
    return true;
  }
}

template <typename T>
NppStatus launchSetScalar(T nValue, T *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  const size_t rowBytes = (size_t)oSizeROI.width * sizeof(T);
  uint4 packet{};
  int elemsPerPacket = 0;
  const bool vec = rowBytes >= 16 && ((uintptr_t)pDst & 15u) == 0u && (nDstStep & 15) == 0 &&
                   make_set_packet(nValue, packet, elemsPerPacket);
  if (vec) {
    const int packetsPerRow = (int)(rowBytes / 16);
    const int tailStart = packetsPerRow * elemsPerPacket;
    dim3 blockSize(16, 16);
    dim3 gridSize((unsigned)(packetsPerRow + 1), (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    nppiSet_C1R_vec_kernel<T><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        packet, nValue, pDst, nDstStep, oSizeROI.width, oSizeROI.height, packetsPerRow, tailStart);
  } else {
    dim3 blockSize(16, 16);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    nppiSet_C1R_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(nValue, pDst, nDstStep, oSizeROI.width,
                                                                          oSizeROI.height);
  }
  return cudaGetLastError() == cudaSuccess ? NPP_SUCCESS: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

template <typename T, int CHANNELS>
NppStatus launchSetVector(const T *aValue, T *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  // Pass the channel values by value so the Ctx entry point stays asynchronous
  // and can be used inside a CUDA graph capture (no device allocation/copy).
  NppiSetValue<T, CHANNELS> value{};
  for (int c = 0; c < CHANNELS; ++c) {
    value.v[c] = aValue[c];
  }

  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  nppiSet_CxR_kernel<T, CHANNELS><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(value, pDst, nDstStep,
                                                                                      oSizeROI.width, oSizeROI.height);

  return cudaGetLastError() == cudaSuccess ? NPP_SUCCESS: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

extern "C" {

NppStatus nppiSet_8u_C1R_Ctx_impl(Npp8u nValue, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  return launchSetScalar(nValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_8u_C3R_Ctx_impl(const Npp8u aValue[3], Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  return launchSetVector<Npp8u, 3>(aValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_8u_C4R_Ctx_impl(const Npp8u aValue[4], Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  return launchSetVector<Npp8u, 4>(aValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_16u_C1R_Ctx_impl(Npp16u nValue, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchSetScalar(nValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_16u_C3R_Ctx_impl(const Npp16u aValue[3], Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchSetVector<Npp16u, 3>(aValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_16u_C4R_Ctx_impl(const Npp16u aValue[4], Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchSetVector<Npp16u, 4>(aValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_16s_C1R_Ctx_impl(Npp16s nValue, Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchSetScalar(nValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_16s_C3R_Ctx_impl(const Npp16s aValue[3], Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchSetVector<Npp16s, 3>(aValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_16s_C4R_Ctx_impl(const Npp16s aValue[4], Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchSetVector<Npp16s, 4>(aValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_32f_C1R_Ctx_impl(Npp32f nValue, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchSetScalar(nValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_32f_C3R_Ctx_impl(const Npp32f aValue[3], Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchSetVector<Npp32f, 3>(aValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_32f_C4R_Ctx_impl(const Npp32f aValue[4], Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return launchSetVector<Npp32f, 4>(aValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

} // extern "C"
