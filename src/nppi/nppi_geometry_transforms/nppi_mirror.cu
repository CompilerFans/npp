#include "nppdefs.h"
#include <climits>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ============ general mirror kernel ============
template <typename T, int channels>
__global__ void nppiMirror_kernel_template(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, int width, int height,
                                           NppiAxis flip) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int srcX = x;
    int srcY = y;
    int dstX = x;
    int dstY = y;

    // Compute source coordinates based on flip axis
    if (flip == NPP_HORIZONTAL_AXIS) {
      // Horizontal flip (top-bottom flip)
      srcY = height - 1 - y;
    } else if (flip == NPP_VERTICAL_AXIS) {
      // Vertical flip (left-right flip)
      srcX = width - 1 - x;
    } else if (flip == NPP_BOTH_AXIS) {
      // Dual axis flip (180 degree rotation)
      srcX = width - 1 - x;
      srcY = height - 1 - y;
    }

    // Calculate byte offsets
    int srcByteOffset = srcY * nSrcStep + srcX * channels * sizeof(T);
    int dstByteOffset = dstY * nDstStep + dstX * channels * sizeof(T);

    // Copy all channels using unrolled loops for better performance
    if (channels == 1) {
      *((T *)((char *)pDst + dstByteOffset)) = *((const T *)((const char *)pSrc + srcByteOffset));
    } else if (channels == 3) {
      T *dstPtr = (T *)((char *)pDst + dstByteOffset);
      const T *srcPtr = (const T *)((const char *)pSrc + srcByteOffset);
      dstPtr[0] = srcPtr[0];
      dstPtr[1] = srcPtr[1];
      dstPtr[2] = srcPtr[2];
    } else if (channels == 4) {
      T *dstPtr = (T *)((char *)pDst + dstByteOffset);
      const T *srcPtr = (const T *)((const char *)pSrc + srcByteOffset);
      dstPtr[0] = srcPtr[0];
      dstPtr[1] = srcPtr[1];
      dstPtr[2] = srcPtr[2];
      dstPtr[3] = srcPtr[3];
    }
  }
}

// ============ in-place mirror kernel ============
template <typename T, int channels>
__global__ void nppiMirrorInPlace_kernel_template(T *pSrcDst, int nSrcDstStep, int width, int height, NppiAxis flip) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    // Only process necessary pixels for in-place operation
    int processX = x;
    int processY = y;

    if (flip == NPP_VERTICAL_AXIS) {
      if (x >= (width + 1) / 2)
        return;
      processX = width - 1 - x;
    } else if (flip == NPP_HORIZONTAL_AXIS) {
      if (y >= (height + 1) / 2)
        return;
      processY = height - 1 - y;
    } else if (flip == NPP_BOTH_AXIS) {
      if (x >= (width + 1) / 2 || y >= (height + 1) / 2)
        return;
      processX = width - 1 - x;
      processY = height - 1 - y;
    }

    // Calculate byte offsets
    int srcByteOffset = y * nSrcDstStep + x * channels * sizeof(T);
    int dstByteOffset = processY * nSrcDstStep + processX * channels * sizeof(T);

    // Skip if source and destination are the same
    if (srcByteOffset == dstByteOffset)
      return;

    // Swap pixels
    T temp[4]; // Maximum 4 channels
    T *srcPtr = (T *)((char *)pSrcDst + srcByteOffset);
    T *dstPtr = (T *)((char *)pSrcDst + dstByteOffset);

    for (int c = 0; c < channels; c++) {
      temp[c] = srcPtr[c];
    }
    for (int c = 0; c < channels; c++) {
      srcPtr[c] = dstPtr[c];
    }
    for (int c = 0; c < channels; c++) {
      dstPtr[c] = temp[c];
    }
  }
}

// ============ launch kernel ============
template <typename T>
NppStatus launch_mirror_kernel(const T *pSrc, int nSrcStep, T *pDst, int nDstStep, int width, int height, NppiAxis flip,
                               int channels, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

  if (channels == 1) {
    nppiMirror_kernel_template<T, 1>
        <<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, width, height, flip);
  } else if (channels == 3) {
    nppiMirror_kernel_template<T, 3>
        <<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, width, height, flip);
  } else if (channels == 4) {
    nppiMirror_kernel_template<T, 4>
        <<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, width, height, flip);
  } else {
    return NPP_CHANNEL_ERROR;
  }

  cudaError_t err = cudaGetLastError();
  return (err == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

template <typename T>
NppStatus launch_mirror_inplace_kernel(T *pSrcDst, int nSrcDstStep, int width, int height, NppiAxis flip, int channels,
                                       cudaStream_t stream) {

  // Use pitched temporary copy to preserve row stride and avoid read-after-write hazards
  size_t tempPitchBytes = 0;
  T *d_temp = nullptr;
  cudaError_t cudaErr =
      cudaMallocPitch(reinterpret_cast<void **>(&d_temp), &tempPitchBytes, width * channels * sizeof(T), height);
  if (cudaErr != cudaSuccess) {
    return NPP_MEMORY_ALLOCATION_ERR;
  }

  cudaErr = cudaMemcpy2DAsync(d_temp, tempPitchBytes, pSrcDst, nSrcDstStep, width * channels * sizeof(T), height,
                              cudaMemcpyDeviceToDevice, stream);
  if (cudaErr != cudaSuccess) {
    cudaFree(d_temp);
    return NPP_MEMCPY_ERROR;
  }

  if (tempPitchBytes > static_cast<size_t>(INT_MAX)) {
    cudaFree(d_temp);
    return NPP_STEP_ERROR;
  }

  NppStatus status = launch_mirror_kernel<T>(d_temp, static_cast<int>(tempPitchBytes), pSrcDst, nSrcDstStep, width,
                                             height, flip, channels, stream);

  cudaFree(d_temp);
  return status;
}

extern "C" {

// 16-bit unsigned C1R impl
NppStatus nppiMirror_16u_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oROI,
                                      NppiAxis flip, cudaStream_t stream) {
  return launch_mirror_kernel<Npp16u>(pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, flip, 1, stream);
}

// 16-bit unsigned C1IR impl
NppStatus nppiMirror_16u_C1IR_Ctx_impl(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                       cudaStream_t stream) {
  return launch_mirror_inplace_kernel<Npp16u>(pSrcDst, nSrcDstStep, oROI.width, oROI.height, flip, 1, stream);
}

// 16-bit unsigned C3R impl
NppStatus nppiMirror_16u_C3R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oROI,
                                      NppiAxis flip, cudaStream_t stream) {
  return launch_mirror_kernel<Npp16u>(pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, flip, 3, stream);
}

// 16-bit unsigned C3IR impl
NppStatus nppiMirror_16u_C3IR_Ctx_impl(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                       cudaStream_t stream) {
  return launch_mirror_inplace_kernel<Npp16u>(pSrcDst, nSrcDstStep, oROI.width, oROI.height, flip, 3, stream);
}

// 16-bit unsigned C4R impl
NppStatus nppiMirror_16u_C4R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oROI,
                                      NppiAxis flip, cudaStream_t stream) {
  return launch_mirror_kernel<Npp16u>(pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, flip, 4, stream);
}

// 16-bit unsigned C4IR impl
NppStatus nppiMirror_16u_C4IR_Ctx_impl(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                       cudaStream_t stream) {
  return launch_mirror_inplace_kernel<Npp16u>(pSrcDst, nSrcDstStep, oROI.width, oROI.height, flip, 4, stream);
}

// 32-bit float C1R impl
NppStatus nppiMirror_32f_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI,
                                      NppiAxis flip, cudaStream_t stream) {
  return launch_mirror_kernel<Npp32f>(pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, flip, 1, stream);
}

// 32-bit float C1IR impl
NppStatus nppiMirror_32f_C1IR_Ctx_impl(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                       cudaStream_t stream) {
  return launch_mirror_inplace_kernel<Npp32f>(pSrcDst, nSrcDstStep, oROI.width, oROI.height, flip, 1, stream);
}

// 32-bit float C3R impl
NppStatus nppiMirror_32f_C3R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI,
                                      NppiAxis flip, cudaStream_t stream) {
  return launch_mirror_kernel<Npp32f>(pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, flip, 3, stream);
}

// 32-bit float C3IR impl
NppStatus nppiMirror_32f_C3IR_Ctx_impl(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                       cudaStream_t stream) {
  return launch_mirror_inplace_kernel<Npp32f>(pSrcDst, nSrcDstStep, oROI.width, oROI.height, flip, 3, stream);
}

// 32-bit float C4R impl
NppStatus nppiMirror_32f_C4R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI,
                                      NppiAxis flip, cudaStream_t stream) {
  return launch_mirror_kernel<Npp32f>(pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, flip, 4, stream);
}

// 32-bit float C4IR impl
NppStatus nppiMirror_32f_C4IR_Ctx_impl(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                       cudaStream_t stream) {
  return launch_mirror_inplace_kernel<Npp32f>(pSrcDst, nSrcDstStep, oROI.width, oROI.height, flip, 4, stream);
}

// 32-bit signed integer C1R impl
NppStatus nppiMirror_32s_C1R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oROI,
                                      NppiAxis flip, cudaStream_t stream) {
  return launch_mirror_kernel<Npp32s>(pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, flip, 1, stream);
}

// 32-bit signed integer C1IR impl
NppStatus nppiMirror_32s_C1IR_Ctx_impl(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                       cudaStream_t stream) {
  return launch_mirror_inplace_kernel<Npp32s>(pSrcDst, nSrcDstStep, oROI.width, oROI.height, flip, 1, stream);
}

// 32-bit signed integer C3R impl
NppStatus nppiMirror_32s_C3R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oROI,
                                      NppiAxis flip, cudaStream_t stream) {
  return launch_mirror_kernel<Npp32s>(pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, flip, 3, stream);
}

// 32-bit signed integer C3IR impl
NppStatus nppiMirror_32s_C3IR_Ctx_impl(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                       cudaStream_t stream) {
  return launch_mirror_inplace_kernel<Npp32s>(pSrcDst, nSrcDstStep, oROI.width, oROI.height, flip, 3, stream);
}

// 32-bit signed integer C4R impl
NppStatus nppiMirror_32s_C4R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oROI,
                                      NppiAxis flip, cudaStream_t stream) {
  return launch_mirror_kernel<Npp32s>(pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, flip, 4, stream);
}

// 32-bit signed integer C4IR impl
NppStatus nppiMirror_32s_C4IR_Ctx_impl(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                       cudaStream_t stream) {
  return launch_mirror_inplace_kernel<Npp32s>(pSrcDst, nSrcDstStep, oROI.width, oROI.height, flip, 4, stream);
}

NppStatus nppiMirror_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI,
                                     NppiAxis flip, cudaStream_t stream) {
  return launch_mirror_kernel<Npp8u>(pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, flip, 1, stream);
}

// 8-bit unsigned C1IR impl
NppStatus nppiMirror_8u_C1IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                      cudaStream_t stream) {
  return launch_mirror_inplace_kernel<Npp8u>(pSrcDst, nSrcDstStep, oROI.width, oROI.height, flip, 1, stream);
}

// 8-bit unsigned C3R impl
NppStatus nppiMirror_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI,
                                     NppiAxis flip, cudaStream_t stream) {
  return launch_mirror_kernel<Npp8u>(pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, flip, 3, stream);
}

// 8-bit unsigned C3IR impl
NppStatus nppiMirror_8u_C3IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                      cudaStream_t stream) {
  return launch_mirror_inplace_kernel<Npp8u>(pSrcDst, nSrcDstStep, oROI.width, oROI.height, flip, 3, stream);
}

// 8-bit unsigned C4R impl
NppStatus nppiMirror_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI,
                                     NppiAxis flip, cudaStream_t stream) {
  return launch_mirror_kernel<Npp8u>(pSrc, nSrcStep, pDst, nDstStep, oROI.width, oROI.height, flip, 4, stream);
}

// 8-bit unsigned C4IR impl
NppStatus nppiMirror_8u_C4IR_Ctx_impl(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                      cudaStream_t stream) {
  return launch_mirror_inplace_kernel<Npp8u>(pSrcDst, nSrcDstStep, oROI.width, oROI.height, flip, 4, stream);
}
}
