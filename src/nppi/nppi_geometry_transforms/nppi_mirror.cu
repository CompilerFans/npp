#include "nppdefs.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 镜像翻转kernel实现
__global__ void nppiMirror_8u_C1R_kernel_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                              int height, NppiAxis flip) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int srcX = x;
    int srcY = y;
    int dstX = x;
    int dstY = y;

    // 根据翻转轴计算源坐标
    if (flip == NPP_HORIZONTAL_AXIS) {
      // 水平翻转（上下翻转）
      srcY = height - 1 - y;
    } else if (flip == NPP_VERTICAL_AXIS) {
      // 垂直翻转（左右翻转）
      srcX = width - 1 - x;
    } else if (flip == NPP_BOTH_AXIS) {
      // 双轴翻转（180度旋转）
      srcX = width - 1 - x;
      srcY = height - 1 - y;
    }

    // 读取源像素并写入目标
    const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + srcY * nSrcStep);
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + dstY * nDstStep);

    pDstRow[dstX] = pSrcRow[srcX];
  }
}

extern "C" {
cudaError_t nppiMirror_8u_C1R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI,
                                     NppiAxis flip, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oROI.width + blockSize.x - 1) / blockSize.x, (oROI.height + blockSize.y - 1) / blockSize.y);

  nppiMirror_8u_C1R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oROI.width,
                                                                    oROI.height, flip);

  return cudaGetLastError();
}
}
