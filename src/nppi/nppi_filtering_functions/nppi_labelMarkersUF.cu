#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// GPU-based Union-Find for Connected Component Labeling
// This implementation uses the label image itself as the Union-Find structure
// Each pixel stores its parent's linear index (y * width + x + 1, with 0 for background)

// Helper: convert linear index to 2D coordinates
__device__ void linearTo2D(Npp32u linearIdx, int width, int &x, int &y) {
  // linearIdx is 1-based (0 is background)
  Npp32u idx = linearIdx - 1;
  y = idx / width;
  x = idx % width;
}

// Helper: get label value at 2D position
__device__ Npp32u getLabel2D(Npp32u *pLabels, int nLabelsStep, int x, int y) {
  Npp32u *row = (Npp32u *)((char *)pLabels + y * nLabelsStep);
  return row[x];
}

// Helper: set label value at 2D position
__device__ void setLabel2D(Npp32u *pLabels, int nLabelsStep, int x, int y, Npp32u value) {
  Npp32u *row = (Npp32u *)((char *)pLabels + y * nLabelsStep);
  row[x] = value;
}

// Atomic CAS on 2D label array
__device__ Npp32u atomicCAS2D(Npp32u *pLabels, int nLabelsStep, int x, int y, Npp32u compare, Npp32u val) {
  Npp32u *row = (Npp32u *)((char *)pLabels + y * nLabelsStep);
  return atomicCAS(&row[x], compare, val);
}

// Find root with path compression (2D version)
__device__ Npp32u find2D(Npp32u *pLabels, int nLabelsStep, int width, Npp32u label) {
  if (label == 0) return 0;

  int x, y;
  linearTo2D(label, width, x, y);
  Npp32u parent = getLabel2D(pLabels, nLabelsStep, x, y);

  while (parent != label) {
    int px, py;
    linearTo2D(parent, width, px, py);
    Npp32u grandparent = getLabel2D(pLabels, nLabelsStep, px, py);

    // Path compression: point to grandparent
    if (grandparent != parent) {
      atomicCAS2D(pLabels, nLabelsStep, x, y, parent, grandparent);
    }

    label = parent;
    x = px;
    y = py;
    parent = grandparent;
  }

  return label;
}

// Union operation (2D version)
__device__ void union2D(Npp32u *pLabels, int nLabelsStep, int width, Npp32u labelA, Npp32u labelB) {
  if (labelA == 0 || labelB == 0) return;

  Npp32u rootA = find2D(pLabels, nLabelsStep, width, labelA);
  Npp32u rootB = find2D(pLabels, nLabelsStep, width, labelB);

  while (rootA != rootB) {
    Npp32u minRoot = min(rootA, rootB);
    Npp32u maxRoot = max(rootA, rootB);

    int mx, my;
    linearTo2D(maxRoot, width, mx, my);

    Npp32u old = atomicCAS2D(pLabels, nLabelsStep, mx, my, maxRoot, minRoot);
    if (old == maxRoot) {
      break;
    }

    rootA = find2D(pLabels, nLabelsStep, width, rootA);
    rootB = find2D(pLabels, nLabelsStep, width, rootB);
  }
}

// Initialize labels: each foreground pixel gets its linear index as initial label
__global__ void initLabels_kernel(const Npp8u *pSrc, int nSrcStep,
                                   Npp32u *pLabels, int nLabelsStep,
                                   int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  Npp32u *labels_row = (Npp32u *)((char *)pLabels + y * nLabelsStep);

  if (src_row[x] != 0) {
    // Foreground pixel: assign linear index + 1 (to avoid 0 which is background)
    labels_row[x] = y * width + x + 1;
  } else {
    // Background pixel
    labels_row[x] = 0;
  }
}

// Local merge: merge neighboring pixels (4-way connectivity)
__global__ void localMerge4Way_kernel(Npp32u *pLabels, int nLabelsStep,
                                       int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  Npp32u *labels_row = (Npp32u *)((char *)pLabels + y * nLabelsStep);
  Npp32u label = labels_row[x];

  if (label == 0) return; // Skip background

  // Check right neighbor
  if (x + 1 < width) {
    Npp32u rightLabel = labels_row[x + 1];
    if (rightLabel != 0) {
      union2D(pLabels, nLabelsStep, width, label, rightLabel);
    }
  }

  // Check bottom neighbor
  if (y + 1 < height) {
    Npp32u *next_row = (Npp32u *)((char *)pLabels + (y + 1) * nLabelsStep);
    Npp32u bottomLabel = next_row[x];
    if (bottomLabel != 0) {
      union2D(pLabels, nLabelsStep, width, label, bottomLabel);
    }
  }
}

// Local merge: merge neighboring pixels (8-way connectivity)
__global__ void localMerge8Way_kernel(Npp32u *pLabels, int nLabelsStep,
                                       int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  Npp32u *labels_row = (Npp32u *)((char *)pLabels + y * nLabelsStep);
  Npp32u label = labels_row[x];

  if (label == 0) return; // Skip background

  // Right neighbor
  if (x + 1 < width) {
    Npp32u rightLabel = labels_row[x + 1];
    if (rightLabel != 0) {
      union2D(pLabels, nLabelsStep, width, label, rightLabel);
    }
  }

  if (y + 1 < height) {
    Npp32u *next_row = (Npp32u *)((char *)pLabels + (y + 1) * nLabelsStep);

    // Bottom-left neighbor
    if (x > 0) {
      Npp32u blLabel = next_row[x - 1];
      if (blLabel != 0) {
        union2D(pLabels, nLabelsStep, width, label, blLabel);
      }
    }

    // Bottom neighbor
    Npp32u bottomLabel = next_row[x];
    if (bottomLabel != 0) {
      union2D(pLabels, nLabelsStep, width, label, bottomLabel);
    }

    // Bottom-right neighbor
    if (x + 1 < width) {
      Npp32u brLabel = next_row[x + 1];
      if (brLabel != 0) {
        union2D(pLabels, nLabelsStep, width, label, brLabel);
      }
    }
  }
}

// Path compression: flatten the label tree
__global__ void pathCompression_kernel(Npp32u *pLabels, int nLabelsStep,
                                        int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  Npp32u *labels_row = (Npp32u *)((char *)pLabels + y * nLabelsStep);
  Npp32u label = labels_row[x];

  if (label == 0) return; // Skip background

  // Find root and update
  Npp32u root = find2D(pLabels, nLabelsStep, width, label);
  labels_row[x] = root;
}

// Batch processing kernel: initialize labels for multiple images
__global__ void batchInitLabels_kernel(const NppiImageDescriptor *pSrcBatchList,
                                        NppiImageDescriptor *pDstBatchList,
                                        int batchSize, int maxWidth, int maxHeight) {
  int imgIdx = blockIdx.z;
  if (imgIdx >= batchSize) return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  NppiSize imgSize = pDstBatchList[imgIdx].oSize;
  if (x >= imgSize.width || y >= imgSize.height) return;

  const Npp8u *pSrc = (const Npp8u *)pSrcBatchList[imgIdx].pData;
  int nSrcStep = pSrcBatchList[imgIdx].nStep;
  Npp32u *pDst = (Npp32u *)pDstBatchList[imgIdx].pData;
  int nDstStep = pDstBatchList[imgIdx].nStep;

  const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  Npp32u *dst_row = (Npp32u *)((char *)pDst + y * nDstStep);

  if (src_row[x] != 0) {
    dst_row[x] = y * imgSize.width + x + 1;
  } else {
    dst_row[x] = 0;
  }
}

// Batch processing kernel: local merge for multiple images (8-way)
__global__ void batchLocalMerge8Way_kernel(NppiImageDescriptor *pDstBatchList,
                                            int batchSize, int maxWidth, int maxHeight) {
  int imgIdx = blockIdx.z;
  if (imgIdx >= batchSize) return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  NppiSize imgSize = pDstBatchList[imgIdx].oSize;
  if (x >= imgSize.width || y >= imgSize.height) return;

  Npp32u *pLabels = (Npp32u *)pDstBatchList[imgIdx].pData;
  int nLabelsStep = pDstBatchList[imgIdx].nStep;
  int width = imgSize.width;
  int height = imgSize.height;

  Npp32u *labels_row = (Npp32u *)((char *)pLabels + y * nLabelsStep);
  Npp32u label = labels_row[x];

  if (label == 0) return;

  // Right neighbor
  if (x + 1 < width) {
    Npp32u rightLabel = labels_row[x + 1];
    if (rightLabel != 0) {
      union2D(pLabels, nLabelsStep, width, label, rightLabel);
    }
  }

  if (y + 1 < height) {
    Npp32u *next_row = (Npp32u *)((char *)pLabels + (y + 1) * nLabelsStep);

    if (x > 0) {
      Npp32u blLabel = next_row[x - 1];
      if (blLabel != 0) {
        union2D(pLabels, nLabelsStep, width, label, blLabel);
      }
    }

    Npp32u bottomLabel = next_row[x];
    if (bottomLabel != 0) {
      union2D(pLabels, nLabelsStep, width, label, bottomLabel);
    }

    if (x + 1 < width) {
      Npp32u brLabel = next_row[x + 1];
      if (brLabel != 0) {
        union2D(pLabels, nLabelsStep, width, label, brLabel);
      }
    }
  }
}

// Batch processing kernel: path compression for multiple images
__global__ void batchPathCompression_kernel(NppiImageDescriptor *pDstBatchList,
                                             int batchSize, int maxWidth, int maxHeight) {
  int imgIdx = blockIdx.z;
  if (imgIdx >= batchSize) return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  NppiSize imgSize = pDstBatchList[imgIdx].oSize;
  if (x >= imgSize.width || y >= imgSize.height) return;

  Npp32u *pLabels = (Npp32u *)pDstBatchList[imgIdx].pData;
  int nLabelsStep = pDstBatchList[imgIdx].nStep;
  int width = imgSize.width;

  Npp32u *labels_row = (Npp32u *)((char *)pLabels + y * nLabelsStep);
  Npp32u label = labels_row[x];

  if (label == 0) return;

  Npp32u root = find2D(pLabels, nLabelsStep, width, label);
  labels_row[x] = root;
}

extern "C" {

// Get required buffer size for label markers
NppStatus nppiLabelMarkersUFGetBufferSize_32u_C1R_Ctx_impl(NppiSize oSizeROI, int *hpBufferSize) {
  // Buffer size is proportional to image size for compatibility with NVIDIA NPP
  // Even though our implementation uses the label image itself as Union-Find structure,
  // we report a size proportional to the image for API compatibility
  size_t imagePixels = (size_t)oSizeROI.width * oSizeROI.height;
  size_t bufferSize = imagePixels * sizeof(Npp32u) + 1024; // Extra space for alignment
  size_t alignedSize = (bufferSize + 511) & ~511;

  *hpBufferSize = (int)alignedSize;
  return NPP_SUCCESS;
}

// Single image label markers using Union-Find
NppStatus nppiLabelMarkersUF_8u32u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep,
                                                 Npp32u *pDst, int nDstStep,
                                                 NppiSize oSizeROI, NppiNorm eNorm,
                                                 Npp8u *pBuffer,
                                                 NppStreamContext nppStreamCtx) {
  int width = oSizeROI.width;
  int height = oSizeROI.height;

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  // Step 1: Initialize labels
  initLabels_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, width, height);

  // Step 2: Local merge based on connectivity
  int numIterations = 10; // Multiple iterations for convergence
  for (int i = 0; i < numIterations; i++) {
    if (eNorm == nppiNormInf) {
      // 8-way connectivity
      localMerge8Way_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
          pDst, nDstStep, width, height);
    } else {
      // 4-way connectivity (nppiNormL1)
      localMerge4Way_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
          pDst, nDstStep, width, height);
    }
  }

  // Step 3: Path compression
  for (int i = 0; i < 5; i++) {
    pathCompression_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pDst, nDstStep, width, height);
  }

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

// Batch image label markers using Union-Find
NppStatus nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx_impl(
    const NppiImageDescriptor *pSrcBatchList,
    NppiImageDescriptor *pDstBatchList,
    int nBatchSize, NppiSize oMaxSizeROI,
    NppiNorm eNorm, NppStreamContext nppStreamCtx) {

  int maxWidth = oMaxSizeROI.width;
  int maxHeight = oMaxSizeROI.height;

  dim3 blockSize(16, 16);
  dim3 gridSize((maxWidth + blockSize.x - 1) / blockSize.x,
                (maxHeight + blockSize.y - 1) / blockSize.y,
                nBatchSize);

  // Step 1: Initialize labels for all images
  batchInitLabels_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrcBatchList, pDstBatchList, nBatchSize, maxWidth, maxHeight);

  // Step 2: Local merge (currently only 8-way supported for batch)
  int numIterations = 10;
  for (int i = 0; i < numIterations; i++) {
    batchLocalMerge8Way_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pDstBatchList, nBatchSize, maxWidth, maxHeight);
  }

  // Step 3: Path compression
  for (int i = 0; i < 5; i++) {
    batchPathCompression_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pDstBatchList, nBatchSize, maxWidth, maxHeight);
  }

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

} // extern "C"
