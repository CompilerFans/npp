#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// GPU-based Union-Find for Connected Component Labeling
// This implementation uses the label image itself as the Union-Find structure
// Each pixel stores its parent's linear index (y * width + x)
// Foreground pixels keep their initial label; only background pixels are merged

// Helper: convert 1-based linear index to 2D coordinates
__device__ void linearTo2D(Npp32u linearIdx, int width, int &x, int &y) {
  // linearIdx is 1-based (y * width + x + 1), so subtract 1 first
  Npp32u idx0 = linearIdx - 1;
  y = idx0 / width;
  x = idx0 % width;
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

// Initialize labels:
// ALL pixels get their 1-based linear index as initial label (y * width + x + 1)
// Using 1-based indexing so that label 0 can be used as "no label" sentinel
// Only BACKGROUND pixels (value == 0) will be merged via Union-Find
// FOREGROUND pixels keep their initial linear index unchanged
// This matches NVIDIA NPP behavior
__global__ void initLabels_kernel(const Npp8u *pSrc, int nSrcStep,
                                   Npp32u *pLabels, int nLabelsStep,
                                   int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  Npp32u *labels_row = (Npp32u *)((char *)pLabels + y * nLabelsStep);

  // All pixels get their 1-based linear index as initial label
  // Background pixels will be merged later; foreground pixels keep this value
  labels_row[x] = y * width + x + 1;
}

// Local merge: merge neighboring pixels with SAME VALUE (4-way connectivity)
// Pixels are merged only if they have the exact same source value
// This matches NVIDIA NPP behavior
__global__ void localMerge4Way_kernel(const Npp8u *pSrc, int nSrcStep,
                                       Npp32u *pLabels, int nLabelsStep,
                                       int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  Npp8u myValue = src_row[x];

  Npp32u *labels_row = (Npp32u *)((char *)pLabels + y * nLabelsStep);
  Npp32u label = labels_row[x];

  // Check right neighbor (merge only if same value)
  if (x + 1 < width && src_row[x + 1] == myValue) {
    Npp32u rightLabel = labels_row[x + 1];
    union2D(pLabels, nLabelsStep, width, label, rightLabel);
  }

  // Check bottom neighbor (merge only if same value)
  if (y + 1 < height) {
    const Npp8u *next_src_row = (const Npp8u *)((const char *)pSrc + (y + 1) * nSrcStep);
    if (next_src_row[x] == myValue) {
      Npp32u *next_row = (Npp32u *)((char *)pLabels + (y + 1) * nLabelsStep);
      Npp32u bottomLabel = next_row[x];
      union2D(pLabels, nLabelsStep, width, label, bottomLabel);
    }
  }
}

// Local merge: merge neighboring pixels with SAME VALUE (8-way connectivity)
// Pixels are merged only if they have the exact same source value
// Only check forward directions (right, bottom-left, bottom, bottom-right) to avoid redundant work
__global__ void localMerge8Way_kernel(const Npp8u *pSrc, int nSrcStep,
                                       Npp32u *pLabels, int nLabelsStep,
                                       int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  Npp8u myValue = src_row[x];

  Npp32u *labels_row = (Npp32u *)((char *)pLabels + y * nLabelsStep);
  Npp32u label = labels_row[x];

  // Right neighbor (merge only if same value)
  if (x + 1 < width && src_row[x + 1] == myValue) {
    Npp32u rightLabel = labels_row[x + 1];
    union2D(pLabels, nLabelsStep, width, label, rightLabel);
  }

  if (y + 1 < height) {
    const Npp8u *next_src_row = (const Npp8u *)((const char *)pSrc + (y + 1) * nSrcStep);
    Npp32u *next_row = (Npp32u *)((char *)pLabels + (y + 1) * nLabelsStep);

    // Bottom-left neighbor (merge only if same value)
    if (x > 0 && next_src_row[x - 1] == myValue) {
      Npp32u blLabel = next_row[x - 1];
      union2D(pLabels, nLabelsStep, width, label, blLabel);
    }

    // Bottom neighbor (merge only if same value)
    if (next_src_row[x] == myValue) {
      Npp32u bottomLabel = next_row[x];
      union2D(pLabels, nLabelsStep, width, label, bottomLabel);
    }

    // Bottom-right neighbor (merge only if same value)
    if (x + 1 < width && next_src_row[x + 1] == myValue) {
      Npp32u brLabel = next_row[x + 1];
      union2D(pLabels, nLabelsStep, width, label, brLabel);
    }
  }
}

// Path compression: flatten the label tree for ALL pixels
__global__ void pathCompression_kernel(Npp32u *pLabels, int nLabelsStep,
                                        int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  Npp32u *labels_row = (Npp32u *)((char *)pLabels + y * nLabelsStep);
  Npp32u label = labels_row[x];

  // Find root and update
  Npp32u root = find2D(pLabels, nLabelsStep, width, label);
  labels_row[x] = root;
}

// Final pass: convert 1-based labels to 0-based (subtract 1 from all labels)
__global__ void convertTo0Based_kernel(Npp32u *pLabels, int nLabelsStep,
                                        int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  Npp32u *labels_row = (Npp32u *)((char *)pLabels + y * nLabelsStep);
  labels_row[x] = labels_row[x] - 1;
}

// Batch processing kernel: initialize labels for multiple images
// ALL pixels get their 1-based linear index; only background pixels will be merged later
__global__ void batchInitLabels_kernel(const NppiImageDescriptor *pSrcBatchList,
                                        NppiImageDescriptor *pDstBatchList,
                                        int batchSize, int maxWidth, int maxHeight) {
  int imgIdx = blockIdx.z;
  if (imgIdx >= batchSize) return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  NppiSize imgSize = pDstBatchList[imgIdx].oSize;
  if (x >= imgSize.width || y >= imgSize.height) return;

  Npp32u *pDst = (Npp32u *)pDstBatchList[imgIdx].pData;
  int nDstStep = pDstBatchList[imgIdx].nStep;

  Npp32u *dst_row = (Npp32u *)((char *)pDst + y * nDstStep);

  // All pixels get their 1-based linear index as initial label
  dst_row[x] = y * imgSize.width + x + 1;
}

// Batch processing kernel: local merge for multiple images (8-way)
// Only check forward directions (right, bottom-left, bottom, bottom-right) to avoid redundant work
__global__ void batchLocalMerge8Way_kernel(const NppiImageDescriptor *pSrcBatchList,
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
  Npp32u *pLabels = (Npp32u *)pDstBatchList[imgIdx].pData;
  int nLabelsStep = pDstBatchList[imgIdx].nStep;
  int width = imgSize.width;
  int height = imgSize.height;

  const Npp8u *src_row = (const Npp8u *)((const char *)pSrc + y * nSrcStep);
  Npp8u myValue = src_row[x];

  Npp32u *labels_row = (Npp32u *)((char *)pLabels + y * nLabelsStep);
  Npp32u label = labels_row[x];

  // Right neighbor (merge only if same value)
  if (x + 1 < width && src_row[x + 1] == myValue) {
    Npp32u rightLabel = labels_row[x + 1];
    union2D(pLabels, nLabelsStep, width, label, rightLabel);
  }

  if (y + 1 < height) {
    const Npp8u *next_src_row = (const Npp8u *)((const char *)pSrc + (y + 1) * nSrcStep);
    Npp32u *next_row = (Npp32u *)((char *)pLabels + (y + 1) * nLabelsStep);

    // Bottom-left neighbor (merge only if same value)
    if (x > 0 && next_src_row[x - 1] == myValue) {
      Npp32u blLabel = next_row[x - 1];
      union2D(pLabels, nLabelsStep, width, label, blLabel);
    }

    // Bottom neighbor (merge only if same value)
    if (next_src_row[x] == myValue) {
      Npp32u bottomLabel = next_row[x];
      union2D(pLabels, nLabelsStep, width, label, bottomLabel);
    }

    // Bottom-right neighbor (merge only if same value)
    if (x + 1 < width && next_src_row[x + 1] == myValue) {
      Npp32u brLabel = next_row[x + 1];
      union2D(pLabels, nLabelsStep, width, label, brLabel);
    }
  }
}

// Batch processing kernel: path compression for multiple images
// Compresses ALL pixels
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

  Npp32u root = find2D(pLabels, nLabelsStep, width, label);
  labels_row[x] = root;
}

// Batch processing kernel: convert 1-based labels to 0-based
__global__ void batchConvertTo0Based_kernel(NppiImageDescriptor *pDstBatchList,
                                             int batchSize, int maxWidth, int maxHeight) {
  int imgIdx = blockIdx.z;
  if (imgIdx >= batchSize) return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  NppiSize imgSize = pDstBatchList[imgIdx].oSize;
  if (x >= imgSize.width || y >= imgSize.height) return;

  Npp32u *pLabels = (Npp32u *)pDstBatchList[imgIdx].pData;
  int nLabelsStep = pDstBatchList[imgIdx].nStep;

  Npp32u *labels_row = (Npp32u *)((char *)pLabels + y * nLabelsStep);
  labels_row[x] = labels_row[x] - 1;
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
  // Use more iterations for larger images to ensure convergence
  int numIterations = 20; // Increased iterations for better convergence
  for (int i = 0; i < numIterations; i++) {
    if (eNorm == nppiNormInf) {
      // 8-way connectivity
      localMerge8Way_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
          pSrc, nSrcStep, pDst, nDstStep, width, height);
    } else {
      // 4-way connectivity (nppiNormL1)
      localMerge4Way_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
          pSrc, nSrcStep, pDst, nDstStep, width, height);
    }
    // Interleave path compression for faster convergence
    if (i % 4 == 3) {
      pathCompression_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
          pDst, nDstStep, width, height);
    }
  }

  // Step 3: Final path compression (for all pixels)
  for (int i = 0; i < 10; i++) {
    pathCompression_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pDst, nDstStep, width, height);
  }

  // Step 4: Convert from 1-based to 0-based labels
  convertTo0Based_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pDst, nDstStep, width, height);

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
  // Use more iterations for better convergence
  int numIterations = 20;
  for (int i = 0; i < numIterations; i++) {
    batchLocalMerge8Way_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pSrcBatchList, pDstBatchList, nBatchSize, maxWidth, maxHeight);
    // Interleave path compression for faster convergence
    if (i % 4 == 3) {
      batchPathCompression_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
          pDstBatchList, nBatchSize, maxWidth, maxHeight);
    }
  }

  // Step 3: Final path compression (for all pixels)
  for (int i = 0; i < 10; i++) {
    batchPathCompression_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pDstBatchList, nBatchSize, maxWidth, maxHeight);
  }

  // Step 4: Convert from 1-based to 0-based labels
  batchConvertTo0Based_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pDstBatchList, nBatchSize, maxWidth, maxHeight);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

} // extern "C"
