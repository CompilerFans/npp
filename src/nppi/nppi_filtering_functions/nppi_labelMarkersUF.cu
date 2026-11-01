#include "npp.h"
#include <cuda_runtime.h>


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
template <typename Src>
__global__ void initLabels_kernel(const Src *pSrc, int nSrcStep,
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
template <typename Src>
__global__ void localMerge4Way_kernel(const Src *pSrc, int nSrcStep,
                                       Npp32u *pLabels, int nLabelsStep,
                                       int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  const Src *src_row = (const Src *)((const char *)pSrc + y * nSrcStep);
  Src myValue = src_row[x];

  Npp32u *labels_row = (Npp32u *)((char *)pLabels + y * nLabelsStep);
  Npp32u label = labels_row[x];

  // Check right neighbor (merge only if same value)
  if (x + 1 < width && src_row[x + 1] == myValue) {
    Npp32u rightLabel = labels_row[x + 1];
    union2D(pLabels, nLabelsStep, width, label, rightLabel);
  }

  // Check bottom neighbor (merge only if same value)
  if (y + 1 < height) {
    const Src *next_src_row = (const Src *)((const char *)pSrc + (y + 1) * nSrcStep);
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
template <typename Src>
__global__ void localMerge8Way_kernel(const Src *pSrc, int nSrcStep,
                                       Npp32u *pLabels, int nLabelsStep,
                                       int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  const Src *src_row = (const Src *)((const char *)pSrc + y * nSrcStep);
  Src myValue = src_row[x];

  Npp32u *labels_row = (Npp32u *)((char *)pLabels + y * nLabelsStep);
  Npp32u label = labels_row[x];

  // Right neighbor (merge only if same value)
  if (x + 1 < width && src_row[x + 1] == myValue) {
    Npp32u rightLabel = labels_row[x + 1];
    union2D(pLabels, nLabelsStep, width, label, rightLabel);
  }

  if (y + 1 < height) {
    const Src *next_src_row = (const Src *)((const char *)pSrc + (y + 1) * nSrcStep);
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
template <typename Src>
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
template <typename Src>
__global__ void batchLocalMerge8Way_kernel(const NppiImageDescriptor *pSrcBatchList,
                                            NppiImageDescriptor *pDstBatchList,
                                            int batchSize, int maxWidth, int maxHeight) {
  int imgIdx = blockIdx.z;
  if (imgIdx >= batchSize) return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  NppiSize imgSize = pDstBatchList[imgIdx].oSize;
  if (x >= imgSize.width || y >= imgSize.height) return;

  const Src *pSrc = (const Src *)pSrcBatchList[imgIdx].pData;
  int nSrcStep = pSrcBatchList[imgIdx].nStep;
  Npp32u *pLabels = (Npp32u *)pDstBatchList[imgIdx].pData;
  int nLabelsStep = pDstBatchList[imgIdx].nStep;
  int width = imgSize.width;
  int height = imgSize.height;

  const Src *src_row = (const Src *)((const char *)pSrc + y * nSrcStep);
  Src myValue = src_row[x];

  Npp32u *labels_row = (Npp32u *)((char *)pLabels + y * nLabelsStep);
  Npp32u label = labels_row[x];

  // Right neighbor (merge only if same value)
  if (x + 1 < width && src_row[x + 1] == myValue) {
    Npp32u rightLabel = labels_row[x + 1];
    union2D(pLabels, nLabelsStep, width, label, rightLabel);
  }

  if (y + 1 < height) {
    const Src *next_src_row = (const Src *)((const char *)pSrc + (y + 1) * nSrcStep);
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

// Batch processing kernel: local merge for multiple images (4-way)
template <typename Src>
__global__ void batchLocalMerge4Way_kernel(const NppiImageDescriptor *pSrcBatchList,
                                            NppiImageDescriptor *pDstBatchList,
                                            int batchSize, int maxWidth, int maxHeight) {
  int imgIdx = blockIdx.z;
  if (imgIdx >= batchSize) return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  NppiSize imgSize = pDstBatchList[imgIdx].oSize;
  if (x >= imgSize.width || y >= imgSize.height) return;

  const Src *pSrc = (const Src *)pSrcBatchList[imgIdx].pData;
  int nSrcStep = pSrcBatchList[imgIdx].nStep;
  Npp32u *pLabels = (Npp32u *)pDstBatchList[imgIdx].pData;
  int nLabelsStep = pDstBatchList[imgIdx].nStep;
  int width = imgSize.width;
  int height = imgSize.height;

  const Src *src_row = (const Src *)((const char *)pSrc + y * nSrcStep);
  Src myValue = src_row[x];

  Npp32u *labels_row = (Npp32u *)((char *)pLabels + y * nLabelsStep);
  Npp32u label = labels_row[x];

  if (x + 1 < width && src_row[x + 1] == myValue) {
    union2D(pLabels, nLabelsStep, width, label, labels_row[x + 1]);
  }
  if (y + 1 < height) {
    const Src *next_src_row = (const Src *)((const char *)pSrc + (y + 1) * nSrcStep);
    Npp32u *next_row = (Npp32u *)((char *)pLabels + (y + 1) * nLabelsStep);
    if (next_src_row[x] == myValue) {
      union2D(pLabels, nLabelsStep, width, label, next_row[x]);
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

namespace {

// Shared implementation for the single-image label-markers variants.  Src is
// the source pixel type (8u / 16u / 32u); the destination label image is
// always 32u and the Union-Find lives inside it.
template <typename Src>
NppStatus labelMarkersSingleImpl(const Src *pSrc, int nSrcStep, Npp32u *pDst, int nDstStep,
                                 NppiSize oSizeROI, NppiNorm eNorm, Npp8u * /*pBuffer*/,
                                 NppStreamContext nppStreamCtx) {
  int width = oSizeROI.width;
  int height = oSizeROI.height;

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  // Step 1: initialize every pixel with its 1-based linear index.
  initLabels_kernel<Src><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrc, nSrcStep, pDst, nDstStep, width, height);

  // Step 2: local merge; L1 is 4-way, Inf is 8-way.
  const int numIterations = 20;
  for (int i = 0; i < numIterations; i++) {
    if (eNorm == nppiNormInf) {
      localMerge8Way_kernel<Src><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
          pSrc, nSrcStep, pDst, nDstStep, width, height);
    } else {
      localMerge4Way_kernel<Src><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
          pSrc, nSrcStep, pDst, nDstStep, width, height);
    }
    if (i % 4 == 3) {
      pathCompression_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
          pDst, nDstStep, width, height);
    }
  }

  for (int i = 0; i < 10; i++) {
    pathCompression_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pDst, nDstStep, width, height);
  }
  convertTo0Based_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pDst, nDstStep, width, height);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_SUCCESS;
}

template <typename Src>
NppStatus labelMarkersBatchImpl(const NppiImageDescriptor *pSrcBatchList,
                                NppiImageDescriptor *pDstBatchList, int nBatchSize,
                                NppiSize oMaxSizeROI, NppiNorm eNorm,
                                NppStreamContext nppStreamCtx) {
  int maxWidth = oMaxSizeROI.width;
  int maxHeight = oMaxSizeROI.height;

  // NPP documents the batch lists as device memory, but the differential
  // harness (and plenty of caller code) passes host arrays.  Stage them so the
  // kernels always see device memory.
  NppiImageDescriptor *dSrc = nullptr;
  NppiImageDescriptor *dDst = nullptr;
  const size_t listBytes = (size_t)nBatchSize * sizeof(NppiImageDescriptor);
  if (cudaMallocAsync(reinterpret_cast<void **>(&dSrc), listBytes, nppStreamCtx.hStream) !=
          cudaSuccess ||
      cudaMallocAsync(reinterpret_cast<void **>(&dDst), listBytes, nppStreamCtx.hStream) !=
          cudaSuccess) {
    if (dSrc != nullptr) cudaFreeAsync(dSrc, nppStreamCtx.hStream);
    return NPP_MEMORY_ALLOCATION_ERR;
  }
  if (cudaMemcpyAsync(dSrc, pSrcBatchList, listBytes, cudaMemcpyDefault, nppStreamCtx.hStream) !=
          cudaSuccess ||
      cudaMemcpyAsync(dDst, pDstBatchList, listBytes, cudaMemcpyDefault, nppStreamCtx.hStream) !=
          cudaSuccess) {
    cudaFreeAsync(dSrc, nppStreamCtx.hStream);
    cudaFreeAsync(dDst, nppStreamCtx.hStream);
    return NPP_MEMCPY_ERROR;
  }
  pSrcBatchList = dSrc;
  pDstBatchList = dDst;

  dim3 blockSize(16, 16);
  dim3 gridSize((maxWidth + blockSize.x - 1) / blockSize.x,
                (maxHeight + blockSize.y - 1) / blockSize.y, nBatchSize);

  batchInitLabels_kernel<Src><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pSrcBatchList, pDstBatchList, nBatchSize, maxWidth, maxHeight);

  const int numIterations = 20;
  for (int i = 0; i < numIterations; i++) {
    if (eNorm == nppiNormInf) {
      batchLocalMerge8Way_kernel<Src><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
          pSrcBatchList, pDstBatchList, nBatchSize, maxWidth, maxHeight);
    } else {
      batchLocalMerge4Way_kernel<Src><<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
          pSrcBatchList, pDstBatchList, nBatchSize, maxWidth, maxHeight);
    }
    if (i % 4 == 3) {
      batchPathCompression_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
          pDstBatchList, nBatchSize, maxWidth, maxHeight);
    }
  }

  for (int i = 0; i < 10; i++) {
    batchPathCompression_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pDstBatchList, nBatchSize, maxWidth, maxHeight);
  }
  batchConvertTo0Based_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
      pDstBatchList, nBatchSize, maxWidth, maxHeight);

  cudaError_t cudaStatus = cudaGetLastError();
  cudaFreeAsync(dSrc, nppStreamCtx.hStream);
  cudaFreeAsync(dDst, nppStreamCtx.hStream);
  if (cudaStatus != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_SUCCESS;
}

}  // namespace

extern "C" {

// Get required buffer size for label markers
NppStatus nppiLabelMarkersUFGetBufferSize_32u_C1R_Ctx_impl(NppiSize oSizeROI, int *hpBufferSize) {
  size_t imagePixels = (size_t)oSizeROI.width * oSizeROI.height;
  size_t bufferSize = imagePixels * sizeof(Npp32u) + 1024;
  size_t alignedSize = (bufferSize + 511) & ~511;
  *hpBufferSize = (int)alignedSize;
  return NPP_SUCCESS;
}

NppStatus nppiLabelMarkersUF_8u32u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp32u *pDst,
                                                 int nDstStep, NppiSize oSizeROI, NppiNorm eNorm,
                                                 Npp8u *pBuffer, NppStreamContext nppStreamCtx) {
  return labelMarkersSingleImpl<Npp8u>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm, pBuffer,
                                       nppStreamCtx);
}

NppStatus nppiLabelMarkersUF_16u32u_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp32u *pDst,
                                                  int nDstStep, NppiSize oSizeROI, NppiNorm eNorm,
                                                  Npp8u *pBuffer, NppStreamContext nppStreamCtx) {
  return labelMarkersSingleImpl<Npp16u>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm, pBuffer,
                                        nppStreamCtx);
}

NppStatus nppiLabelMarkersUF_32u_C1R_Ctx_impl(const Npp32u *pSrc, int nSrcStep, Npp32u *pDst,
                                               int nDstStep, NppiSize oSizeROI, NppiNorm eNorm,
                                               Npp8u *pBuffer, NppStreamContext nppStreamCtx) {
  return labelMarkersSingleImpl<Npp32u>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eNorm, pBuffer,
                                        nppStreamCtx);
}

NppStatus nppiLabelMarkersUFBatch_8u32u_C1R_Advanced_Ctx_impl(
    const NppiImageDescriptor *pSrcBatchList, NppiImageDescriptor *pDstBatchList, int nBatchSize,
    NppiSize oMaxSizeROI, NppiNorm eNorm, NppStreamContext nppStreamCtx) {
  return labelMarkersBatchImpl<Npp8u>(pSrcBatchList, pDstBatchList, nBatchSize, oMaxSizeROI, eNorm,
                                      nppStreamCtx);
}

NppStatus nppiLabelMarkersUFBatch_16u32u_C1R_Advanced_Ctx_impl(
    const NppiImageDescriptor *pSrcBatchList, NppiImageDescriptor *pDstBatchList, int nBatchSize,
    NppiSize oMaxSizeROI, NppiNorm eNorm, NppStreamContext nppStreamCtx) {
  return labelMarkersBatchImpl<Npp16u>(pSrcBatchList, pDstBatchList, nBatchSize, oMaxSizeROI, eNorm,
                                       nppStreamCtx);
}

NppStatus nppiLabelMarkersUFBatch_32u_C1R_Advanced_Ctx_impl(
    const NppiImageDescriptor *pSrcBatchList, NppiImageDescriptor *pDstBatchList, int nBatchSize,
    NppiSize oMaxSizeROI, NppiNorm eNorm, NppStreamContext nppStreamCtx) {
  return labelMarkersBatchImpl<Npp32u>(pSrcBatchList, pDstBatchList, nBatchSize, oMaxSizeROI, eNorm,
                                       nppStreamCtx);
}

}  // extern "C"
