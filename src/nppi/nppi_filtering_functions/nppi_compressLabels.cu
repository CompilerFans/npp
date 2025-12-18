#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

// Kernel: Collect all unique labels from the image
// Note: label 0 is valid (foreground pixels), so we collect all labels including 0
__global__ void collectLabels_kernel(const Npp32u *pMarkerLabels, int nStep, int width, int height,
                                     Npp32u *labelFlags, int maxLabels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp32u *row = (const Npp32u *)((const char *)pMarkerLabels + y * nStep);
  Npp32u label = row[x];

  // Collect all labels (including 0) that are within range
  if (label < (Npp32u)maxLabels) {
    labelFlags[label] = 1;
  }
}

// Kernel: Build compressed label mapping using prefix sum results
__global__ void buildMapping_kernel(const Npp32u *labelFlags, const Npp32u *prefixSum,
                                    Npp32u *labelMapping, int maxLabels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= maxLabels)
    return;

  if (labelFlags[idx] == 1) {
    // This label exists, map it to its compressed value
    // prefixSum[idx] gives the count of labels before this one
    labelMapping[idx] = prefixSum[idx];
  } else {
    labelMapping[idx] = 0;
  }
}

// Kernel: Apply the compressed label mapping to the image
// Note: label 0 is valid, so we apply mapping to all labels including 0
__global__ void applyMapping_kernel(Npp32u *pMarkerLabels, int nStep, int width, int height,
                                    const Npp32u *labelMapping, int maxLabels) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  Npp32u *row = (Npp32u *)((char *)pMarkerLabels + y * nStep);
  Npp32u label = row[x];

  if (label < (Npp32u)maxLabels) {
    row[x] = labelMapping[label];
  }
}

// Simple prefix sum kernel (for small arrays, run on single block)
__global__ void prefixSum_kernel(const Npp32u *input, Npp32u *output, int n) {
  extern __shared__ Npp32u temp[];

  int tid = threadIdx.x;
  int offset = 1;

  // Load input into shared memory
  if (tid < n)
    temp[tid] = input[tid];
  else
    temp[tid] = 0;

  // Build sum in place up the tree
  for (int d = n >> 1; d > 0; d >>= 1) {
    __syncthreads();
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      if (bi < n)
        temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  // Clear the last element
  if (tid == 0)
    temp[n - 1] = 0;

  // Traverse down tree & build scan
  for (int d = 1; d < n; d *= 2) {
    offset >>= 1;
    __syncthreads();
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      if (bi < n) {
        Npp32u t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
      }
    }
  }

  __syncthreads();

  // Write results to output (exclusive scan, so add 1 for 1-based labels)
  if (tid < n)
    output[tid] = temp[tid] + 1;
}

// Large array prefix sum using multiple blocks
__global__ void blockPrefixSum_kernel(const Npp32u *input, Npp32u *output, Npp32u *blockSums,
                                       int n, int blockSize) {
  extern __shared__ Npp32u sdata[];

  int tid = threadIdx.x;
  int blockId = blockIdx.x;
  int globalIdx = blockId * blockSize + tid;

  // Load data
  sdata[tid] = (globalIdx < n) ? input[globalIdx] : 0;
  __syncthreads();

  // Reduce within block
  for (int stride = 1; stride < blockSize; stride *= 2) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index < blockSize) {
      sdata[index] += sdata[index - stride];
    }
    __syncthreads();
  }

  // Store block sum and clear last element
  if (tid == blockSize - 1) {
    if (blockSums) blockSums[blockId] = sdata[tid];
    sdata[tid] = 0;
  }
  __syncthreads();

  // Down-sweep
  for (int stride = blockSize / 2; stride > 0; stride /= 2) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index < blockSize) {
      Npp32u temp = sdata[index - stride];
      sdata[index - stride] = sdata[index];
      sdata[index] += temp;
    }
    __syncthreads();
  }

  // Write output
  if (globalIdx < n) {
    output[globalIdx] = sdata[tid] + 1; // 1-based labels
  }
}

__global__ void addBlockOffsets_kernel(Npp32u *data, const Npp32u *blockOffsets, int n, int blockSize) {
  int blockId = blockIdx.x;
  int globalIdx = blockId * blockSize + threadIdx.x;

  if (globalIdx < n && blockId > 0) {
    data[globalIdx] += blockOffsets[blockId];
  }
}

extern "C" {

// Get required buffer size
NppStatus nppiCompressMarkerLabelsGetBufferSize_32u_C1R_Ctx_impl(int nStartingNumber, int *hpBufferSize) {
  // Buffer layout:
  // 1. labelFlags array (Npp32u * maxLabels) - marks which labels exist
  // 2. labelMapping array (Npp32u * maxLabels) - compressed label mapping

  size_t maxLabels = (size_t)nStartingNumber + 1; // +1 for safety

  size_t flagsSize = maxLabels * sizeof(Npp32u);
  size_t mappingSize = maxLabels * sizeof(Npp32u);

  size_t totalSize = flagsSize + mappingSize;
  size_t alignedSize = (totalSize + 511) & ~511; // 512-byte alignment

  *hpBufferSize = (int)alignedSize;
  return NPP_SUCCESS;
}

// Compress marker labels implementation
NppStatus nppiCompressMarkerLabelsUF_32u_C1IR_Ctx_impl(Npp32u *pMarkerLabels, int nMarkerLabelsStep,
                                                       NppiSize oMarkerLabelsROI, int nStartingNumber,
                                                       int *pNewMarkerLabelsNumber, Npp8u *pDeviceBuffer,
                                                       NppStreamContext nppStreamCtx) {
  int width = oMarkerLabelsROI.width;
  int height = oMarkerLabelsROI.height;
  int maxLabels = nStartingNumber + 1; // nStartingNumber = width * height

  // Setup buffer pointers
  Npp32u *labelFlags = (Npp32u *)pDeviceBuffer;
  Npp32u *labelMapping = labelFlags + maxLabels;

  cudaStream_t stream = nppStreamCtx.hStream;

  // Step 1: Initialize labelFlags to 0
  cudaMemsetAsync(labelFlags, 0, maxLabels * sizeof(Npp32u), stream);

  // Step 2: Collect all unique labels from the image
  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                (height + blockSize.y - 1) / blockSize.y);

  collectLabels_kernel<<<gridSize, blockSize, 0, stream>>>(
      pMarkerLabels, nMarkerLabelsStep, width, height, labelFlags, maxLabels);

  // Step 3: Compute prefix sum to create compressed label mapping
  // For simplicity, use CPU-based approach for large arrays
  // Copy labelFlags to host, compute prefix sum, copy back

  std::vector<Npp32u> h_flags(maxLabels);
  std::vector<Npp32u> h_prefixSum(maxLabels);

  cudaMemcpyAsync(h_flags.data(), labelFlags, maxLabels * sizeof(Npp32u),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // Compute exclusive prefix sum on CPU
  Npp32u count = 0;
  for (int i = 0; i < maxLabels; i++) {
    if (h_flags[i] == 1) {
      count++;
      h_prefixSum[i] = count; // 1-based label
    } else {
      h_prefixSum[i] = 0;
    }
  }

  *pNewMarkerLabelsNumber = count;

  // Copy prefix sum back to device as labelMapping
  cudaMemcpyAsync(labelMapping, h_prefixSum.data(), maxLabels * sizeof(Npp32u),
                  cudaMemcpyHostToDevice, stream);

  // Step 4: Apply the mapping to relabel the image
  applyMapping_kernel<<<gridSize, blockSize, 0, stream>>>(
      pMarkerLabels, nMarkerLabelsStep, width, height, labelMapping, maxLabels);

  cudaError_t cudaStatus = cudaStreamSynchronize(stream);
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

} // extern "C"
