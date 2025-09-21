#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

// Union-Find data structure
struct UnionFind {
  Npp32u *parent;
  int *rank;
  int size;

  __device__ UnionFind(Npp32u *p, int *r, int s) : parent(p), rank(r), size(s) {}

  __device__ Npp32u find(Npp32u x) {
    if (x >= size)
      return x;

    // Iterative path compression to avoid recursive calls
    Npp32u root = x;
    while (parent[root] != root) {
      root = parent[root];
    }

    // Path compression: point all nodes on path directly to root
    while (parent[x] != x) {
      Npp32u next = parent[x];
      parent[x] = root;
      x = next;
    }

    return root;
  }

  __device__ void unite(Npp32u x, Npp32u y) {
    if (x >= size || y >= size)
      return;

    Npp32u rootX = find(x);
    Npp32u rootY = find(y);

    if (rootX != rootY) {
      // Union by rank
      if (rank[rootX] < rank[rootY]) {
        parent[rootX] = rootY;
      } else if (rank[rootX] > rank[rootY]) {
        parent[rootY] = rootX;
      } else {
        parent[rootY] = rootX;
        rank[rootX]++;
      }
    }
  }
};

// Initialize Union-Find structure

__global__ void initUnionFind_kernel(Npp32u *parent, int *rank, int maxLabels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < maxLabels) {
    parent[idx] = idx;
    rank[idx] = 0;
  }
}

// First pass: establish connectivity
__global__ void buildConnectivity_kernel(const Npp32u *pMarkerLabels, int nMarkerLabelsStep, int width, int height,
                                         Npp32u *parent, int *rank) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp32u *current_row = (const Npp32u *)((const char *)pMarkerLabels + y * nMarkerLabelsStep);
  Npp32u currentLabel = current_row[x];

  if (currentLabel == 0)
    return; // Skip background

  UnionFind uf(parent, rank, 65536); // Assume max label count is 65536

  // Check 4-connected neighborhood
  int dx[] = {-1, 1, 0, 0};
  int dy[] = {0, 0, -1, 1};

  for (int i = 0; i < 4; i++) {
    int nx = x + dx[i];
    int ny = y + dy[i];

    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
      const Npp32u *neighbor_row = (const Npp32u *)((const char *)pMarkerLabels + ny * nMarkerLabelsStep);
      Npp32u neighborLabel = neighbor_row[nx];

      if (neighborLabel != 0 && neighborLabel == currentLabel) {
        uf.unite(currentLabel, neighborLabel);
      }
    }
  }
}

// Path compression optimization

__global__ void pathCompression_kernel(Npp32u *parent, int maxLabels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < maxLabels) {
    UnionFind uf(parent, nullptr, maxLabels);
    parent[idx] = uf.find(idx);
  }
}

// Collect unique root labels
__global__ void collectUniqueRoots_kernel(const Npp32u *pMarkerLabels, int nMarkerLabelsStep, int width, int height,
                                          const Npp32u *parent, Npp32u *uniqueLabels, bool *labelUsed) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  const Npp32u *current_row = (const Npp32u *)((const char *)pMarkerLabels + y * nMarkerLabelsStep);
  Npp32u label = current_row[x];

  if (label != 0) {
    Npp32u root = parent[label];
    if (root < 65536 && !labelUsed[root]) {
      labelUsed[root] = true;
    }
  }
}

// Relabel
__global__ void relabelImage_kernel(Npp32u *pMarkerLabels, int nMarkerLabelsStep, int width, int height,
                                    const Npp32u *parent, const Npp32u *labelMapping, int startingNumber) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  Npp32u *current_row = (Npp32u *)((char *)pMarkerLabels + y * nMarkerLabelsStep);
  Npp32u label = current_row[x];

  if (label != 0) {
    Npp32u root = parent[label];
    if (root < 65536) {
      // Find new label through labelMapping
      for (int i = 0; i < 65536; i++) {
        if (labelMapping[i] == root) {
          current_row[x] = startingNumber + i;
          return;
        }
      }
    }
  }
}

extern "C" {

// Get required buffer size

NppStatus nppiCompressMarkerLabelsGetBufferSize_32u_C1R_Ctx_impl(int nMarkerLabels, int *hpBufferSize) {
  // Space required for Union-Find:
  // 1. parent array (Npp32u * maxLabels)
  // 2. rank array (int * maxLabels)
  // 3. temporary label array (Npp32u * maxLabels)
  // 4. label usage markers (bool * maxLabels)

  int maxLabels = 65536; // Assume max label count
  size_t parentSize = maxLabels * sizeof(Npp32u);
  size_t rankSize = maxLabels * sizeof(int);
  size_t tempLabelSize = maxLabels * sizeof(Npp32u);
  size_t usedSize = maxLabels * sizeof(bool);

  size_t totalSize = parentSize + rankSize + tempLabelSize + usedSize;
  size_t alignedSize = (totalSize + 511) & ~511; // 512byte alignment

  *hpBufferSize = (int)alignedSize;
  return NPP_SUCCESS;
}

// Union-Find标签压缩implementation
NppStatus nppiCompressMarkerLabelsUF_32u_C1IR_Ctx_impl(Npp32u *pMarkerLabels, int nMarkerLabelsStep,
                                                       NppiSize oMarkerLabelsROI, int nStartingNumber,
                                                       int *pNewMarkerLabelsNumber, Npp8u *pDeviceBuffer,
                                                       NppStreamContext nppStreamCtx) {
  int width = oMarkerLabelsROI.width;
  int height = oMarkerLabelsROI.height;
  int maxLabels = 65536;

  // Setup buffers
  Npp32u *parent = (Npp32u *)pDeviceBuffer;
  int *rank = (int *)(parent + maxLabels);
  Npp32u *labelMapping = (Npp32u *)(rank + maxLabels);
  bool *labelUsed = (bool *)(labelMapping + maxLabels);

  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

  dim3 linearBlockSize(256);
  dim3 linearGridSize((maxLabels + linearBlockSize.x - 1) / linearBlockSize.x);

  // 第一步：初始化Union-Find
  initUnionFind_kernel<<<linearGridSize, linearBlockSize, 0, nppStreamCtx.hStream>>>(parent, rank, maxLabels);

  // 第二步：建立连通性
  buildConnectivity_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pMarkerLabels, nMarkerLabelsStep, width,
                                                                             height, parent, rank);

  // 第三步：路径压缩
  for (int iter = 0; iter < 10; iter++) { // 多次迭代确保完全压缩
    pathCompression_kernel<<<linearGridSize, linearBlockSize, 0, nppStreamCtx.hStream>>>(parent, maxLabels);
  }

  // 第四步：初始化label usage markers
  cudaMemsetAsync(labelUsed, 0, maxLabels * sizeof(bool), nppStreamCtx.hStream);

  // 第五步：Collect unique root labels
  collectUniqueRoots_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pMarkerLabels, nMarkerLabelsStep, width,
                                                                              height, parent, labelMapping, labelUsed);

  // 第六步：在CPU上处理标签映射（简化版）
  std::vector<bool> h_labelUsed(maxLabels);
  std::vector<char> h_labelUsedChar(maxLabels); // 使用char而非bool避免vector<bool>问题
  cudaMemcpyAsync(h_labelUsedChar.data(), labelUsed, maxLabels * sizeof(bool), cudaMemcpyDeviceToHost,
                  nppStreamCtx.hStream);
  cudaStreamSynchronize(nppStreamCtx.hStream);

  std::vector<Npp32u> h_labelMapping(maxLabels, 0);
  int newLabelCount = 0;

  // 转换char到bool
  for (int i = 0; i < maxLabels; i++) {
    h_labelUsed[i] = (h_labelUsedChar[i] != 0);
  }

  for (int i = 1; i < maxLabels; i++) { // 跳过0（背景）
    if (h_labelUsed[i]) {
      h_labelMapping[newLabelCount] = i;
      newLabelCount++;
    }
  }

  *pNewMarkerLabelsNumber = newLabelCount;

  // 第七步：将映射拷贝回GPU
  cudaMemcpyAsync(labelMapping, h_labelMapping.data(), maxLabels * sizeof(Npp32u), cudaMemcpyHostToDevice,
                  nppStreamCtx.hStream);

  // 第八步：Relabel图像
  relabelImage_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(pMarkerLabels, nMarkerLabelsStep, width, height,
                                                                        parent, labelMapping, nStartingNumber);

  cudaError_t cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}
}
