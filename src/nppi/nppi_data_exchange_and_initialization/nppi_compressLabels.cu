#include "npp.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/sequence.h>

/**
 * CUDA kernels for Label Compression using Union-Find
 */

// Union-Find数据结构
struct UnionFind {
    Npp32u* parent;
    int* rank;
    int size;
    
    __device__ UnionFind(Npp32u* p, int* r, int s) : parent(p), rank(r), size(s) {}
    
    __device__ Npp32u find(Npp32u x) {
        if (x >= size) return x;
        
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // 路径压缩
        }
        return parent[x];
    }
    
    __device__ void unite(Npp32u x, Npp32u y) {
        if (x >= size || y >= size) return;
        
        Npp32u rootX = find(x);
        Npp32u rootY = find(y);
        
        if (rootX != rootY) {
            // 按秩合并
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

// 初始化Union-Find结构
__global__ void initUnionFind_kernel(Npp32u* parent, int* rank, int maxLabels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < maxLabels) {
        parent[idx] = idx;
        rank[idx] = 0;
    }
}

// 第一遍：建立连通性
__global__ void buildConnectivity_kernel(const Npp32u* pMarkerLabels, int nMarkerLabelsStep,
                                         int width, int height, Npp32u* parent, int* rank) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const Npp32u* current_row = (const Npp32u*)((const char*)pMarkerLabels + y * nMarkerLabelsStep);
    Npp32u currentLabel = current_row[x];
    
    if (currentLabel == 0) return;  // 跳过背景
    
    UnionFind uf(parent, rank, 65536);  // 假设最大标签数为65536
    
    // 检查4连通邻域
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};
    
    for (int i = 0; i < 4; i++) {
        int nx = x + dx[i];
        int ny = y + dy[i];
        
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            const Npp32u* neighbor_row = (const Npp32u*)((const char*)pMarkerLabels + ny * nMarkerLabelsStep);
            Npp32u neighborLabel = neighbor_row[nx];
            
            if (neighborLabel != 0 && neighborLabel == currentLabel) {
                uf.unite(currentLabel, neighborLabel);
            }
        }
    }
}

// 路径压缩优化
__global__ void pathCompression_kernel(Npp32u* parent, int maxLabels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < maxLabels) {
        UnionFind uf(parent, nullptr, maxLabels);
        parent[idx] = uf.find(idx);
    }
}

// 收集唯一根标签
__global__ void collectUniqueRoots_kernel(const Npp32u* pMarkerLabels, int nMarkerLabelsStep,
                                          int width, int height, const Npp32u* parent,
                                          Npp32u* uniqueLabels, bool* labelUsed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const Npp32u* current_row = (const Npp32u*)((const char*)pMarkerLabels + y * nMarkerLabelsStep);
    Npp32u label = current_row[x];
    
    if (label != 0) {
        Npp32u root = parent[label];
        if (root < 65536 && !labelUsed[root]) {
            labelUsed[root] = true;
        }
    }
}

// 重新标记
__global__ void relabelImage_kernel(Npp32u* pMarkerLabels, int nMarkerLabelsStep,
                                   int width, int height, const Npp32u* parent,
                                   const Npp32u* labelMapping, int startingNumber) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    Npp32u* current_row = (Npp32u*)((char*)pMarkerLabels + y * nMarkerLabelsStep);
    Npp32u label = current_row[x];
    
    if (label != 0) {
        Npp32u root = parent[label];
        if (root < 65536) {
            // 通过labelMapping找到新标签
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

// 获取标签压缩所需缓冲区大小
NppStatus nppiCompressMarkerLabelsGetBufferSize_32u_C1R_Ctx_cuda(int nMarkerLabels, int* hpBufferSize) {
    // Union-Find需要的空间：
    // 1. parent数组 (Npp32u * maxLabels)
    // 2. rank数组 (int * maxLabels) 
    // 3. 临时标签数组 (Npp32u * maxLabels)
    // 4. 标签使用标记 (bool * maxLabels)
    
    int maxLabels = 65536;  // 假设最大标签数
    size_t parentSize = maxLabels * sizeof(Npp32u);
    size_t rankSize = maxLabels * sizeof(int);
    size_t tempLabelSize = maxLabels * sizeof(Npp32u);
    size_t usedSize = maxLabels * sizeof(bool);
    
    size_t totalSize = parentSize + rankSize + tempLabelSize + usedSize;
    size_t alignedSize = (totalSize + 511) & ~511;  // 512字节对齐
    
    *hpBufferSize = (int)alignedSize;
    return NPP_SUCCESS;
}

// Union-Find标签压缩实现
NppStatus nppiCompressMarkerLabelsUF_32u_C1IR_Ctx_cuda(Npp32u* pMarkerLabels, int nMarkerLabelsStep,
                                                       NppiSize oMarkerLabelsROI, int nStartingNumber,
                                                       int* pNewMarkerLabelsNumber, Npp8u* pDeviceBuffer, 
                                                       NppStreamContext nppStreamCtx) {
    int width = oMarkerLabelsROI.width;
    int height = oMarkerLabelsROI.height;
    int maxLabels = 65536;
    
    // 设置缓冲区
    Npp32u* parent = (Npp32u*)pDeviceBuffer;
    int* rank = (int*)(parent + maxLabels);
    Npp32u* labelMapping = (Npp32u*)(rank + maxLabels);
    bool* labelUsed = (bool*)(labelMapping + maxLabels);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    dim3 linearBlockSize(256);
    dim3 linearGridSize((maxLabels + linearBlockSize.x - 1) / linearBlockSize.x);
    
    // 第一步：初始化Union-Find
    initUnionFind_kernel<<<linearGridSize, linearBlockSize, 0, nppStreamCtx.hStream>>>(
        parent, rank, maxLabels);
    
    // 第二步：建立连通性
    buildConnectivity_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pMarkerLabels, nMarkerLabelsStep, width, height, parent, rank);
    
    // 第三步：路径压缩
    for (int iter = 0; iter < 10; iter++) {  // 多次迭代确保完全压缩
        pathCompression_kernel<<<linearGridSize, linearBlockSize, 0, nppStreamCtx.hStream>>>(
            parent, maxLabels);
    }
    
    // 第四步：初始化标签使用标记
    cudaMemsetAsync(labelUsed, 0, maxLabels * sizeof(bool), nppStreamCtx.hStream);
    
    // 第五步：收集唯一根标签
    collectUniqueRoots_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pMarkerLabels, nMarkerLabelsStep, width, height, parent, labelMapping, labelUsed);
    
    // 第六步：在CPU上处理标签映射（简化版）
    std::vector<bool> h_labelUsed(maxLabels);
    std::vector<char> h_labelUsedChar(maxLabels);  // 使用char而非bool避免vector<bool>问题
    cudaMemcpyAsync(h_labelUsedChar.data(), labelUsed, maxLabels * sizeof(bool), 
                    cudaMemcpyDeviceToHost, nppStreamCtx.hStream);
    cudaStreamSynchronize(nppStreamCtx.hStream);
    
    std::vector<Npp32u> h_labelMapping(maxLabels, 0);
    int newLabelCount = 0;
    
    // 转换char到bool
    for (int i = 0; i < maxLabels; i++) {
        h_labelUsed[i] = (h_labelUsedChar[i] != 0);
    }
    
    for (int i = 1; i < maxLabels; i++) {  // 跳过0（背景）
        if (h_labelUsed[i]) {
            h_labelMapping[newLabelCount] = i;
            newLabelCount++;
        }
    }
    
    *pNewMarkerLabelsNumber = newLabelCount;
    
    // 第七步：将映射拷贝回GPU
    cudaMemcpyAsync(labelMapping, h_labelMapping.data(), maxLabels * sizeof(Npp32u),
                    cudaMemcpyHostToDevice, nppStreamCtx.hStream);
    
    // 第八步：重新标记图像
    relabelImage_kernel<<<gridSize, blockSize, 0, nppStreamCtx.hStream>>>(
        pMarkerLabels, nMarkerLabelsStep, width, height, parent, labelMapping, nStartingNumber);
    
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    return NPP_SUCCESS;
}

} // extern "C"