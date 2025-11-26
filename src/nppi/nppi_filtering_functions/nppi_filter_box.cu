#include "nppdefs.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void nppiFilterBox_8u_C1R_kernel_impl(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                                 int width, int height, int maskWidth, int maskHeight, int anchorX,
                                                 int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    int startX = x - anchorX;
    int startY = y - anchorY;
    int endX = startX + maskWidth;
    int endY = startY + maskHeight;

    int sum = 0;
    int totalMaskPixels = maskWidth * maskHeight;

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
        // Check if coordinates are within image bounds
        if (i >= 0 && i < width && j >= 0 && j < height) {
          const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
          sum += pSrcRow[i];
        }
        // Out-of-bounds
      }
    }

    // Compute average using total mask size with truncation (toward zero)
    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);
    pDstRow[x] = sum / totalMaskPixels; // (truncation)
  }
}

// Single-pass 2D convolution kernel for small odd square masks
// Uses 2D tiled shared memory with halo for efficient computation
template<int BLOCK_W, int BLOCK_H, int RADIUS>
__global__ void filterBoxOptimized_8u_C1R(
    const Npp8u* __restrict__ pSrc, int nSrcStep,
    Npp8u* __restrict__ pDst, int nDstStep,
    int width, int height, int maskWidth, int maskHeight)
{
    extern __shared__ Npp8u sharedMem[];

    const int SHARED_W = BLOCK_W + 2 * RADIUS;
    const int SHARED_H = BLOCK_H + 2 * RADIUS;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * BLOCK_W + tx;
    const int y = blockIdx.y * BLOCK_H + ty;

    // Cooperatively load tile with halo into shared memory with zero padding
    for (int sy = ty; sy < SHARED_H; sy += BLOCK_H) {
        for (int sx = tx; sx < SHARED_W; sx += BLOCK_W) {
            int srcX = blockIdx.x * BLOCK_W + sx - RADIUS;
            int srcY = blockIdx.y * BLOCK_H + sy - RADIUS;

            // Zero padding for out-of-bounds pixels
            Npp8u value = 0;
            if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
                const Npp8u* srcRow = (const Npp8u*)((const char*)pSrc + srcY * nSrcStep);
                value = srcRow[srcX];
            }

            sharedMem[sy * SHARED_W + sx] = value;
        }
    }

    __syncthreads();

    if (x < width && y < height) {
        int sum = 0;
        const int baseIdx = (ty + RADIUS) * SHARED_W + (tx + RADIUS);

        #pragma unroll
        for (int dy = -RADIUS; dy <= RADIUS; dy++) {
            #pragma unroll
            for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                sum += sharedMem[baseIdx + dy * SHARED_W + dx];
            }
        }

        Npp8u* dstRow = (Npp8u*)((char*)pDst + y * nDstStep);
        dstRow[x] = sum / (maskWidth * maskHeight);
    }
}

// Each thread processes 2x2 tile for maximum shared memory reuse
template<int BLOCK_W, int BLOCK_H>
__global__ void filterBoxOptimized_8u_C1R_3x3(
    const Npp8u* __restrict__ pSrc, int nSrcStep,
    Npp8u* __restrict__ pDst, int nDstStep,
    int width, int height)
{
    const int RADIUS = 1;
    const int TILE_SIZE = 2;
    const int SHARED_W = BLOCK_W * TILE_SIZE + 2 * RADIUS;
    const int SHARED_H = BLOCK_H * TILE_SIZE + 2 * RADIUS;

    // add padding to avoid bank conflict
    __shared__ Npp8u smem[SHARED_H][SHARED_W + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x_base = (blockIdx.x * BLOCK_W + tx) * TILE_SIZE;
    const int y_base = (blockIdx.y * BLOCK_H + ty) * TILE_SIZE;

    // Cooperative loading with vectorization
    const int tid = ty * BLOCK_W + tx;
    const int totalThreads = BLOCK_W * BLOCK_H;

    #pragma unroll
    for (int i = tid; i < SHARED_W * SHARED_H; i += totalThreads) {
        int sy = i / SHARED_W;
        int sx = i % SHARED_W;

        int srcX = blockIdx.x * BLOCK_W * TILE_SIZE + sx - RADIUS;
        int srcY = blockIdx.y * BLOCK_H * TILE_SIZE + sy - RADIUS;

        Npp8u value = 0;
        if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
            value = __ldg((const Npp8u*)((const char*)pSrc + srcY * nSrcStep) + srcX);
        }

        smem[sy][sx] = value;
    }

    __syncthreads();

    // Process 2x2 tile with optimized memory access
    #pragma unroll
    for (int dy = 0; dy < TILE_SIZE; dy++) {
        #pragma unroll
        for (int dx = 0; dx < TILE_SIZE; dx++) {
            int x = x_base + dx;
            int y = y_base + dy;

            if (x < width && y < height) {
                const int base_y = ty * TILE_SIZE + dy + RADIUS;
                const int base_x = tx * TILE_SIZE + dx + RADIUS;

                int sum = smem[base_y - 1][base_x - 1] +
                          smem[base_y - 1][base_x]     +
                          smem[base_y - 1][base_x + 1] +
                          smem[base_y][base_x - 1]     +
                          smem[base_y][base_x]         +
                          smem[base_y][base_x + 1]     +
                          smem[base_y + 1][base_x - 1] +
                          smem[base_y + 1][base_x]     +
                          smem[base_y + 1][base_x + 1];

                Npp8u* dstRow = (Npp8u*)((char*)pDst + y * nDstStep);
                dstRow[x] = sum / 9;
            }
        }
    }
}

// Each thread processes 2x2 tile with row caching
template<int BLOCK_W, int BLOCK_H>
__global__ void filterBoxOptimized_8u_C1R_5x5(
    const Npp8u* __restrict__ pSrc, int nSrcStep,
    Npp8u* __restrict__ pDst, int nDstStep,
    int width, int height)
{
    const int RADIUS = 2;
    const int PAD = 1;
    const int TILE_SIZE = 2;
    const int SHARED_W = BLOCK_W * TILE_SIZE + 2 * RADIUS + PAD;
    const int SHARED_H = BLOCK_H * TILE_SIZE + 2 * RADIUS;

    __shared__ Npp8u smem[SHARED_H][SHARED_W];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x_base = (blockIdx.x * BLOCK_W + tx) * TILE_SIZE;
    const int y_base = (blockIdx.y * BLOCK_H + ty) * TILE_SIZE;

    // Cooperative loading
    const int tid = ty * BLOCK_W + tx;
    const int totalPixels = (BLOCK_W * TILE_SIZE + 2 * RADIUS) * SHARED_H;

    for (int i = tid; i < totalPixels; i += BLOCK_W * BLOCK_H) {
        int sy = i / (BLOCK_W * TILE_SIZE + 2 * RADIUS);
        int sx = i % (BLOCK_W * TILE_SIZE + 2 * RADIUS);

        int srcX = blockIdx.x * BLOCK_W * TILE_SIZE + sx - RADIUS;
        int srcY = blockIdx.y * BLOCK_H * TILE_SIZE + sy - RADIUS;

        Npp8u value = 0;
        if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
            const Npp8u* srcRow = (const Npp8u*)((const char*)pSrc + srcY * nSrcStep);
            value = __ldg(&srcRow[srcX]);
        }

        smem[sy][sx] = value;
    }

    __syncthreads();

    // Process 2x2 tile - cache rows in registers for better performance
    #pragma unroll
    for (int dy = 0; dy < TILE_SIZE; dy++) {
        #pragma unroll
        for (int dx = 0; dx < TILE_SIZE; dx++) {
            int x = x_base + dx;
            int y = y_base + dy;

            if (x < width && y < height) {
                const int base_y = ty * TILE_SIZE + dy + RADIUS;
                const int base_x = tx * TILE_SIZE + dx + RADIUS;

                // Cache 5 rows in registers
                int r0 = smem[base_y-2][base_x-2] + smem[base_y-2][base_x-1] + smem[base_y-2][base_x] + smem[base_y-2][base_x+1] + smem[base_y-2][base_x+2];
                int r1 = smem[base_y-1][base_x-2] + smem[base_y-1][base_x-1] + smem[base_y-1][base_x] + smem[base_y-1][base_x+1] + smem[base_y-1][base_x+2];
                int r2 = smem[base_y  ][base_x-2] + smem[base_y  ][base_x-1] + smem[base_y  ][base_x] + smem[base_y  ][base_x+1] + smem[base_y  ][base_x+2];
                int r3 = smem[base_y+1][base_x-2] + smem[base_y+1][base_x-1] + smem[base_y+1][base_x] + smem[base_y+1][base_x+1] + smem[base_y+1][base_x+2];
                int r4 = smem[base_y+2][base_x-2] + smem[base_y+2][base_x-1] + smem[base_y+2][base_x] + smem[base_y+2][base_x+1] + smem[base_y+2][base_x+2];

                int sum = r0 + r1 + r2 + r3 + r4;

                Npp8u* dstRow = (Npp8u*)((char*)pDst + y * nDstStep);
                dstRow[x] = sum / 25;
            }
        }
    }
}

// Box filter kernel for 4-channel 8-bit unsigned with zero padding
__global__ void nppiFilterBox_8u_C4R_kernel_impl(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                                 int width, int height, int maskWidth, int maskHeight, int anchorX,
                                                 int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    // Calculate filter coverage area
    int startX = x - anchorX;
    int startY = y - anchorY;
    int endX = startX + maskWidth;
    int endY = startY + maskHeight;

    int sum[4] = {0, 0, 0, 0};
    int totalMaskPixels = maskWidth * maskHeight;

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
        // Check if coordinates are within image bounds
        if (i >= 0 && i < width && j >= 0 && j < height) {
          const Npp8u *pSrcRow = (const Npp8u *)((const char *)pSrc + j * nSrcStep);
          for (int c = 0; c < 4; c++) {
            sum[c] += pSrcRow[i * 4 + c];
          }
        }
        // Out-of-bounds pixels contribute 0 to sum for all channels (zero padding)
      }
    }

    Npp8u *pDstRow = (Npp8u *)((char *)pDst + y * nDstStep);
    for (int c = 0; c < 4; c++) {
      pDstRow[x * 4 + c] = sum[c] / totalMaskPixels;
    }
  }
}

// Specialized 3x3 box filter with additional optimizations
__global__ void nppiFilterBox_8u_C4R_shared_3x3(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                             int width, int height) {
    const int MASK_WIDTH = 3;
    const int MASK_HEIGHT = 3;
    const int ANCHOR_X = 1;
    const int ANCHOR_Y = 1;
    const int TILE_WIDTH = blockDim.x + MASK_WIDTH - 1;
    const int TILE_HEIGHT = blockDim.y + MASK_HEIGHT - 1;

    extern __shared__ uchar4 shared_mem[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tile_x = blockIdx.x * blockDim.x;
    int tile_y = blockIdx.y * blockDim.y;

    // load data to shared memory
    for (int i = ty; i < TILE_HEIGHT; i += blockDim.y) {
        for (int j = tx; j < TILE_WIDTH; j += blockDim.x) {
            int global_x = tile_x + j - ANCHOR_X;
            int global_y = tile_y + i - ANCHOR_Y;

            uchar4 value = make_uchar4(0, 0, 0, 0);
            if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height) {
                const uchar4 *src_row = (const uchar4 *)((const char *)pSrc + global_y * nSrcStep);
                value = src_row[global_x];
            }

            shared_mem[i * TILE_WIDTH + j] = value;
        }
    }

    __syncthreads();

    int out_x = tile_x + tx;
    int out_y = tile_y + ty;

    if (out_x < width && out_y < height) {
        // read form shared memory
        uchar4 p00 = shared_mem[ty * TILE_WIDTH + tx];
        uchar4 p01 = shared_mem[ty * TILE_WIDTH + tx + 1];
        uchar4 p02 = shared_mem[ty * TILE_WIDTH + tx + 2];
        uchar4 p10 = shared_mem[(ty + 1) * TILE_WIDTH + tx];
        uchar4 p11 = shared_mem[(ty + 1) * TILE_WIDTH + tx + 1];
        uchar4 p12 = shared_mem[(ty + 1) * TILE_WIDTH + tx + 2];
        uchar4 p20 = shared_mem[(ty + 2) * TILE_WIDTH + tx];
        uchar4 p21 = shared_mem[(ty + 2) * TILE_WIDTH + tx + 1];
        uchar4 p22 = shared_mem[(ty + 2) * TILE_WIDTH + tx + 2];

        uchar4 SumVal;
        SumVal.x = (p00.x + p01.x + p02.x + p10.x + p11.x + p12.x + p20.x + p21.x + p22.x) / 9;
        SumVal.y = (p00.y + p01.y + p02.y + p10.y + p11.y + p12.y + p20.y + p21.y + p22.y) / 9;
        SumVal.z = (p00.z + p01.z + p02.z + p10.z + p11.z + p12.z + p20.z + p21.z + p22.z) / 9;
        SumVal.w = (p00.w + p01.w + p02.w + p10.w + p11.w + p12.w + p20.w + p21.w + p22.w) / 9;

        uchar4 *dst_row = (uchar4 *)((char *)pDst + out_y * nDstStep);
        dst_row[out_x] = SumVal;
    }
}

// Optimized NxN box filter
template<int Mask_Width, int Mask_Heigth, int Anchor_X, int Anchor_Y>
__global__ void nppiFilterBox_8u_C4R_shared_NxN(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                             int width, int height) {
    const int MASK_WIDTH = Mask_Width;
    const int MASK_HEIGHT = Mask_Heigth;
    const int ANCHOR_X = Anchor_X;
    const int ANCHOR_Y = Anchor_Y;
    const int TILE_WIDTH = blockDim.x + MASK_WIDTH - 1;
    const int TILE_HEIGHT = blockDim.y + MASK_HEIGHT -1;

    extern __shared__ uchar4 shared_mem[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tile_x = blockIdx.x * blockDim.x;
    int tile_y = blockIdx.y * blockDim.y;

    // load data to shared memory
    for (int i = ty; i < TILE_HEIGHT; i += blockDim.y) {
        for (int j = tx; j < TILE_WIDTH; j += blockDim.x) {
            int global_x = tile_x + j - ANCHOR_X;
            int global_y = tile_y + i - ANCHOR_Y;

            uchar4 value = make_uchar4(0, 0, 0, 0);
            if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height) {
                const uchar4 *src_row = (const uchar4 *)((const char *)pSrc + global_y * nSrcStep);
                value = src_row[global_x];
            }

            shared_mem[i * TILE_WIDTH + j] = value;
        }
    }

    __syncthreads();

    int out_x = tile_x + tx;
    int out_y = tile_y + ty;

    if (out_x < width && out_y < height) {
        uint4 sumVal = make_uint4(0, 0, 0, 0);

        #pragma unroll
        for (int dy = 0; dy < Mask_Width; dy++) {
            #pragma unroll
            for (int dx = 0; dx < Mask_Heigth; dx++) {
                uchar4 pixel = shared_mem[(ty + dy) * TILE_WIDTH + (tx + dx)];
                sumVal.x += pixel.x;
                sumVal.y += pixel.y;
                sumVal.z += pixel.z;
                sumVal.w += pixel.w;
            }
        }
        uchar4 sum_value = make_uchar4(0, 0, 0, 0);
        sum_value.x = sumVal.x / (Mask_Width * Mask_Heigth);
        sum_value.y = sumVal.y / (Mask_Width * Mask_Heigth);
        sum_value.z = sumVal.z / (Mask_Width * Mask_Heigth);
        sum_value.w = sumVal.w / (Mask_Width * Mask_Heigth);

        uchar4 *dst_row = (uchar4 *)((char *)pDst + out_y * nDstStep);
        dst_row[out_x] = sum_value;
    }
}

__global__ void nppiFilterBox_8u_C4R_kernel_vec(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int width,
                                             int height,int maskWidth, int maskHeight, int anchorX, int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {

    uchar4 *dst_pixel = (uchar4 *)((char *)pDst + y * nDstStep) + x;
    uint4 sumVal_vec = make_uint4(0, 0, 0, 0);

    for (int my = 0; my < maskHeight; my++) {
      for (int mx = 0; mx < maskWidth; mx++) {
          int srcY = y + my - anchorY;
          int srcX = x + mx - anchorX;
          uchar4 pixelValue_vec = make_uchar4(0, 0, 0, 0);
          if(0 <= srcY && srcY <= (height - 1) && 0 <= srcX && srcX <= (width - 1)) {
            const uchar4 *src_pixel = (const uchar4 *)((const char *)pSrc + srcY * nSrcStep) + srcX;
            pixelValue_vec = *src_pixel;
          }

          sumVal_vec.x += pixelValue_vec.x;
          sumVal_vec.y += pixelValue_vec.y;
          sumVal_vec.z += pixelValue_vec.z;
          sumVal_vec.w += pixelValue_vec.w;
      }
    }
    uchar4 sum_value = make_uchar4(0, 0, 0, 0);
    sum_value.x = sumVal_vec.x / (maskHeight * maskWidth);
    sum_value.y = sumVal_vec.y / (maskHeight * maskWidth);
    sum_value.z = sumVal_vec.z / (maskHeight * maskWidth);
    sum_value.w = sumVal_vec.w / (maskHeight * maskWidth);
    *dst_pixel = sum_value;
  }
}

// Box filter kernel for single-channel 32-bit float with zero padding
__global__ void nppiFilterBox_32f_C1R_kernel_impl(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                                  int width, int height, int maskWidth, int maskHeight, int anchorX,
                                                  int anchorY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    // Calculate filter coverage area
    int startX = x - anchorX;
    int startY = y - anchorY;
    int endX = startX + maskWidth;
    int endY = startY + maskHeight;

    float sum = 0.0f;
    int totalMaskPixels = maskWidth * maskHeight;

    for (int j = startY; j < endY; j++) {
      for (int i = startX; i < endX; i++) {
        // Check if coordinates are within image bounds
        if (i >= 0 && i < width && j >= 0 && j < height) {
          const Npp32f *pSrcRow = (const Npp32f *)((const char *)pSrc + j * nSrcStep);
          sum += pSrcRow[i];
        }
        // Out-of-bounds
      }
    }

    Npp32f *pDstRow = (Npp32f *)((char *)pDst + y * nDstStep);
    pDstRow[x] = sum / totalMaskPixels;
  }
}

__device__ __forceinline__ float getPixel(const Npp32f *pSrc, int nSrcStep, int x, int y, int width, int height) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        const Npp32f *pSrcRow = (const Npp32f *)((const char *)pSrc + y * nSrcStep);
        return pSrcRow[x];
    }
    return 0.0f;
}

__global__ void nppiFilterBox_32f_C1R_3x3_shared(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                                int width, int height) {
    const int MASK_WIDTH = 3;
    const int MASK_HEIGHT = 3;
    const int ANCHOR_X = 1;
    const int ANCHOR_Y = 1;
    const int BLOCK_WIDTH = 16;
    const int BLOCK_HEIGHT = 16;

    __shared__ float shared_mem[BLOCK_HEIGHT + 2][BLOCK_WIDTH + 2];

    int blockStartX = blockIdx.x * BLOCK_WIDTH;
    int blockStartY = blockIdx.y * BLOCK_HEIGHT;

    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

    for (int loadY = threadY; loadY < BLOCK_HEIGHT + 2; loadY += BLOCK_HEIGHT) {
        for (int loadX = threadX; loadX < BLOCK_WIDTH + 2; loadX += BLOCK_WIDTH) {
            int globalX = blockStartX + loadX - ANCHOR_X;
            int globalY = blockStartY + loadY - ANCHOR_Y;

            shared_mem[loadY][loadX] = getPixel(pSrc, nSrcStep, globalX, globalY, width, height);
        }
    }

    __syncthreads();

    int x = blockStartX + threadIdx.x;
    int y = blockStartY + threadIdx.y;

    if (x < width && y < height) {
        float sum =
            shared_mem[threadIdx.y][threadIdx.x] +
            shared_mem[threadIdx.y][threadIdx.x + 1] +
            shared_mem[threadIdx.y][threadIdx.x + 2] +
            shared_mem[threadIdx.y + 1][threadIdx.x] +
            shared_mem[threadIdx.y + 1][threadIdx.x + 1] +
            shared_mem[threadIdx.y + 1][threadIdx.x + 2] +
            shared_mem[threadIdx.y + 2][threadIdx.x] +
            shared_mem[threadIdx.y + 2][threadIdx.x + 1] +
            shared_mem[threadIdx.y + 2][threadIdx.x + 2];

        Npp32f *pDstRow = (Npp32f *)((char *)pDst + y * nDstStep);
        pDstRow[x] = sum * 0.111111111f; // 1/9
    }
}

__global__ void nppiFilterBox_32f_C1R_5x5_shared(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                                int width, int height) {
    const int MASK_WIDTH = 5;
    const int MASK_HEIGHT = 5;
    const int ANCHOR_X = 2;
    const int ANCHOR_Y = 2;
    const int BLOCK_WIDTH = 16;
    const int BLOCK_HEIGHT = 16;

    // shared memory: block size + 4 pixel bound
    __shared__ float shared_mem[BLOCK_HEIGHT + 4][BLOCK_WIDTH + 4];

    int blockStartX = blockIdx.x * BLOCK_WIDTH;
    int blockStartY = blockIdx.y * BLOCK_HEIGHT;

    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

    for (int loadY = threadY; loadY < BLOCK_HEIGHT + 4; loadY += BLOCK_HEIGHT) {
        for (int loadX = threadX; loadX < BLOCK_WIDTH + 4; loadX += BLOCK_WIDTH) {
            int globalX = blockStartX + loadX - ANCHOR_X;
            int globalY = blockStartY + loadY - ANCHOR_Y;

            shared_mem[loadY][loadX] = getPixel(pSrc, nSrcStep, globalX, globalY, width, height);
        }
    }

    __syncthreads();

    int x = blockStartX + threadIdx.x;
    int y = blockStartY + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;

        #pragma unroll
        for (int j = 0; j < 5; j++) {
            sum += shared_mem[threadIdx.y + j][threadIdx.x] +
                   shared_mem[threadIdx.y + j][threadIdx.x + 1] +
                   shared_mem[threadIdx.y + j][threadIdx.x + 2] +
                   shared_mem[threadIdx.y + j][threadIdx.x + 3] +
                   shared_mem[threadIdx.y + j][threadIdx.x + 4];
        }

        Npp32f *pDstRow = (Npp32f *)((char *)pDst + y * nDstStep);
        pDstRow[x] = sum * 0.04f; // 1/25
    }
}

__global__ void nppiFilterBox_32f_C1R_7x7_shared(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                                int width, int height) {
    const int MASK_WIDTH = 7;
    const int MASK_HEIGHT = 7;
    const int ANCHOR_X = 3;
    const int ANCHOR_Y = 3;
    const int BLOCK_WIDTH = 16;
    const int BLOCK_HEIGHT = 16;

    __shared__ float shared_mem[BLOCK_HEIGHT + 6][BLOCK_WIDTH + 6];

    int blockStartX = blockIdx.x * BLOCK_WIDTH;
    int blockStartY = blockIdx.y * BLOCK_HEIGHT;

    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

    for (int loadY = threadY; loadY < BLOCK_HEIGHT + 6; loadY += BLOCK_HEIGHT) {
        for (int loadX = threadX; loadX < BLOCK_WIDTH + 6; loadX += BLOCK_WIDTH) {
            int globalX = blockStartX + loadX - ANCHOR_X;
            int globalY = blockStartY + loadY - ANCHOR_Y;

            shared_mem[loadY][loadX] = getPixel(pSrc, nSrcStep, globalX, globalY, width, height);
        }
    }

    __syncthreads();

    int x = blockStartX + threadIdx.x;
    int y = blockStartY + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;

        #pragma unroll
        for (int j = 0; j < 7; j++) {
            #pragma unroll
            for (int i = 0; i < 7; i++) {
                sum += shared_mem[threadIdx.y + j][threadIdx.x + i];
            }
        }

        Npp32f *pDstRow = (Npp32f *)((char *)pDst + y * nDstStep);
        pDstRow[x] = sum * 0.020408163f; // 1/49
    }
}

extern "C" {
cudaError_t nppiFilterBox_8u_C1R_kernel(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                        NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream) {

    const int BLOCK_W = 16;
    const int BLOCK_H = 16;
    dim3 blockSize(BLOCK_W, BLOCK_H);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    const int width = oSizeROI.width;
    const int height = oSizeROI.height;
    const int maskWidth = oMaskSize.width;
    const int maskHeight = oMaskSize.height;
    const int anchorX = oAnchor.x;
    const int anchorY = oAnchor.y;

    const int sharedMemSize = (BLOCK_W + maskWidth -1) * (BLOCK_H + maskHeight - 1) * sizeof(Npp8u);

    // For small odd square masks (3x3, 5x5, 7x7), use single-pass kernel
    if (maskWidth == 3 && maskHeight == 3 && anchorX == 1 && anchorY == 1) {

        dim3 gridSize(((width + 1) / 2 + BLOCK_W - 1) / BLOCK_W,
                        ((height + 1) / 2 + BLOCK_H - 1) / BLOCK_H);
        // Use optimized 3x3 specialization
        filterBoxOptimized_8u_C1R_3x3<BLOCK_W, BLOCK_H><<<gridSize, blockSize, 0, stream>>>(
            pSrc, nSrcStep, pDst, nDstStep, width, height);
    } else if (maskWidth == 5 && maskHeight == 5 && anchorX == 2 && anchorY == 2) {
        dim3 gridSize(((width + 1) / 2 + BLOCK_W - 1) / BLOCK_W,
                        ((height + 1) / 2 + BLOCK_H - 1) / BLOCK_H);
        // Use optimized 5x5 specialization
        filterBoxOptimized_8u_C1R_5x5<BLOCK_W, BLOCK_H><<<gridSize, blockSize, 0, stream>>>(
            pSrc, nSrcStep, pDst, nDstStep, width, height);
    } else if (maskWidth == 7 && maskHeight == 7 && anchorX == 3 && anchorY == 3) {
        filterBoxOptimized_8u_C1R<BLOCK_W, BLOCK_H, 3><<<gridSize, blockSize, sharedMemSize, stream>>>(
            pSrc, nSrcStep, pDst, nDstStep, width, height, maskWidth, maskHeight);
    } else {
        nppiFilterBox_8u_C1R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                            oSizeROI.height, oMaskSize.width,
                                                                            oMaskSize.height, oAnchor.x, oAnchor.y);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (stream == 0) {
        cudaDeviceSynchronize();
        }
    }
    return err;
}

cudaError_t nppiFilterBox_8u_C4R_kernel(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                        NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  // Select optimized kernel based on filter size
  if (oMaskSize.width == 3 && oMaskSize.height == 3 && oAnchor.x == 1 && oAnchor.y == 1) {
    int tile_width = blockSize.x + oMaskSize.width - 1;
    int tile_height = blockSize.y + oMaskSize.height - 1;
    size_t shared_mem_size = tile_width * tile_height * sizeof(uchar4);

    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    nppiFilterBox_8u_C4R_shared_3x3<<<gridSize, blockSize, shared_mem_size, stream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
  } else if (oMaskSize.width == 5 && oMaskSize.height == 5 && oAnchor.x == 2 && oAnchor.y == 2) {
    int tile_width = blockSize.x + oMaskSize.width - 1;
    int tile_height = blockSize.y + oMaskSize.height - 1;
    size_t shared_mem_size = tile_width * tile_height * sizeof(uchar4);

    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    nppiFilterBox_8u_C4R_shared_NxN<5, 5, 2, 2><<<gridSize, blockSize, shared_mem_size, stream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
  } else if (oMaskSize.width == 7 && oMaskSize.height == 7 && oAnchor.x == 3 && oAnchor.y == 3) {
    int tile_width = blockSize.x + oMaskSize.width - 1;
    int tile_height = blockSize.y + oMaskSize.height - 1;
    size_t shared_mem_size = tile_width * tile_height * sizeof(uchar4);

    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);

    nppiFilterBox_8u_C4R_shared_NxN<7, 7, 3, 3><<<gridSize, blockSize, shared_mem_size, stream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height);
  } else {
    // Calculate halo region and shared memory size
    dim3 blockSize(32, 8);
    dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x,
                  (oSizeROI.height + blockSize.y - 1) / blockSize.y);
    nppiFilterBox_8u_C4R_kernel_vec<<<gridSize, blockSize, 0, stream>>>(
        pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width, oSizeROI.height, oMaskSize.width, oMaskSize.height,
        oAnchor.x, oAnchor.y);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    if (stream == 0) {
      cudaDeviceSynchronize();
    }
  }
  return err;
}

cudaError_t nppiFilterBox_32f_C1R_kernel(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                         NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor,
                                         cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((oSizeROI.width + blockSize.x - 1) / blockSize.x, (oSizeROI.height + blockSize.y - 1) / blockSize.y);

  if (oMaskSize.width == 3 && oMaskSize.height == 3 && oAnchor.x == 1 && oAnchor.y == 1) {
    nppiFilterBox_32f_C1R_3x3_shared<<<gridSize, blockSize>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                              oSizeROI.width, oSizeROI.height);
  } else if (oMaskSize.width == 5 && oMaskSize.height == 5 && oAnchor.x == 2 && oAnchor.y == 2) {
    nppiFilterBox_32f_C1R_5x5_shared<<<gridSize, blockSize>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                              oSizeROI.width, oSizeROI.height);
  } else if (oMaskSize.width == 7 && oMaskSize.height == 7 && oAnchor.x == 3 && oAnchor.y == 3) {
    nppiFilterBox_32f_C1R_7x7_shared<<<gridSize, blockSize>>>(pSrc, nSrcStep, pDst, nDstStep,
                                                              oSizeROI.width, oSizeROI.height);
  } else {
    nppiFilterBox_32f_C1R_kernel_impl<<<gridSize, blockSize, 0, stream>>>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI.width,
                                                                          oSizeROI.height, oMaskSize.width,
                                                                          oMaskSize.height, oAnchor.x, oAnchor.y);
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    if (stream == 0) {
      cudaDeviceSynchronize();
    }
  }
  return err;
}
}
