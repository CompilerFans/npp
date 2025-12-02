#include "npp.h"
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

class NppiMirrorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化CUDA
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess);
    }

    void TearDown() override {
        // 清理所有分配的GPU内存
        for (void* ptr : gpu_memory_allocated) {
            cudaFree(ptr);
        }
        gpu_memory_allocated.clear();
    }

    template<typename T>
    T* allocateGpuMemory(size_t size) {
        T* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, size * sizeof(T));
        EXPECT_EQ(err, cudaSuccess);
        if (err == cudaSuccess) {
            gpu_memory_allocated.push_back(ptr);
        }
        return ptr;
    }

    template<typename T>
    void copyToGpu(const std::vector<T>& src, T* dst) {
        cudaError_t err = cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice);
        EXPECT_EQ(err, cudaSuccess);
    }

    template<typename T>
    std::vector<T> copyFromGpu(const T* src, size_t size) {
        std::vector<T> dst(size);
        cudaError_t err = cudaMemcpy(dst.data(), src, size * sizeof(T), cudaMemcpyDeviceToHost);
        EXPECT_EQ(err, cudaSuccess);
        return dst;
    }

    template<typename T>
    std::vector<T> createTestPattern(int width, int height, int channels) {
        std::vector<T> data(width * height * channels);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                for (int c = 0; c < channels; ++c) {
                    int idx = (y * width + x) * channels + c;
                    if constexpr (std::is_same<T, Npp8u>::value) {
                        data[idx] = static_cast<T>((x + y * 10 + c * 100) % 256);
                    } else if constexpr (std::is_same<T, Npp16u>::value) {
                        data[idx] = static_cast<T>((x + y * 10 + c * 100) % 65536);
                    } else if constexpr (std::is_same<T, Npp32s>::value) {
                        data[idx] = static_cast<T>((x + y * 10 + c * 100) % 10000);
                    } else if constexpr (std::is_same<T, Npp32f>::value) {
                        data[idx] = static_cast<T>((x + y * 10 + c * 100) % 10000) / 1000.0f;
                    }
                }
            }
        }
        return data;
    }

private:
    std::vector<void*> gpu_memory_allocated;
};

// Horizontal mirror image: Flip up and down
template<typename T>
void verifyHorizontalMirror(const std::vector<T>& original, const std::vector<T>& mirrored,
                           int width, int height, int channels) {
    for (int y = 0; y < height; ++y) {
        int srcY = height - 1 - y;
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                int origIdx = (srcY * width + x) * channels + c;
                int mirrorIdx = (y * width + x) * channels + c;
                EXPECT_EQ(mirrored[mirrorIdx], original[origIdx])
                    << "Horizontal mirror failed at (x=" << x << ", y=" << y << ", c=" << c << ")";
            }
        }
    }
}
// Vertical mirror image: Flip left and right
template<typename T>
void verifyVerticalMirror(const std::vector<T>& original, const std::vector<T>& mirrored,
                         int width, int height, int channels) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int srcX = width - 1 - x;
            for (int c = 0; c < channels; ++c) {
                int origIdx = (y * width + srcX) * channels + c;
                int mirrorIdx = (y * width + x) * channels + c;
                EXPECT_EQ(mirrored[mirrorIdx], original[origIdx])
                    << "Vertical mirror failed at (x=" << x << ", y=" << y << ", c=" << c << ")";
            }
        }
    }
}
// Horizontal and vertical mirror images
template<typename T>
void verifyBothAxisMirror(const std::vector<T>& original, const std::vector<T>& mirrored,
                         int width, int height, int channels) {
    for (int y = 0; y < height; ++y) {
        int srcY = height - 1 - y;
        for (int x = 0; x < width; ++x) {
            int srcX = width - 1 - x;
            for (int c = 0; c < channels; ++c) {
                int origIdx = (srcY * width + srcX) * channels + c;
                int mirrorIdx = (y * width + x) * channels + c;
                EXPECT_EQ(mirrored[mirrorIdx], original[origIdx])
                    << "Both axis mirror failed at (x=" << x << ", y=" << y << ", c=" << c << ")";
            }
        }
    }
}

TEST_F(NppiMirrorTest, 8u_C1R_Horizontal) {
    const int width = 32;
    const int height = 32;
    const int channels = 1;

    auto srcData = createTestPattern<Npp8u>(width, height, channels);
    auto dstData = std::vector<Npp8u>(width * height * channels, 0);

    Npp8u* d_src = allocateGpuMemory<Npp8u>(srcData.size());
    Npp8u* d_dst = allocateGpuMemory<Npp8u>(dstData.size());

    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_8u_C1R(d_src, width * sizeof(Npp8u), d_dst, width * sizeof(Npp8u),
                                        roi, NPP_HORIZONTAL_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_dst, dstData.size());
    verifyHorizontalMirror(srcData, result, width, height, channels);
}

TEST_F(NppiMirrorTest, 8u_C1R_Vertical) {
    const int width = 32;
    const int height = 32;
    const int channels = 1;

    auto srcData = createTestPattern<Npp8u>(width, height, channels);
    auto dstData = std::vector<Npp8u>(width * height * channels, 0);

    Npp8u* d_src = allocateGpuMemory<Npp8u>(srcData.size());
    Npp8u* d_dst = allocateGpuMemory<Npp8u>(dstData.size());

    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_8u_C1R(d_src, width * sizeof(Npp8u), d_dst, width * sizeof(Npp8u),
                                        roi, NPP_VERTICAL_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_dst, dstData.size());
    verifyVerticalMirror(srcData, result, width, height, channels);
}

TEST_F(NppiMirrorTest, 8u_C1R_Both) {
    const int width = 32;
    const int height = 32;
    const int channels = 1;

    auto srcData = createTestPattern<Npp8u>(width, height, channels);
    auto dstData = std::vector<Npp8u>(width * height * channels, 0);

    Npp8u* d_src = allocateGpuMemory<Npp8u>(srcData.size());
    Npp8u* d_dst = allocateGpuMemory<Npp8u>(dstData.size());

    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_8u_C1R(d_src, width * sizeof(Npp8u), d_dst, width * sizeof(Npp8u),
                                        roi, NPP_BOTH_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_dst, dstData.size());
    verifyBothAxisMirror(srcData, result, width, height, channels);
}

TEST_F(NppiMirrorTest, 8u_C3R_Horizontal) {
    const int width = 32;
    const int height = 32;
    const int channels = 3;

    auto srcData = createTestPattern<Npp8u>(width, height, channels);
    auto dstData = std::vector<Npp8u>(width * height * channels, 0);

    Npp8u* d_src = allocateGpuMemory<Npp8u>(srcData.size());
    Npp8u* d_dst = allocateGpuMemory<Npp8u>(dstData.size());

    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_8u_C3R(d_src, width * channels * sizeof(Npp8u),
                                        d_dst, width * channels * sizeof(Npp8u),
                                        roi, NPP_HORIZONTAL_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_dst, dstData.size());
    verifyHorizontalMirror(srcData, result, width, height, channels);
}

TEST_F(NppiMirrorTest, 8u_C4R_Horizontal) {
    const int width = 32;
    const int height = 32;
    const int channels = 4;

    auto srcData = createTestPattern<Npp8u>(width, height, channels);
    auto dstData = std::vector<Npp8u>(width * height * channels, 0);

    Npp8u* d_src = allocateGpuMemory<Npp8u>(srcData.size());
    Npp8u* d_dst = allocateGpuMemory<Npp8u>(dstData.size());

    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_8u_C4R(d_src, width * channels * sizeof(Npp8u),
                                        d_dst, width * channels * sizeof(Npp8u),
                                        roi, NPP_HORIZONTAL_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_dst, dstData.size());
    verifyHorizontalMirror(srcData, result, width, height, channels);
}

TEST_F(NppiMirrorTest, 16u_C1R_Horizontal) {
    const int width = 32;
    const int height = 32;
    const int channels = 1;

    auto srcData = createTestPattern<Npp16u>(width, height, channels);
    auto dstData = std::vector<Npp16u>(width * height * channels, 0);

    Npp16u* d_src = allocateGpuMemory<Npp16u>(srcData.size());
    Npp16u* d_dst = allocateGpuMemory<Npp16u>(dstData.size());

    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_16u_C1R(d_src, width * sizeof(Npp16u),
                                         d_dst, width * sizeof(Npp16u),
                                         roi, NPP_HORIZONTAL_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_dst, dstData.size());
    verifyHorizontalMirror(srcData, result, width, height, channels);
}

TEST_F(NppiMirrorTest, 32f_C1R_Horizontal) {
    const int width = 32;
    const int height = 32;
    const int channels = 1;

    auto srcData = createTestPattern<Npp32f>(width, height, channels);
    auto dstData = std::vector<Npp32f>(width * height * channels, 0);

    Npp32f* d_src = allocateGpuMemory<Npp32f>(srcData.size());
    Npp32f* d_dst = allocateGpuMemory<Npp32f>(dstData.size());

    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_32f_C1R(d_src, width * sizeof(Npp32f),
                                         d_dst, width * sizeof(Npp32f),
                                         roi, NPP_HORIZONTAL_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_dst, dstData.size());
    verifyHorizontalMirror(srcData, result, width, height, channels);
}

TEST_F(NppiMirrorTest, 32s_C1R_Horizontal) {
    const int width = 32;
    const int height = 32;
    const int channels = 1;

    auto srcData = createTestPattern<Npp32s>(width, height, channels);
    auto dstData = std::vector<Npp32s>(width * height * channels, 0);

    Npp32s* d_src = allocateGpuMemory<Npp32s>(srcData.size());
    Npp32s* d_dst = allocateGpuMemory<Npp32s>(dstData.size());

    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_32s_C1R(d_src, width * sizeof(Npp32s),
                                         d_dst, width * sizeof(Npp32s),
                                         roi, NPP_HORIZONTAL_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_dst, dstData.size());
    verifyHorizontalMirror(srcData, result, width, height, channels);
}

TEST_F(NppiMirrorTest, 8u_C1IR_Horizontal) {
    const int width = 32;
    const int height = 32;
    const int channels = 1;

    auto srcData = createTestPattern<Npp8u>(width, height, channels);
    auto expectedData = std::vector<Npp8u>(width * height * channels);

    // 计算期望结果
    for (int y = 0; y < height; ++y) {
        int srcY = height - 1 - y;
        for (int x = 0; x < width; ++x) {
            expectedData[y * width + x] = srcData[srcY * width + x];
        }
    }

    Npp8u* d_src = allocateGpuMemory<Npp8u>(srcData.size());
    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_8u_C1IR(d_src, width * sizeof(Npp8u), roi, NPP_HORIZONTAL_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_src, srcData.size());
    EXPECT_EQ(result, expectedData);
}

TEST_F(NppiMirrorTest, 8u_C3IR_Vertical) {
    const int width = 32;
    const int height = 32;
    const int channels = 3;

    auto srcData = createTestPattern<Npp8u>(width, height, channels);
    auto expectedData = std::vector<Npp8u>(width * height * channels);

    // 计算期望结果
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int srcX = width - 1 - x;
            for (int c = 0; c < channels; ++c) {
                expectedData[(y * width + x) * channels + c] =
                    srcData[(y * width + srcX) * channels + c];
            }
        }
    }

    Npp8u* d_src = allocateGpuMemory<Npp8u>(srcData.size());
    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_8u_C3IR(d_src, width * channels * sizeof(Npp8u),
                                         roi, NPP_VERTICAL_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_src, srcData.size());
    EXPECT_EQ(result, expectedData);
}

TEST_F(NppiMirrorTest, 8u_C1R_Ctx) {
    const int width = 32;
    const int height = 32;
    const int channels = 1;

    auto srcData = createTestPattern<Npp8u>(width, height, channels);
    auto dstData = std::vector<Npp8u>(width * height * channels, 0);

    Npp8u* d_src = allocateGpuMemory<Npp8u>(srcData.size());
    Npp8u* d_dst = allocateGpuMemory<Npp8u>(dstData.size());

    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStreamContext ctx;
    ctx.hStream = 0;  // 默认stream

    NppStatus status = nppiMirror_8u_C1R_Ctx(d_src, width * sizeof(Npp8u),
                                            d_dst, width * sizeof(Npp8u),
                                            roi, NPP_HORIZONTAL_AXIS, ctx);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_dst, dstData.size());
    verifyHorizontalMirror(srcData, result, width, height, channels);
}

// ============ 16-bit unsigned tests ============
// Test 16-bit unsigned C3R (3 channels)
TEST_F(NppiMirrorTest, 16u_C3R_Vertical) {
    const int width = 32;
    const int height = 32;
    const int channels = 3;

    auto srcData = createTestPattern<Npp16u>(width, height, channels);
    auto dstData = std::vector<Npp16u>(width * height * channels, 0);

    Npp16u* d_src = allocateGpuMemory<Npp16u>(srcData.size());
    Npp16u* d_dst = allocateGpuMemory<Npp16u>(dstData.size());

    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_16u_C3R(d_src, width * channels * sizeof(Npp16u),
                                         d_dst, width * channels * sizeof(Npp16u),
                                         roi, NPP_VERTICAL_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_dst, dstData.size());
    verifyVerticalMirror(srcData, result, width, height, channels);
}

// Test 16-bit unsigned C4R (4 channels)
TEST_F(NppiMirrorTest, 16u_C4R_Both) {
    const int width = 32;
    const int height = 32;
    const int channels = 4;

    auto srcData = createTestPattern<Npp16u>(width, height, channels);
    auto dstData = std::vector<Npp16u>(width * height * channels, 0);

    Npp16u* d_src = allocateGpuMemory<Npp16u>(srcData.size());
    Npp16u* d_dst = allocateGpuMemory<Npp16u>(dstData.size());

    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_16u_C4R(d_src, width * channels * sizeof(Npp16u),
                                         d_dst, width * channels * sizeof(Npp16u),
                                         roi, NPP_BOTH_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_dst, dstData.size());
    verifyBothAxisMirror(srcData, result, width, height, channels);
}

// Test 16-bit unsigned C1IR (in-place, 1 channel)
TEST_F(NppiMirrorTest, 16u_C1IR_Both) {
    const int width = 32;
    const int height = 32;
    const int channels = 1;

    auto srcData = createTestPattern<Npp16u>(width, height, channels);
    auto expectedData = std::vector<Npp16u>(width * height * channels);

    // Calculate expected result for both axis mirror
    for (int y = 0; y < height; ++y) {
        int srcY = height - 1 - y;
        for (int x = 0; x < width; ++x) {
            int srcX = width - 1 - x;
            expectedData[y * width + x] = srcData[srcY * width + srcX];
        }
    }

    Npp16u* d_src = allocateGpuMemory<Npp16u>(srcData.size());
    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_16u_C1IR(d_src, width * sizeof(Npp16u), roi, NPP_BOTH_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_src, srcData.size());
    EXPECT_EQ(result, expectedData);
}

// Test 16-bit unsigned C3IR (in-place, 3 channels)
TEST_F(NppiMirrorTest, 16u_C3IR_Horizontal) {
    const int width = 32;
    const int height = 32;
    const int channels = 3;

    auto srcData = createTestPattern<Npp16u>(width, height, channels);
    auto expectedData = std::vector<Npp16u>(width * height * channels);

    // Calculate expected result for horizontal mirror
    for (int y = 0; y < height; ++y) {
        int srcY = height - 1 - y;
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                expectedData[(y * width + x) * channels + c] =
                    srcData[(srcY * width + x) * channels + c];
            }
        }
    }

    Npp16u* d_src = allocateGpuMemory<Npp16u>(srcData.size());
    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_16u_C3IR(d_src, width * channels * sizeof(Npp16u),
                                          roi, NPP_HORIZONTAL_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_src, srcData.size());
    EXPECT_EQ(result, expectedData);
}

// Test 16-bit unsigned C4IR (in-place, 4 channels)
TEST_F(NppiMirrorTest, 16u_C4IR_Vertical) {
    const int width = 32;
    const int height = 32;
    const int channels = 4;

    auto srcData = createTestPattern<Npp16u>(width, height, channels);
    auto expectedData = std::vector<Npp16u>(width * height * channels);

    // Calculate expected result for vertical mirror
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int srcX = width - 1 - x;
            for (int c = 0; c < channels; ++c) {
                expectedData[(y * width + x) * channels + c] =
                    srcData[(y * width + srcX) * channels + c];
            }
        }
    }

    Npp16u* d_src = allocateGpuMemory<Npp16u>(srcData.size());
    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_16u_C4IR(d_src, width * channels * sizeof(Npp16u),
                                          roi, NPP_VERTICAL_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_src, srcData.size());
    EXPECT_EQ(result, expectedData);
}

// Test 16-bit unsigned Ctx versions
TEST_F(NppiMirrorTest, 16u_C1R_Ctx) {
    const int width = 32;
    const int height = 32;
    const int channels = 1;

    auto srcData = createTestPattern<Npp16u>(width, height, channels);
    auto dstData = std::vector<Npp16u>(width * height * channels, 0);

    Npp16u* d_src = allocateGpuMemory<Npp16u>(srcData.size());
    Npp16u* d_dst = allocateGpuMemory<Npp16u>(dstData.size());

    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStreamContext ctx;
    ctx.hStream = 0;

    NppStatus status = nppiMirror_16u_C1R_Ctx(d_src, width * sizeof(Npp16u),
                                             d_dst, width * sizeof(Npp16u),
                                             roi, NPP_VERTICAL_AXIS, ctx);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_dst, dstData.size());
    verifyVerticalMirror(srcData, result, width, height, channels);
}

// ============ 32-bit float tests ============

// Test 32-bit float C3R (3 channels)
TEST_F(NppiMirrorTest, 32f_C3R_Vertical) {
    const int width = 32;
    const int height = 32;
    const int channels = 3;

    auto srcData = createTestPattern<Npp32f>(width, height, channels);
    auto dstData = std::vector<Npp32f>(width * height * channels, 0);

    Npp32f* d_src = allocateGpuMemory<Npp32f>(srcData.size());
    Npp32f* d_dst = allocateGpuMemory<Npp32f>(dstData.size());

    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_32f_C3R(d_src, width * channels * sizeof(Npp32f),
                                         d_dst, width * channels * sizeof(Npp32f),
                                         roi, NPP_VERTICAL_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_dst, dstData.size());
    verifyVerticalMirror(srcData, result, width, height, channels);
}

// Test 32-bit float C4R (4 channels)
TEST_F(NppiMirrorTest, 32f_C4R_Both) {
    const int width = 32;
    const int height = 32;
    const int channels = 4;

    auto srcData = createTestPattern<Npp32f>(width, height, channels);
    auto dstData = std::vector<Npp32f>(width * height * channels, 0);

    Npp32f* d_src = allocateGpuMemory<Npp32f>(srcData.size());
    Npp32f* d_dst = allocateGpuMemory<Npp32f>(dstData.size());

    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_32f_C4R(d_src, width * channels * sizeof(Npp32f),
                                         d_dst, width * channels * sizeof(Npp32f),
                                         roi, NPP_BOTH_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_dst, dstData.size());
    verifyBothAxisMirror(srcData, result, width, height, channels);
}

// Test 32-bit float C1IR (in-place, 1 channel)
TEST_F(NppiMirrorTest, 32f_C1IR_Vertical) {
    const int width = 32;
    const int height = 32;
    const int channels = 1;

    auto srcData = createTestPattern<Npp32f>(width, height, channels);
    auto expectedData = std::vector<Npp32f>(width * height * channels);

    // Calculate expected result for vertical mirror
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int srcX = width - 1 - x;
            expectedData[y * width + x] = srcData[y * width + srcX];
        }
    }

    Npp32f* d_src = allocateGpuMemory<Npp32f>(srcData.size());
    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_32f_C1IR(d_src, width * sizeof(Npp32f), roi, NPP_VERTICAL_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_src, srcData.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expectedData[i]);
    }
}

// Test 32-bit float C3IR (in-place, 3 channels)
TEST_F(NppiMirrorTest, 32f_C3IR_Both) {
    const int width = 32;
    const int height = 32;
    const int channels = 3;

    auto srcData = createTestPattern<Npp32f>(width, height, channels);
    auto expectedData = std::vector<Npp32f>(width * height * channels);

    // Calculate expected result for both axis mirror
    for (int y = 0; y < height; ++y) {
        int srcY = height - 1 - y;
        for (int x = 0; x < width; ++x) {
            int srcX = width - 1 - x;
            for (int c = 0; c < channels; ++c) {
                expectedData[(y * width + x) * channels + c] =
                    srcData[(srcY * width + srcX) * channels + c];
            }
        }
    }

    Npp32f* d_src = allocateGpuMemory<Npp32f>(srcData.size());
    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_32f_C3IR(d_src, width * channels * sizeof(Npp32f),
                                          roi, NPP_BOTH_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_src, srcData.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expectedData[i]);
    }
}

// Test 32-bit float C4IR (in-place, 4 channels)
TEST_F(NppiMirrorTest, 32f_C4IR_Horizontal) {
    const int width = 32;
    const int height = 32;
    const int channels = 4;

    auto srcData = createTestPattern<Npp32f>(width, height, channels);
    auto expectedData = std::vector<Npp32f>(width * height * channels);

    // Calculate expected result for horizontal mirror
    for (int y = 0; y < height; ++y) {
        int srcY = height - 1 - y;
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                expectedData[(y * width + x) * channels + c] =
                    srcData[(srcY * width + x) * channels + c];
            }
        }
    }

    Npp32f* d_src = allocateGpuMemory<Npp32f>(srcData.size());
    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_32f_C4IR(d_src, width * channels * sizeof(Npp32f),
                                          roi, NPP_HORIZONTAL_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_src, srcData.size());
    for (size_t i = 0; i < result.size(); ++i) {
        EXPECT_FLOAT_EQ(result[i], expectedData[i]);
    }
}

// Test 32-bit float Ctx versions
TEST_F(NppiMirrorTest, 32f_C3R_Ctx) {
    const int width = 32;
    const int height = 32;
    const int channels = 3;

    auto srcData = createTestPattern<Npp32f>(width, height, channels);
    auto dstData = std::vector<Npp32f>(width * height * channels, 0);

    Npp32f* d_src = allocateGpuMemory<Npp32f>(srcData.size());
    Npp32f* d_dst = allocateGpuMemory<Npp32f>(dstData.size());

    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStreamContext ctx;
    ctx.hStream = 0;

    NppStatus status = nppiMirror_32f_C3R_Ctx(d_src, width * channels * sizeof(Npp32f),
                                             d_dst, width * channels * sizeof(Npp32f),
                                             roi, NPP_HORIZONTAL_AXIS, ctx);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_dst, dstData.size());
    verifyHorizontalMirror(srcData, result, width, height, channels);
}

// ============ 32-bit signed integer tests ============

// Test 32-bit signed integer C3R (3 channels)
TEST_F(NppiMirrorTest, 32s_C3R_Vertical) {
    const int width = 32;
    const int height = 32;
    const int channels = 3;

    auto srcData = createTestPattern<Npp32s>(width, height, channels);
    auto dstData = std::vector<Npp32s>(width * height * channels, 0);

    Npp32s* d_src = allocateGpuMemory<Npp32s>(srcData.size());
    Npp32s* d_dst = allocateGpuMemory<Npp32s>(dstData.size());

    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_32s_C3R(d_src, width * channels * sizeof(Npp32s),
                                         d_dst, width * channels * sizeof(Npp32s),
                                         roi, NPP_VERTICAL_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_dst, dstData.size());
    verifyVerticalMirror(srcData, result, width, height, channels);
}

// Test 32-bit signed integer C4R (4 channels)
TEST_F(NppiMirrorTest, 32s_C4R_Both) {
    const int width = 32;
    const int height = 32;
    const int channels = 4;

    auto srcData = createTestPattern<Npp32s>(width, height, channels);
    auto dstData = std::vector<Npp32s>(width * height * channels, 0);

    Npp32s* d_src = allocateGpuMemory<Npp32s>(srcData.size());
    Npp32s* d_dst = allocateGpuMemory<Npp32s>(dstData.size());

    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_32s_C4R(d_src, width * channels * sizeof(Npp32s),
                                         d_dst, width * channels * sizeof(Npp32s),
                                         roi, NPP_BOTH_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_dst, dstData.size());
    verifyBothAxisMirror(srcData, result, width, height, channels);
}

// Test 32-bit signed integer C1IR (in-place, 1 channel)
TEST_F(NppiMirrorTest, 32s_C1IR_Horizontal) {
    const int width = 32;
    const int height = 32;
    const int channels = 1;

    auto srcData = createTestPattern<Npp32s>(width, height, channels);
    auto expectedData = std::vector<Npp32s>(width * height * channels);

    // Calculate expected result for horizontal mirror
    for (int y = 0; y < height; ++y) {
        int srcY = height - 1 - y;
        for (int x = 0; x < width; ++x) {
            expectedData[y * width + x] = srcData[srcY * width + x];
        }
    }

    Npp32s* d_src = allocateGpuMemory<Npp32s>(srcData.size());
    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_32s_C1IR(d_src, width * sizeof(Npp32s), roi, NPP_HORIZONTAL_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_src, srcData.size());
    EXPECT_EQ(result, expectedData);
}

// Test 32-bit signed integer C3IR (in-place, 3 channels)
TEST_F(NppiMirrorTest, 32s_C3IR_Vertical) {
    const int width = 32;
    const int height = 32;
    const int channels = 3;

    auto srcData = createTestPattern<Npp32s>(width, height, channels);
    auto expectedData = std::vector<Npp32s>(width * height * channels);

    // Calculate expected result for vertical mirror
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int srcX = width - 1 - x;
            for (int c = 0; c < channels; ++c) {
                expectedData[(y * width + x) * channels + c] =
                    srcData[(y * width + srcX) * channels + c];
            }
        }
    }

    Npp32s* d_src = allocateGpuMemory<Npp32s>(srcData.size());
    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_32s_C3IR(d_src, width * channels * sizeof(Npp32s),
                                          roi, NPP_VERTICAL_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_src, srcData.size());
    EXPECT_EQ(result, expectedData);
}

// Test 32-bit signed integer C4IR (in-place, 4 channels)
TEST_F(NppiMirrorTest, 32s_C4IR_Both) {
    const int width = 32;
    const int height = 32;
    const int channels = 4;

    auto srcData = createTestPattern<Npp32s>(width, height, channels);
    auto expectedData = std::vector<Npp32s>(width * height * channels);

    // Calculate expected result for both axis mirror
    for (int y = 0; y < height; ++y) {
        int srcY = height - 1 - y;
        for (int x = 0; x < width; ++x) {
            int srcX = width - 1 - x;
            for (int c = 0; c < channels; ++c) {
                expectedData[(y * width + x) * channels + c] =
                    srcData[(srcY * width + srcX) * channels + c];
            }
        }
    }

    Npp32s* d_src = allocateGpuMemory<Npp32s>(srcData.size());
    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_32s_C4IR(d_src, width * channels * sizeof(Npp32s),
                                          roi, NPP_BOTH_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_src, srcData.size());
    EXPECT_EQ(result, expectedData);
}

// Test 32-bit signed integer Ctx versions
TEST_F(NppiMirrorTest, 32s_C4R_Ctx) {
    const int width = 32;
    const int height = 32;
    const int channels = 4;

    auto srcData = createTestPattern<Npp32s>(width, height, channels);
    auto dstData = std::vector<Npp32s>(width * height * channels, 0);

    Npp32s* d_src = allocateGpuMemory<Npp32s>(srcData.size());
    Npp32s* d_dst = allocateGpuMemory<Npp32s>(dstData.size());

    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStreamContext ctx;
    ctx.hStream = 0;

    NppStatus status = nppiMirror_32s_C4R_Ctx(d_src, width * channels * sizeof(Npp32s),
                                             d_dst, width * channels * sizeof(Npp32s),
                                             roi, NPP_BOTH_AXIS, ctx);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_dst, dstData.size());
    verifyBothAxisMirror(srcData, result, width, height, channels);
}

// ============ 8-bit unsigned additional tests ============

// Test 8-bit unsigned Ctx versions for C3
TEST_F(NppiMirrorTest, 8u_C3R_Ctx) {
    const int width = 32;
    const int height = 32;
    const int channels = 3;

    auto srcData = createTestPattern<Npp8u>(width, height, channels);
    auto dstData = std::vector<Npp8u>(width * height * channels, 0);

    Npp8u* d_src = allocateGpuMemory<Npp8u>(srcData.size());
    Npp8u* d_dst = allocateGpuMemory<Npp8u>(dstData.size());

    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStreamContext ctx;
    ctx.hStream = 0;

    NppStatus status = nppiMirror_8u_C3R_Ctx(d_src, width * channels * sizeof(Npp8u),
                                            d_dst, width * channels * sizeof(Npp8u),
                                            roi, NPP_VERTICAL_AXIS, ctx);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_dst, dstData.size());
    verifyVerticalMirror(srcData, result, width, height, channels);
}

// Test 8-bit unsigned Ctx versions for C4
TEST_F(NppiMirrorTest, 8u_C4R_Ctx) {
    const int width = 32;
    const int height = 32;
    const int channels = 4;

    auto srcData = createTestPattern<Npp8u>(width, height, channels);
    auto dstData = std::vector<Npp8u>(width * height * channels, 0);

    Npp8u* d_src = allocateGpuMemory<Npp8u>(srcData.size());
    Npp8u* d_dst = allocateGpuMemory<Npp8u>(dstData.size());

    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStreamContext ctx;
    ctx.hStream = 0;

    NppStatus status = nppiMirror_8u_C4R_Ctx(d_src, width * channels * sizeof(Npp8u),
                                            d_dst, width * channels * sizeof(Npp8u),
                                            roi, NPP_BOTH_AXIS, ctx);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_dst, dstData.size());
    verifyBothAxisMirror(srcData, result, width, height, channels);
}

// Test 8-bit unsigned C1IR_Ctx (in-place with context)
TEST_F(NppiMirrorTest, 8u_C1IR_Ctx) {
    const int width = 32;
    const int height = 32;
    const int channels = 1;

    auto srcData = createTestPattern<Npp8u>(width, height, channels);
    auto expectedData = std::vector<Npp8u>(width * height * channels);

    // Calculate expected result for horizontal mirror
    for (int y = 0; y < height; ++y) {
        int srcY = height - 1 - y;
        for (int x = 0; x < width; ++x) {
            expectedData[y * width + x] = srcData[srcY * width + x];
        }
    }

    Npp8u* d_src = allocateGpuMemory<Npp8u>(srcData.size());
    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStreamContext ctx;
    ctx.hStream = 0;

    NppStatus status = nppiMirror_8u_C1IR_Ctx(d_src, width * sizeof(Npp8u),
                                             roi, NPP_HORIZONTAL_AXIS, ctx);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_src, srcData.size());
    EXPECT_EQ(result, expectedData);
}

// Test 8-bit unsigned C3IR_Ctx (in-place with context)
TEST_F(NppiMirrorTest, 8u_C3IR_Ctx) {
    const int width = 32;
    const int height = 32;
    const int channels = 3;

    auto srcData = createTestPattern<Npp8u>(width, height, channels);
    auto expectedData = std::vector<Npp8u>(width * height * channels);

    // Calculate expected result for vertical mirror
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int srcX = width - 1 - x;
            for (int c = 0; c < channels; ++c) {
                expectedData[(y * width + x) * channels + c] =
                    srcData[(y * width + srcX) * channels + c];
            }
        }
    }

    Npp8u* d_src = allocateGpuMemory<Npp8u>(srcData.size());
    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStreamContext ctx;
    ctx.hStream = 0;

    NppStatus status = nppiMirror_8u_C3IR_Ctx(d_src, width * channels * sizeof(Npp8u),
                                             roi, NPP_VERTICAL_AXIS, ctx);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_src, srcData.size());
    EXPECT_EQ(result, expectedData);
}

// Test 8-bit unsigned C4IR_Ctx (in-place with context)
TEST_F(NppiMirrorTest, 8u_C4IR_Ctx) {
    const int width = 32;
    const int height = 32;
    const int channels = 4;

    auto srcData = createTestPattern<Npp8u>(width, height, channels);
    auto expectedData = std::vector<Npp8u>(width * height * channels);

    // Calculate expected result for both axis mirror
    for (int y = 0; y < height; ++y) {
        int srcY = height - 1 - y;
        for (int x = 0; x < width; ++x) {
            int srcX = width - 1 - x;
            for (int c = 0; c < channels; ++c) {
                expectedData[(y * width + x) * channels + c] =
                    srcData[(srcY * width + srcX) * channels + c];
            }
        }
    }

    Npp8u* d_src = allocateGpuMemory<Npp8u>(srcData.size());
    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStreamContext ctx;
    ctx.hStream = 0;

    NppStatus status = nppiMirror_8u_C4IR_Ctx(d_src, width * channels * sizeof(Npp8u),
                                             roi, NPP_BOTH_AXIS, ctx);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_src, srcData.size());
    EXPECT_EQ(result, expectedData);
}

// Test with maximum step size (padding)
TEST_F(NppiMirrorTest, WithPaddedStep) {
    const int width = 31;  // Odd width
    const int height = 31; // Odd height
    const int channels = 3;

    // Add padding to make step a multiple of 8
    int actualStep = width * channels * sizeof(Npp8u);
    int paddedStep = ((actualStep + 7) / 8) * 8;

    auto srcData = createTestPattern<Npp8u>(width, height, channels);
    auto dstData = std::vector<Npp8u>(width * height * channels, 0);

    // Create padded source data
    std::vector<Npp8u> srcPadded(height * paddedStep, 0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                srcPadded[y * paddedStep + x * channels + c] =
                    srcData[(y * width + x) * channels + c];
            }
        }
    }

    Npp8u* d_src = allocateGpuMemory<Npp8u>(height * paddedStep);
    Npp8u* d_dst = allocateGpuMemory<Npp8u>(height * paddedStep);

    copyToGpu(srcPadded, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_8u_C3R(d_src, paddedStep, d_dst, paddedStep,
                                        roi, NPP_HORIZONTAL_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto resultPadded = copyFromGpu(d_dst, height * paddedStep);
    std::vector<Npp8u> result(width * height * channels);

    // Extract actual data from padded result
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                result[(y * width + x) * channels + c] =
                    resultPadded[y * paddedStep + x * channels + c];
            }
        }
    }

    verifyHorizontalMirror(srcData, result, width, height, channels);
}

// Test overlapping source and destination (should work)
TEST_F(NppiMirrorTest, OverlappingMemory) {
    const int width = 64;
    const int height = 64;
    const int channels = 1;
    const int size = width * height * channels;

    // Allocate a single buffer twice as large
    Npp8u* d_buffer = allocateGpuMemory<Npp8u>(size * 2);

    // Use first half as source, second half as destination
    Npp8u* d_src = d_buffer;
    Npp8u* d_dst = d_buffer + size;

    auto srcData = createTestPattern<Npp8u>(width, height, channels);
    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_8u_C1R(d_src, width * sizeof(Npp8u),
                                        d_dst, width * sizeof(Npp8u),
                                        roi, NPP_VERTICAL_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_dst, size);
    verifyVerticalMirror(srcData, result, width, height, channels);
}

// Test large image
TEST_F(NppiMirrorTest, 8u_C3R_LargeImage) {
    const int width = 1024;
    const int height = 1024;
    const int channels = 3;

    auto srcData = createTestPattern<Npp8u>(width, height, channels);
    auto dstData = std::vector<Npp8u>(width * height * channels, 0);

    Npp8u* d_src = allocateGpuMemory<Npp8u>(srcData.size());
    Npp8u* d_dst = allocateGpuMemory<Npp8u>(dstData.size());

    copyToGpu(srcData, d_src);

    NppiSize roi = {width, height};
    NppStatus status = nppiMirror_8u_C3R(d_src, width * channels * sizeof(Npp8u),
                                        d_dst, width * channels * sizeof(Npp8u),
                                        roi, NPP_VERTICAL_AXIS);

    EXPECT_EQ(status, NPP_SUCCESS);

    auto result = copyFromGpu(d_dst, dstData.size());
    verifyVerticalMirror(srcData, result, width, height, channels);
}
