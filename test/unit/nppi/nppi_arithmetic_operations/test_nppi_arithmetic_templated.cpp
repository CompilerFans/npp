#include "npp.h"
#include "npp_test_base.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

template<typename T>
class NppiArithmeticTemplatedTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaError_t err = cudaSetDevice(0);
        ASSERT_EQ(err, cudaSuccess);
        
        // Initialize random generator
        gen.seed(12345);
        
        // Set up test data sizes
        width = 128;
        height = 128;
        totalPixels = width * height;
    }

    void TearDown() override {
        cudaError_t err = cudaDeviceSynchronize();
        EXPECT_EQ(err, cudaSuccess);
    }

    // Generate random test data
    std::vector<T> generateRandomData(size_t count) {
        std::vector<T> data(count);
        if constexpr (std::is_same_v<T, Npp8u>) {
            std::uniform_int_distribution<int> dis(0, 255);
            for (size_t i = 0; i < count; ++i) {
                data[i] = static_cast<T>(dis(gen));
            }
        } else if constexpr (std::is_same_v<T, Npp8s>) {
            std::uniform_int_distribution<int> dis(-128, 127);
            for (size_t i = 0; i < count; ++i) {
                data[i] = static_cast<T>(dis(gen));
            }
        } else if constexpr (std::is_same_v<T, Npp16u>) {
            std::uniform_int_distribution<int> dis(0, 65535);
            for (size_t i = 0; i < count; ++i) {
                data[i] = static_cast<T>(dis(gen));
            }
        } else if constexpr (std::is_same_v<T, Npp16s>) {
            std::uniform_int_distribution<int> dis(-32768, 32767);
            for (size_t i = 0; i < count; ++i) {
                data[i] = static_cast<T>(dis(gen));
            }
        } else if constexpr (std::is_same_v<T, Npp32f>) {
            std::uniform_real_distribution<float> dis(0.0f, 100.0f);
            for (size_t i = 0; i < count; ++i) {
                data[i] = static_cast<T>(dis(gen));
            }
        }
        return data;
    }

    // Helper function to allocate and copy data to GPU
    template<int Channels>
    void setupGPUData(const std::vector<T>& hostSrc1, const std::vector<T>& hostSrc2,
                      T*& d_src1, T*& d_src2, T*& d_dst, int& step) {
        if constexpr (std::is_same_v<T, Npp8u>) {
            if constexpr (Channels == 1) {
                d_src1 = nppiMalloc_8u_C1(width, height, &step);
                d_src2 = nppiMalloc_8u_C1(width, height, &step);
                d_dst = nppiMalloc_8u_C1(width, height, &step);
            } else if constexpr (Channels == 3) {
                d_src1 = nppiMalloc_8u_C3(width, height, &step);
                d_src2 = nppiMalloc_8u_C3(width, height, &step);
                d_dst = nppiMalloc_8u_C3(width, height, &step);
            } else if constexpr (Channels == 4) {
                d_src1 = nppiMalloc_8u_C4(width, height, &step);
                d_src2 = nppiMalloc_8u_C4(width, height, &step);
                d_dst = nppiMalloc_8u_C4(width, height, &step);
            }
        } else if constexpr (std::is_same_v<T, Npp16u>) {
            if constexpr (Channels == 1) {
                d_src1 = nppiMalloc_16u_C1(width, height, &step);
                d_src2 = nppiMalloc_16u_C1(width, height, &step);
                d_dst = nppiMalloc_16u_C1(width, height, &step);
            }
        } else if constexpr (std::is_same_v<T, Npp32f>) {
            if constexpr (Channels == 1) {
                d_src1 = nppiMalloc_32f_C1(width, height, &step);
                d_src2 = nppiMalloc_32f_C1(width, height, &step);
                d_dst = nppiMalloc_32f_C1(width, height, &step);
            } else if constexpr (Channels == 3) {
                d_src1 = nppiMalloc_32f_C3(width, height, &step);
                d_src2 = nppiMalloc_32f_C3(width, height, &step);
                d_dst = nppiMalloc_32f_C3(width, height, &step);
            }
        }

        ASSERT_NE(d_src1, nullptr);
        ASSERT_NE(d_src2, nullptr);
        ASSERT_NE(d_dst, nullptr);

        // Copy data to GPU
        int hostStep = width * Channels * sizeof(T);
        cudaMemcpy2D(d_src1, step, hostSrc1.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
        cudaMemcpy2D(d_src2, step, hostSrc2.data(), hostStep, hostStep, height, cudaMemcpyHostToDevice);
    }

    // Reference implementation for different operations
    template<int Channels>
    std::vector<T> calculateReference(const std::vector<T>& src1, const std::vector<T>& src2,
                                      const std::string& operation, int scaleFactor = 0) {
        std::vector<T> result(src1.size());
        
        for (size_t i = 0; i < src1.size(); ++i) {
            double val1 = static_cast<double>(src1[i]);
            double val2 = static_cast<double>(src2[i]);
            double res = 0.0;
            
            if (operation == "add") {
                res = val1 + val2;
            } else if (operation == "sub") {
                res = val2 - val1;  // NPP Sub convention: pSrc2 - pSrc1
            } else if (operation == "mul") {
                res = val1 * val2;
            } else if (operation == "div") {
                res = (val2 != 0) ? val1 / val2 : 0.0;
            } else if (operation == "absdiff") {
                res = std::abs(val1 - val2);
            }
            
            // Apply scaling for integer types
            // NVIDIA NPP uses truncation (not rounding) for scale factors
            if (scaleFactor > 0 && std::is_integral_v<T>) {
                res = res / (1 << scaleFactor);
            }
            
            // Saturate to type range
            result[i] = saturateToType(res);
        }
        
        return result;
    }

    T saturateToType(double value) {
        if constexpr (std::is_same_v<T, Npp8u>) {
            return static_cast<T>(std::max(0.0, std::min(255.0, value)));
        } else if constexpr (std::is_same_v<T, Npp8s>) {
            return static_cast<T>(std::max(-128.0, std::min(127.0, value)));
        } else if constexpr (std::is_same_v<T, Npp16u>) {
            return static_cast<T>(std::max(0.0, std::min(65535.0, value)));
        } else if constexpr (std::is_same_v<T, Npp16s>) {
            return static_cast<T>(std::max(-32768.0, std::min(32767.0, value)));
        } else if constexpr (std::is_same_v<T, Npp32f>) {
            return static_cast<T>(value);
        }
        return T{};
    }

    // Test binary operations
    template<int Channels>
    void testBinaryOperation(const std::string& operation, int scaleFactor = 0) {
        // Generate test data
        std::vector<T> hostSrc1 = generateRandomData(totalPixels * Channels);
        std::vector<T> hostSrc2 = generateRandomData(totalPixels * Channels);
        
        // Allocate GPU memory
        T *d_src1 = nullptr, *d_src2 = nullptr, *d_dst = nullptr;
        int step = 0;
        setupGPUData<Channels>(hostSrc1, hostSrc2, d_src1, d_src2, d_dst, step);
        
        NppiSize oSizeROI = {width, height};
        NppStatus status = NPP_SUCCESS;
        
        // Call appropriate NPP function based on type and operation
        if constexpr (std::is_same_v<T, Npp8u>) {
            if (operation == "add") {
                if constexpr (Channels == 1) {
                    status = nppiAdd_8u_C1RSfs(d_src1, step, d_src2, step, d_dst, step, oSizeROI, scaleFactor);
                } else if constexpr (Channels == 3) {
                    status = nppiAdd_8u_C3RSfs(d_src1, step, d_src2, step, d_dst, step, oSizeROI, scaleFactor);
                }
            } else if (operation == "sub") {
                if constexpr (Channels == 1) {
                    status = nppiSub_8u_C1RSfs(d_src1, step, d_src2, step, d_dst, step, oSizeROI, scaleFactor);
                }
            }
        } else if constexpr (std::is_same_v<T, Npp32f>) {
            if (operation == "add") {
                if constexpr (Channels == 1) {
                    status = nppiAdd_32f_C1R(d_src1, step, d_src2, step, d_dst, step, oSizeROI);
                } else if constexpr (Channels == 3) {
                    status = nppiAdd_32f_C3R(d_src1, step, d_src2, step, d_dst, step, oSizeROI);
                }
            }
        }
        
        ASSERT_EQ(status, NPP_SUCCESS);
        
        // Copy result back and verify
        std::vector<T> hostResult(totalPixels * Channels);
        int hostStep = width * Channels * sizeof(T);
        cudaMemcpy2D(hostResult.data(), hostStep, d_dst, step, hostStep, height, cudaMemcpyDeviceToHost);
        
        // Calculate reference
        std::vector<T> reference = calculateReference<Channels>(hostSrc1, hostSrc2, operation, scaleFactor);
        
        // Verify results with tolerance for scale factor differences
        for (size_t i = 0; i < hostResult.size(); ++i) {
            if constexpr (std::is_integral_v<T>) {
                // For integer types with scale factors, allow tolerance of 1
                // Due to different rounding implementations between NVIDIA NPP and reference
                if (scaleFactor > 0) {
                    EXPECT_NEAR(static_cast<double>(hostResult[i]), static_cast<double>(reference[i]), 1.0) 
                        << "Mismatch at index " << i << " (scaleFactor=" << scaleFactor << ")";
                } else {
                    EXPECT_EQ(hostResult[i], reference[i]) << "Mismatch at index " << i;
                }
            } else {
                EXPECT_NEAR(static_cast<float>(hostResult[i]), static_cast<float>(reference[i]), 1e-5f) 
                    << "Mismatch at index " << i;
            }
        }
        
        // Cleanup
        nppiFree(d_src1);
        nppiFree(d_src2);
        nppiFree(d_dst);
    }

protected:
    std::mt19937 gen;
    int width, height;
    size_t totalPixels;
};

// Instantiate test for different types
using TestTypes = ::testing::Types<Npp8u, Npp16u, Npp32f>;
TYPED_TEST_SUITE(NppiArithmeticTemplatedTest, TestTypes);

TYPED_TEST(NppiArithmeticTemplatedTest, AddOperation_C1) {
    if constexpr (std::is_same_v<TypeParam, Npp8u>) {
        this->template testBinaryOperation<1>("add", 2); // Test with scaling
    } else if constexpr (std::is_same_v<TypeParam, Npp32f>) {
        this->template testBinaryOperation<1>("add", 0); // No scaling for float
    }
}

TYPED_TEST(NppiArithmeticTemplatedTest, AddOperation_C3) {
    if constexpr (std::is_same_v<TypeParam, Npp8u>) {
        this->template testBinaryOperation<3>("add", 1);
    } else if constexpr (std::is_same_v<TypeParam, Npp32f>) {
        this->template testBinaryOperation<3>("add", 0);
    }
}

TYPED_TEST(NppiArithmeticTemplatedTest, SubOperation_C1) {
    if constexpr (std::is_same_v<TypeParam, Npp8u>) {
        this->template testBinaryOperation<1>("sub", 1);
    }
}
