/**
 * @file test_dual_source_comparison.cpp
 * @brief 双源算术运算与NVIDIA NPP的对比测试
 * 
 * 测试Add、Sub、Mul、Div双源函数的功能一致性和性能对比
 */

#include "test/framework/nvidia_comparison_test_base.h"
#include <random>
#include <algorithm>

using namespace test_framework;

class NPPIDualSourceComparisonTest : public NvidiaComparisonTestBase {
protected:
    // 生成随机测试数据
    template<typename T>
    void generateRandomData(std::vector<T>& data, T minVal, T maxVal) {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        if constexpr (std::is_floating_point<T>::value) {
            std::uniform_real_distribution<T> dis(minVal, maxVal);
            for (auto& val : data) {
                val = dis(gen);
            }
        } else {
            std::uniform_int_distribution<int> dis(minVal, maxVal);
            for (auto& val : data) {
                val = static_cast<T>(dis(gen));
            }
        }
    }
    
    // 通用的双源算术运算对比测试
    template<typename T>
    void compareDualSourceOperation(
        const std::string& opName,
        int width, int height,
        const std::vector<T>& src1Data,
        const std::vector<T>& src2Data,
        int scaleFactor,
        std::function<NppStatus(const T*, int, const T*, int, T*, int, NppiSize, int)> openFunc,
        std::function<NppStatus(const T*, int, const T*, int, T*, int, NppiSize, int)> nvFunc) {
        
        skipIfNoNvidiaNpp("Skipping " + opName + " dual source comparison");
        
        // 如果没有NVIDIA NPP，直接返回
        if (!hasNvidiaNpp()) {
            return;
        }
        
        // 分配内存
        int src1Step, src2Step, dstStep;
        T* openSrc1 = allocateAndInitialize(width, height, &src1Step, src1Data);
        T* openSrc2 = allocateAndInitialize(width, height, &src2Step, src2Data);
        T* openDst = allocateAndInitialize<T>(width, height, &dstStep, {});
        
        int nvSrc1Step, nvSrc2Step, nvDstStep;
        T* nvSrc1 = allocateWithNvidia<T>(width, height, &nvSrc1Step);
        T* nvSrc2 = allocateWithNvidia<T>(width, height, &nvSrc2Step);
        T* nvDst = allocateWithNvidia<T>(width, height, &nvDstStep);
        
        ASSERT_NE(openSrc1, nullptr);
        ASSERT_NE(openSrc2, nullptr);
        ASSERT_NE(openDst, nullptr);
        ASSERT_NE(nvSrc1, nullptr);
        ASSERT_NE(nvSrc2, nullptr);
        ASSERT_NE(nvDst, nullptr);
        
        // 复制数据到NVIDIA缓冲区
        cudaMemcpy2D(nvSrc1, nvSrc1Step, src1Data.data(), width * sizeof(T),
                    width * sizeof(T), height, cudaMemcpyHostToDevice);
        cudaMemcpy2D(nvSrc2, nvSrc2Step, src2Data.data(), width * sizeof(T),
                    width * sizeof(T), height, cudaMemcpyHostToDevice);
        
        // 执行操作
        NppiSize roi = {width, height};
        NppStatus openStatus = openFunc(openSrc1, src1Step, openSrc2, src2Step, openDst, dstStep, roi, scaleFactor);
        NppStatus nvStatus = nvFunc(nvSrc1, nvSrc1Step, nvSrc2, nvSrc2Step, nvDst, nvDstStep, roi, scaleFactor);
        
        EXPECT_EQ(openStatus, NPP_NO_ERROR) << "OpenNPP " << opName << " failed";
        EXPECT_EQ(nvStatus, NPP_NO_ERROR) << "NVIDIA " << opName << " failed";
        
        // 比较结果
        std::vector<T> openResult(width * height);
        std::vector<T> nvResult(width * height);
        
        cudaMemcpy2D(openResult.data(), width * sizeof(T), openDst, dstStep,
                    width * sizeof(T), height, cudaMemcpyDeviceToHost);
        cudaMemcpy2D(nvResult.data(), width * sizeof(T), nvDst, nvDstStep,
                    width * sizeof(T), height, cudaMemcpyDeviceToHost);
        
        ConsistencyResult consistency;
        consistency.totalElements = width * height;
        
        if constexpr (std::is_floating_point<T>::value) {
            consistency.passed = compareArrays(openResult.data(), nvResult.data(),
                                             consistency.totalElements, T(1e-5), &consistency.mismatchCount);
        } else {
            consistency.passed = compareArrays(openResult.data(), nvResult.data(),
                                             consistency.totalElements, 1, &consistency.mismatchCount);
        }
        
        consistency.mismatchRate = (consistency.mismatchCount * 100.0) / consistency.totalElements;
        
        printConsistencyResult(consistency, opName + " " + std::to_string(width) + "x" + std::to_string(height));
        
        // 清理
        freeMemory(openSrc1);
        freeMemory(openSrc2);
        freeMemory(openDst);
        freeMemory(nvSrc1, true);
        freeMemory(nvSrc2, true);
        freeMemory(nvDst, true);
    }
    
    // 32-bit float版本（无缩放因子）
    template<typename T>
    void compareDualSourceOperationFloat(
        const std::string& opName,
        int width, int height,
        const std::vector<T>& src1Data,
        const std::vector<T>& src2Data,
        std::function<NppStatus(const T*, int, const T*, int, T*, int, NppiSize)> openFunc,
        std::function<NppStatus(const T*, int, const T*, int, T*, int, NppiSize)> nvFunc) {
        
        skipIfNoNvidiaNpp("Skipping " + opName + " dual source comparison");
        
        // 如果没有NVIDIA NPP，直接返回
        if (!hasNvidiaNpp()) {
            return;
        }
        
        // 分配内存
        int src1Step, src2Step, dstStep;
        T* openSrc1 = allocateAndInitialize(width, height, &src1Step, src1Data);
        T* openSrc2 = allocateAndInitialize(width, height, &src2Step, src2Data);
        T* openDst = allocateAndInitialize<T>(width, height, &dstStep, {});
        
        int nvSrc1Step, nvSrc2Step, nvDstStep;
        T* nvSrc1 = allocateWithNvidia<T>(width, height, &nvSrc1Step);
        T* nvSrc2 = allocateWithNvidia<T>(width, height, &nvSrc2Step);
        T* nvDst = allocateWithNvidia<T>(width, height, &nvDstStep);
        
        ASSERT_NE(openSrc1, nullptr);
        ASSERT_NE(openSrc2, nullptr);
        ASSERT_NE(openDst, nullptr);
        ASSERT_NE(nvSrc1, nullptr);
        ASSERT_NE(nvSrc2, nullptr);
        ASSERT_NE(nvDst, nullptr);
        
        // 复制数据到NVIDIA缓冲区
        cudaMemcpy2D(nvSrc1, nvSrc1Step, src1Data.data(), width * sizeof(T),
                    width * sizeof(T), height, cudaMemcpyHostToDevice);
        cudaMemcpy2D(nvSrc2, nvSrc2Step, src2Data.data(), width * sizeof(T),
                    width * sizeof(T), height, cudaMemcpyHostToDevice);
        
        // 执行操作
        NppiSize roi = {width, height};
        NppStatus openStatus = openFunc(openSrc1, src1Step, openSrc2, src2Step, openDst, dstStep, roi);
        NppStatus nvStatus = nvFunc(nvSrc1, nvSrc1Step, nvSrc2, nvSrc2Step, nvDst, nvDstStep, roi);
        
        EXPECT_EQ(openStatus, NPP_NO_ERROR) << "OpenNPP " << opName << " failed";
        EXPECT_EQ(nvStatus, NPP_NO_ERROR) << "NVIDIA " << opName << " failed";
        
        // 比较结果
        std::vector<T> openResult(width * height);
        std::vector<T> nvResult(width * height);
        
        cudaMemcpy2D(openResult.data(), width * sizeof(T), openDst, dstStep,
                    width * sizeof(T), height, cudaMemcpyDeviceToHost);
        cudaMemcpy2D(nvResult.data(), width * sizeof(T), nvDst, nvDstStep,
                    width * sizeof(T), height, cudaMemcpyDeviceToHost);
        
        ConsistencyResult consistency;
        consistency.totalElements = width * height;
        consistency.passed = compareArrays(openResult.data(), nvResult.data(),
                                         consistency.totalElements, T(1e-5), &consistency.mismatchCount);
        consistency.mismatchRate = (consistency.mismatchCount * 100.0) / consistency.totalElements;
        
        printConsistencyResult(consistency, opName + " " + std::to_string(width) + "x" + std::to_string(height));
        
        // 清理
        freeMemory(openSrc1);
        freeMemory(openSrc2);
        freeMemory(openDst);
        freeMemory(nvSrc1, true);
        freeMemory(nvSrc2, true);
        freeMemory(nvDst, true);
    }
};

// ==================== Add双源对比测试 ====================

TEST_F(NPPIDualSourceComparisonTest, Add_8u_C1RSfs_Comparison) {
    const int width = 512;
    const int height = 512;
    const int scaleFactor = 0;
    
    // 准备测试数据
    std::vector<Npp8u> src1Data(width * height);
    std::vector<Npp8u> src2Data(width * height);
    generateRandomData(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(100));
    generateRandomData(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(100));
    
    compareDualSourceOperation<Npp8u>(
        "Add_8u_C1RSfs",
        width, height,
        src1Data, src2Data,
        scaleFactor,
        [](const Npp8u* pSrc1, int nSrc1Step, const Npp8u* pSrc2, int nSrc2Step,
           Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return nppiAdd_8u_C1RSfs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor);
        },
        [this](const Npp8u* pSrc1, int nSrc1Step, const Npp8u* pSrc2, int nSrc2Step,
               Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return nvidiaLoader_->nv_nppiAdd_8u_C1RSfs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor);
        }
    );
}

TEST_F(NPPIDualSourceComparisonTest, Add_32f_C1R_Comparison) {
    const int width = 640;
    const int height = 480;
    
    // 准备测试数据
    std::vector<Npp32f> src1Data(width * height);
    std::vector<Npp32f> src2Data(width * height);
    generateRandomData(src1Data, -50.0f, 50.0f);
    generateRandomData(src2Data, -50.0f, 50.0f);
    
    compareDualSourceOperationFloat<Npp32f>(
        "Add_32f_C1R",
        width, height,
        src1Data, src2Data,
        [](const Npp32f* pSrc1, int nSrc1Step, const Npp32f* pSrc2, int nSrc2Step,
           Npp32f* pDst, int nDstStep, NppiSize oSizeROI) {
            return nppiAdd_32f_C1R(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI);
        },
        [this](const Npp32f* pSrc1, int nSrc1Step, const Npp32f* pSrc2, int nSrc2Step,
               Npp32f* pDst, int nDstStep, NppiSize oSizeROI) {
            return nvidiaLoader_->nv_nppiAdd_32f_C1R(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI);
        }
    );
}

// ==================== Sub双源对比测试 ====================

TEST_F(NPPIDualSourceComparisonTest, Sub_8u_C1RSfs_Comparison) {
    const int width = 256;
    const int height = 256;
    const int scaleFactor = 0;
    
    // 准备测试数据
    std::vector<Npp8u> src1Data(width * height);
    std::vector<Npp8u> src2Data(width * height);
    generateRandomData(src1Data, static_cast<Npp8u>(100), static_cast<Npp8u>(255));
    generateRandomData(src2Data, static_cast<Npp8u>(0), static_cast<Npp8u>(100));
    
    compareDualSourceOperation<Npp8u>(
        "Sub_8u_C1RSfs",
        width, height,
        src1Data, src2Data,
        scaleFactor,
        [](const Npp8u* pSrc1, int nSrc1Step, const Npp8u* pSrc2, int nSrc2Step,
           Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return nppiSub_8u_C1RSfs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor);
        },
        [this](const Npp8u* pSrc1, int nSrc1Step, const Npp8u* pSrc2, int nSrc2Step,
               Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return nvidiaLoader_->nv_nppiSub_8u_C1RSfs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor);
        }
    );
}

TEST_F(NPPIDualSourceComparisonTest, Sub_32f_C1R_Comparison) {
    const int width = 1024;
    const int height = 768;
    
    // 准备测试数据
    std::vector<Npp32f> src1Data(width * height);
    std::vector<Npp32f> src2Data(width * height);
    generateRandomData(src1Data, -100.0f, 100.0f);
    generateRandomData(src2Data, -50.0f, 50.0f);
    
    compareDualSourceOperationFloat<Npp32f>(
        "Sub_32f_C1R",
        width, height,
        src1Data, src2Data,
        [](const Npp32f* pSrc1, int nSrc1Step, const Npp32f* pSrc2, int nSrc2Step,
           Npp32f* pDst, int nDstStep, NppiSize oSizeROI) {
            return nppiSub_32f_C1R(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI);
        },
        [this](const Npp32f* pSrc1, int nSrc1Step, const Npp32f* pSrc2, int nSrc2Step,
               Npp32f* pDst, int nDstStep, NppiSize oSizeROI) {
            return nvidiaLoader_->nv_nppiSub_32f_C1R(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI);
        }
    );
}

// ==================== Mul双源对比测试 ====================

TEST_F(NPPIDualSourceComparisonTest, Mul_8u_C1RSfs_Comparison) {
    const int width = 128;
    const int height = 128;
    const int scaleFactor = 8;  // 乘法需要缩放因子避免溢出
    
    // 准备测试数据
    std::vector<Npp8u> src1Data(width * height);
    std::vector<Npp8u> src2Data(width * height);
    generateRandomData(src1Data, static_cast<Npp8u>(1), static_cast<Npp8u>(16));
    generateRandomData(src2Data, static_cast<Npp8u>(1), static_cast<Npp8u>(16));
    
    compareDualSourceOperation<Npp8u>(
        "Mul_8u_C1RSfs",
        width, height,
        src1Data, src2Data,
        scaleFactor,
        [](const Npp8u* pSrc1, int nSrc1Step, const Npp8u* pSrc2, int nSrc2Step,
           Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return nppiMul_8u_C1RSfs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor);
        },
        [this](const Npp8u* pSrc1, int nSrc1Step, const Npp8u* pSrc2, int nSrc2Step,
               Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return nvidiaLoader_->nv_nppiMul_8u_C1RSfs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor);
        }
    );
}

TEST_F(NPPIDualSourceComparisonTest, Mul_32f_C1R_Comparison) {
    const int width = 800;
    const int height = 600;
    
    // 准备测试数据
    std::vector<Npp32f> src1Data(width * height);
    std::vector<Npp32f> src2Data(width * height);
    generateRandomData(src1Data, -10.0f, 10.0f);
    generateRandomData(src2Data, -5.0f, 5.0f);
    
    compareDualSourceOperationFloat<Npp32f>(
        "Mul_32f_C1R",
        width, height,
        src1Data, src2Data,
        [](const Npp32f* pSrc1, int nSrc1Step, const Npp32f* pSrc2, int nSrc2Step,
           Npp32f* pDst, int nDstStep, NppiSize oSizeROI) {
            return nppiMul_32f_C1R(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI);
        },
        [this](const Npp32f* pSrc1, int nSrc1Step, const Npp32f* pSrc2, int nSrc2Step,
               Npp32f* pDst, int nDstStep, NppiSize oSizeROI) {
            return nvidiaLoader_->nv_nppiMul_32f_C1R(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI);
        }
    );
}

// ==================== Div双源对比测试 ====================

TEST_F(NPPIDualSourceComparisonTest, Div_8u_C1RSfs_Comparison) {
    const int width = 256;
    const int height = 256;
    const int scaleFactor = 0;
    
    // 准备测试数据 - 避免除零
    std::vector<Npp8u> src1Data(width * height);
    std::vector<Npp8u> src2Data(width * height);
    generateRandomData(src1Data, static_cast<Npp8u>(0), static_cast<Npp8u>(255));
    generateRandomData(src2Data, static_cast<Npp8u>(1), static_cast<Npp8u>(255));  // 避免除零
    
    compareDualSourceOperation<Npp8u>(
        "Div_8u_C1RSfs",
        width, height,
        src1Data, src2Data,
        scaleFactor,
        [](const Npp8u* pSrc1, int nSrc1Step, const Npp8u* pSrc2, int nSrc2Step,
           Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return nppiDiv_8u_C1RSfs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor);
        },
        [this](const Npp8u* pSrc1, int nSrc1Step, const Npp8u* pSrc2, int nSrc2Step,
               Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return nvidiaLoader_->nv_nppiDiv_8u_C1RSfs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nScaleFactor);
        }
    );
}

TEST_F(NPPIDualSourceComparisonTest, Div_32f_C1R_Comparison) {
    const int width = 512;
    const int height = 384;
    
    // 准备测试数据
    std::vector<Npp32f> src1Data(width * height);
    std::vector<Npp32f> src2Data(width * height);
    generateRandomData(src1Data, -100.0f, 100.0f);
    generateRandomData(src2Data, -10.0f, 10.0f);
    
    // 避免除零：将接近零的值替换
    for (auto& val : src2Data) {
        if (std::abs(val) < 1e-6f) {
            val = val >= 0 ? 1e-6f : -1e-6f;
        }
    }
    
    compareDualSourceOperationFloat<Npp32f>(
        "Div_32f_C1R",
        width, height,
        src1Data, src2Data,
        [](const Npp32f* pSrc1, int nSrc1Step, const Npp32f* pSrc2, int nSrc2Step,
           Npp32f* pDst, int nDstStep, NppiSize oSizeROI) {
            return nppiDiv_32f_C1R(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI);
        },
        [this](const Npp32f* pSrc1, int nSrc1Step, const Npp32f* pSrc2, int nSrc2Step,
               Npp32f* pDst, int nDstStep, NppiSize oSizeROI) {
            return nvidiaLoader_->nv_nppiDiv_32f_C1R(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI);
        }
    );
}

// ==================== 性能对比测试 ====================

TEST_F(NPPIDualSourceComparisonTest, DualSource_PerformanceComparison) {
    skipIfNoNvidiaNpp("Skipping dual source performance comparison");
    
    // 如果没有NVIDIA NPP，直接返回
    if (!hasNvidiaNpp()) {
        return;
    }
    
    const int width = 1024;
    const int height = 1024;
    
    std::cout << "\n=== Dual Source Performance Comparison ===\n";
    std::cout << "Image size: " << width << "x" << height << "\n";
    std::cout << std::setw(15) << "Operation"
              << std::setw(20) << "OpenNPP (ms)"
              << std::setw(20) << "NVIDIA (ms)"
              << std::setw(15) << "Speedup" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    // 准备测试数据
    std::vector<Npp32f> src1Data(width * height);
    std::vector<Npp32f> src2Data(width * height);
    generateRandomData(src1Data, -50.0f, 50.0f);
    generateRandomData(src2Data, -25.0f, 25.0f);
    
    // 避免除零
    for (auto& val : src2Data) {
        if (std::abs(val) < 1e-6f) {
            val = val >= 0 ? 1e-6f : -1e-6f;
        }
    }
    
    struct TestOp {
        std::string name;
        std::function<double()> openTest;
        std::function<double()> nvTest;
    };
    
    // 分配内存
    int src1Step, src2Step, dstStep;
    Npp32f* openSrc1 = allocateAndInitialize(width, height, &src1Step, src1Data);
    Npp32f* openSrc2 = allocateAndInitialize(width, height, &src2Step, src2Data);
    Npp32f* openDst = allocateAndInitialize<Npp32f>(width, height, &dstStep, {});
    
    int nvSrc1Step, nvSrc2Step, nvDstStep;
    Npp32f* nvSrc1 = allocateWithNvidia<Npp32f>(width, height, &nvSrc1Step);
    Npp32f* nvSrc2 = allocateWithNvidia<Npp32f>(width, height, &nvSrc2Step);
    Npp32f* nvDst = allocateWithNvidia<Npp32f>(width, height, &nvDstStep);
    
    // 复制数据到NVIDIA缓冲区
    cudaMemcpy2D(nvSrc1, nvSrc1Step, src1Data.data(), width * sizeof(Npp32f),
                width * sizeof(Npp32f), height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(nvSrc2, nvSrc2Step, src2Data.data(), width * sizeof(Npp32f),
                width * sizeof(Npp32f), height, cudaMemcpyHostToDevice);
    
    NppiSize roi = {width, height};
    
    std::vector<TestOp> operations = {
        {"Add_32f_C1R", 
         [&]() { return measureExecutionTime([&]() {
             nppiAdd_32f_C1R(openSrc1, src1Step, openSrc2, src2Step, openDst, dstStep, roi);
         }); },
         [&]() { return measureExecutionTime([&]() {
             nvidiaLoader_->nv_nppiAdd_32f_C1R(nvSrc1, nvSrc1Step, nvSrc2, nvSrc2Step, nvDst, nvDstStep, roi);
         }); }
        },
        {"Sub_32f_C1R", 
         [&]() { return measureExecutionTime([&]() {
             nppiSub_32f_C1R(openSrc1, src1Step, openSrc2, src2Step, openDst, dstStep, roi);
         }); },
         [&]() { return measureExecutionTime([&]() {
             nvidiaLoader_->nv_nppiSub_32f_C1R(nvSrc1, nvSrc1Step, nvSrc2, nvSrc2Step, nvDst, nvDstStep, roi);
         }); }
        },
        {"Mul_32f_C1R", 
         [&]() { return measureExecutionTime([&]() {
             nppiMul_32f_C1R(openSrc1, src1Step, openSrc2, src2Step, openDst, dstStep, roi);
         }); },
         [&]() { return measureExecutionTime([&]() {
             nvidiaLoader_->nv_nppiMul_32f_C1R(nvSrc1, nvSrc1Step, nvSrc2, nvSrc2Step, nvDst, nvDstStep, roi);
         }); }
        },
        {"Div_32f_C1R", 
         [&]() { return measureExecutionTime([&]() {
             nppiDiv_32f_C1R(openSrc1, src1Step, openSrc2, src2Step, openDst, dstStep, roi);
         }); },
         [&]() { return measureExecutionTime([&]() {
             nvidiaLoader_->nv_nppiDiv_32f_C1R(nvSrc1, nvSrc1Step, nvSrc2, nvSrc2Step, nvDst, nvDstStep, roi);
         }); }
        }
    };
    
    for (const auto& op : operations) {
        double openTime = op.openTest();
        double nvTime = op.nvTest();
        double speedup = nvTime > 0 ? openTime / nvTime : 0;
        
        std::cout << std::setw(15) << op.name
                  << std::setw(20) << std::fixed << std::setprecision(3) << openTime
                  << std::setw(20) << (nvTime > 0 ? std::to_string(nvTime) : "N/A")
                  << std::setw(15) << std::fixed << std::setprecision(2)
                  << (speedup > 0 ? std::to_string(speedup) + "x" : "N/A")
                  << std::endl;
    }
    
    // 清理
    freeMemory(openSrc1);
    freeMemory(openSrc2);
    freeMemory(openDst);
    freeMemory(nvSrc1, true);
    freeMemory(nvSrc2, true);
    freeMemory(nvDst, true);
}