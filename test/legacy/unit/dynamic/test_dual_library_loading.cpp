/**
 * @file test_dual_library_loading.cpp
 * @brief 双库动态加载测试
 * 
 * 测试同时动态加载OpenNPP和NVIDIA NPP，验证符号解析和基本功能
 */

#include "test/framework/dual_library_test_base.h"
#include <vector>

using namespace test_framework;

class DualLibraryLoadingTest : public DualLibraryTestBase {
protected:
    void SetUp() override {
        DualLibraryTestBase::SetUp();
    }
};

// ==================== 基本加载测试 ====================

TEST_F(DualLibraryLoadingTest, LibraryAvailability) {
    std::cout << "\n=== Library Availability Test ===" << std::endl;
    
    std::cout << "OpenNPP: " << (hasOpenNpp() ? "Available ✓" : "Not Available ✗") << std::endl;
    if (!hasOpenNpp()) {
        std::cout << "  Error: " << openNppLoader_->getErrorMessage() << std::endl;
    }
    
    std::cout << "NVIDIA NPP: " << (hasNvidiaNpp() ? "Available ✓" : "Not Available ✗") << std::endl;
    if (!hasNvidiaNpp()) {
        std::cout << "  Error: " << nvidiaLoader_->getErrorMessage() << std::endl;
    }
    
    std::cout << "Both libraries: " << (hasBothLibraries() ? "Available ✓" : "Not Available ✗") << std::endl;
    
    // 至少有一个库可用
    EXPECT_TRUE(hasOpenNpp() || hasNvidiaNpp()) << "At least one library should be available";
}

// ==================== 核心函数对比测试 ====================

TEST_F(DualLibraryLoadingTest, CoreFunctions_DeviceProperties) {
    std::cout << "\n=== Core Functions: Device Properties ===" << std::endl;
    
    if (hasOpenNpp()) {
        std::cout << "OpenNPP Device Properties:" << std::endl;
        if (openNppLoader_->op_nppGetGpuNumSMs) {
            int numSMs = openNppLoader_->op_nppGetGpuNumSMs();
            std::cout << "  Number of SMs: " << numSMs << std::endl;
            EXPECT_GT(numSMs, 0) << "OpenNPP should report positive SM count";
        }
        
        if (openNppLoader_->op_nppGetGpuName) {
            const char* gpuName = openNppLoader_->op_nppGetGpuName();
            std::cout << "  GPU Name: " << (gpuName ? gpuName : "NULL") << std::endl;
            EXPECT_NE(gpuName, nullptr) << "OpenNPP should report GPU name";
        }
    }
    
    if (hasNvidiaNpp()) {
        std::cout << "NVIDIA NPP Device Properties:" << std::endl;
        if (nvidiaLoader_->nv_nppGetGpuNumSMs) {
            int numSMs = nvidiaLoader_->nv_nppGetGpuNumSMs();
            std::cout << "  Number of SMs: " << numSMs << std::endl;
            EXPECT_GT(numSMs, 0) << "NVIDIA NPP should report positive SM count";
        }
        
        if (nvidiaLoader_->nv_nppGetGpuName) {
            const char* gpuName = nvidiaLoader_->nv_nppGetGpuName();
            std::cout << "  GPU Name: " << (gpuName ? gpuName : "NULL") << std::endl;
            EXPECT_NE(gpuName, nullptr) << "NVIDIA NPP should report GPU name";
        }
    }
    
    // 如果两个库都可用，比较结果
    if (hasBothLibraries()) {
        std::cout << "Comparing results between libraries..." << std::endl;
        
        if (openNppLoader_->op_nppGetGpuNumSMs && nvidiaLoader_->nv_nppGetGpuNumSMs) {
            int openSMs = openNppLoader_->op_nppGetGpuNumSMs();
            int nvidiaSMs = nvidiaLoader_->nv_nppGetGpuNumSMs();
            EXPECT_EQ(openSMs, nvidiaSMs) << "Both libraries should report same SM count";
        }
    }
}

TEST_F(DualLibraryLoadingTest, CoreFunctions_StreamManagement) {
    std::cout << "\n=== Core Functions: Stream Management ===" << std::endl;
    
    if (hasOpenNpp()) {
        std::cout << "Testing OpenNPP stream management..." << std::endl;
        
        if (openNppLoader_->op_nppGetStream) {
            cudaStream_t stream = openNppLoader_->op_nppGetStream();
            std::cout << "  Default stream: " << (stream == nullptr ? "NULL" : 
                (stream == cudaStreamLegacy ? "Legacy" : "Custom")) << std::endl;
        }
        
        if (openNppLoader_->op_nppSetStream) {
            NppStatus status = openNppLoader_->op_nppSetStream(nullptr);
            std::cout << "  Set NULL stream status: " << status << std::endl;
            EXPECT_EQ(status, NPP_NO_ERROR) << "OpenNPP should accept NULL stream";
        }
    }
    
    if (hasNvidiaNpp()) {
        std::cout << "Testing NVIDIA NPP stream management..." << std::endl;
        
        if (nvidiaLoader_->nv_nppGetStream) {
            cudaStream_t stream = nvidiaLoader_->nv_nppGetStream();
            std::cout << "  Default stream: " << (stream == nullptr ? "NULL" : 
                (stream == cudaStreamLegacy ? "Legacy" : "Custom")) << std::endl;
        }
        
        if (nvidiaLoader_->nv_nppSetStream) {
            NppStatus status = nvidiaLoader_->nv_nppSetStream(nullptr);
            std::cout << "  Set NULL stream status: " << status << std::endl;
            EXPECT_EQ(status, NPP_NO_ERROR) << "NVIDIA NPP should accept NULL stream";
        }
    }
}

// ==================== 内存分配测试 ====================

TEST_F(DualLibraryLoadingTest, MemoryAllocation) {
    std::cout << "\n=== Memory Allocation Test ===" << std::endl;
    
    const int width = 64;
    const int height = 32;
    int step_open = 0, step_nvidia = 0;
    
    if (hasOpenNpp()) {
        std::cout << "Testing OpenNPP memory allocation..." << std::endl;
        
        Npp8u* ptr_open = allocateWithOpenNpp<Npp8u>(width, height, &step_open);
        if (ptr_open) {
            std::cout << "  OpenNPP 8u allocation: SUCCESS (step=" << step_open << ")" << std::endl;
            freeOpenNppMemory(ptr_open);
            EXPECT_GT(step_open, 0) << "OpenNPP step should be positive";
        } else {
            std::cout << "  OpenNPP 8u allocation: FAILED" << std::endl;
        }
    }
    
    if (hasNvidiaNpp()) {
        std::cout << "Testing NVIDIA NPP memory allocation..." << std::endl;
        
        Npp8u* ptr_nvidia = allocateWithNvidia<Npp8u>(width, height, &step_nvidia);
        if (ptr_nvidia) {
            std::cout << "  NVIDIA NPP 8u allocation: SUCCESS (step=" << step_nvidia << ")" << std::endl;
            freeNvidiaMemory(ptr_nvidia);
            EXPECT_GT(step_nvidia, 0) << "NVIDIA NPP step should be positive";
        } else {
            std::cout << "  NVIDIA NPP 8u allocation: FAILED" << std::endl;
        }
    }
    
    // 如果两个库都可用，比较步长
    if (hasBothLibraries() && step_open > 0 && step_nvidia > 0) {
        std::cout << "Comparing memory layouts..." << std::endl;
        std::cout << "  OpenNPP step: " << step_open << std::endl;
        std::cout << "  NVIDIA step: " << step_nvidia << std::endl;
        
        // 步长可能不同（由于对齐策略），但都应该大于等于width
        EXPECT_GE(step_open, width) << "OpenNPP step should be at least width";
        EXPECT_GE(step_nvidia, width) << "NVIDIA step should be at least width";
    }
}

// ==================== 简单算术运算测试 ====================

TEST_F(DualLibraryLoadingTest, SimpleArithmetic_AddC) {
    std::cout << "\n=== Simple Arithmetic: AddC Test ===" << std::endl;
    
    const int width = 8;
    const int height = 4;
    const int size = width * height;
    
    // 创建测试数据
    std::vector<Npp8u> hostData(size);
    std::vector<Npp8u> resultOpen(size), resultNvidia(size);
    
    for (int i = 0; i < size; i++) {
        hostData[i] = i % 100;  // 避免溢出
    }
    
    const Npp8u constant = 50;
    const int scaleFactor = 0;
    
    if (hasOpenNpp()) {
        std::cout << "Testing OpenNPP AddC..." << std::endl;
        
        int step = 0;
        Npp8u* d_src = allocateWithOpenNpp<Npp8u>(width, height, &step);
        Npp8u* d_dst = allocateWithOpenNpp<Npp8u>(width, height, &step);
        
        if (d_src && d_dst && openNppLoader_->op_nppiAddC_8u_C1RSfs_Ctx) {
            // 拷贝数据到GPU
            initializeDeviceData(d_src, step, width, height, hostData);
            
            // 创建流上下文
            NppStreamContext ctx;
            if (openNppLoader_->op_nppGetStreamContext) {
                openNppLoader_->op_nppGetStreamContext(&ctx);
            } else {
                memset(&ctx, 0, sizeof(ctx));
                ctx.hStream = cudaStreamLegacy;
            }
            
            // 执行AddC
            NppiSize roiSize = {width, height};
            NppStatus status = openNppLoader_->op_nppiAddC_8u_C1RSfs_Ctx(
                d_src, step, constant, d_dst, step, roiSize, scaleFactor, ctx);
            
            std::cout << "  OpenNPP AddC status: " << status << std::endl;
            EXPECT_EQ(status, NPP_NO_ERROR) << "OpenNPP AddC should succeed";
            
            if (status == NPP_NO_ERROR) {
                // 拷贝结果回CPU
                cudaMemcpy2D(resultOpen.data(), width * sizeof(Npp8u),
                            d_dst, step, width * sizeof(Npp8u), height,
                            cudaMemcpyDeviceToHost);
            }
        }
        
        freeOpenNppMemory(d_src);
        freeOpenNppMemory(d_dst);
    }
    
    if (hasNvidiaNpp()) {
        std::cout << "Testing NVIDIA NPP AddC..." << std::endl;
        
        int step = 0;
        Npp8u* d_src = allocateWithNvidia<Npp8u>(width, height, &step);
        Npp8u* d_dst = allocateWithNvidia<Npp8u>(width, height, &step);
        
        if (d_src && d_dst && nvidiaLoader_->nv_nppiAddC_8u_C1RSfs_Ctx) {
            // 拷贝数据到GPU
            initializeDeviceData(d_src, step, width, height, hostData);
            
            // 创建流上下文
            NppStreamContext ctx;
            if (nvidiaLoader_->nv_nppGetStreamContext) {
                nvidiaLoader_->nv_nppGetStreamContext(&ctx);
            } else {
                memset(&ctx, 0, sizeof(ctx));
                ctx.hStream = cudaStreamLegacy;
            }
            
            // 执行AddC
            NppiSize roiSize = {width, height};
            NppStatus status = nvidiaLoader_->nv_nppiAddC_8u_C1RSfs_Ctx(
                d_src, step, constant, d_dst, step, roiSize, scaleFactor, ctx);
            
            std::cout << "  NVIDIA NPP AddC status: " << status << std::endl;
            EXPECT_EQ(status, NPP_NO_ERROR) << "NVIDIA NPP AddC should succeed";
            
            if (status == NPP_NO_ERROR) {
                // 拷贝结果回CPU
                cudaMemcpy2D(resultNvidia.data(), width * sizeof(Npp8u),
                            d_dst, step, width * sizeof(Npp8u), height,
                            cudaMemcpyDeviceToHost);
            }
        }
        
        freeNvidiaMemory(d_src);
        freeNvidiaMemory(d_dst);
    }
    
    // 如果两个库都可用，比较结果
    if (hasBothLibraries()) {
        std::cout << "Comparing AddC results..." << std::endl;
        
        bool resultsMatch = compareArrays(
            resultOpen.data(), resultNvidia.data(), size, 
            1, nullptr, "OpenNPP", "NVIDIA");
        
        if (resultsMatch) {
            std::cout << "  Results match ✓" << std::endl;
        } else {
            std::cout << "  Results differ ✗" << std::endl;
        }
        
        EXPECT_TRUE(resultsMatch) << "AddC results should match between libraries";
    }
}

// ==================== 符号解析完整性测试 ====================

TEST_F(DualLibraryLoadingTest, SymbolResolution_Completeness) {
    std::cout << "\n=== Symbol Resolution Completeness ===" << std::endl;
    
    if (hasOpenNpp()) {
        std::cout << "OpenNPP symbol resolution:" << std::endl;
        
        int resolvedCount = 0, totalCount = 0;
        
        // 检查核心函数
        #define CHECK_SYMBOL(ptr, name) \
            totalCount++; \
            if (ptr) { \
                resolvedCount++; \
                std::cout << "  " << name << ": ✓" << std::endl; \
            } else { \
                std::cout << "  " << name << ": ✗" << std::endl; \
            }
        
        CHECK_SYMBOL(openNppLoader_->op_nppGetGpuNumSMs, "nppGetGpuNumSMs");
        CHECK_SYMBOL(openNppLoader_->op_nppGetStream, "nppGetStream");
        CHECK_SYMBOL(openNppLoader_->op_nppSetStream, "nppSetStream");
        CHECK_SYMBOL(openNppLoader_->op_nppiMalloc_8u_C1, "nppiMalloc_8u_C1");
        CHECK_SYMBOL(openNppLoader_->op_nppiFree, "nppiFree");
        CHECK_SYMBOL(openNppLoader_->op_nppiAddC_8u_C1RSfs_Ctx, "nppiAddC_8u_C1RSfs_Ctx");
        
        std::cout << "  Total: " << resolvedCount << "/" << totalCount 
                  << " (" << (resolvedCount * 100.0 / totalCount) << "%)" << std::endl;
        
        EXPECT_GT(resolvedCount, totalCount / 2) << "Most OpenNPP symbols should be resolved";
    }
    
    if (hasNvidiaNpp()) {
        std::cout << "NVIDIA NPP symbol resolution:" << std::endl;
        
        int resolvedCount = 0, totalCount = 0;
        
        CHECK_SYMBOL(nvidiaLoader_->nv_nppGetGpuNumSMs, "nppGetGpuNumSMs");
        CHECK_SYMBOL(nvidiaLoader_->nv_nppGetStream, "nppGetStream");
        CHECK_SYMBOL(nvidiaLoader_->nv_nppSetStream, "nppSetStream");
        CHECK_SYMBOL(nvidiaLoader_->nv_nppiMalloc_8u_C1, "nppiMalloc_8u_C1");
        CHECK_SYMBOL(nvidiaLoader_->nv_nppiFree, "nppiFree");
        CHECK_SYMBOL(nvidiaLoader_->nv_nppiAddC_8u_C1RSfs_Ctx, "nppiAddC_8u_C1RSfs_Ctx");
        
        std::cout << "  Total: " << resolvedCount << "/" << totalCount 
                  << " (" << (resolvedCount * 100.0 / totalCount) << "%)" << std::endl;
        
        EXPECT_GT(resolvedCount, totalCount / 2) << "Most NVIDIA NPP symbols should be resolved";
        
        #undef CHECK_SYMBOL
    }
}