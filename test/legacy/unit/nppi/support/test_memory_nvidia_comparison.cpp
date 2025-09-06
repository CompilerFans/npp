/**
 * @file test_memory_nvidia_comparison.cpp
 * @brief NPPI内存管理与NVIDIA NPP的对比测试
 * 
 * 使用统一的测试框架进行内存分配和管理的对比
 */

#include "test/framework/nvidia_comparison_test_base.h"
#include <map>
#include <vector>
#include <algorithm>

using namespace test_framework;

class NPPIMemoryComparisonTest : public NvidiaComparisonTestBase {
protected:
    // 测试不同宽度的step值
    void testStepAlignment(const std::string& dataType, 
                          const std::vector<int>& widths,
                          int elementSize,
                          int channels) {
        std::cout << "\n=== Step Alignment Comparison: " << dataType << " ===" << std::endl;
        std::cout << std::setw(10) << "Width"
                  << std::setw(15) << "Min Step"
                  << std::setw(15) << "OpenNPP"
                  << std::setw(15) << "NVIDIA"
                  << std::setw(10) << "Match"
                  << std::setw(15) << "Difference" << std::endl;
        std::cout << std::string(85, '-') << std::endl;
        
        int matchCount = 0;
        std::map<int, int> stepDifferences;
        
        for (int width : widths) {
            int minStep = width * elementSize * channels;
            int openStep = 0, nvStep = 0;
            void* openPtr = nullptr;
            void* nvPtr = nullptr;
            
            // 根据数据类型分配内存
            if (dataType == "8u_C1") {
                openPtr = nppiMalloc_8u_C1(width, 1, &openStep);
                if (hasNvidiaNpp() && nvidiaLoader_->nv_nppiMalloc_8u_C1) {
                    nvPtr = nvidiaLoader_->nv_nppiMalloc_8u_C1(width, 1, &nvStep);
                }
            } else if (dataType == "8u_C3") {
                openPtr = nppiMalloc_8u_C3(width, 1, &openStep);
                if (hasNvidiaNpp() && nvidiaLoader_->nv_nppiMalloc_8u_C3) {
                    nvPtr = nvidiaLoader_->nv_nppiMalloc_8u_C3(width, 1, &nvStep);
                }
            } else if (dataType == "16u_C1") {
                openPtr = nppiMalloc_16u_C1(width, 1, &openStep);
                if (hasNvidiaNpp() && nvidiaLoader_->nv_nppiMalloc_16u_C1) {
                    nvPtr = nvidiaLoader_->nv_nppiMalloc_16u_C1(width, 1, &nvStep);
                }
            } else if (dataType == "32f_C1") {
                openPtr = nppiMalloc_32f_C1(width, 1, &openStep);
                if (hasNvidiaNpp() && nvidiaLoader_->nv_nppiMalloc_32f_C1) {
                    nvPtr = nvidiaLoader_->nv_nppiMalloc_32f_C1(width, 1, &nvStep);
                }
            } else if (dataType == "32f_C4") {
                openPtr = nppiMalloc_32f_C4(width, 1, &openStep);
                if (hasNvidiaNpp() && nvidiaLoader_->nv_nppiMalloc_32f_C4) {
                    nvPtr = nvidiaLoader_->nv_nppiMalloc_32f_C4(width, 1, &nvStep);
                }
            }
            
            ASSERT_NE(openPtr, nullptr) << "OpenNPP allocation failed for width " << width;
            
            bool match = false;
            int diff = 0;
            
            if (nvPtr) {
                match = (openStep == nvStep);
                diff = openStep - nvStep;
                if (match) matchCount++;
                stepDifferences[diff]++;
            }
            
            std::cout << std::setw(10) << width
                      << std::setw(15) << minStep
                      << std::setw(15) << openStep
                      << std::setw(15) << (nvPtr ? std::to_string(nvStep) : "N/A")
                      << std::setw(10) << (match ? "✓" : "✗")
                      << std::setw(15) << (nvPtr ? std::to_string(diff) : "N/A")
                      << std::endl;
            
            // 清理
            nppiFree(openPtr);
            if (nvPtr && nvidiaLoader_->nv_nppiFree) {
                nvidiaLoader_->nv_nppiFree(nvPtr);
            }
        }
        
        std::cout << std::string(85, '-') << std::endl;
        std::cout << "Match rate: " << matchCount << "/" << widths.size() 
                  << " (" << (matchCount * 100.0 / widths.size()) << "%)" << std::endl;
        
        // 分析差异模式
        if (!stepDifferences.empty()) {
            std::cout << "\nStep difference distribution:" << std::endl;
            for (const auto& [diff, count] : stepDifferences) {
                std::cout << "  Difference " << diff << ": " << count << " cases" << std::endl;
            }
        }
    }
};

// ==================== 基础内存分配对比 ====================

TEST_F(NPPIMemoryComparisonTest, BasicAllocation_8u_C1) {
    skipIfNoNvidiaNpp("Skipping memory allocation comparison");
    
    const std::vector<int> testWidths = {
        1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33,
        63, 64, 65, 127, 128, 129, 255, 256, 257,
        511, 512, 513, 640, 800, 1024, 1280, 1920, 3840
    };
    
    testStepAlignment("8u_C1", testWidths, 1, 1);
}

TEST_F(NPPIMemoryComparisonTest, BasicAllocation_8u_C3) {
    skipIfNoNvidiaNpp("Skipping memory allocation comparison");
    
    const std::vector<int> testWidths = {
        1, 5, 10, 21, 32, 43, 64, 85, 128, 171, 256, 341, 512, 640, 1024, 1920
    };
    
    testStepAlignment("8u_C3", testWidths, 1, 3);
}

TEST_F(NPPIMemoryComparisonTest, BasicAllocation_16u_C1) {
    skipIfNoNvidiaNpp("Skipping memory allocation comparison");
    
    const std::vector<int> testWidths = {
        1, 8, 15, 16, 17, 32, 64, 128, 256, 512, 640, 1024, 1920
    };
    
    testStepAlignment("16u_C1", testWidths, 2, 1);
}

TEST_F(NPPIMemoryComparisonTest, BasicAllocation_32f_C1) {
    skipIfNoNvidiaNpp("Skipping memory allocation comparison");
    
    const std::vector<int> testWidths = {
        1, 4, 7, 8, 9, 16, 32, 64, 128, 256, 512, 640, 1024, 1920
    };
    
    testStepAlignment("32f_C1", testWidths, 4, 1);
}

TEST_F(NPPIMemoryComparisonTest, BasicAllocation_32f_C4) {
    skipIfNoNvidiaNpp("Skipping memory allocation comparison");
    
    const std::vector<int> testWidths = {
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
    };
    
    testStepAlignment("32f_C4", testWidths, 4, 4);
}

// ==================== 大规模内存分配测试 ====================

TEST_F(NPPIMemoryComparisonTest, LargeAllocation_Performance) {
    skipIfNoNvidiaNpp("Skipping large allocation performance comparison");
    
    struct AllocationTest {
        int width;
        int height;
        const char* description;
    };
    
    std::vector<AllocationTest> tests = {
        {640, 480, "VGA"},
        {1280, 720, "HD"},
        {1920, 1080, "Full HD"},
        {3840, 2160, "4K"},
        {7680, 4320, "8K"}
    };
    
    std::cout << "\n=== Large Allocation Performance Comparison ===" << std::endl;
    std::cout << std::setw(15) << "Resolution"
              << std::setw(15) << "Dimensions"
              << std::setw(20) << "OpenNPP Time (ms)"
              << std::setw(20) << "NVIDIA Time (ms)"
              << std::setw(15) << "Speedup" << std::endl;
    std::cout << std::string(85, '-') << std::endl;
    
    for (const auto& test : tests) {
        int openStep, nvStep;
        
        // 测量OpenNPP分配时间
        double openTime = measureExecutionTime([&]() {
            Npp8u* ptr = nppiMalloc_8u_C3(test.width, test.height, &openStep);
            if (ptr) nppiFree(ptr);
        }, 10);
        
        // 测量NVIDIA NPP分配时间
        double nvTime = 0;
        if (nvidiaLoader_->nv_nppiMalloc_8u_C3 && nvidiaLoader_->nv_nppiFree) {
            nvTime = measureExecutionTime([&]() {
                Npp8u* ptr = nvidiaLoader_->nv_nppiMalloc_8u_C3(test.width, test.height, &nvStep);
                if (ptr) nvidiaLoader_->nv_nppiFree(ptr);
            }, 10);
        }
        
        double speedup = nvTime > 0 ? openTime / nvTime : 0;
        
        std::cout << std::setw(15) << test.description
                  << std::setw(7) << test.width << "x" << std::setw(6) << test.height
                  << std::setw(20) << std::fixed << std::setprecision(3) << openTime
                  << std::setw(20) << (nvTime > 0 ? std::to_string(nvTime) : "N/A")
                  << std::setw(15) << std::fixed << std::setprecision(2) 
                  << (speedup > 0 ? std::to_string(speedup) + "x" : "N/A")
                  << std::endl;
        
        // 验证是否能成功分配
        Npp8u* testPtr = nppiMalloc_8u_C3(test.width, test.height, &openStep);
        if (testPtr) {
            nppiFree(testPtr);
        } else {
            std::cout << "  WARNING: Failed to allocate " << test.description << std::endl;
        }
    }
}

// ==================== 内存对齐模式分析 ====================

TEST_F(NPPIMemoryComparisonTest, AlignmentPatternAnalysis) {
    std::cout << "\n=== Memory Alignment Pattern Analysis ===" << std::endl;
    
    // 分析不同数据类型的对齐模式
    struct AlignmentTest {
        std::string type;
        int elementSize;
        int channels;
        std::vector<int> commonAlignments;
    };
    
    std::vector<AlignmentTest> tests = {
        {"8u_C1", 1, 1, {}},
        {"8u_C3", 1, 3, {}},
        {"16u_C1", 2, 1, {}},
        {"32f_C1", 4, 1, {}},
        {"32f_C4", 4, 4, {}}
    };
    
    // 测试各种宽度以发现对齐模式
    for (auto& test : tests) {
        std::map<int, int> alignmentCount;
        
        for (int width = 1; width <= 256; width++) {
            int step = 0;
            void* ptr = nullptr;
            
            if (test.type == "8u_C1") {
                ptr = nppiMalloc_8u_C1(width, 1, &step);
            } else if (test.type == "8u_C3") {
                ptr = nppiMalloc_8u_C3(width, 1, &step);
            } else if (test.type == "16u_C1") {
                ptr = nppiMalloc_16u_C1(width, 1, &step);
            } else if (test.type == "32f_C1") {
                ptr = nppiMalloc_32f_C1(width, 1, &step);
            } else if (test.type == "32f_C4") {
                ptr = nppiMalloc_32f_C4(width, 1, &step);
            }
            
            if (ptr) {
                int minStep = width * test.elementSize * test.channels;
                
                // 查找最大的2的幂次对齐
                int alignTo = 1;
                while (alignTo <= 512) {
                    if (step % alignTo == 0 && (step - alignTo) < minStep) {
                        alignmentCount[alignTo]++;
                        break;
                    }
                    alignTo *= 2;
                }
                
                nppiFree(ptr);
            }
        }
        
        // 找出最常见的对齐值
        std::vector<std::pair<int, int>> sortedAlignments(alignmentCount.begin(), alignmentCount.end());
        std::sort(sortedAlignments.begin(), sortedAlignments.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        std::cout << "\n" << test.type << " alignment distribution:" << std::endl;
        int shown = 0;
        for (const auto& [align, count] : sortedAlignments) {
            if (shown++ < 5) {  // 只显示前5个
                std::cout << "  " << align << "-byte aligned: " << count << " cases" << std::endl;
                test.commonAlignments.push_back(align);
            }
        }
    }
}

// ==================== 内存访问性能测试 ====================

TEST_F(NPPIMemoryComparisonTest, MemoryAccessPerformance) {
    const int width = 1024;
    const int height = 1024;
    const int iterations = 100;
    
    std::cout << "\n=== Memory Access Performance Test ===" << std::endl;
    
    // 测试不同step值对性能的影响
    struct AccessTest {
        int customStep;
        const char* description;
    };
    
    int minStep = width * sizeof(float);
    std::vector<AccessTest> tests = {
        {minStep, "Minimum step (no padding)"},
        {(minStep + 31) & ~31, "32-byte aligned"},
        {(minStep + 63) & ~63, "64-byte aligned"},
        {(minStep + 127) & ~127, "128-byte aligned"},
        {(minStep + 255) & ~255, "256-byte aligned"}
    };
    
    for (const auto& test : tests) {
        // 使用cudaMalloc2D分配特定step的内存
        float* devPtr;
        size_t pitch;
        cudaMallocPitch(&devPtr, &pitch, test.customStep, height);
        
        if (pitch >= static_cast<size_t>(test.customStep)) {
            // 准备测试数据
            std::vector<float> hostData(width * height, 1.0f);
            
            // 测量内存拷贝性能
            double copyTime = measureExecutionTime([&]() {
                cudaMemcpy2D(devPtr, pitch, hostData.data(), minStep,
                            minStep, height, cudaMemcpyHostToDevice);
            }, iterations);
            
            std::cout << std::setw(30) << test.description
                      << ": " << std::fixed << std::setprecision(3) 
                      << copyTime << " ms" << std::endl;
            
            cudaFree(devPtr);
        }
    }
}

// ==================== 错误处理和边界条件 ====================

TEST_F(NPPIMemoryComparisonTest, ErrorHandling_InvalidParameters) {
    std::cout << "\n=== Error Handling Test ===" << std::endl;
    
    struct ErrorTest {
        int width;
        int height;
        const char* description;
        bool shouldFail;
    };
    
    std::vector<ErrorTest> tests = {
        {0, 100, "Zero width", true},
        {100, 0, "Zero height", true},
        {-100, 100, "Negative width", true},
        {100, -100, "Negative height", true},
        {INT_MAX, 1, "Maximum width", true},
        {1, INT_MAX, "Maximum height", true},
        {1, 1, "Minimum valid size", false},
        {16384, 16384, "Large but valid", false}
    };
    
    for (const auto& test : tests) {
        int step;
        Npp8u* openPtr = nppiMalloc_8u_C1(test.width, test.height, &step);
        bool openFailed = (openPtr == nullptr);
        
        std::cout << std::setw(25) << test.description << ": ";
        std::cout << "OpenNPP " << (openFailed ? "failed" : "succeeded");
        
        if (hasNvidiaNpp() && nvidiaLoader_->nv_nppiMalloc_8u_C1) {
            Npp8u* nvPtr = nvidiaLoader_->nv_nppiMalloc_8u_C1(test.width, test.height, &step);
            bool nvFailed = (nvPtr == nullptr);
            std::cout << ", NVIDIA " << (nvFailed ? "failed" : "succeeded");
            
            if (nvPtr && nvidiaLoader_->nv_nppiFree) {
                nvidiaLoader_->nv_nppiFree(nvPtr);
            }
        }
        
        std::cout << " - " << (openFailed == test.shouldFail ? "✓ Expected" : "✗ Unexpected") << std::endl;
        
        if (openPtr) {
            nppiFree(openPtr);
        }
    }
}

/*  DISABLE temp */
TEST_F(NPPIMemoryComparisonTest, DISABLED_MemoryExhaustion) {
    std::cout << "\n=== Memory Exhaustion Test ===" << std::endl;
    
    const int hugeSize = 16384;  // 16K x 16K
    std::vector<Npp32f*> allocations;
    int successCount = 0;
    
    // 尝试分配直到失败
    for (int i = 0; i < 10; i++) {
        int step;
        Npp32f* ptr = nppiMalloc_32f_C4(hugeSize, hugeSize, &step);
        
        if (ptr) {
            allocations.push_back(ptr);
            successCount++;
            
            size_t expectedSize = static_cast<size_t>(hugeSize) * hugeSize * 4 * sizeof(Npp32f);
            std::cout << "Allocation " << i << " succeeded: ~" 
                      << (expectedSize / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
        } else {
            std::cout << "Allocation " << i << " failed - memory exhausted" << std::endl;
            break;
        }
    }
    
    std::cout << "Successfully allocated " << successCount << " huge buffers" << std::endl;
    
    // 清理
    for (auto ptr : allocations) {
        nppiFree(ptr);
    }
    
    // 验证内存是否正确释放
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    std::cout << "After cleanup: " << (freeMem / (1024.0 * 1024.0 * 1024.0)) 
              << " GB free of " << (totalMem / (1024.0 * 1024.0 * 1024.0)) << " GB total" << std::endl;
}