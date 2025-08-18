/**
 * OpenNPP算术运算综合测试
 * 使用重构的测试框架
 */

#include "framework/npp_test_common.h"
#include "framework/npp_image_buffer.h"
#include "npp.h"
#include <functional>

using namespace opennpp::test;

// CPU参考实现
namespace nppi_reference {
    extern NppStatus nppiAddC_8u_C1RSfs_reference(const Npp8u* pSrc, int nSrcStep, const Npp8u nConstant,
                                                  Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    extern NppStatus nppiAddC_16u_C1RSfs_reference(const Npp16u* pSrc, int nSrcStep, const Npp16u nConstant,
                                                   Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    extern NppStatus nppiAddC_16s_C1RSfs_reference(const Npp16s* pSrc, int nSrcStep, const Npp16s nConstant,
                                                   Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    extern NppStatus nppiAddC_32f_C1R_reference(const Npp32f* pSrc, int nSrcStep, const Npp32f nConstant,
                                                Npp32f* pDst, int nDstStep, NppiSize oSizeROI);
    
    extern NppStatus nppiSubC_8u_C1RSfs_reference(const Npp8u* pSrc, int nSrcStep, const Npp8u nConstant,
                                                  Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    extern NppStatus nppiSubC_16u_C1RSfs_reference(const Npp16u* pSrc, int nSrcStep, const Npp16u nConstant,
                                                   Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    
    extern NppStatus nppiMulC_8u_C1RSfs_reference(const Npp8u* pSrc, int nSrcStep, const Npp8u nConstant,
                                                  Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    
    extern NppStatus nppiDivC_8u_C1RSfs_reference(const Npp8u* pSrc, int nSrcStep, const Npp8u nConstant,
                                                  Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
}

class ArithmeticTest {
private:
    TestConfig config_;
    TestStatistics stats_;
    
    template<typename T>
    TestResult testOperation(
        const std::string& opName,
        std::function<NppStatus(const T*, int, T, T*, int, NppiSize, int)> openFunc,
        std::function<NppStatus(const T*, int, T, T*, int, NppiSize, int)> nvidiaFunc,
        std::function<NppStatus(const T*, int, T, T*, int, NppiSize, int)> cpuFunc,
        T sourceValue,
        T constant,
        int scaleFactor,
        TestPattern pattern = TestPattern::GRADIENT) {
        
        TestResult result(opName);
        Timer timer;
        
        try {
            // 创建测试缓冲区
            ImageBuffer<T> src(config_.defaultWidth, config_.defaultHeight);
            ImageBuffer<T> openDst(config_.defaultWidth, config_.defaultHeight);
            ImageBuffer<T> cpuDst(config_.defaultWidth, config_.defaultHeight);
            
            // 填充源数据
            src.fill(pattern, sourceValue);
            
            // 测试OpenNPP
            timer.start();
            NppStatus openStatus = openFunc(
                src.devicePtr(), src.deviceStep(), constant,
                openDst.devicePtr(), openDst.deviceStep(),
                src.size(), scaleFactor
            );
            cudaDeviceSynchronize();
            result.executionTime = timer.elapsedMilliseconds();
            
            if (openStatus != NPP_NO_ERROR) {
                result.errorMessage = "OpenNPP failed with status: " + std::to_string(openStatus);
                return result;
            }
            
            openDst.copyFromDevice();
            
            // 测试CPU参考
            if (config_.compareWithCpu && cpuFunc) {
                src.copyFromDevice();
                NppStatus cpuStatus = cpuFunc(
                    src.hostPtr(), src.hostStep(), constant,
                    cpuDst.hostPtr(), cpuDst.hostStep(),
                    src.size(), scaleFactor
                );
                
                if (cpuStatus == NPP_NO_ERROR) {
                    result.maxDifference = ImageBuffer<T>::compare(openDst, cpuDst, config_.integerTolerance);
                    result.passed = (result.maxDifference <= config_.integerTolerance);
                } else {
                    result.errorMessage = "CPU reference failed";
                }
            }
            
#ifdef HAVE_NVIDIA_NPP
            // 测试NVIDIA NPP
            if (config_.compareWithNvidia && nvidiaFunc) {
                ImageBuffer<T> nvidiaDst(config_.defaultWidth, config_.defaultHeight);
                
                NppStatus nvidiaStatus = nvidiaFunc(
                    src.devicePtr(), src.deviceStep(), constant,
                    nvidiaDst.devicePtr(), nvidiaDst.deviceStep(),
                    src.size(), scaleFactor
                );
                
                if (nvidiaStatus == NPP_NO_ERROR) {
                    nvidiaDst.copyFromDevice();
                    double nvidiaVsOpen = ImageBuffer<T>::compare(openDst, nvidiaDst, config_.integerTolerance);
                    
                    if (nvidiaVsOpen > config_.integerTolerance) {
                        result.passed = false;
                        result.errorMessage = "Mismatch with NVIDIA NPP";
                        result.maxDifference = std::max(result.maxDifference, nvidiaVsOpen);
                    }
                }
            }
#else
            if (!cpuFunc) {
                result.passed = true;  // 如果没有参考实现，假设通过
            }
#endif
            
        } catch (const std::exception& e) {
            result.errorMessage = e.what();
            result.passed = false;
        }
        
        return result;
    }
    
    // 32位浮点数版本（无缩放因子）
    TestResult testOperation32f(
        const std::string& opName,
        std::function<NppStatus(const Npp32f*, int, Npp32f, Npp32f*, int, NppiSize)> openFunc,
        std::function<NppStatus(const Npp32f*, int, Npp32f, Npp32f*, int, NppiSize)> nvidiaFunc,
        std::function<NppStatus(const Npp32f*, int, Npp32f, Npp32f*, int, NppiSize)> cpuFunc,
        Npp32f sourceValue,
        Npp32f constant,
        TestPattern pattern = TestPattern::GRADIENT) {
        
        TestResult result(opName);
        Timer timer;
        
        try {
            ImageBuffer<Npp32f> src(config_.defaultWidth, config_.defaultHeight);
            ImageBuffer<Npp32f> openDst(config_.defaultWidth, config_.defaultHeight);
            ImageBuffer<Npp32f> cpuDst(config_.defaultWidth, config_.defaultHeight);
            
            src.fill(pattern, sourceValue);
            
            timer.start();
            NppStatus openStatus = openFunc(
                src.devicePtr(), src.deviceStep(), constant,
                openDst.devicePtr(), openDst.deviceStep(),
                src.size()
            );
            cudaDeviceSynchronize();
            result.executionTime = timer.elapsedMilliseconds();
            
            if (openStatus != NPP_NO_ERROR) {
                result.errorMessage = "OpenNPP failed";
                return result;
            }
            
            openDst.copyFromDevice();
            
            if (config_.compareWithCpu && cpuFunc) {
                src.copyFromDevice();
                NppStatus cpuStatus = cpuFunc(
                    src.hostPtr(), src.hostStep(), constant,
                    cpuDst.hostPtr(), cpuDst.hostStep(),
                    src.size()
                );
                
                if (cpuStatus == NPP_NO_ERROR) {
                    result.maxDifference = ImageBuffer<Npp32f>::compare(openDst, cpuDst, config_.floatTolerance);
                    result.passed = (result.maxDifference <= config_.floatTolerance);
                }
            }
            
#ifdef HAVE_NVIDIA_NPP
            if (config_.compareWithNvidia && nvidiaFunc) {
                ImageBuffer<Npp32f> nvidiaDst(config_.defaultWidth, config_.defaultHeight);
                
                NppStatus nvidiaStatus = nvidiaFunc(
                    src.devicePtr(), src.deviceStep(), constant,
                    nvidiaDst.devicePtr(), nvidiaDst.deviceStep(),
                    src.size()
                );
                
                if (nvidiaStatus == NPP_NO_ERROR) {
                    nvidiaDst.copyFromDevice();
                    double nvidiaVsOpen = ImageBuffer<Npp32f>::compare(openDst, nvidiaDst, config_.floatTolerance);
                    
                    if (nvidiaVsOpen > config_.floatTolerance) {
                        result.passed = false;
                        result.maxDifference = std::max(result.maxDifference, nvidiaVsOpen);
                    }
                }
            }
#else
            if (!cpuFunc) {
                result.passed = true;
            }
#endif
            
        } catch (const std::exception& e) {
            result.errorMessage = e.what();
            result.passed = false;
        }
        
        return result;
    }
    
public:
    ArithmeticTest(const TestConfig& config = TestConfig()) : config_(config) {}
    
    void runAllTests() {
        std::cout << "\n=== OpenNPP算术运算综合测试 ===" << std::endl;
        std::cout << "配置: " << config_.defaultWidth << "x" << config_.defaultHeight 
                  << ", 容差: 整数=" << config_.integerTolerance 
                  << ", 浮点=" << config_.floatTolerance << std::endl;
        
        // AddC测试
        std::cout << "\n--- AddC运算 ---" << std::endl;
        runAddCTests();
        
        // SubC测试
        std::cout << "\n--- SubC运算 ---" << std::endl;
        runSubCTests();
        
        // MulC测试
        std::cout << "\n--- MulC运算 ---" << std::endl;
        runMulCTests();
        
        // DivC测试
        std::cout << "\n--- DivC运算 ---" << std::endl;
        runDivCTests();
        
        // 打印统计
        stats_.printSummary();
        if (config_.verbose) {
            stats_.printDetails();
        }
    }
    
    bool allTestsPassed() const {
        return stats_.allPassed();
    }
    
private:
    void runAddCTests() {
        // 8u
        auto result = testOperation<Npp8u>(
            "nppiAddC_8u_C1RSfs",
            nppiAddC_8u_C1RSfs,
#ifdef HAVE_NVIDIA_NPP
            ::nppiAddC_8u_C1RSfs,
#else
            nullptr,
#endif
            nppi_reference::nppiAddC_8u_C1RSfs_reference,
            50, 10, 0
        );
        printResult(result);
        stats_.addResult(result);
        
        // 16u
        result = testOperation<Npp16u>(
            "nppiAddC_16u_C1RSfs",
            nppiAddC_16u_C1RSfs,
#ifdef HAVE_NVIDIA_NPP
            ::nppiAddC_16u_C1RSfs,
#else
            nullptr,
#endif
            nppi_reference::nppiAddC_16u_C1RSfs_reference,
            500, 100, 0
        );
        printResult(result);
        stats_.addResult(result);
        
        // 16s
        result = testOperation<Npp16s>(
            "nppiAddC_16s_C1RSfs",
            nppiAddC_16s_C1RSfs,
#ifdef HAVE_NVIDIA_NPP
            ::nppiAddC_16s_C1RSfs,
#else
            nullptr,
#endif
            nppi_reference::nppiAddC_16s_C1RSfs_reference,
            -500, 100, 0
        );
        printResult(result);
        stats_.addResult(result);
        
        // 32f
        auto result32f = testOperation32f(
            "nppiAddC_32f_C1R",
            nppiAddC_32f_C1R,
#ifdef HAVE_NVIDIA_NPP
            ::nppiAddC_32f_C1R,
#else
            nullptr,
#endif
            nppi_reference::nppiAddC_32f_C1R_reference,
            50.0f, 10.5f
        );
        printResult(result32f);
        stats_.addResult(result32f);
    }
    
    void runSubCTests() {
        auto result = testOperation<Npp8u>(
            "nppiSubC_8u_C1RSfs",
            nppiSubC_8u_C1RSfs,
#ifdef HAVE_NVIDIA_NPP
            ::nppiSubC_8u_C1RSfs,
#else
            nullptr,
#endif
            nppi_reference::nppiSubC_8u_C1RSfs_reference,
            100, 50, 0
        );
        printResult(result);
        stats_.addResult(result);
        
        result = testOperation<Npp16u>(
            "nppiSubC_16u_C1RSfs",
            nppiSubC_16u_C1RSfs,
#ifdef HAVE_NVIDIA_NPP
            ::nppiSubC_16u_C1RSfs,
#else
            nullptr,
#endif
            nppi_reference::nppiSubC_16u_C1RSfs_reference,
            1000, 500, 0
        );
        printResult(result);
        stats_.addResult(result);
    }
    
    void runMulCTests() {
        auto result = testOperation<Npp8u>(
            "nppiMulC_8u_C1RSfs",
            nppiMulC_8u_C1RSfs,
#ifdef HAVE_NVIDIA_NPP
            ::nppiMulC_8u_C1RSfs,
#else
            nullptr,
#endif
            nppi_reference::nppiMulC_8u_C1RSfs_reference,
            10, 5, 0
        );
        printResult(result);
        stats_.addResult(result);
    }
    
    void runDivCTests() {
        auto result = testOperation<Npp8u>(
            "nppiDivC_8u_C1RSfs",
            nppiDivC_8u_C1RSfs,
#ifdef HAVE_NVIDIA_NPP
            ::nppiDivC_8u_C1RSfs,
#else
            nullptr,
#endif
            nppi_reference::nppiDivC_8u_C1RSfs_reference,
            100, 5, 0
        );
        printResult(result);
        stats_.addResult(result);
    }
    
    void printResult(const TestResult& result) {
        if (config_.verbose) {
            std::cout << "  " << (result.passed ? "✓" : "✗") << " "
                      << std::left << std::setw(30) << result.testName
                      << " [" << std::fixed << std::setprecision(3) 
                      << result.executionTime << "ms]";
            
            if (!result.passed && !result.errorMessage.empty()) {
                std::cout << " - " << result.errorMessage;
            }
            
            if (result.maxDifference > 0) {
                std::cout << " (diff=" << std::scientific << std::setprecision(2) 
                         << result.maxDifference << ")";
            }
            
            std::cout << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    TestConfig config;
    
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--quiet" || arg == "-q") {
            config.verbose = false;
        } else if (arg == "--size" && i + 2 < argc) {
            config.defaultWidth = std::atoi(argv[++i]);
            config.defaultHeight = std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "用法: " << argv[0] << " [选项]" << std::endl;
            std::cout << "选项:" << std::endl;
            std::cout << "  --quiet, -q       安静模式" << std::endl;
            std::cout << "  --size W H        设置测试图像大小" << std::endl;
            std::cout << "  --help, -h        显示帮助" << std::endl;
            return 0;
        }
    }
    
    ArithmeticTest test(config);
    test.runAllTests();
    
    return test.allTestsPassed() ? 0 : 1;
}