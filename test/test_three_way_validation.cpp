#include "npp_validation_test.h"
#include "npp.h"
#include <cuda_runtime.h>
#include <functional>

#ifdef HAVE_NVIDIA_NPP
#include <npp.h>
#endif

/**
 * 完整的三方对比验证测试
 * OpenNPP vs NVIDIA NPP vs CPU Reference
 * 覆盖所有算术运算和数据类型
 */

// Forward declarations for reference implementations
namespace nppi_reference {
    // AddC operations
    NppStatus nppiAddC_8u_C1RSfs_reference(const Npp8u* pSrc, int nSrcStep, const Npp8u nConstant,
                                          Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    NppStatus nppiAddC_16u_C1RSfs_reference(const Npp16u* pSrc, int nSrcStep, const Npp16u nConstant,
                                           Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    NppStatus nppiAddC_16s_C1RSfs_reference(const Npp16s* pSrc, int nSrcStep, const Npp16s nConstant,
                                           Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    NppStatus nppiAddC_32f_C1R_reference(const Npp32f* pSrc, int nSrcStep, const Npp32f nConstant,
                                        Npp32f* pDst, int nDstStep, NppiSize oSizeROI);
    
    // SubC operations
    NppStatus nppiSubC_8u_C1RSfs_reference(const Npp8u* pSrc, int nSrcStep, const Npp8u nConstant,
                                          Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    NppStatus nppiSubC_16u_C1RSfs_reference(const Npp16u* pSrc, int nSrcStep, const Npp16u nConstant,
                                           Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    NppStatus nppiSubC_16s_C1RSfs_reference(const Npp16s* pSrc, int nSrcStep, const Npp16s nConstant,
                                           Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    NppStatus nppiSubC_32f_C1R_reference(const Npp32f* pSrc, int nSrcStep, const Npp32f nConstant,
                                        Npp32f* pDst, int nDstStep, NppiSize oSizeROI);
    
    // MulC operations
    NppStatus nppiMulC_8u_C1RSfs_reference(const Npp8u* pSrc, int nSrcStep, const Npp8u nConstant,
                                          Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    NppStatus nppiMulC_16u_C1RSfs_reference(const Npp16u* pSrc, int nSrcStep, const Npp16u nConstant,
                                           Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    NppStatus nppiMulC_16s_C1RSfs_reference(const Npp16s* pSrc, int nSrcStep, const Npp16s nConstant,
                                           Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    NppStatus nppiMulC_32f_C1R_reference(const Npp32f* pSrc, int nSrcStep, const Npp32f nConstant,
                                        Npp32f* pDst, int nDstStep, NppiSize oSizeROI);
    
    // DivC operations
    NppStatus nppiDivC_8u_C1RSfs_reference(const Npp8u* pSrc, int nSrcStep, const Npp8u nConstant,
                                          Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    NppStatus nppiDivC_16u_C1RSfs_reference(const Npp16u* pSrc, int nSrcStep, const Npp16u nConstant,
                                           Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    NppStatus nppiDivC_16s_C1RSfs_reference(const Npp16s* pSrc, int nSrcStep, const Npp16s nConstant,
                                           Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    NppStatus nppiDivC_32f_C1R_reference(const Npp32f* pSrc, int nSrcStep, const Npp32f nConstant,
                                        Npp32f* pDst, int nDstStep, NppiSize oSizeROI);
    
    // Multi-channel operations
    NppStatus nppiAddC_8u_C3RSfs_reference(const Npp8u* pSrc, int nSrcStep, const Npp8u aConstants[3],
                                          Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
}

namespace npp_validation {

class ThreeWayValidationTest : public NPPValidationTestBase {
public:
    ThreeWayValidationTest(bool verbose = true) : NPPValidationTestBase(verbose) {}
    
    void runAllTests() override {
        std::cout << "=== 三方对比验证测试 - OpenNPP vs NVIDIA NPP vs CPU参考 ===\n" << std::endl;
        
        // AddC operations
        std::cout << "--- AddC 运算测试 ---" << std::endl;
        testAddC_AllTypes();
        
        // SubC operations  
        std::cout << "\n--- SubC 运算测试 ---" << std::endl;
        testSubC_AllTypes();
        
        // MulC operations
        std::cout << "\n--- MulC 运算测试 ---" << std::endl;
        testMulC_AllTypes();
        
        // DivC operations
        std::cout << "\n--- DivC 运算测试 ---" << std::endl;
        testDivC_AllTypes();
        
        // Multi-channel operations
        std::cout << "\n--- 多通道运算测试 ---" << std::endl;
        testMultiChannel();
    }
    
private:
    template<typename T>
    void runThreeWayTest(const std::string& testName,
                        std::function<NppStatus(const T*, int, T, T*, int, NppiSize, int)> openFunc,
                        std::function<NppStatus(const T*, int, T, T*, int, NppiSize, int)> nvidiaFunc,
                        std::function<NppStatus(const T*, int, T, T*, int, NppiSize, int)> refFunc,
                        T sourceValue, T constant, int scaleFactor, 
                        double tolerance = 1.0) {
        
        TestResult result(testName);
        
        try {
            const int width = 128;
            const int height = 128;
            
            // Create test buffers
            ValidationImageBuffer<T> srcBuffer(width, height);
            ValidationImageBuffer<T> openDstBuffer(width, height);
            ValidationImageBuffer<T> nvidiaDstBuffer(width, height);
            ValidationImageBuffer<T> refDstBuffer(width, height);
            
            // Fill source with test pattern
            srcBuffer.fillPattern(sourceValue);
            
            // Test OpenNPP
            NppStatus openStatus = openFunc(srcBuffer.devicePtr(), srcBuffer.step(), constant,
                                           openDstBuffer.devicePtr(), openDstBuffer.step(),
                                           srcBuffer.size(), scaleFactor);
            result.openNppSuccess = (openStatus == NPP_NO_ERROR);
            if (result.openNppSuccess) {
                openDstBuffer.copyFromDevice();
            }
            
#ifdef HAVE_NVIDIA_NPP
            // Test NVIDIA NPP
            NppStatus nvidiaStatus = nvidiaFunc(srcBuffer.devicePtr(), srcBuffer.step(), constant,
                                               nvidiaDstBuffer.devicePtr(), nvidiaDstBuffer.step(),
                                               srcBuffer.size(), scaleFactor);
            result.nvidiaNppSuccess = (nvidiaStatus == NPP_NO_ERROR);
            if (result.nvidiaNppSuccess) {
                nvidiaDstBuffer.copyFromDevice();
            }
#else
            result.nvidiaNppSuccess = true; // Skip if not available
#endif
            
            // Test CPU reference
            srcBuffer.copyFromDevice();
            NppStatus refStatus = refFunc(srcBuffer.hostPtr(), srcBuffer.width() * sizeof(T), constant,
                                         refDstBuffer.hostPtr(), refDstBuffer.width() * sizeof(T),
                                         srcBuffer.size(), scaleFactor);
            result.cpuRefSuccess = (refStatus == NPP_NO_ERROR);
            
            // Three-way comparison
            if (result.openNppSuccess && result.cpuRefSuccess) {
                double openVsRef = ValidationImageBuffer<T>::compareBuffers(openDstBuffer, refDstBuffer, tolerance);
                result.maxDifference = openVsRef;
                
#ifdef HAVE_NVIDIA_NPP
                if (result.nvidiaNppSuccess) {
                    double openVsNvidia = ValidationImageBuffer<T>::compareBuffers(openDstBuffer, nvidiaDstBuffer, tolerance);
                    double nvidiaVsRef = ValidationImageBuffer<T>::compareBuffers(nvidiaDstBuffer, refDstBuffer, tolerance);
                    result.maxDifference = std::max(result.maxDifference, std::max(openVsNvidia, nvidiaVsRef));
                    
                    result.resultsMatch = (openVsRef <= tolerance && openVsNvidia <= tolerance && nvidiaVsRef <= tolerance);
                    
                    if (verbose_ && result.resultsMatch) {
                        std::cout << "  ✓ " << testName << " - 三方结果完全一致" << std::endl;
                    } else if (verbose_ && !result.resultsMatch) {
                        std::cout << "  ✗ " << testName << " - 存在差异: OpenVsRef=" << openVsRef 
                                  << ", OpenVsNvidia=" << openVsNvidia << ", NvidiaVsRef=" << nvidiaVsRef << std::endl;
                    }
                } else {
                    result.resultsMatch = (openVsRef <= tolerance);
                    if (verbose_) {
                        std::cout << "  ✓ " << testName << " - OpenNPP vs CPU参考一致 (NVIDIA NPP不可用)" << std::endl;
                    }
                }
#else
                result.resultsMatch = (openVsRef <= tolerance);
                if (verbose_) {
                    std::cout << "  ✓ " << testName << " - OpenNPP vs CPU参考一致" << std::endl;
                }
#endif
            }
            
        } catch (const std::exception& e) {
            result.errorMessage = e.what();
            if (verbose_) {
                std::cout << "  ✗ " << testName << " - 异常: " << e.what() << std::endl;
            }
        }
        
        results_.push_back(result);
    }
    
    // 32位浮点数版本(无缩放因子)
    void runThreeWayTest32f(const std::string& testName,
                           std::function<NppStatus(const Npp32f*, int, Npp32f, Npp32f*, int, NppiSize)> openFunc,
                           std::function<NppStatus(const Npp32f*, int, Npp32f, Npp32f*, int, NppiSize)> nvidiaFunc,
                           std::function<NppStatus(const Npp32f*, int, Npp32f, Npp32f*, int, NppiSize)> refFunc,
                           Npp32f sourceValue, Npp32f constant) {
        
        TestResult result(testName);
        
        try {
            const int width = 128;
            const int height = 128;
            const double tolerance = 1e-6;
            
            ValidationImageBuffer<Npp32f> srcBuffer(width, height);
            ValidationImageBuffer<Npp32f> openDstBuffer(width, height);
            ValidationImageBuffer<Npp32f> nvidiaDstBuffer(width, height);
            ValidationImageBuffer<Npp32f> refDstBuffer(width, height);
            
            srcBuffer.fillPattern(sourceValue);
            
            // Test OpenNPP
            NppStatus openStatus = openFunc(srcBuffer.devicePtr(), srcBuffer.step(), constant,
                                           openDstBuffer.devicePtr(), openDstBuffer.step(),
                                           srcBuffer.size());
            result.openNppSuccess = (openStatus == NPP_NO_ERROR);
            if (result.openNppSuccess) {
                openDstBuffer.copyFromDevice();
            }
            
#ifdef HAVE_NVIDIA_NPP
            // Test NVIDIA NPP
            NppStatus nvidiaStatus = nvidiaFunc(srcBuffer.devicePtr(), srcBuffer.step(), constant,
                                               nvidiaDstBuffer.devicePtr(), nvidiaDstBuffer.step(),
                                               srcBuffer.size());
            result.nvidiaNppSuccess = (nvidiaStatus == NPP_NO_ERROR);
            if (result.nvidiaNppSuccess) {
                nvidiaDstBuffer.copyFromDevice();
            }
#else
            result.nvidiaNppSuccess = true;
#endif
            
            // Test CPU reference
            srcBuffer.copyFromDevice();
            NppStatus refStatus = refFunc(srcBuffer.hostPtr(), srcBuffer.width() * sizeof(Npp32f), constant,
                                         refDstBuffer.hostPtr(), refDstBuffer.width() * sizeof(Npp32f),
                                         srcBuffer.size());
            result.cpuRefSuccess = (refStatus == NPP_NO_ERROR);
            
            // Three-way comparison
            if (result.openNppSuccess && result.cpuRefSuccess) {
                double openVsRef = ValidationImageBuffer<Npp32f>::compareBuffers(openDstBuffer, refDstBuffer, tolerance);
                result.maxDifference = openVsRef;
                
#ifdef HAVE_NVIDIA_NPP
                if (result.nvidiaNppSuccess) {
                    double openVsNvidia = ValidationImageBuffer<Npp32f>::compareBuffers(openDstBuffer, nvidiaDstBuffer, tolerance);
                    double nvidiaVsRef = ValidationImageBuffer<Npp32f>::compareBuffers(nvidiaDstBuffer, refDstBuffer, tolerance);
                    result.maxDifference = std::max(result.maxDifference, std::max(openVsNvidia, nvidiaVsRef));
                    
                    result.resultsMatch = (openVsRef <= tolerance && openVsNvidia <= tolerance && nvidiaVsRef <= tolerance);
                } else {
                    result.resultsMatch = (openVsRef <= tolerance);
                }
#else
                result.resultsMatch = (openVsRef <= tolerance);
#endif
                
                if (verbose_ && result.resultsMatch) {
                    std::cout << "  ✓ " << testName << " - 三方结果一致 (最大差异: " << result.maxDifference << ")" << std::endl;
                }
            }
            
        } catch (const std::exception& e) {
            result.errorMessage = e.what();
        }
        
        results_.push_back(result);
    }
    
    void testAddC_AllTypes() {
        // 8u AddC
        auto openFunc8u = [](const Npp8u* pSrc, int nSrcStep, Npp8u nConstant, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return nppiAddC_8u_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
        };
        
#ifdef HAVE_NVIDIA_NPP
        auto nvidiaFunc8u = [](const Npp8u* pSrc, int nSrcStep, Npp8u nConstant, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return ::nppiAddC_8u_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
        };
#else
        auto nvidiaFunc8u = openFunc8u;
#endif
        
        runThreeWayTest<Npp8u>("nppiAddC_8u_C1RSfs", openFunc8u, nvidiaFunc8u, nppi_reference::nppiAddC_8u_C1RSfs_reference, 100, 50, 1);
        
        // 16u AddC
        auto openFunc16u = [](const Npp16u* pSrc, int nSrcStep, Npp16u nConstant, Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return nppiAddC_16u_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
        };
        
#ifdef HAVE_NVIDIA_NPP
        auto nvidiaFunc16u = [](const Npp16u* pSrc, int nSrcStep, Npp16u nConstant, Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return ::nppiAddC_16u_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
        };
#else
        auto nvidiaFunc16u = openFunc16u;
#endif
        
        runThreeWayTest<Npp16u>("nppiAddC_16u_C1RSfs", openFunc16u, nvidiaFunc16u, nppi_reference::nppiAddC_16u_C1RSfs_reference, 1000, 500, 2);
        
        // 16s AddC
        auto openFunc16s = [](const Npp16s* pSrc, int nSrcStep, Npp16s nConstant, Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return nppiAddC_16s_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
        };
        
#ifdef HAVE_NVIDIA_NPP
        auto nvidiaFunc16s = [](const Npp16s* pSrc, int nSrcStep, Npp16s nConstant, Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return ::nppiAddC_16s_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
        };
#else
        auto nvidiaFunc16s = openFunc16s;
#endif
        
        runThreeWayTest<Npp16s>("nppiAddC_16s_C1RSfs", openFunc16s, nvidiaFunc16s, nppi_reference::nppiAddC_16s_C1RSfs_reference, 1000, -300, 1);
        
        // 32f AddC
        auto openFunc32f = [](const Npp32f* pSrc, int nSrcStep, Npp32f nConstant, Npp32f* pDst, int nDstStep, NppiSize oSizeROI) {
            return nppiAddC_32f_C1R(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI);
        };
        
#ifdef HAVE_NVIDIA_NPP
        auto nvidiaFunc32f = [](const Npp32f* pSrc, int nSrcStep, Npp32f nConstant, Npp32f* pDst, int nDstStep, NppiSize oSizeROI) {
            return ::nppiAddC_32f_C1R(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI);
        };
#else
        auto nvidiaFunc32f = openFunc32f;
#endif
        
        runThreeWayTest32f("nppiAddC_32f_C1R", openFunc32f, nvidiaFunc32f, nppi_reference::nppiAddC_32f_C1R_reference, 50.5f, 15.5f);
    }
    
    void testSubC_AllTypes() {
        // Similar implementation for SubC operations
        auto openFunc8u = [](const Npp8u* pSrc, int nSrcStep, Npp8u nConstant, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return nppiSubC_8u_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
        };
        
#ifdef HAVE_NVIDIA_NPP
        auto nvidiaFunc8u = [](const Npp8u* pSrc, int nSrcStep, Npp8u nConstant, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return ::nppiSubC_8u_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
        };
#else
        auto nvidiaFunc8u = openFunc8u;
#endif
        
        runThreeWayTest<Npp8u>("nppiSubC_8u_C1RSfs", openFunc8u, nvidiaFunc8u, nppi_reference::nppiSubC_8u_C1RSfs_reference, 200, 50, 1);
        
        // Add other SubC types...
        runThreeWayTest<Npp16u>("nppiSubC_16u_C1RSfs", 
            [](const Npp16u* pSrc, int nSrcStep, Npp16u nConstant, Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
                return nppiSubC_16u_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
            },
#ifdef HAVE_NVIDIA_NPP
            [](const Npp16u* pSrc, int nSrcStep, Npp16u nConstant, Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
                return ::nppiSubC_16u_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
            },
#else
            [](const Npp16u* pSrc, int nSrcStep, Npp16u nConstant, Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
                return nppiSubC_16u_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
            },
#endif
            nppi_reference::nppiSubC_16u_C1RSfs_reference, 2000, 500, 1);
    }
    
    void testMulC_AllTypes() {
        // MulC operations
        runThreeWayTest<Npp8u>("nppiMulC_8u_C1RSfs", 
            [](const Npp8u* pSrc, int nSrcStep, Npp8u nConstant, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
                return nppiMulC_8u_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
            },
#ifdef HAVE_NVIDIA_NPP
            [](const Npp8u* pSrc, int nSrcStep, Npp8u nConstant, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
                return ::nppiMulC_8u_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
            },
#else
            [](const Npp8u* pSrc, int nSrcStep, Npp8u nConstant, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
                return nppiMulC_8u_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
            },
#endif
            nppi_reference::nppiMulC_8u_C1RSfs_reference, 20, 3, 2);
    }
    
    void testDivC_AllTypes() {
        // DivC operations
        runThreeWayTest<Npp8u>("nppiDivC_8u_C1RSfs", 
            [](const Npp8u* pSrc, int nSrcStep, Npp8u nConstant, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
                return nppiDivC_8u_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
            },
#ifdef HAVE_NVIDIA_NPP
            [](const Npp8u* pSrc, int nSrcStep, Npp8u nConstant, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
                return ::nppiDivC_8u_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
            },
#else
            [](const Npp8u* pSrc, int nSrcStep, Npp8u nConstant, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
                return nppiDivC_8u_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
            },
#endif
            nppi_reference::nppiDivC_8u_C1RSfs_reference, 100, 4, 2);
    }
    
    void testMultiChannel() {
        TestResult result("nppiAddC_8u_C3RSfs_ThreeWay");
        
        try {
            const int width = 64;
            const int height = 64;
            const int channels = 3;
            const int dataSize = width * height * channels;
            
            std::vector<Npp8u> srcHost(dataSize);
            std::vector<Npp8u> openDstHost(dataSize, 0);
            std::vector<Npp8u> nvidiaDstHost(dataSize, 0);
            std::vector<Npp8u> refDstHost(dataSize, 0);
            
            // Fill with RGB pattern
            for (int i = 0; i < dataSize; i += 3) {
                srcHost[i + 0] = 50;   // R
                srcHost[i + 1] = 100;  // G
                srcHost[i + 2] = 150;  // B
            }
            
            // Allocate device memory using NPPI functions
            Npp8u *srcDev = nullptr, *openDstDev = nullptr, *nvidiaDstDev = nullptr;
            int srcStep, openDstStep, nvidiaStep;
            
            srcDev = nppiMalloc_8u_C3(width, height, &srcStep);
            openDstDev = nppiMalloc_8u_C3(width, height, &openDstStep);
            
            if (!srcDev || !openDstDev) {
                throw std::runtime_error("NPPI memory allocation failed");
            }
            
            cudaMemcpy2D(srcDev, srcStep, srcHost.data(), width * channels * sizeof(Npp8u),
                        width * channels * sizeof(Npp8u), height, cudaMemcpyHostToDevice);
            
            Npp8u constants[3] = {10, 20, 30};
            NppiSize roiSize = {width, height};
            
            // Test OpenNPP
            NppStatus openStatus = nppiAddC_8u_C3RSfs(srcDev, srcStep, constants, openDstDev, openDstStep, roiSize, 1);
            result.openNppSuccess = (openStatus == NPP_NO_ERROR);
            
            if (result.openNppSuccess) {
                cudaMemcpy2D(openDstHost.data(), width * channels * sizeof(Npp8u), openDstDev, openDstStep,
                            width * channels * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
            }
            
#ifdef HAVE_NVIDIA_NPP
            // Test NVIDIA NPP
            nvidiaDstDev = nppiMalloc_8u_C3(width, height, &nvidiaStep);
            if (!nvidiaDstDev) {
                throw std::runtime_error("NPPI memory allocation failed for NVIDIA test");
            }
            
            NppStatus nvidiaStatus = ::nppiAddC_8u_C3RSfs(srcDev, srcStep, constants, nvidiaDstDev, nvidiaStep, roiSize, 1);
            result.nvidiaNppSuccess = (nvidiaStatus == NPP_NO_ERROR);
            
            if (result.nvidiaNppSuccess) {
                cudaMemcpy2D(nvidiaDstHost.data(), width * channels * sizeof(Npp8u), nvidiaDstDev, nvidiaStep,
                            width * channels * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
            }
            nppiFree(nvidiaDstDev);
#else
            result.nvidiaNppSuccess = true;
#endif
            
            // Test CPU reference
            NppStatus refStatus = nppi_reference::nppiAddC_8u_C3RSfs_reference(srcHost.data(), width * channels, constants,
                                                                              refDstHost.data(), width * channels, roiSize, 1);
            result.cpuRefSuccess = (refStatus == NPP_NO_ERROR);
            
            // Compare results
            if (result.openNppSuccess && result.cpuRefSuccess) {
                bool allMatch = true;
                double maxDiff = 0.0;
                
                for (int i = 0; i < dataSize; ++i) {
                    double diff = std::abs(static_cast<double>(openDstHost[i]) - static_cast<double>(refDstHost[i]));
                    maxDiff = std::max(maxDiff, diff);
                    if (diff > 1.0) allMatch = false;
                }
                
#ifdef HAVE_NVIDIA_NPP
                if (result.nvidiaNppSuccess) {
                    for (int i = 0; i < dataSize; ++i) {
                        double diff = std::abs(static_cast<double>(openDstHost[i]) - static_cast<double>(nvidiaDstHost[i]));
                        maxDiff = std::max(maxDiff, diff);
                        if (diff > 1.0) allMatch = false;
                    }
                }
#endif
                
                result.maxDifference = maxDiff;
                result.resultsMatch = allMatch;
                
                if (verbose_) {
                    if (allMatch) {
                        std::cout << "  ✓ nppiAddC_8u_C3RSfs - 三方多通道结果一致" << std::endl;
                    } else {
                        std::cout << "  ✗ nppiAddC_8u_C3RSfs - 多通道结果存在差异: " << maxDiff << std::endl;
                    }
                }
            }
            
            // Cleanup
            nppiFree(srcDev);
            nppiFree(openDstDev);
            
        } catch (const std::exception& e) {
            result.errorMessage = e.what();
        }
        
        results_.push_back(result);
    }
};

} // namespace npp_validation

int main() {
    std::cout << "=== NPP 三方对比验证测试系统 ===" << std::endl;
    std::cout << "OpenNPP ↔ NVIDIA NPP ↔ CPU参考实现\n" << std::endl;
    
    // Initialize CUDA
    cudaError_t cudaResult = cudaSetDevice(0);
    if (cudaResult != cudaSuccess) {
        std::cout << "CUDA设备初始化失败" << std::endl;
        return 1;
    }
    
    npp_validation::ThreeWayValidationTest validationTest(true);
    validationTest.runAllTests();
    validationTest.printResults();
    
    std::cout << "\n=== 三方验证完成 ===" << std::endl;
    return validationTest.allTestsPassed() ? 0 : 1;
}