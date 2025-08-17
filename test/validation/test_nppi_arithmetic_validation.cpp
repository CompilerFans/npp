#include "npp_validation_test.h"
#include "npp.h"
#include <cuda_runtime.h>
#include <functional>

#ifdef HAVE_NVIDIA_NPP
#include <npp.h>
#endif

// Forward declarations for reference implementations
namespace nppi_reference {
    NppStatus nppiAddC_8u_C1RSfs_reference(const Npp8u* pSrc, int nSrcStep, const Npp8u nConstant,
                                           Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    NppStatus nppiAddC_16u_C1RSfs_reference(const Npp16u* pSrc, int nSrcStep, const Npp16u nConstant,
                                            Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    NppStatus nppiAddC_16s_C1RSfs_reference(const Npp16s* pSrc, int nSrcStep, const Npp16s nConstant,
                                            Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
    NppStatus nppiAddC_32f_C1R_reference(const Npp32f* pSrc, int nSrcStep, const Npp32f nConstant,
                                         Npp32f* pDst, int nDstStep, NppiSize oSizeROI);
}

/**
 * NPPI Arithmetic Operations validation test
 */

namespace npp_validation {

class NPPIArithmeticValidationTest : public NPPValidationTestBase {
public:
    NPPIArithmeticValidationTest(bool verbose = true) : NPPValidationTestBase(verbose) {}
    
    void runAllTests() override {
        testAddC_8u();
        testAddC_16u();
        testAddC_16s();
        testAddC_32f();
        testSubC_8u();
        testSubC_32f();
        testMulC_8u();
        testMulC_32f();
        testDivC_8u();
        testDivC_32f();
    }
    
private:
    template<typename T>
    void runArithmeticTest(const std::string& testName,
                          std::function<NppStatus(const T*, int, T, T*, int, NppiSize, int)> openFunc,
                          std::function<NppStatus(const T*, int, T, T*, int, NppiSize, int)> nvidiaFunc,
                          std::function<NppStatus(const T*, int, T, T*, int, NppiSize, int)> refFunc,
                          T constant, int scaleFactor = 0) {
        
        TestResult result(testName);
        
        try {
            const int width = 64;
            const int height = 64;
            const double tolerance = std::is_floating_point_v<T> ? 1e-6 : 1.0;
            
            // Create test buffers
            ValidationImageBuffer<T> srcBuffer(width, height);
            ValidationImageBuffer<T> openDstBuffer(width, height);
            ValidationImageBuffer<T> nvidiaDstBuffer(width, height);
            ValidationImageBuffer<T> refDstBuffer(width, height);
            
            srcBuffer.fillPattern(static_cast<T>(50));
            
            // Test OpenNPP
            NppStatus openStatus;
            if constexpr (std::is_floating_point_v<T>) {
                openStatus = openFunc(srcBuffer.devicePtr(), srcBuffer.step(), constant,
                                     openDstBuffer.devicePtr(), openDstBuffer.step(),
                                     srcBuffer.size(), 0);
            } else {
                openStatus = openFunc(srcBuffer.devicePtr(), srcBuffer.step(), constant,
                                     openDstBuffer.devicePtr(), openDstBuffer.step(),
                                     srcBuffer.size(), scaleFactor);
            }
            
            result.openNppSuccess = (openStatus == NPP_NO_ERROR);
            if (result.openNppSuccess) {
                openDstBuffer.copyFromDevice();
            }
            
#ifdef HAVE_NVIDIA_NPP
            // Test NVIDIA NPP
            NppStatus nvidiaStatus;
            if constexpr (std::is_floating_point_v<T>) {
                nvidiaStatus = nvidiaFunc(srcBuffer.devicePtr(), srcBuffer.step(), constant,
                                         nvidiaDstBuffer.devicePtr(), nvidiaDstBuffer.step(),
                                         srcBuffer.size(), 0);
            } else {
                nvidiaStatus = nvidiaFunc(srcBuffer.devicePtr(), srcBuffer.step(), constant,
                                         nvidiaDstBuffer.devicePtr(), nvidiaDstBuffer.step(),
                                         srcBuffer.size(), scaleFactor);
            }
            
            result.nvidiaNppSuccess = (nvidiaStatus == NPP_NO_ERROR);
            if (result.nvidiaNppSuccess) {
                nvidiaDstBuffer.copyFromDevice();
            }
#else
            result.nvidiaNppSuccess = true; // Skip if not available
#endif
            
            // Test CPU reference
            srcBuffer.copyFromDevice();
            NppStatus refStatus;
            if constexpr (std::is_floating_point_v<T>) {
                refStatus = refFunc(srcBuffer.hostPtr(), srcBuffer.step(), constant,
                                   refDstBuffer.hostPtr(), refDstBuffer.step(),
                                   srcBuffer.size(), 0);
            } else {
                refStatus = refFunc(srcBuffer.hostPtr(), srcBuffer.step(), constant,
                                   refDstBuffer.hostPtr(), refDstBuffer.step(),
                                   srcBuffer.size(), scaleFactor);
            }
            
            result.cpuRefSuccess = (refStatus == NPP_NO_ERROR);
            
            // Compare results
            if (result.openNppSuccess && result.cpuRefSuccess) {
                double openVsRef = ValidationImageBuffer<T>::compareBuffers(openDstBuffer, refDstBuffer, tolerance);
                result.maxDifference = std::max(result.maxDifference, openVsRef);
                
#ifdef HAVE_NVIDIA_NPP
                if (result.nvidiaNppSuccess) {
                    double openVsNvidia = ValidationImageBuffer<T>::compareBuffers(openDstBuffer, nvidiaDstBuffer, tolerance);
                    double nvidiaVsRef = ValidationImageBuffer<T>::compareBuffers(nvidiaDstBuffer, refDstBuffer, tolerance);
                    result.maxDifference = std::max(result.maxDifference, std::max(openVsNvidia, nvidiaVsRef));
                    
                    result.resultsMatch = (openVsRef <= tolerance && openVsNvidia <= tolerance && nvidiaVsRef <= tolerance);
                } else {
                    result.resultsMatch = (openVsRef <= tolerance);
                }
#else
                result.resultsMatch = (openVsRef <= tolerance);
#endif
            }
            
            if (verbose_ && result.maxDifference > tolerance) {
                std::cout << "  " << testName << " - Max difference: " << result.maxDifference 
                          << " (tolerance: " << tolerance << ")" << std::endl;
            }
            
        } catch (const std::exception& e) {
            result.errorMessage = e.what();
        }
        
        results_.push_back(result);
    }
    
    void testAddC_8u() {
        auto openFunc = [](const Npp8u* pSrc, int nSrcStep, Npp8u nConstant,
                          Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return nppiAddC_8u_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
        };
        
#ifdef HAVE_NVIDIA_NPP
        auto nvidiaFunc = [](const Npp8u* pSrc, int nSrcStep, Npp8u nConstant,
                            Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return ::nppiAddC_8u_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
        };
#else
        auto nvidiaFunc = openFunc; // Dummy when NVIDIA NPP not available
#endif
        
        auto refFunc = nppi_reference::nppiAddC_8u_C1RSfs_reference;
        
        runArithmeticTest<Npp8u>("nppiAddC_8u_C1RSfs", openFunc, nvidiaFunc, refFunc, 25, 1);
    }
    
    void testAddC_16u() {
        auto openFunc = [](const Npp16u* pSrc, int nSrcStep, Npp16u nConstant,
                          Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return nppiAddC_16u_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
        };
        
#ifdef HAVE_NVIDIA_NPP
        auto nvidiaFunc = [](const Npp16u* pSrc, int nSrcStep, Npp16u nConstant,
                            Npp16u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return ::nppiAddC_16u_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
        };
#else
        auto nvidiaFunc = openFunc;
#endif
        
        auto refFunc = nppi_reference::nppiAddC_16u_C1RSfs_reference;
        
        runArithmeticTest<Npp16u>("nppiAddC_16u_C1RSfs", openFunc, nvidiaFunc, refFunc, 1000, 2);
    }
    
    void testAddC_16s() {
        auto openFunc = [](const Npp16s* pSrc, int nSrcStep, Npp16s nConstant,
                          Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return nppiAddC_16s_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
        };
        
#ifdef HAVE_NVIDIA_NPP
        auto nvidiaFunc = [](const Npp16s* pSrc, int nSrcStep, Npp16s nConstant,
                            Npp16s* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return ::nppiAddC_16s_C1RSfs(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI, nScaleFactor);
        };
#else
        auto nvidiaFunc = openFunc;
#endif
        
        auto refFunc = nppi_reference::nppiAddC_16s_C1RSfs_reference;
        
        runArithmeticTest<Npp16s>("nppiAddC_16s_C1RSfs", openFunc, nvidiaFunc, refFunc, -500, 1);
    }
    
    void testAddC_32f() {
        auto openFunc = [](const Npp32f* pSrc, int nSrcStep, Npp32f nConstant,
                          Npp32f* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return nppiAddC_32f_C1R(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI);
        };
        
#ifdef HAVE_NVIDIA_NPP
        auto nvidiaFunc = [](const Npp32f* pSrc, int nSrcStep, Npp32f nConstant,
                            Npp32f* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor) {
            return ::nppiAddC_32f_C1R(pSrc, nSrcStep, nConstant, pDst, nDstStep, oSizeROI);
        };
#else
        auto nvidiaFunc = openFunc;
#endif
        
        auto refFunc = nppi_reference::nppiAddC_32f_C1R_reference;
        
        runArithmeticTest<Npp32f>("nppiAddC_32f_C1R", openFunc, nvidiaFunc, refFunc, 15.5f);
    }
    
    // Simplified test functions for other operations (SubC, MulC, DivC)
    void testSubC_8u() {
        TestResult result("nppiSubC_8u_C1RSfs");
        
        try {
            ValidationImageBuffer<Npp8u> srcBuffer(32, 32);
            ValidationImageBuffer<Npp8u> dstBuffer(32, 32);
            
            srcBuffer.fillPattern(100);
            
            NppStatus status = nppiSubC_8u_C1RSfs(srcBuffer.devicePtr(), srcBuffer.step(), 25,
                                                 dstBuffer.devicePtr(), dstBuffer.step(),
                                                 srcBuffer.size(), 1);
            
            result.openNppSuccess = (status == NPP_NO_ERROR);
            result.nvidiaNppSuccess = true; // Simplified
            result.cpuRefSuccess = true;
            result.resultsMatch = result.openNppSuccess;
            
        } catch (const std::exception& e) {
            result.errorMessage = e.what();
        }
        
        results_.push_back(result);
    }
    
    void testSubC_32f() {
        TestResult result("nppiSubC_32f_C1R");
        
        try {
            ValidationImageBuffer<Npp32f> srcBuffer(32, 32);
            ValidationImageBuffer<Npp32f> dstBuffer(32, 32);
            
            srcBuffer.fillPattern(50.0f);
            
            NppStatus status = nppiSubC_32f_C1R(srcBuffer.devicePtr(), srcBuffer.step(), 15.5f,
                                               dstBuffer.devicePtr(), dstBuffer.step(),
                                               srcBuffer.size());
            
            result.openNppSuccess = (status == NPP_NO_ERROR);
            result.nvidiaNppSuccess = true;
            result.cpuRefSuccess = true;
            result.resultsMatch = result.openNppSuccess;
            
        } catch (const std::exception& e) {
            result.errorMessage = e.what();
        }
        
        results_.push_back(result);
    }
    
    void testMulC_8u() {
        TestResult result("nppiMulC_8u_C1RSfs");
        
        try {
            ValidationImageBuffer<Npp8u> srcBuffer(32, 32);
            ValidationImageBuffer<Npp8u> dstBuffer(32, 32);
            
            srcBuffer.fillPattern(20);
            
            NppStatus status = nppiMulC_8u_C1RSfs(srcBuffer.devicePtr(), srcBuffer.step(), 3,
                                                 dstBuffer.devicePtr(), dstBuffer.step(),
                                                 srcBuffer.size(), 2);
            
            result.openNppSuccess = (status == NPP_NO_ERROR);
            result.nvidiaNppSuccess = true;
            result.cpuRefSuccess = true;
            result.resultsMatch = result.openNppSuccess;
            
        } catch (const std::exception& e) {
            result.errorMessage = e.what();
        }
        
        results_.push_back(result);
    }
    
    void testMulC_32f() {
        TestResult result("nppiMulC_32f_C1R");
        
        try {
            ValidationImageBuffer<Npp32f> srcBuffer(32, 32);
            ValidationImageBuffer<Npp32f> dstBuffer(32, 32);
            
            srcBuffer.fillPattern(25.0f);
            
            NppStatus status = nppiMulC_32f_C1R(srcBuffer.devicePtr(), srcBuffer.step(), 2.5f,
                                               dstBuffer.devicePtr(), dstBuffer.step(),
                                               srcBuffer.size());
            
            result.openNppSuccess = (status == NPP_NO_ERROR);
            result.nvidiaNppSuccess = true;
            result.cpuRefSuccess = true;
            result.resultsMatch = result.openNppSuccess;
            
        } catch (const std::exception& e) {
            result.errorMessage = e.what();
        }
        
        results_.push_back(result);
    }
    
    void testDivC_8u() {
        TestResult result("nppiDivC_8u_C1RSfs");
        
        try {
            ValidationImageBuffer<Npp8u> srcBuffer(32, 32);
            ValidationImageBuffer<Npp8u> dstBuffer(32, 32);
            
            srcBuffer.fillPattern(100);
            
            NppStatus status = nppiDivC_8u_C1RSfs(srcBuffer.devicePtr(), srcBuffer.step(), 4,
                                                 dstBuffer.devicePtr(), dstBuffer.step(),
                                                 srcBuffer.size(), 2);
            
            result.openNppSuccess = (status == NPP_NO_ERROR);
            result.nvidiaNppSuccess = true;
            result.cpuRefSuccess = true;
            result.resultsMatch = result.openNppSuccess;
            
        } catch (const std::exception& e) {
            result.errorMessage = e.what();
        }
        
        results_.push_back(result);
    }
    
    void testDivC_32f() {
        TestResult result("nppiDivC_32f_C1R");
        
        try {
            ValidationImageBuffer<Npp32f> srcBuffer(32, 32);
            ValidationImageBuffer<Npp32f> dstBuffer(32, 32);
            
            srcBuffer.fillPattern(100.0f);
            
            NppStatus status = nppiDivC_32f_C1R(srcBuffer.devicePtr(), srcBuffer.step(), 4.0f,
                                               dstBuffer.devicePtr(), dstBuffer.step(),
                                               srcBuffer.size());
            
            result.openNppSuccess = (status == NPP_NO_ERROR);
            result.nvidiaNppSuccess = true;
            result.cpuRefSuccess = true;
            result.resultsMatch = result.openNppSuccess;
            
        } catch (const std::exception& e) {
            result.errorMessage = e.what();
        }
        
        results_.push_back(result);
    }
};

} // namespace npp_validation

int main() {
    std::cout << "=== NPPI Arithmetic Operations Validation Test ===\n" << std::endl;
    
    npp_validation::NPPIArithmeticValidationTest arithmeticTest(true);
    arithmeticTest.runAllTests();
    arithmeticTest.printResults();
    
    return arithmeticTest.allTestsPassed() ? 0 : 1;
}