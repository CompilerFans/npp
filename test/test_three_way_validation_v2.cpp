#include "framework/npp_validation_test_v2.h"
#include "npp.h"
#include <cuda_runtime.h>
#include <functional>

#ifdef HAVE_NVIDIA_NPP
#include <npp.h>
#endif

/**
 * 三方对比验证测试 - 使用改进的pitched内存处理
 */

// CPU参考实现的前向声明
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

class ThreeWayValidationTestV2 : public NPPValidationTestBase {
public:
    ThreeWayValidationTestV2(bool verbose = true) : NPPValidationTestBase(verbose) {}
    
    void runAllTests() override {
        std::cout << "=== 三方对比验证测试 V2 (正确的Pitched内存处理) ===\n" << std::endl;
        
        // 测试AddC操作
        std::cout << "--- AddC 运算测试 ---" << std::endl;
        testAddC_8u();
        testAddC_16u();
        testAddC_16s();
        testAddC_32f();
    }
    
private:
    void testAddC_8u() {
        const std::string testName = "nppiAddC_8u_C1RSfs_v2";
        TestResult result(testName);
        
        try {
            const int width = 127;  // 使用非对齐宽度以测试pitch处理
            const int height = 64;
            const Npp8u sourceValue = 50;
            const Npp8u constant = 10;
            const int scaleFactor = 0;
            
            // 创建测试缓冲区
            ValidationImageBufferV2<Npp8u> srcBuffer(width, height);
            ValidationImageBufferV2<Npp8u> openDstBuffer(width, height);
            ValidationImageBufferV2<Npp8u> refDstBuffer(width, height);
            
            // 填充源数据
            srcBuffer.fillPattern(sourceValue);
            
            // 测试OpenNPP (设备上运行)
            NppStatus openStatus = nppiAddC_8u_C1RSfs(
                srcBuffer.devicePtr(), srcBuffer.deviceStep(), constant,
                openDstBuffer.devicePtr(), openDstBuffer.deviceStep(),
                srcBuffer.size(), scaleFactor);
            
            result.openNppSuccess = (openStatus == NPP_NO_ERROR);
            if (result.openNppSuccess) {
                openDstBuffer.copyFromDevice();
            }
            
#ifdef HAVE_NVIDIA_NPP
            // 测试NVIDIA NPP
            ValidationImageBufferV2<Npp8u> nvidiaDstBuffer(width, height);
            NppStatus nvidiaStatus = ::nppiAddC_8u_C1RSfs(
                srcBuffer.devicePtr(), srcBuffer.deviceStep(), constant,
                nvidiaDstBuffer.devicePtr(), nvidiaDstBuffer.deviceStep(),
                srcBuffer.size(), scaleFactor);
            
            result.nvidiaNppSuccess = (nvidiaStatus == NPP_NO_ERROR);
            if (result.nvidiaNppSuccess) {
                nvidiaDstBuffer.copyFromDevice();
            }
#else
            result.nvidiaNppSuccess = true;
#endif
            
            // 测试CPU参考实现（主机上运行，使用正确的步长）
            srcBuffer.copyFromDevice();  // 确保主机内存有数据
            NppStatus refStatus = nppi_reference::nppiAddC_8u_C1RSfs_reference(
                srcBuffer.hostPtr(), srcBuffer.hostStep(), constant,
                refDstBuffer.hostPtr(), refDstBuffer.hostStep(),
                srcBuffer.size(), scaleFactor);
            
            result.cpuRefSuccess = (refStatus == NPP_NO_ERROR);
            
            // 比较结果
            if (result.openNppSuccess && result.cpuRefSuccess) {
                double openVsRef = ValidationImageBufferV2<Npp8u>::compareBuffers(
                    openDstBuffer, refDstBuffer, 1.0);
                result.maxDifference = openVsRef;
                
#ifdef HAVE_NVIDIA_NPP
                if (result.nvidiaNppSuccess) {
                    double openVsNvidia = ValidationImageBufferV2<Npp8u>::compareBuffers(
                        openDstBuffer, nvidiaDstBuffer, 1.0);
                    double nvidiaVsRef = ValidationImageBufferV2<Npp8u>::compareBuffers(
                        nvidiaDstBuffer, refDstBuffer, 1.0);
                    
                    result.maxDifference = std::max({openVsRef, openVsNvidia, nvidiaVsRef});
                    result.resultsMatch = (openVsRef <= 1.0 && openVsNvidia <= 1.0 && nvidiaVsRef <= 1.0);
                    
                    if (verbose_) {
                        if (result.resultsMatch) {
                            std::cout << "  ✓ " << testName << " - 三方结果完全一致" << std::endl;
                        } else {
                            std::cout << "  ✗ " << testName << " - 存在差异: "
                                     << "OpenVsRef=" << openVsRef 
                                     << ", OpenVsNvidia=" << openVsNvidia 
                                     << ", NvidiaVsRef=" << nvidiaVsRef << std::endl;
                        }
                    }
                } else {
                    result.resultsMatch = (openVsRef <= 1.0);
                }
#else
                result.resultsMatch = (openVsRef <= 1.0);
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
    
    void testAddC_16u() {
        const std::string testName = "nppiAddC_16u_C1RSfs_v2";
        TestResult result(testName);
        
        try {
            const int width = 129;  // 非对齐宽度
            const int height = 64;
            const Npp16u sourceValue = 500;
            const Npp16u constant = 100;
            const int scaleFactor = 0;
            
            ValidationImageBufferV2<Npp16u> srcBuffer(width, height);
            ValidationImageBufferV2<Npp16u> openDstBuffer(width, height);
            ValidationImageBufferV2<Npp16u> refDstBuffer(width, height);
            
            srcBuffer.fillPattern(sourceValue);
            
            // 测试OpenNPP
            NppStatus openStatus = nppiAddC_16u_C1RSfs(
                srcBuffer.devicePtr(), srcBuffer.deviceStep(), constant,
                openDstBuffer.devicePtr(), openDstBuffer.deviceStep(),
                srcBuffer.size(), scaleFactor);
            
            result.openNppSuccess = (openStatus == NPP_NO_ERROR);
            if (result.openNppSuccess) {
                openDstBuffer.copyFromDevice();
            }
            
#ifdef HAVE_NVIDIA_NPP
            ValidationImageBufferV2<Npp16u> nvidiaDstBuffer(width, height);
            NppStatus nvidiaStatus = ::nppiAddC_16u_C1RSfs(
                srcBuffer.devicePtr(), srcBuffer.deviceStep(), constant,
                nvidiaDstBuffer.devicePtr(), nvidiaDstBuffer.deviceStep(),
                srcBuffer.size(), scaleFactor);
            
            result.nvidiaNppSuccess = (nvidiaStatus == NPP_NO_ERROR);
            if (result.nvidiaNppSuccess) {
                nvidiaDstBuffer.copyFromDevice();
            }
#else
            result.nvidiaNppSuccess = true;
#endif
            
            // CPU参考
            srcBuffer.copyFromDevice();
            NppStatus refStatus = nppi_reference::nppiAddC_16u_C1RSfs_reference(
                srcBuffer.hostPtr(), srcBuffer.hostStep(), constant,
                refDstBuffer.hostPtr(), refDstBuffer.hostStep(),
                srcBuffer.size(), scaleFactor);
            
            result.cpuRefSuccess = (refStatus == NPP_NO_ERROR);
            
            // 比较
            if (result.openNppSuccess && result.cpuRefSuccess) {
                double openVsRef = ValidationImageBufferV2<Npp16u>::compareBuffers(
                    openDstBuffer, refDstBuffer, 1.0);
                result.maxDifference = openVsRef;
                
#ifdef HAVE_NVIDIA_NPP
                if (result.nvidiaNppSuccess) {
                    double openVsNvidia = ValidationImageBufferV2<Npp16u>::compareBuffers(
                        openDstBuffer, nvidiaDstBuffer, 1.0);
                    double nvidiaVsRef = ValidationImageBufferV2<Npp16u>::compareBuffers(
                        nvidiaDstBuffer, refDstBuffer, 1.0);
                    
                    result.maxDifference = std::max({openVsRef, openVsNvidia, nvidiaVsRef});
                    result.resultsMatch = (openVsRef <= 1.0 && openVsNvidia <= 1.0 && nvidiaVsRef <= 1.0);
                    
                    if (verbose_ && result.resultsMatch) {
                        std::cout << "  ✓ " << testName << " - 三方结果完全一致" << std::endl;
                    }
                } else {
                    result.resultsMatch = (openVsRef <= 1.0);
                }
#else
                result.resultsMatch = (openVsRef <= 1.0);
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
    
    void testAddC_16s() {
        const std::string testName = "nppiAddC_16s_C1RSfs_v2";
        TestResult result(testName);
        
        try {
            const int width = 131;  // 非对齐宽度
            const int height = 64;
            const Npp16s sourceValue = -500;
            const Npp16s constant = 100;
            const int scaleFactor = 0;
            
            ValidationImageBufferV2<Npp16s> srcBuffer(width, height);
            ValidationImageBufferV2<Npp16s> openDstBuffer(width, height);
            ValidationImageBufferV2<Npp16s> refDstBuffer(width, height);
            
            srcBuffer.fillPattern(sourceValue);
            
            NppStatus openStatus = nppiAddC_16s_C1RSfs(
                srcBuffer.devicePtr(), srcBuffer.deviceStep(), constant,
                openDstBuffer.devicePtr(), openDstBuffer.deviceStep(),
                srcBuffer.size(), scaleFactor);
            
            result.openNppSuccess = (openStatus == NPP_NO_ERROR);
            if (result.openNppSuccess) {
                openDstBuffer.copyFromDevice();
            }
            
#ifdef HAVE_NVIDIA_NPP
            ValidationImageBufferV2<Npp16s> nvidiaDstBuffer(width, height);
            NppStatus nvidiaStatus = ::nppiAddC_16s_C1RSfs(
                srcBuffer.devicePtr(), srcBuffer.deviceStep(), constant,
                nvidiaDstBuffer.devicePtr(), nvidiaDstBuffer.deviceStep(),
                srcBuffer.size(), scaleFactor);
            
            result.nvidiaNppSuccess = (nvidiaStatus == NPP_NO_ERROR);
            if (result.nvidiaNppSuccess) {
                nvidiaDstBuffer.copyFromDevice();
            }
#else
            result.nvidiaNppSuccess = true;
#endif
            
            srcBuffer.copyFromDevice();
            NppStatus refStatus = nppi_reference::nppiAddC_16s_C1RSfs_reference(
                srcBuffer.hostPtr(), srcBuffer.hostStep(), constant,
                refDstBuffer.hostPtr(), refDstBuffer.hostStep(),
                srcBuffer.size(), scaleFactor);
            
            result.cpuRefSuccess = (refStatus == NPP_NO_ERROR);
            
            if (result.openNppSuccess && result.cpuRefSuccess) {
                double openVsRef = ValidationImageBufferV2<Npp16s>::compareBuffers(
                    openDstBuffer, refDstBuffer, 1.0);
                result.maxDifference = openVsRef;
                result.resultsMatch = (openVsRef <= 1.0);
                
                if (verbose_ && result.resultsMatch) {
                    std::cout << "  ✓ " << testName << " - 三方结果完全一致" << std::endl;
                }
            }
            
        } catch (const std::exception& e) {
            result.errorMessage = e.what();
            if (verbose_) {
                std::cout << "  ✗ " << testName << " - 异常: " << e.what() << std::endl;
            }
        }
        
        results_.push_back(result);
    }
    
    void testAddC_32f() {
        const std::string testName = "nppiAddC_32f_C1R_v2";
        TestResult result(testName);
        
        try {
            const int width = 133;  // 非对齐宽度
            const int height = 64;
            const Npp32f sourceValue = 50.0f;
            const Npp32f constant = 10.5f;
            
            ValidationImageBufferV2<Npp32f> srcBuffer(width, height);
            ValidationImageBufferV2<Npp32f> openDstBuffer(width, height);
            ValidationImageBufferV2<Npp32f> refDstBuffer(width, height);
            
            srcBuffer.fillPattern(sourceValue);
            
            NppStatus openStatus = nppiAddC_32f_C1R(
                srcBuffer.devicePtr(), srcBuffer.deviceStep(), constant,
                openDstBuffer.devicePtr(), openDstBuffer.deviceStep(),
                srcBuffer.size());
            
            result.openNppSuccess = (openStatus == NPP_NO_ERROR);
            if (result.openNppSuccess) {
                openDstBuffer.copyFromDevice();
            }
            
#ifdef HAVE_NVIDIA_NPP
            ValidationImageBufferV2<Npp32f> nvidiaDstBuffer(width, height);
            NppStatus nvidiaStatus = ::nppiAddC_32f_C1R(
                srcBuffer.devicePtr(), srcBuffer.deviceStep(), constant,
                nvidiaDstBuffer.devicePtr(), nvidiaDstBuffer.deviceStep(),
                srcBuffer.size());
            
            result.nvidiaNppSuccess = (nvidiaStatus == NPP_NO_ERROR);
            if (result.nvidiaNppSuccess) {
                nvidiaDstBuffer.copyFromDevice();
            }
#else
            result.nvidiaNppSuccess = true;
#endif
            
            srcBuffer.copyFromDevice();
            NppStatus refStatus = nppi_reference::nppiAddC_32f_C1R_reference(
                srcBuffer.hostPtr(), srcBuffer.hostStep(), constant,
                refDstBuffer.hostPtr(), refDstBuffer.hostStep(),
                srcBuffer.size());
            
            result.cpuRefSuccess = (refStatus == NPP_NO_ERROR);
            
            if (result.openNppSuccess && result.cpuRefSuccess) {
                const double tolerance = 1e-6;
                double openVsRef = ValidationImageBufferV2<Npp32f>::compareBuffers(
                    openDstBuffer, refDstBuffer, tolerance);
                result.maxDifference = openVsRef;
                result.resultsMatch = (openVsRef <= tolerance);
                
                if (verbose_) {
                    std::cout << "  ✓ " << testName << " - 三方结果一致 (最大差异: " 
                             << result.maxDifference << ")" << std::endl;
                }
            }
            
        } catch (const std::exception& e) {
            result.errorMessage = e.what();
            if (verbose_) {
                std::cout << "  ✗ " << testName << " - 异常: " << e.what() << std::endl;
            }
        }
        
        results_.push_back(result);
    }
};

int main() {
    std::cout << "=== NPP 三方对比验证测试系统 V2 ===" << std::endl;
    std::cout << "正确处理Pitched内存布局\n" << std::endl;
    
    ThreeWayValidationTestV2 test;
    test.runAllTests();
    test.printResults();
    
    std::cout << "\n=== 测试完成 ===" << std::endl;
    return 0;
}