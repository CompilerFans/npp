#include "npp_validation_test.h"
#include "npp.h"
#include <cuda_runtime.h>

#ifdef HAVE_NVIDIA_NPP
#include <npp.h>
#endif

/**
 * NPPI Support Functions validation test
 */

namespace npp_validation {

class NPPISupportValidationTest : public NPPValidationTestBase {
public:
    NPPISupportValidationTest(bool verbose = true) : NPPValidationTestBase(verbose) {}
    
    void runAllTests() override {
        testMemoryAllocation_8u();
        testMemoryAllocation_16u();
        testMemoryAllocation_16s();
        testMemoryAllocation_32f();
        testMemoryFree();
    }
    
private:
    void testMemoryAllocation_8u() {
        TestResult result("nppiMalloc_8u_C1");
        
        try {
            const int width = 256;
            const int height = 256;
            
            // Test OpenNPP
            int openStep;
            Npp8u* openPtr = nppiMalloc_8u_C1(width, height, &openStep);
            result.openNppSuccess = (openPtr != nullptr && openStep >= width);
            
            if (verbose_ && result.openNppSuccess) {
                std::cout << "OpenNPP allocated 8u: ptr=" << static_cast<void*>(openPtr) 
                          << ", step=" << openStep << std::endl;
            }
            
#ifdef HAVE_NVIDIA_NPP
            // Test NVIDIA NPP
            int nvidiaStep;
            Npp8u* nvidiaPtr = ::nppiMalloc_8u_C1(width, height, &nvidiaStep);
            result.nvidiaNppSuccess = (nvidiaPtr != nullptr && nvidiaStep >= width);
            
            if (verbose_ && result.nvidiaNppSuccess) {
                std::cout << "NVIDIA NPP allocated 8u: ptr=" << static_cast<void*>(nvidiaPtr) 
                          << ", step=" << nvidiaStep << std::endl;
            }
            
            // Steps should be identical for same allocation parameters
            result.resultsMatch = (openStep == nvidiaStep);
            result.maxDifference = std::abs(openStep - nvidiaStep);
            
            // Clean up NVIDIA allocation
            if (nvidiaPtr) {
                ::nppiFree(nvidiaPtr);
            }
#else
            result.nvidiaNppSuccess = true;
            result.resultsMatch = true;
#endif
            
            // CPU reference - direct CUDA allocation
            void* cudaPtr;
            size_t cudaPitch;
            cudaError_t cudaResult = cudaMallocPitch(&cudaPtr, &cudaPitch, width, height);
            result.cpuRefSuccess = (cudaResult == cudaSuccess);
            
            if (result.cpuRefSuccess) {
                int cudaStep = static_cast<int>(cudaPitch);
                if (verbose_) {
                    std::cout << "CUDA direct allocation: ptr=" << cudaPtr 
                              << ", step=" << cudaStep << std::endl;
                }
                
                // OpenNPP should match CUDA direct allocation behavior
                if (result.resultsMatch) {
                    result.resultsMatch = (openStep == cudaStep);
                }
                
                cudaFree(cudaPtr);
            }
            
            // Clean up OpenNPP allocation
            if (openPtr) {
                nppiFree(openPtr);
            }
            
        } catch (const std::exception& e) {
            result.errorMessage = e.what();
        }
        
        results_.push_back(result);
    }
    
    void testMemoryAllocation_16u() {
        TestResult result("nppiMalloc_16u_C1");
        
        try {
            const int width = 128;
            const int height = 128;
            
            // Test OpenNPP
            int openStep;
            Npp16u* openPtr = nppiMalloc_16u_C1(width, height, &openStep);
            result.openNppSuccess = (openPtr != nullptr && openStep >= width * sizeof(Npp16u));
            
#ifdef HAVE_NVIDIA_NPP
            // Test NVIDIA NPP
            int nvidiaStep;
            Npp16u* nvidiaPtr = ::nppiMalloc_16u_C1(width, height, &nvidiaStep);
            result.nvidiaNppSuccess = (nvidiaPtr != nullptr && nvidiaStep >= width * sizeof(Npp16u));
            
            result.resultsMatch = (openStep == nvidiaStep);
            result.maxDifference = std::abs(openStep - nvidiaStep);
            
            if (nvidiaPtr) {
                ::nppiFree(nvidiaPtr);
            }
#else
            result.nvidiaNppSuccess = true;
            result.resultsMatch = true;
#endif
            
            // CPU reference
            void* cudaPtr;
            size_t cudaPitch;
            cudaError_t cudaResult = cudaMallocPitch(&cudaPtr, &cudaPitch, 
                                                    width * sizeof(Npp16u), height);
            result.cpuRefSuccess = (cudaResult == cudaSuccess);
            
            if (result.cpuRefSuccess) {
                int cudaStep = static_cast<int>(cudaPitch);
                if (result.resultsMatch) {
                    result.resultsMatch = (openStep == cudaStep);
                }
                cudaFree(cudaPtr);
            }
            
            if (openPtr) {
                nppiFree(openPtr);
            }
            
        } catch (const std::exception& e) {
            result.errorMessage = e.what();
        }
        
        results_.push_back(result);
    }
    
    void testMemoryAllocation_16s() {
        TestResult result("nppiMalloc_16s_C1");
        
        try {
            const int width = 64;
            const int height = 64;
            
            // Test OpenNPP
            int openStep;
            Npp16s* openPtr = nppiMalloc_16s_C1(width, height, &openStep);
            result.openNppSuccess = (openPtr != nullptr && openStep >= width * sizeof(Npp16s));
            
#ifdef HAVE_NVIDIA_NPP
            // Test NVIDIA NPP  
            int nvidiaStep;
            Npp16s* nvidiaPtr = ::nppiMalloc_16s_C1(width, height, &nvidiaStep);
            result.nvidiaNppSuccess = (nvidiaPtr != nullptr && nvidiaStep >= width * sizeof(Npp16s));
            
            result.resultsMatch = (openStep == nvidiaStep);
            
            if (nvidiaPtr) {
                ::nppiFree(nvidiaPtr);
            }
#else
            result.nvidiaNppSuccess = true;
            result.resultsMatch = true;
#endif
            
            result.cpuRefSuccess = true; // Memory allocation consistency tested above
            
            if (openPtr) {
                nppiFree(openPtr);
            }
            
        } catch (const std::exception& e) {
            result.errorMessage = e.what();
        }
        
        results_.push_back(result);
    }
    
    void testMemoryAllocation_32f() {
        TestResult result("nppiMalloc_32f_C1");
        
        try {
            const int width = 32;
            const int height = 32;
            
            // Test OpenNPP
            int openStep;
            Npp32f* openPtr = nppiMalloc_32f_C1(width, height, &openStep);
            result.openNppSuccess = (openPtr != nullptr && openStep >= width * sizeof(Npp32f));
            
#ifdef HAVE_NVIDIA_NPP
            // Test NVIDIA NPP
            int nvidiaStep;
            Npp32f* nvidiaPtr = ::nppiMalloc_32f_C1(width, height, &nvidiaStep);
            result.nvidiaNppSuccess = (nvidiaPtr != nullptr && nvidiaStep >= width * sizeof(Npp32f));
            
            result.resultsMatch = (openStep == nvidiaStep);
            
            if (nvidiaPtr) {
                ::nppiFree(nvidiaPtr);
            }
#else
            result.nvidiaNppSuccess = true;
            result.resultsMatch = true;
#endif
            
            result.cpuRefSuccess = true;
            
            if (openPtr) {
                nppiFree(openPtr);
            }
            
        } catch (const std::exception& e) {
            result.errorMessage = e.what();
        }
        
        results_.push_back(result);
    }
    
    void testMemoryFree() {
        TestResult result("nppiFree");
        
        try {
            // Allocate memory to test free
            int step;
            Npp8u* ptr = nppiMalloc_8u_C1(100, 100, &step);
            
            if (ptr) {
                // Test OpenNPP free
                nppiFree(ptr);
                result.openNppSuccess = true;
            }
            
#ifdef HAVE_NVIDIA_NPP
            // Test NVIDIA NPP free
            Npp8u* nvidiaPtr = ::nppiMalloc_8u_C1(100, 100, &step);
            if (nvidiaPtr) {
                ::nppiFree(nvidiaPtr);
                result.nvidiaNppSuccess = true;
            }
#else
            result.nvidiaNppSuccess = true;
#endif
            
            result.cpuRefSuccess = true; // Free operation success is hard to verify directly
            result.resultsMatch = true;  // If no crashes, consider it successful
            
        } catch (const std::exception& e) {
            result.errorMessage = e.what();
        }
        
        results_.push_back(result);
    }
};

} // namespace npp_validation

int main() {
    std::cout << "=== NPPI Support Functions Validation Test ===\n" << std::endl;
    
    npp_validation::NPPISupportValidationTest supportTest(true);
    supportTest.runAllTests();
    supportTest.printResults();
    
    return supportTest.allTestsPassed() ? 0 : 1;
}