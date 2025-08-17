#include "npp_validation_test.h"
#include "npp.h"
#include <cuda_runtime.h>

#ifdef HAVE_NVIDIA_NPP
#include <npp.h>
#endif

/**
 * NPP Core functions validation test
 */

namespace npp_validation {

class NPPCoreValidationTest : public NPPValidationTestBase {
public:
    NPPCoreValidationTest(bool verbose = true) : NPPValidationTestBase(verbose) {}
    
    void runAllTests() override {
        testGetLibVersion();
        testGetGpuProperties();
        testStreamManagement();
    }
    
private:
    void testGetLibVersion() {
        TestResult result("nppGetLibVersion");
        
        try {
            // Test OpenNPP
            const NppLibraryVersion* openVersion = nppGetLibVersion();
            result.openNppSuccess = (openVersion != nullptr);
            
            if (verbose_ && openVersion) {
                std::cout << "OpenNPP Version: " << openVersion->major 
                          << "." << openVersion->minor 
                          << "." << openVersion->build << std::endl;
            }
            
#ifdef HAVE_NVIDIA_NPP
            // Test NVIDIA NPP
            const NppLibraryVersion* nvidiaVersion = ::nppGetLibVersion();
            result.nvidiaNppSuccess = (nvidiaVersion != nullptr);
            
            if (verbose_ && nvidiaVersion) {
                std::cout << "NVIDIA NPP Version: " << nvidiaVersion->major 
                          << "." << nvidiaVersion->minor 
                          << "." << nvidiaVersion->build << std::endl;
            }
            
            // Compare structures (versions may differ, that's expected)
            result.resultsMatch = (openVersion != nullptr && nvidiaVersion != nullptr);
#else
            result.nvidiaNppSuccess = true; // Skip if not available
            result.resultsMatch = true;
#endif
            
            result.cpuRefSuccess = true; // No CPU ref needed for version info
            
        } catch (const std::exception& e) {
            result.errorMessage = e.what();
        }
        
        results_.push_back(result);
    }
    
    void testGetGpuProperties() {
        TestResult result("nppGetGpuProperties");
        
        try {
            // Test OpenNPP
            int openSMs = nppGetGpuNumSMs();
            int openMaxThreads = nppGetMaxThreadsPerBlock();
            int openMaxThreadsPerSM = nppGetMaxThreadsPerSM();
            
            result.openNppSuccess = (openSMs > 0 && openMaxThreads > 0 && openMaxThreadsPerSM > 0);
            
            if (verbose_) {
                std::cout << "OpenNPP GPU Info - SMs: " << openSMs 
                          << ", MaxThreads/Block: " << openMaxThreads
                          << ", MaxThreads/SM: " << openMaxThreadsPerSM << std::endl;
            }
            
#ifdef HAVE_NVIDIA_NPP
            // Test NVIDIA NPP
            int nvidiaSMs = ::nppGetGpuNumSMs();
            int nvidiaMaxThreads = ::nppGetMaxThreadsPerBlock();
            int nvidiaMaxThreadsPerSM = ::nppGetMaxThreadsPerSM();
            
            result.nvidiaNppSuccess = (nvidiaSMs > 0 && nvidiaMaxThreads > 0 && nvidiaMaxThreadsPerSM > 0);
            
            if (verbose_) {
                std::cout << "NVIDIA NPP GPU Info - SMs: " << nvidiaSMs 
                          << ", MaxThreads/Block: " << nvidiaMaxThreads
                          << ", MaxThreads/SM: " << nvidiaMaxThreadsPerSM << std::endl;
            }
            
            // Values should match exactly
            result.resultsMatch = (openSMs == nvidiaSMs && 
                                 openMaxThreads == nvidiaMaxThreads &&
                                 openMaxThreadsPerSM == nvidiaMaxThreadsPerSM);
            
            if (!result.resultsMatch) {
                double diff1 = std::abs(openSMs - nvidiaSMs);
                double diff2 = std::abs(openMaxThreads - nvidiaMaxThreads);
                double diff3 = std::abs(openMaxThreadsPerSM - nvidiaMaxThreadsPerSM);
                result.maxDifference = std::max(diff1, std::max(diff2, diff3));
            }
#else
            result.nvidiaNppSuccess = true;
            result.resultsMatch = true;
#endif
            
            // CPU reference - get values directly from CUDA
            cudaDeviceProp prop;
            cudaError_t cudaResult = cudaGetDeviceProperties(&prop, 0);
            result.cpuRefSuccess = (cudaResult == cudaSuccess);
            
            if (result.cpuRefSuccess && result.resultsMatch) {
                result.resultsMatch = (openSMs == prop.multiProcessorCount &&
                                     openMaxThreads == prop.maxThreadsPerBlock &&
                                     openMaxThreadsPerSM == prop.maxThreadsPerMultiProcessor);
            }
            
        } catch (const std::exception& e) {
            result.errorMessage = e.what();
        }
        
        results_.push_back(result);
    }
    
    void testStreamManagement() {
        TestResult result("nppStreamManagement");
        
        try {
            // Create test stream
            cudaStream_t testStream;
            cudaError_t cudaResult = cudaStreamCreate(&testStream);
            if (cudaResult != cudaSuccess) {
                result.errorMessage = "Failed to create CUDA stream";
                results_.push_back(result);
                return;
            }
            
            // Test OpenNPP
            cudaStream_t originalStream = nppGetStream();
            NppStatus status = nppSetStream(testStream);
            cudaStream_t currentStream = nppGetStream();
            
            result.openNppSuccess = (status == NPP_NO_ERROR && currentStream == testStream);
            
            // Test stream context
            NppStreamContext streamCtx;
            status = nppGetStreamContext(&streamCtx);
            result.openNppSuccess = result.openNppSuccess && 
                                   (status == NPP_NO_ERROR && streamCtx.hStream == testStream);
            
#ifdef HAVE_NVIDIA_NPP
            // Test NVIDIA NPP
            cudaStream_t nvidiaOriginalStream = ::nppGetStream();
            NppStatus nvidiaStatus = ::nppSetStream(testStream);
            cudaStream_t nvidiaCurrentStream = ::nppGetStream();
            
            result.nvidiaNppSuccess = (nvidiaStatus == NPP_NO_ERROR && nvidiaCurrentStream == testStream);
            
            // Test NVIDIA stream context
            NppStreamContext nvidiaStreamCtx;
            nvidiaStatus = ::nppGetStreamContext(&nvidiaStreamCtx);
            result.nvidiaNppSuccess = result.nvidiaNppSuccess && 
                                     (nvidiaStatus == NPP_NO_ERROR && nvidiaStreamCtx.hStream == testStream);
            
            result.resultsMatch = (currentStream == nvidiaCurrentStream);
            
            // Restore original streams
            ::nppSetStream(nvidiaOriginalStream);
#else
            result.nvidiaNppSuccess = true;
            result.resultsMatch = true;
#endif
            
            result.cpuRefSuccess = true; // Stream management is GPU-specific
            
            // Restore original stream
            nppSetStream(originalStream);
            
            // Clean up
            cudaStreamDestroy(testStream);
            
        } catch (const std::exception& e) {
            result.errorMessage = e.what();
        }
        
        results_.push_back(result);
    }
};

} // namespace npp_validation

int main() {
    std::cout << "=== NPP Core Functions Validation Test ===\n" << std::endl;
    
    npp_validation::NPPCoreValidationTest coreTest(true);
    coreTest.runAllTests();
    coreTest.printResults();
    
    return coreTest.allTestsPassed() ? 0 : 1;
}