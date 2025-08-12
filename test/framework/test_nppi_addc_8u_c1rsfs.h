#ifndef TEST_NPPI_ADDC_8U_C1RSFS_H
#define TEST_NPPI_ADDC_8U_C1RSFS_H

#include "npp_test_framework.h"
#include "npp_image_arithmetic.h"
#include <cassert>
#include <dlfcn.h>

// Forward declaration of NVIDIA NPP function
extern "C" {
    NppStatus nppiAddC_8u_C1RSfs_Ctx(const Npp8u* pSrc1, int nSrc1Step, const Npp8u nConstant,
                                     Npp8u* pDst, int nDstStep, NppiSize oSizeROI, 
                                     int nScaleFactor, NppStreamContext nppStreamCtx);
}

/**
 * @brief Test parameters for nppiAddC_8u_C1RSfs_Ctx
 */
class AddC8uTestParameters : public NPPTest::TestParameters {
public:
    int width;
    int height;
    Npp8u constant;
    int scale_factor;
    int pattern_type;
    double tolerance;
    
    AddC8uTestParameters(int w, int h, Npp8u c, int sf, int pt = 0, double tol = 0.0)
        : width(w), height(h), constant(c), scale_factor(sf), pattern_type(pt), tolerance(tol) {}
    
    std::string toString() const override {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "%dx%d, const=%u, scale=%d, pattern=%d", 
                width, height, constant, scale_factor, pattern_type);
        return std::string(buffer);
    }
    
    bool isValid() const override {
        return width > 0 && height > 0 && scale_factor >= 0 && scale_factor <= 16;
    }
};

/**
 * @brief Test class for nppiAddC_8u_C1RSfs_Ctx
 */
class TestNppiAddC8uC1RSfs : public NPPTest::NPPTestBase {
private:
    bool nvidia_available_;

public:
    TestNppiAddC8uC1RSfs() : NPPTestBase("nppiAddC_8u_C1RSfs_Ctx", "Add Constant to 8-bit Image") {
        // Check if NVIDIA NPP is available
        nvidia_available_ = checkNVIDIAAvailability();
    }
    
    bool isNVIDIAAvailable() const override {
        return nvidia_available_;
    }
    
    NPPTest::TestResult run(const NPPTest::TestParameters& params) override {
        const AddC8uTestParameters& add_params = static_cast<const AddC8uTestParameters&>(params);
        NPPTest::TestResult result;
        
        try {
            // Create buffers
            NPPTest::ImageBuffer<Npp8u> src(add_params.width, add_params.height);
            NPPTest::ImageBuffer<Npp8u> dst_cpu(add_params.width, add_params.height);
            NPPTest::ImageBuffer<Npp8u> dst_cuda(add_params.width, add_params.height);
            NPPTest::ImageBuffer<Npp8u> dst_nvidia(add_params.width, add_params.height);
            
            // Fill test data
            src.fillPattern(add_params.pattern_type);
            src.copyToDevice();
            
            NppiSize roi = {add_params.width, add_params.height};
            
            // Run CPU implementation
            NPPTest::Timer cpu_timer(false);
            cpu_timer.start();
            
            NppStatus cpu_status = nppiAddC_8u_C1RSfs_Ctx_reference(
                src.host(), add_params.width, add_params.constant,
                dst_cpu.host(), add_params.width, roi, add_params.scale_factor
            );
            
            result.cpu_time_ms = cpu_timer.stop();
            
            if (cpu_status != NPP_NO_ERROR) {
                result.error_message = "CPU implementation failed with status " + std::to_string(cpu_status);
                return result;
            }
            
            // Run CUDA implementation
            NPPTest::Timer gpu_timer(true);
            gpu_timer.start();
            
            NppStatus cuda_status = nppiAddC_8u_C1RSfs_Ctx_cuda(
                src.device(), add_params.width, add_params.constant,
                dst_cuda.device(), add_params.width, roi, add_params.scale_factor, stream_ctx_
            );
            
            cudaDeviceSynchronize();
            result.gpu_time_ms = gpu_timer.stop();
            
            if (cuda_status != NPP_NO_ERROR) {
                result.error_message = "CUDA implementation failed with status " + std::to_string(cuda_status);
                return result;
            }
            
            dst_cuda.copyFromDevice();
            
            // Run NVIDIA implementation if available
            if (nvidia_available_) {
                NPPTest::Timer nvidia_timer(true);
                nvidia_timer.start();
                
                NppStatus nvidia_status = nppiAddC_8u_C1RSfs_Ctx(
                    src.device(), add_params.width, add_params.constant,
                    dst_nvidia.device(), add_params.width, roi, 
                    add_params.scale_factor, stream_ctx_
                );
                
                cudaDeviceSynchronize();
                result.nvidia_time_ms = nvidia_timer.stop();
                
                if (nvidia_status != NPP_NO_ERROR) {
                    result.error_message = "NVIDIA implementation failed with status " + std::to_string(nvidia_status);
                    return result;
                }
                
                dst_nvidia.copyFromDevice();
                
                // Compare CUDA vs NVIDIA
                double error = dst_cuda.compare(dst_nvidia, add_params.tolerance);
                result.max_error = error;
                
                if (error > add_params.tolerance) {
                    result.error_message = "CUDA vs NVIDIA error: " + std::to_string(error);
                    return result;
                }
            }
            
            // Compare CPU vs CUDA
            double cpu_cuda_error = dst_cpu.compare(dst_cuda, add_params.tolerance);
            if (cpu_cuda_error > add_params.tolerance) {
                result.error_message = "CPU vs CUDA error: " + std::to_string(cpu_cuda_error);
                return result;
            }
            
            result.max_error = std::max(result.max_error, cpu_cuda_error);
            result.passed = true;
            
        } catch (const std::exception& e) {
            result.error_message = std::string("Exception: ") + e.what();
        }
        
        return result;
    }
    
    std::vector<std::unique_ptr<NPPTest::TestParameters>> getTestCases() const override {
        std::vector<std::unique_ptr<NPPTest::TestParameters>> test_cases;
        
        // Standard test cases
        std::vector<std::pair<int, int>> sizes = {
            {1, 1}, {32, 32}, {64, 64}, {128, 128}, {256, 256}, {512, 512},
            {1024, 768}, {1920, 1080}, {2048, 2048}, {100, 200}
        };
        
        std::vector<Npp8u> constants = {0, 1, 50, 100, 128, 200, 255};
        std::vector<int> scale_factors = {0, 1, 2, 3, 4, 8};
        std::vector<int> patterns = {0, 1, 2}; // gradient, checkerboard, solid
        
        for (const auto& [w, h] : sizes) {
            for (Npp8u c : constants) {
                for (int sf : scale_factors) {
                    for (int pt : patterns) {
                        test_cases.emplace_back(
                            std::make_unique<AddC8uTestParameters>(w, h, c, sf, pt, 0.0)
                        );
                    }
                }
            }
        }
        
        // Edge cases
        test_cases.emplace_back(std::make_unique<AddC8uTestParameters>(1, 1000, 128, 1, 0, 0.0));
        test_cases.emplace_back(std::make_unique<AddC8uTestParameters>(1000, 1, 128, 1, 0, 0.0));
        
        return test_cases;
    }
    
private:
    bool checkNVIDIAAvailability() {
        // Try to load NVIDIA NPP dynamically
        void* handle = dlopen("libnppial.so", RTLD_LAZY);
        if (!handle) {
            handle = dlopen("nppial64_12.dll", RTLD_LAZY);
        }
        
        if (handle) {
            dlclose(handle);
            return true;
        }
        return false;
    }
};

// Auto-register the test
static bool register_addc_test = []() {
    NPPTest::TestRegistry::registerTest(std::make_unique<TestNppiAddC8uC1RSfs>());
    return true;
}();

#endif // TEST_NPPI_ADDC_8U_C1RSFS_H
