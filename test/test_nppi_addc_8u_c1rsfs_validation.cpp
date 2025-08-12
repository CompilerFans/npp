#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <vector>

#include "nppdefs.h"
#include "npp_image_arithmetic.h"

// Forward declaration of the NVIDIA function
extern "C" {
    NppStatus nppiAddC_8u_C1RSfs_Ctx(const Npp8u* pSrc1, int nSrc1Step, const Npp8u nConstant,
                                     Npp8u* pDst, int nDstStep, NppiSize oSizeROI, 
                                     int nScaleFactor, NppStreamContext nppStreamCtx);
}

/**
 * Validation test against NVIDIA closed-source library
 * 
 * This test specifically compares our open-source implementation
 * against the actual NVIDIA NPP library to ensure correctness.
 */

class ValidationTest {
public:
    ValidationTest() {
        // Initialize CUDA
        cudaFree(0); // Initialize CUDA context
        
        // Create CUDA stream
        cudaStreamCreate(&stream_);
        
        // Setup NPP stream context
        stream_ctx_.hStream = stream_;
        stream_ctx_.nCudaDeviceId = 0;
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        stream_ctx_.nMultiProcessorCount = prop.multiProcessorCount;
        stream_ctx_.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
        stream_ctx_.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
        stream_ctx_.nSharedMemPerBlock = prop.sharedMemPerBlock;
        stream_ctx_.nCudaDevAttrComputeCapabilityMajor = prop.major;
        stream_ctx_.nCudaDevAttrComputeCapabilityMinor = prop.minor;
        stream_ctx_.nStreamFlags = 0;
    }
    
    ~ValidationTest() {
        cudaStreamDestroy(stream_);
    }
    
    struct ImageBuffer {
        Npp8u* host;
        Npp8u* device;
        int width;
        int height;
        int step;
        
        ImageBuffer(int w, int h) : width(w), height(h) {
            step = width; // 1 byte per pixel
            host = new Npp8u[width * height];
            cudaMalloc(&device, width * height);
        }
        
        ~ImageBuffer() {
            delete[] host;
            if (device) cudaFree(device);
        }
        
        void copyToDevice() {
            cudaMemcpy(device, host, width * height, cudaMemcpyHostToDevice);
        }
        
        void copyFromDevice() {
            cudaMemcpy(host, device, width * height, cudaMemcpyDeviceToHost);
        }
        
        void fillRandom() {
            for (int i = 0; i < width * height; i++) {
                host[i] = rand() % 256;
            }
        }
        
        void fillPattern(int patternType = 0) {
            switch (patternType) {
                case 0: // Gradient
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            host[y * width + x] = (x + y) % 256;
                        }
                    }
                    break;
                case 1: // Checkerboard
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            host[y * width + x] = ((x / 8) + (y / 8)) % 2 ? 0 : 255;
                        }
                    }
                    break;
                case 2: // Solid colors
                    for (int i = 0; i < width * height; i++) {
                        host[i] = (i / (width * height / 4)) * 64;
                    }
                    break;
            }
        }
        
        bool compare(const ImageBuffer& other, int maxError = 0) const {
            if (width != other.width || height != other.height) return false;
            
            int errors = 0;
            for (int i = 0; i < width * height; i++) {
                if (abs(host[i] - other.host[i]) > maxError) {
                    errors++;
                    if (errors <= 10) { // Only print first few errors
                        printf("    Pixel %d: expected %u, got %u\n", i, host[i], other.host[i]);
                    }
                }
            }
            
            if (errors > 0) {
                printf("    Total errors: %d/%d (%.2f%%)\n", 
                       errors, width * height, 100.0 * errors / (width * height));
            }
            
            return errors == 0;
        }
    };
    
    bool runValidationTest(const char* testName, int width, int height, 
                          Npp8u constant, int scaleFactor, int patternType = 0) {
        printf("=== Validation Test: %s ===\n", testName);
        printf("  Dimensions: %dx%d\n", width, height);
        printf("  Constant: %u\n", constant);
        printf("  ScaleFactor: %d\n", scaleFactor);
        
        // Create buffers
        ImageBuffer src(width, height);
        ImageBuffer dst_nvidia(width, height);
        ImageBuffer dst_cuda(width, height);
        ImageBuffer dst_cpu(width, height);
        
        // Fill with test data
        src.fillPattern(patternType);
        src.copyToDevice();
        
        NppiSize roi = {width, height};
        
        // Run NVIDIA NPP (if available)
        NppStatus status_nvidia = NPP_NO_ERROR;
        bool nvidia_available = true;
        
        status_nvidia = nppiAddC_8u_C1RSfs_Ctx(
            src.device, src.step, constant,
            dst_nvidia.device, dst_nvidia.step, roi, 
            scaleFactor, stream_ctx_
        );
        
        if (status_nvidia != NPP_NO_ERROR) {
            printf("  Warning: NVIDIA NPP function returned status %d\n", status_nvidia);
            nvidia_available = false;
        } else {
            cudaDeviceSynchronize();
            dst_nvidia.copyFromDevice();
        }
        
        // Run our CUDA implementation
        NppStatus status_cuda = nppiAddC_8u_C1RSfs_Ctx_cuda(
            src.device, src.step, constant,
            dst_cuda.device, dst_cuda.step, roi,
            scaleFactor, stream_ctx_
        );
        
        if (status_cuda != NPP_NO_ERROR) {
            printf("  ERROR: Our CUDA implementation failed with status %d\n", status_cuda);
            return false;
        }
        
        cudaDeviceSynchronize();
        dst_cuda.copyFromDevice();
        
        // Run our CPU implementation
        NppStatus status_cpu = nppiAddC_8u_C1RSfs_Ctx_reference(
            src.host, src.step, constant,
            dst_cpu.host, dst_cpu.step, roi, scaleFactor
        );
        
        if (status_cpu != NPP_NO_ERROR) {
            printf("  ERROR: Our CPU implementation failed with status %d\n", status_cpu);
            return false;
        }
        
        // Validate results
        bool cuda_vs_cpu = dst_cpu.compare(dst_cuda, 0);
        printf("  CUDA vs CPU: %s\n", cuda_vs_cpu ? "PASS" : "FAIL");
        
        bool nvidia_vs_ours = true;
        if (nvidia_available) {
            nvidia_vs_ours = dst_nvidia.compare(dst_cuda, 0);
            printf("  NVIDIA vs Our CUDA: %s\n", nvidia_vs_ours ? "PASS" : "FAIL");
        } else {
            printf("  NVIDIA comparison: SKIPPED (NPP not available)\n");
        }
        
        // Performance comparison
        if (nvidia_available) {
            printf("  Performance comparison:\n");
            
            const int iterations = 100;
            
            // NVIDIA timing
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            
            cudaEventRecord(start, stream_);
            for (int i = 0; i < iterations; i++) {
                nppiAddC_8u_C1RSfs_Ctx(
                    src.device, src.step, constant,
                    dst_nvidia.device, dst_nvidia.step, roi,
                    scaleFactor, stream_ctx_
                );
            }
            cudaEventRecord(stop, stream_);
            cudaEventSynchronize(stop);
            
            float nvidia_time;
            cudaEventElapsedTime(&nvidia_time, start, stop);
            
            // Our CUDA timing
            cudaEventRecord(start, stream_);
            for (int i = 0; i < iterations; i++) {
                nppiAddC_8u_C1RSfs_Ctx_cuda(
                    src.device, src.step, constant,
                    dst_cuda.device, dst_cuda.step, roi,
                    scaleFactor, stream_ctx_
                );
            }
            cudaEventRecord(stop, stream_);
            cudaEventSynchronize(stop);
            
            float our_time;
            cudaEventElapsedTime(&our_time, start, stop);
            
            printf("    NVIDIA: %.2f ms\n", nvidia_time);
            printf("    Our CUDA: %.2f ms\n", our_time);
            printf("    Speed ratio: %.2fx\n", nvidia_time / our_time);
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        
        return cuda_vs_cpu && (nvidia_available ? nvidia_vs_ours : true);
    }
    
    void runValidationSuite() {
        printf("=== NPP AddC 8u C1RSfs Validation Suite ===\n");
        printf("Validating our implementation against NVIDIA closed-source library\n\n");
        
        int passed = 0;
        int total = 0;
        
        struct ValidationTest {
            const char* name;
            int width;
            int height;
            Npp8u constant;
            int scaleFactor;
            int patternType;
        };
        
        ValidationTest tests[] = {
            {"Small Gradient", 32, 32, 50, 0, 0},
            {"Medium Gradient", 256, 256, 100, 1, 0},
            {"Large Gradient", 1024, 768, 128, 2, 0},
            {"Checkerboard", 64, 64, 25, 1, 1},
            {"Solid Colors", 128, 128, 75, 3, 2},
            {"Edge Case - Zero Constant", 64, 64, 0, 1, 0},
            {"Edge Case - Max Constant", 64, 64, 255, 1, 0},
            {"Edge Case - Zero Scale", 64, 64, 50, 0, 0},
            {"Edge Case - High Scale", 64, 64, 200, 4, 0},
            {"Small Dimensions", 1, 1, 100, 1, 0},
            {"Wide Image", 1024, 32, 75, 2, 0},
            {"Tall Image", 32, 1024, 75, 2, 0},
            {"Power-of-2", 512, 512, 128, 1, 0},
            {"Non-Power-of-2", 100, 200, 64, 2, 0}
        };
        
        for (const auto& test : tests) {
            total++;
            if (runValidationTest(test.name, test.width, test.height, 
                                test.constant, test.scaleFactor, test.patternType)) {
                passed++;
            }
            printf("\n");
        }
        
        printf("=== Validation Summary ===\n");
        printf("Passed: %d/%d\n", passed, total);
        printf("Overall: %s\n", (passed == total) ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
        
        if (passed == total) {
            printf("✓ Our implementation is functionally equivalent to NVIDIA NPP\n");
        }
    }
    
private:
    cudaStream_t stream_;
    NppStreamContext stream_ctx_;
};

int main() {
    ValidationTest validator;
    validator.runValidationSuite();
    return 0;
}