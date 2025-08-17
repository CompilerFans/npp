#include "npp.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#ifdef HAVE_NVIDIA_NPP
#include <npp.h>
#endif

/**
 * 简单的验证测试，避免复杂内存管理
 */

// Forward declaration for reference implementation
extern "C" {
    NppStatus nppiAddC_8u_C1RSfs_reference(const Npp8u* pSrc, int nSrcStep, const Npp8u nConstant,
                                           Npp8u* pDst, int nDstStep, NppiSize oSizeROI, int nScaleFactor);
}

bool testAddC_8u_simple() {
    std::cout << "Testing nppiAddC_8u_C1RSfs..." << std::endl;
    
    // Simple test parameters
    const int width = 32;
    const int height = 32;
    const int dataSize = width * height;
    
    // Allocate host memory
    std::vector<Npp8u> srcHost(dataSize, 50);
    std::vector<Npp8u> dstHost(dataSize, 0);
    std::vector<Npp8u> expectedHost(dataSize, 0);
    
    // Calculate expected result (CPU reference)
    for (int i = 0; i < dataSize; ++i) {
        expectedHost[i] = static_cast<Npp8u>((srcHost[i] + 25) >> 1); // scale factor 1
    }
    
    // Allocate device memory
    Npp8u *srcDev = nullptr, *dstDev = nullptr;
    size_t srcPitch, dstPitch;
    
    cudaError_t cudaResult = cudaMallocPitch((void**)&srcDev, &srcPitch, width * sizeof(Npp8u), height);
    if (cudaResult != cudaSuccess) {
        std::cout << "Failed to allocate source device memory" << std::endl;
        return false;
    }
    
    cudaResult = cudaMallocPitch((void**)&dstDev, &dstPitch, width * sizeof(Npp8u), height);
    if (cudaResult != cudaSuccess) {
        std::cout << "Failed to allocate destination device memory" << std::endl;
        cudaFree(srcDev);
        return false;
    }
    
    // Copy to device
    cudaResult = cudaMemcpy2D(srcDev, srcPitch, srcHost.data(), width * sizeof(Npp8u),
                             width * sizeof(Npp8u), height, cudaMemcpyHostToDevice);
    if (cudaResult != cudaSuccess) {
        std::cout << "Failed to copy to device" << std::endl;
        cudaFree(srcDev);
        cudaFree(dstDev);
        return false;
    }
    
    // Test our implementation
    NppiSize roiSize = {width, height};
    NppStatus status = nppiAddC_8u_C1RSfs(srcDev, (int)srcPitch, 25, dstDev, (int)dstPitch, roiSize, 1);
    
    bool testPassed = false;
    if (status == NPP_NO_ERROR) {
        // Copy result back
        cudaResult = cudaMemcpy2D(dstHost.data(), width * sizeof(Npp8u), dstDev, dstPitch,
                                 width * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
        
        if (cudaResult == cudaSuccess) {
            // Verify result
            testPassed = true;
            for (int i = 0; i < dataSize && testPassed; ++i) {
                if (dstHost[i] != expectedHost[i]) {
                    std::cout << "Mismatch at index " << i << ": got " << (int)dstHost[i] 
                              << ", expected " << (int)expectedHost[i] << std::endl;
                    testPassed = false;
                }
            }
            
            if (testPassed) {
                std::cout << "nppiAddC_8u_C1RSfs: PASS" << std::endl;
            }
        } else {
            std::cout << "Failed to copy result from device" << std::endl;
        }
    } else {
        std::cout << "nppiAddC_8u_C1RSfs failed with status: " << status << std::endl;
    }
    
#ifdef HAVE_NVIDIA_NPP
    // Also test against NVIDIA NPP
    std::vector<Npp8u> nvidiaDstHost(dataSize, 0);
    Npp8u* nvidiaDstDev = nullptr;
    size_t nvidiaDstPitch;
    
    cudaResult = cudaMallocPitch((void**)&nvidiaDstDev, &nvidiaDstPitch, width * sizeof(Npp8u), height);
    if (cudaResult == cudaSuccess) {
        NppStatus nvidiaStatus = ::nppiAddC_8u_C1RSfs(srcDev, (int)srcPitch, 25, nvidiaDstDev, (int)nvidiaDstPitch, roiSize, 1);
        
        if (nvidiaStatus == NPP_NO_ERROR) {
            cudaResult = cudaMemcpy2D(nvidiaDstHost.data(), width * sizeof(Npp8u), nvidiaDstDev, nvidiaDstPitch,
                                     width * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
            
            if (cudaResult == cudaSuccess) {
                // Compare with NVIDIA result
                bool nvidiaMatch = true;
                for (int i = 0; i < dataSize && nvidiaMatch; ++i) {
                    if (dstHost[i] != nvidiaDstHost[i]) {
                        std::cout << "NVIDIA mismatch at index " << i << ": our " << (int)dstHost[i] 
                                  << ", NVIDIA " << (int)nvidiaDstHost[i] << std::endl;
                        nvidiaMatch = false;
                    }
                }
                
                if (nvidiaMatch) {
                    std::cout << "NVIDIA NPP comparison: PASS" << std::endl;
                } else {
                    std::cout << "NVIDIA NPP comparison: FAIL" << std::endl;
                }
            }
        }
        
        cudaFree(nvidiaDstDev);
    }
#endif
    
    // Clean up
    cudaFree(srcDev);
    cudaFree(dstDev);
    
    return testPassed;
}

bool testAddC_16u_simple() {
    std::cout << "Testing nppiAddC_16u_C1RSfs..." << std::endl;
    
    const int width = 32;
    const int height = 32;
    const int dataSize = width * height;
    
    std::vector<Npp16u> srcHost(dataSize, 1000);
    std::vector<Npp16u> dstHost(dataSize, 0);
    
    Npp16u *srcDev = nullptr, *dstDev = nullptr;
    size_t srcPitch, dstPitch;
    
    cudaError_t cudaResult = cudaMallocPitch((void**)&srcDev, &srcPitch, width * sizeof(Npp16u), height);
    if (cudaResult != cudaSuccess) return false;
    
    cudaResult = cudaMallocPitch((void**)&dstDev, &dstPitch, width * sizeof(Npp16u), height);
    if (cudaResult != cudaSuccess) {
        cudaFree(srcDev);
        return false;
    }
    
    cudaResult = cudaMemcpy2D(srcDev, srcPitch, srcHost.data(), width * sizeof(Npp16u),
                             width * sizeof(Npp16u), height, cudaMemcpyHostToDevice);
    if (cudaResult != cudaSuccess) {
        cudaFree(srcDev);
        cudaFree(dstDev);
        return false;
    }
    
    NppiSize roiSize = {width, height};
    NppStatus status = nppiAddC_16u_C1RSfs(srcDev, (int)srcPitch, 500, dstDev, (int)dstPitch, roiSize, 2);
    
    bool testPassed = false;
    if (status == NPP_NO_ERROR) {
        cudaResult = cudaMemcpy2D(dstHost.data(), width * sizeof(Npp16u), dstDev, dstPitch,
                                 width * sizeof(Npp16u), height, cudaMemcpyDeviceToHost);
        
        if (cudaResult == cudaSuccess) {
            // Expected: (1000 + 500) >> 2 = 375
            Npp16u expected = (1000 + 500) >> 2;
            testPassed = true;
            for (int i = 0; i < dataSize && testPassed; ++i) {
                if (dstHost[i] != expected) {
                    std::cout << "Mismatch at index " << i << ": got " << dstHost[i] 
                              << ", expected " << expected << std::endl;
                    testPassed = false;
                }
            }
            
            if (testPassed) {
                std::cout << "nppiAddC_16u_C1RSfs: PASS" << std::endl;
            }
        }
    } else {
        std::cout << "nppiAddC_16u_C1RSfs failed with status: " << status << std::endl;
    }
    
    cudaFree(srcDev);
    cudaFree(dstDev);
    
    return testPassed;
}

bool testAddC_16s_simple() {
    std::cout << "Testing nppiAddC_16s_C1RSfs..." << std::endl;
    
    const int width = 32;
    const int height = 32;
    const int dataSize = width * height;
    
    std::vector<Npp16s> srcHost(dataSize, 1000);
    std::vector<Npp16s> dstHost(dataSize, 0);
    
    Npp16s *srcDev = nullptr, *dstDev = nullptr;
    size_t srcPitch, dstPitch;
    
    cudaError_t cudaResult = cudaMallocPitch((void**)&srcDev, &srcPitch, width * sizeof(Npp16s), height);
    if (cudaResult != cudaSuccess) return false;
    
    cudaResult = cudaMallocPitch((void**)&dstDev, &dstPitch, width * sizeof(Npp16s), height);
    if (cudaResult != cudaSuccess) {
        cudaFree(srcDev);
        return false;
    }
    
    cudaResult = cudaMemcpy2D(srcDev, srcPitch, srcHost.data(), width * sizeof(Npp16s),
                             width * sizeof(Npp16s), height, cudaMemcpyHostToDevice);
    if (cudaResult != cudaSuccess) {
        cudaFree(srcDev);
        cudaFree(dstDev);
        return false;
    }
    
    NppiSize roiSize = {width, height};
    NppStatus status = nppiAddC_16s_C1RSfs(srcDev, (int)srcPitch, -300, dstDev, (int)dstPitch, roiSize, 1);
    
    bool testPassed = false;
    if (status == NPP_NO_ERROR) {
        cudaResult = cudaMemcpy2D(dstHost.data(), width * sizeof(Npp16s), dstDev, dstPitch,
                                 width * sizeof(Npp16s), height, cudaMemcpyDeviceToHost);
        
        if (cudaResult == cudaSuccess) {
            // Expected: (1000 + (-300)) >> 1 = 350
            Npp16s expected = (1000 + (-300)) >> 1;
            testPassed = true;
            for (int i = 0; i < dataSize && testPassed; ++i) {
                if (dstHost[i] != expected) {
                    std::cout << "Mismatch at index " << i << ": got " << dstHost[i] 
                              << ", expected " << expected << std::endl;
                    testPassed = false;
                }
            }
            
            if (testPassed) {
                std::cout << "nppiAddC_16s_C1RSfs: PASS" << std::endl;
            }
        }
    } else {
        std::cout << "nppiAddC_16s_C1RSfs failed with status: " << status << std::endl;
    }
    
    cudaFree(srcDev);
    cudaFree(dstDev);
    
    return testPassed;
}

bool testSubC_8u_simple() {
    std::cout << "Testing nppiSubC_8u_C1RSfs..." << std::endl;
    
    const int width = 32;
    const int height = 32;
    const int dataSize = width * height;
    
    std::vector<Npp8u> srcHost(dataSize, 100);
    std::vector<Npp8u> dstHost(dataSize, 0);
    
    Npp8u *srcDev = nullptr, *dstDev = nullptr;
    size_t srcPitch, dstPitch;
    
    cudaError_t cudaResult = cudaMallocPitch((void**)&srcDev, &srcPitch, width * sizeof(Npp8u), height);
    if (cudaResult != cudaSuccess) {
        std::cout << "Failed to allocate source device memory" << std::endl;
        return false;
    }
    
    cudaResult = cudaMallocPitch((void**)&dstDev, &dstPitch, width * sizeof(Npp8u), height);
    if (cudaResult != cudaSuccess) {
        std::cout << "Failed to allocate destination device memory" << std::endl;
        cudaFree(srcDev);
        return false;
    }
    
    cudaResult = cudaMemcpy2D(srcDev, srcPitch, srcHost.data(), width * sizeof(Npp8u),
                             width * sizeof(Npp8u), height, cudaMemcpyHostToDevice);
    if (cudaResult != cudaSuccess) {
        std::cout << "Failed to copy to device" << std::endl;
        cudaFree(srcDev);
        cudaFree(dstDev);
        return false;
    }
    
    NppiSize roiSize = {width, height};
    NppStatus status = nppiSubC_8u_C1RSfs(srcDev, (int)srcPitch, 25, dstDev, (int)dstPitch, roiSize, 1);
    
    bool testPassed = false;
    if (status == NPP_NO_ERROR) {
        cudaResult = cudaMemcpy2D(dstHost.data(), width * sizeof(Npp8u), dstDev, dstPitch,
                                 width * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
        
        if (cudaResult == cudaSuccess) {
            // Expected: (100 - 25) >> 1 = 37
            Npp8u expected = (100 - 25) >> 1;
            testPassed = true;
            for (int i = 0; i < dataSize && testPassed; ++i) {
                if (dstHost[i] != expected) {
                    std::cout << "Mismatch at index " << i << ": got " << (int)dstHost[i] 
                              << ", expected " << (int)expected << std::endl;
                    testPassed = false;
                }
            }
            
            if (testPassed) {
                std::cout << "nppiSubC_8u_C1RSfs: PASS" << std::endl;
            }
        }
    } else {
        std::cout << "nppiSubC_8u_C1RSfs failed with status: " << status << std::endl;
    }
    
    cudaFree(srcDev);
    cudaFree(dstDev);
    
    return testPassed;
}

bool testMulC_8u_simple() {
    std::cout << "Testing nppiMulC_8u_C1RSfs..." << std::endl;
    
    const int width = 32;
    const int height = 32;
    const int dataSize = width * height;
    
    std::vector<Npp8u> srcHost(dataSize, 20);
    std::vector<Npp8u> dstHost(dataSize, 0);
    
    Npp8u *srcDev = nullptr, *dstDev = nullptr;
    size_t srcPitch, dstPitch;
    
    cudaError_t cudaResult = cudaMallocPitch((void**)&srcDev, &srcPitch, width * sizeof(Npp8u), height);
    if (cudaResult != cudaSuccess) return false;
    
    cudaResult = cudaMallocPitch((void**)&dstDev, &dstPitch, width * sizeof(Npp8u), height);
    if (cudaResult != cudaSuccess) {
        cudaFree(srcDev);
        return false;
    }
    
    cudaResult = cudaMemcpy2D(srcDev, srcPitch, srcHost.data(), width * sizeof(Npp8u),
                             width * sizeof(Npp8u), height, cudaMemcpyHostToDevice);
    if (cudaResult != cudaSuccess) {
        cudaFree(srcDev);
        cudaFree(dstDev);
        return false;
    }
    
    NppiSize roiSize = {width, height};
    NppStatus status = nppiMulC_8u_C1RSfs(srcDev, (int)srcPitch, 3, dstDev, (int)dstPitch, roiSize, 2);
    
    bool testPassed = false;
    if (status == NPP_NO_ERROR) {
        cudaResult = cudaMemcpy2D(dstHost.data(), width * sizeof(Npp8u), dstDev, dstPitch,
                                 width * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
        
        if (cudaResult == cudaSuccess) {
            // Expected: (20 * 3) >> 2 = 15
            Npp8u expected = (20 * 3) >> 2;
            testPassed = true;
            for (int i = 0; i < dataSize && testPassed; ++i) {
                if (dstHost[i] != expected) {
                    std::cout << "Mismatch at index " << i << ": got " << (int)dstHost[i] 
                              << ", expected " << (int)expected << std::endl;
                    testPassed = false;
                }
            }
            
            if (testPassed) {
                std::cout << "nppiMulC_8u_C1RSfs: PASS" << std::endl;
            }
        }
    } else {
        std::cout << "nppiMulC_8u_C1RSfs failed with status: " << status << std::endl;
    }
    
    cudaFree(srcDev);
    cudaFree(dstDev);
    
    return testPassed;
}

bool testDivC_8u_simple() {
    std::cout << "Testing nppiDivC_8u_C1RSfs..." << std::endl;
    
    const int width = 32;
    const int height = 32;
    const int dataSize = width * height;
    
    std::vector<Npp8u> srcHost(dataSize, 100);
    std::vector<Npp8u> dstHost(dataSize, 0);
    
    Npp8u *srcDev = nullptr, *dstDev = nullptr;
    size_t srcPitch, dstPitch;
    
    cudaError_t cudaResult = cudaMallocPitch((void**)&srcDev, &srcPitch, width * sizeof(Npp8u), height);
    if (cudaResult != cudaSuccess) return false;
    
    cudaResult = cudaMallocPitch((void**)&dstDev, &dstPitch, width * sizeof(Npp8u), height);
    if (cudaResult != cudaSuccess) {
        cudaFree(srcDev);
        return false;
    }
    
    cudaResult = cudaMemcpy2D(srcDev, srcPitch, srcHost.data(), width * sizeof(Npp8u),
                             width * sizeof(Npp8u), height, cudaMemcpyHostToDevice);
    if (cudaResult != cudaSuccess) {
        cudaFree(srcDev);
        cudaFree(dstDev);
        return false;
    }
    
    NppiSize roiSize = {width, height};
    NppStatus status = nppiDivC_8u_C1RSfs(srcDev, (int)srcPitch, 4, dstDev, (int)dstPitch, roiSize, 2);
    
    bool testPassed = false;
    if (status == NPP_NO_ERROR) {
        cudaResult = cudaMemcpy2D(dstHost.data(), width * sizeof(Npp8u), dstDev, dstPitch,
                                 width * sizeof(Npp8u), height, cudaMemcpyDeviceToHost);
        
        if (cudaResult == cudaSuccess) {
            // Expected: (100 / 4) << 2 = 100
            Npp8u expected = (100 / 4) << 2;
            testPassed = true;
            for (int i = 0; i < dataSize && testPassed; ++i) {
                if (dstHost[i] != expected) {
                    std::cout << "Mismatch at index " << i << ": got " << (int)dstHost[i] 
                              << ", expected " << (int)expected << std::endl;
                    testPassed = false;
                }
            }
            
            if (testPassed) {
                std::cout << "nppiDivC_8u_C1RSfs: PASS" << std::endl;
            }
        }
    } else {
        std::cout << "nppiDivC_8u_C1RSfs failed with status: " << status << std::endl;
    }
    
    cudaFree(srcDev);
    cudaFree(dstDev);
    
    return testPassed;
}

bool testAddC_32f_simple() {
    std::cout << "Testing nppiAddC_32f_C1R..." << std::endl;
    
    const int width = 32;
    const int height = 32;
    const int dataSize = width * height;
    
    std::vector<Npp32f> srcHost(dataSize, 50.0f);
    std::vector<Npp32f> dstHost(dataSize, 0.0f);
    
    Npp32f *srcDev = nullptr, *dstDev = nullptr;
    size_t srcPitch, dstPitch;
    
    cudaError_t cudaResult = cudaMallocPitch((void**)&srcDev, &srcPitch, width * sizeof(Npp32f), height);
    if (cudaResult != cudaSuccess) return false;
    
    cudaResult = cudaMallocPitch((void**)&dstDev, &dstPitch, width * sizeof(Npp32f), height);
    if (cudaResult != cudaSuccess) {
        cudaFree(srcDev);
        return false;
    }
    
    cudaResult = cudaMemcpy2D(srcDev, srcPitch, srcHost.data(), width * sizeof(Npp32f),
                             width * sizeof(Npp32f), height, cudaMemcpyHostToDevice);
    if (cudaResult != cudaSuccess) {
        cudaFree(srcDev);
        cudaFree(dstDev);
        return false;
    }
    
    NppiSize roiSize = {width, height};
    NppStatus status = nppiAddC_32f_C1R(srcDev, (int)srcPitch, 15.5f, dstDev, (int)dstPitch, roiSize);
    
    bool testPassed = false;
    if (status == NPP_NO_ERROR) {
        cudaResult = cudaMemcpy2D(dstHost.data(), width * sizeof(Npp32f), dstDev, dstPitch,
                                 width * sizeof(Npp32f), height, cudaMemcpyDeviceToHost);
        
        if (cudaResult == cudaSuccess) {
            // Expected: 50.0 + 15.5 = 65.5
            Npp32f expected = 50.0f + 15.5f;
            testPassed = true;
            for (int i = 0; i < dataSize && testPassed; ++i) {
                if (std::abs(dstHost[i] - expected) > 1e-6f) {
                    std::cout << "Mismatch at index " << i << ": got " << dstHost[i] 
                              << ", expected " << expected << std::endl;
                    testPassed = false;
                }
            }
            
            if (testPassed) {
                std::cout << "nppiAddC_32f_C1R: PASS" << std::endl;
            }
        }
    } else {
        std::cout << "nppiAddC_32f_C1R failed with status: " << status << std::endl;
    }
    
    cudaFree(srcDev);
    cudaFree(dstDev);
    
    return testPassed;
}

int main() {
    std::cout << "=== Simple NPP Validation Test ===" << std::endl;
    
    // Initialize CUDA
    cudaError_t cudaResult = cudaSetDevice(0);
    if (cudaResult != cudaSuccess) {
        std::cout << "Failed to set CUDA device" << std::endl;
        return 1;
    }
    
    bool allPassed = true;
    
    // Test all arithmetic operations
    allPassed &= testAddC_8u_simple();
    allPassed &= testAddC_16u_simple();
    allPassed &= testAddC_16s_simple();
    allPassed &= testSubC_8u_simple();
    allPassed &= testMulC_8u_simple();
    allPassed &= testDivC_8u_simple();
    allPassed &= testAddC_32f_simple();
    
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Overall result: " << (allPassed ? "PASS" : "FAIL") << std::endl;
    
    return allPassed ? 0 : 1;
}