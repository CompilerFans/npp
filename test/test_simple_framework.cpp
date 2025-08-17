#include <iostream>
#include <cuda_runtime.h>
#include "npp.h"

int main() {
    printf("Testing simple framework...\n");
    
    // 检查CUDA设备
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        printf("Error: No CUDA devices found!\n");
        return 1;
    }
    
    printf("Found %d CUDA device(s)\n", device_count);
    
    // 设置CUDA设备
    cudaSetDevice(0);
    
    // 打印设备信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using CUDA device: %s\n", prop.name);
    
    // 检查NPP库版本
    const NppLibraryVersion* npp_version = nppGetLibVersion();
    if (npp_version) {
        printf("NPP Library version: %d.%d.%d\n", 
               npp_version->major, npp_version->minor, npp_version->build);
    }
    
    printf("Simple framework test completed successfully!\n");
    return 0;
}