#include "npp.h"
#include <cuda_runtime.h>
#include <cstring>

/**
 * NPP Core Functions Implementation
 */

// NppStreamContext is already defined in nppdefs.h

// Global NPP stream management
static cudaStream_t g_nppCurrentStream = 0;
static bool g_nppStreamInitialized = false;

static NppLibraryVersion g_nppLibVersion = {
    NPP_VER_MAJOR,
    NPP_VER_MINOR, 
    NPP_VER_BUILD
};

/**
 * Get the NPP library version.
 */
const NppLibraryVersion * nppGetLibVersion(void)
{
    return &g_nppLibVersion;
}

/**
 * Get the number of Streaming Multiprocessors (SM) on the active CUDA device.
 */
int nppGetGpuNumSMs(void)
{
    cudaDeviceProp prop;
    cudaError_t result = cudaGetDeviceProperties(&prop, 0);
    if (result != cudaSuccess) {
        return -1;
    }
    return prop.multiProcessorCount;
}

/**
 * Get the maximum number of threads per block on the active CUDA device.
 */
int nppGetMaxThreadsPerBlock(void)
{
    cudaDeviceProp prop;
    cudaError_t result = cudaGetDeviceProperties(&prop, 0);
    if (result != cudaSuccess) {
        return -1;
    }
    return prop.maxThreadsPerBlock;
}

/**
 * Get the maximum number of threads per SM for the active GPU
 */
int nppGetMaxThreadsPerSM(void)
{
    cudaDeviceProp prop;
    cudaError_t result = cudaGetDeviceProperties(&prop, 0);
    if (result != cudaSuccess) {
        return -1;
    }
    return prop.maxThreadsPerMultiProcessor;
}

/**
 * Get the maximum number of threads per SM, maximum threads per block, and number of SMs for the active GPU
 */
int nppGetGpuDeviceProperties(int * pMaxThreadsPerSM, int * pMaxThreadsPerBlock, int * pNumberOfSMs)
{
    if (!pMaxThreadsPerSM || !pMaxThreadsPerBlock || !pNumberOfSMs) {
        return -1;
    }
    
    cudaDeviceProp prop;
    cudaError_t result = cudaGetDeviceProperties(&prop, 0);
    if (result != cudaSuccess) {
        return -1;
    }
    
    *pMaxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
    *pMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    *pNumberOfSMs = prop.multiProcessorCount;
    
    return 0; // cudaSuccess
}

/**
 * Get the name of the active CUDA device.
 */
const char * nppGetGpuName(void)
{
    static char deviceName[256] = {0};
    static bool initialized = false;
    
    if (!initialized) {
        cudaDeviceProp prop;
        cudaError_t result = cudaGetDeviceProperties(&prop, 0);
        if (result == cudaSuccess) {
            strncpy(deviceName, prop.name, sizeof(deviceName) - 1);
            deviceName[sizeof(deviceName) - 1] = '\0';
        } else {
            strcpy(deviceName, "Unknown Device");
        }
        initialized = true;
    }
    
    return deviceName;
}

/**
 * Get the NPP CUDA stream.
 */
cudaStream_t nppGetStream(void)
{
    return g_nppCurrentStream;
}

/**
 * Set the NPP CUDA stream.
 */
NppStatus nppSetStream(cudaStream_t hStream)
{
    // Synchronize current stream before changing
    if (g_nppStreamInitialized) {
        cudaError_t result = cudaStreamSynchronize(g_nppCurrentStream);
        if (result != cudaSuccess) {
            return NPP_CUDA_KERNEL_EXECUTION_ERROR;
        }
    }
    
    g_nppCurrentStream = hStream;
    g_nppStreamInitialized = true;
    
    return NPP_NO_ERROR;
}

/**
 * Get the current NPP managed CUDA stream context.
 */
NppStatus nppGetStreamContext(NppStreamContext * pNppStreamContext)
{
    if (!pNppStreamContext) {
        return NPP_NULL_POINTER_ERROR;
    }
    
    // Get current device ID
    int deviceId;
    cudaError_t result = cudaGetDevice(&deviceId);
    if (result != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    // Get device properties
    cudaDeviceProp prop;
    result = cudaGetDeviceProperties(&prop, deviceId);
    if (result != cudaSuccess) {
        return NPP_CUDA_KERNEL_EXECUTION_ERROR;
    }
    
    // Fill stream context according to NppStreamContext structure in nppdefs.h
    pNppStreamContext->hStream = g_nppCurrentStream;
    pNppStreamContext->nCudaDeviceId = deviceId;
    pNppStreamContext->nMultiProcessorCount = prop.multiProcessorCount;
    pNppStreamContext->nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    pNppStreamContext->nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    pNppStreamContext->nSharedMemPerBlock = prop.sharedMemPerBlock;
    
    // Get compute capability
    cudaDeviceGetAttribute(&pNppStreamContext->nCudaDevAttrComputeCapabilityMajor, 
                          cudaDevAttrComputeCapabilityMajor, deviceId);
    cudaDeviceGetAttribute(&pNppStreamContext->nCudaDevAttrComputeCapabilityMinor, 
                          cudaDevAttrComputeCapabilityMinor, deviceId);
    
    // Get stream flags
    cudaStreamGetFlags(g_nppCurrentStream, &pNppStreamContext->nStreamFlags);
    
    // Reserved field
    pNppStreamContext->nReserved0 = 0;
    
    return NPP_NO_ERROR;
}

/**
 * Get the number of SMs on the device associated with the current NPP CUDA stream.
 */
unsigned int nppGetStreamNumSMs(void)
{
    int deviceId;
    cudaError_t result = cudaGetDevice(&deviceId);
    if (result != cudaSuccess) {
        return 0;
    }
    
    cudaDeviceProp prop;
    result = cudaGetDeviceProperties(&prop, deviceId);
    if (result != cudaSuccess) {
        return 0;
    }
    
    return static_cast<unsigned int>(prop.multiProcessorCount);
}

/**
 * Get the maximum number of threads per SM on the device associated with the current NPP CUDA stream.
 */
unsigned int nppGetStreamMaxThreadsPerSM(void)
{
    int deviceId;
    cudaError_t result = cudaGetDevice(&deviceId);
    if (result != cudaSuccess) {
        return 0;
    }
    
    cudaDeviceProp prop;
    result = cudaGetDeviceProperties(&prop, deviceId);
    if (result != cudaSuccess) {
        return 0;
    }
    
    return static_cast<unsigned int>(prop.maxThreadsPerMultiProcessor);
}