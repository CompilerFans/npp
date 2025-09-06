#include "npp.h"
#include <cuda_runtime.h>
#include <cstring>

/**
 * NPP Core Functions Implementation - Simplified Version
 * 
 * This implementation directly uses CUDA Runtime API without internal stream management.
 * Stream and context are managed by the user through CUDA Runtime API.
 */

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
    int deviceId;
    if (cudaGetDevice(&deviceId) != cudaSuccess) {
        return -1;
    }
    
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess) {
        return -1;
    }
    return prop.multiProcessorCount;
}

/**
 * Get the maximum number of threads per block on the active CUDA device.
 */
int nppGetMaxThreadsPerBlock(void)
{
    int deviceId;
    if (cudaGetDevice(&deviceId) != cudaSuccess) {
        return -1;
    }
    
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess) {
        return -1;
    }
    return prop.maxThreadsPerBlock;
}

/**
 * Get the maximum number of threads per SM for the active GPU
 */
int nppGetMaxThreadsPerSM(void)
{
    int deviceId;
    if (cudaGetDevice(&deviceId) != cudaSuccess) {
        return -1;
    }
    
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess) {
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
    
    int deviceId;
    if (cudaGetDevice(&deviceId) != cudaSuccess) {
        return -1;
    }
    
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess) {
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
    static int lastDevice = -1;
    
    int currentDevice;
    if (cudaGetDevice(&currentDevice) != cudaSuccess) {
        strcpy(deviceName, "Unknown Device");
        return deviceName;
    }
    
    // Update if device changed or not initialized
    if (!initialized || lastDevice != currentDevice) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, currentDevice) == cudaSuccess) {
            strncpy(deviceName, prop.name, sizeof(deviceName) - 1);
            deviceName[sizeof(deviceName) - 1] = '\0';
        } else {
            strcpy(deviceName, "Unknown Device");
        }
        lastDevice = currentDevice;
        initialized = true;
    }
    
    return deviceName;
}

/**
 * Get the NPP CUDA stream.
 * Since we don't manage streams internally, this returns the current CUDA stream.
 */
cudaStream_t nppGetStream(void)
{
    // Return the default stream (0)
    // In a real application, the user should manage their own streams
    return cudaStreamLegacy;  // Use legacy default stream for compatibility
}

/**
 * Set the NPP CUDA stream.
 * Since we don't manage streams internally, this is a no-op for compatibility.
 */
NppStatus nppSetStream(cudaStream_t hStream)
{
    // This function exists for API compatibility
    // Users should use CUDA Runtime API directly to manage streams
    
    // Special handling for known stream values
    if (hStream == nullptr || hStream == cudaStreamLegacy || 
        hStream == cudaStreamPerThread) {
        return NPP_NO_ERROR;
    }
    
    // For other streams, we can't safely validate without potential segfault
    // Just accept them - the actual CUDA calls will fail if invalid
    return NPP_NO_ERROR;
}

/**
 * Get the current NPP managed CUDA stream context.
 * Fills the context based on current CUDA device and stream.
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
    
    // Fill stream context
    // Use the default stream (user should manage their own streams)
    pNppStreamContext->hStream = cudaStreamLegacy;
    pNppStreamContext->nCudaDeviceId = deviceId;
    pNppStreamContext->nMultiProcessorCount = prop.multiProcessorCount;
    pNppStreamContext->nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    pNppStreamContext->nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    pNppStreamContext->nSharedMemPerBlock = prop.sharedMemPerBlock;
    
    // Get compute capability
    pNppStreamContext->nCudaDevAttrComputeCapabilityMajor = prop.major;
    pNppStreamContext->nCudaDevAttrComputeCapabilityMinor = prop.minor;
    
    // Get stream flags for default stream
    unsigned int flags;
    result = cudaStreamGetFlags(cudaStreamLegacy, &flags);
    if (result == cudaSuccess) {
        pNppStreamContext->nStreamFlags = flags;
    } else {
        pNppStreamContext->nStreamFlags = 0;
    }
    
    // Reserved field
    pNppStreamContext->nReserved0 = 0;
    
    return NPP_NO_ERROR;
}

// nppSetStreamContext - Not part of original NPP API
// Removed to maintain compatibility

/**
 * Get the number of SMs on the device associated with the current NPP CUDA stream.
 */
unsigned int nppGetStreamNumSMs(void)
{
    int deviceId;
    if (cudaGetDevice(&deviceId) != cudaSuccess) {
        return 0;
    }
    
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess) {
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
    if (cudaGetDevice(&deviceId) != cudaSuccess) {
        return 0;
    }
    
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess) {
        return 0;
    }
    
    return static_cast<unsigned int>(prop.maxThreadsPerMultiProcessor);
}

// nppGetGpuComputeCapability - Not part of original NPP API
// Removed to maintain compatibility

// nppGetStatusString - Not part of original NPP API
// This is an internal helper for testing
#ifdef NPP_DEBUG_BUILD
static const char * nppGetStatusStringInternal(NppStatus eStatusCode)
#else
__attribute__((unused))
static const char * nppGetStatusStringInternal(NppStatus eStatusCode)
#endif
{
    switch(eStatusCode) {
        case NPP_NOT_SUPPORTED_MODE_ERROR:
            return "Unsupported mode error";
        case NPP_INVALID_HOST_POINTER_ERROR:
            return "Invalid host pointer error";
        case NPP_INVALID_DEVICE_POINTER_ERROR:
            return "Invalid device pointer error";
        case NPP_LUT_PALETTE_BITSIZE_ERROR:
            return "LUT palette bitsize error";
        case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:
            return "Zero crossing mode not supported error";
        case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
            return "Insufficient compute capability";
        case NPP_TEXTURE_BIND_ERROR:
            return "Texture bind error";
        case NPP_WRONG_INTERSECTION_ROI_ERROR:
            return "Wrong intersection ROI error";
        case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
            return "Haar classifier pixel match error";
        case NPP_MEMFREE_ERROR:
            return "Memory free error";
        case NPP_MEMSET_ERROR:
            return "Memory set error";
        case NPP_MEMCPY_ERROR:
            return "Memory copy error";
        case NPP_ALIGNMENT_ERROR:
            return "Alignment error";
        case NPP_CUDA_KERNEL_EXECUTION_ERROR:
            return "CUDA kernel execution error";
        case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
            return "Round mode not supported error";
        case NPP_QUALITY_INDEX_ERROR:
            return "Quality index error";
        case NPP_RESIZE_NO_OPERATION_ERROR:
            return "Resize no operation error";
        case NPP_OVERFLOW_ERROR:
            return "Overflow error";
        case NPP_NOT_EVEN_STEP_ERROR:
            return "Not even step error";
        case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
            return "Histogram number of levels error";
        case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
            return "LUT number of levels error";
        case NPP_CORRUPTED_DATA_ERROR:
            return "Corrupted data error";
        case NPP_CHANNEL_ORDER_ERROR:
            return "Channel order error";
        case NPP_ZERO_MASK_VALUE_ERROR:
            return "Zero mask value error";
        case NPP_QUADRANGLE_ERROR:
            return "Quadrangle error";
        case NPP_RECTANGLE_ERROR:
            return "Rectangle error";
        case NPP_COEFFICIENT_ERROR:
            return "Coefficient error";
        case NPP_NUMBER_OF_CHANNELS_ERROR:
            return "Number of channels error";
        case NPP_COI_ERROR:
            return "Channel of interest error";
        case NPP_DIVISOR_ERROR:
            return "Divisor error";
        case NPP_CHANNEL_ERROR:
            return "Channel error";
        case NPP_STRIDE_ERROR:
            return "Stride error";
        case NPP_ANCHOR_ERROR:
            return "Anchor error";
        case NPP_MASK_SIZE_ERROR:
            return "Mask size error";
        case NPP_RESIZE_FACTOR_ERROR:
            return "Resize factor error";
        case NPP_INTERPOLATION_ERROR:
            return "Interpolation error";
        case NPP_MIRROR_FLIP_ERROR:
            return "Mirror flip error";
        case NPP_MOMENT_00_ZERO_ERROR:
            return "Moment 00 zero error";
        case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR:
            return "Threshold negative level error";
        case NPP_THRESHOLD_ERROR:
            return "Threshold error";
        case NPP_CONTEXT_MATCH_ERROR:
            return "Context match error";
        case NPP_FFT_FLAG_ERROR:
            return "FFT flag error";
        case NPP_FFT_ORDER_ERROR:
            return "FFT order error";
        case NPP_STEP_ERROR:
            return "Step error";
        case NPP_SCALE_RANGE_ERROR:
            return "Scale range error";
        case NPP_DATA_TYPE_ERROR:
            return "Data type error";
        case NPP_OUT_OFF_RANGE_ERROR:
            return "Out of range error";
        case NPP_DIVIDE_BY_ZERO_ERROR:
            return "Divide by zero error";
        case NPP_MEMORY_ALLOCATION_ERR:
            return "Memory allocation error";
        case NPP_NULL_POINTER_ERROR:
            return "Null pointer error";
        case NPP_RANGE_ERROR:
            return "Range error";
        case NPP_SIZE_ERROR:
            return "Size error";
        case NPP_BAD_ARGUMENT_ERROR:
            return "Bad argument error";
        case NPP_NO_MEMORY_ERROR:
            return "No memory error";
        case NPP_NOT_IMPLEMENTED_ERROR:
            return "Not implemented error";
        case NPP_ERROR:
            return "General error";
        case NPP_ERROR_RESERVED:
            return "Reserved error";
        case NPP_NO_ERROR:  // NPP_SUCCESS is the same as NPP_NO_ERROR
            return "Success";
        case NPP_NO_OPERATION_WARNING:
            return "No operation warning";
        case NPP_DIVIDE_BY_ZERO_WARNING:
            return "Divide by zero warning";
        case NPP_AFFINE_QUAD_INCORRECT_WARNING:
            return "Affine quad incorrect warning";
        case NPP_WRONG_INTERSECTION_ROI_WARNING:
            return "Wrong intersection ROI warning";
        case NPP_WRONG_INTERSECTION_QUAD_WARNING:
            return "Wrong intersection quad warning";
        case NPP_DOUBLE_SIZE_WARNING:
            return "Double size warning";
        case NPP_MISALIGNED_DST_ROI_WARNING:
            return "Misaligned destination ROI warning";
        default:
            return "Unknown error code";
    }
}