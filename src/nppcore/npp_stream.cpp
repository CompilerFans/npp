#include "npp.h"

#include <cuda_runtime.h>
#include <mutex>

// NPP managed stream state (see nppSetStream()/nppGetStream()).
//
// Every non-Ctx entry point and nppGetStreamContext() must submit/report work
// on this stream.  If nppSetStream() is a no-op, MPP kernels stay on the
// legacy stream and an application that synchronizes only its own stream (as
// the ZTE demo does before running ORT inference) consumes the previous
// frame's buffer - the one-frame-late detection result reported on site.
//
// The stream and the device it belongs to are kept in one state object so a
// concurrent reader can never observe a stream paired with another device's
// properties.  A map is not used on purpose: the NPP managed stream is a
// single process-wide state, not a per-device one.
namespace {
struct ManagedStreamState {
  cudaStream_t stream = nullptr;
  int deviceId = -1;
};

std::mutex g_managedStreamMutex;
ManagedStreamState g_managedStreamState;

ManagedStreamState getManagedStreamState() {
  std::lock_guard<std::mutex> lock(g_managedStreamMutex);
  return g_managedStreamState;
}

void setManagedStreamState(cudaStream_t stream, int deviceId) {
  std::lock_guard<std::mutex> lock(g_managedStreamMutex);
  g_managedStreamState.stream = stream;
  g_managedStreamState.deviceId = deviceId;
}
} // namespace

// Device whose properties describe the managed stream: the stream's own device
// when one is set, the current device otherwise.
static int nppResolveManagedStreamDevice(cudaStream_t managedStream, int recordedDeviceId, int *deviceId) {
  int device = recordedDeviceId;
  if (managedStream == nullptr || managedStream == cudaStreamPerThread) {
    device = -1;
  }
  if (device < 0 && cudaGetDevice(&device) != cudaSuccess) {
    return -1;
  }
  *deviceId = device;
  return 0;
}

static int nppGetManagedStreamDevice(int *deviceId) {
  ManagedStreamState state = getManagedStreamState();
  return nppResolveManagedStreamDevice(state.stream, state.deviceId, deviceId);
}

cudaStream_t nppGetStream(void) {
  // Return the stream most recently set by nppSetStream() (NULL by default).
  return getManagedStreamState().stream;
}

NppStatus nppSetStream(cudaStream_t hStream) {
  // cudaStreamLegacy and the NULL stream denote the same stream.  Normalize it
  // so callers that check for stream 0 keep taking the device-synchronize path.
  if (hStream == cudaStreamLegacy) {
    hStream = nullptr;
  }

  int deviceId;
  cudaError_t result = cudaGetDevice(&deviceId);
  if (result != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  // Do not probe the handle here: on some runtimes (e.g. CUDA 13.x)
  // cudaStreamGetFlags() faults on a bogus pointer instead of returning
  // cudaErrorInvalidResourceHandle.  The runtime reports an invalid stream
  // when work is actually submitted, which fails later but never crashes.
  setManagedStreamState(hStream, deviceId);
  return NPP_NO_ERROR;
}

NppStatus nppGetStreamContext(NppStreamContext *pNppStreamContext) {
  if (!pNppStreamContext) {
    return NPP_NULL_POINTER_ERROR;
  }

  // Leave a safe default (NULL stream) even if a later query fails and the
  // caller ignores the returned status.
  *pNppStreamContext = {};

  ManagedStreamState managedState = getManagedStreamState();

  // Use the device associated with the managed stream, falling back to the
  // current device when no stream has been set.
  int deviceId;
  if (nppResolveManagedStreamDevice(managedState.stream, managedState.deviceId, &deviceId) != 0) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  // Get device properties
  cudaDeviceProp prop;
  cudaError_t result = cudaGetDeviceProperties(&prop, deviceId);
  if (result != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  // Fill stream context with the NPP managed stream set by nppSetStream().
  // This is the stream that non-Ctx NPP calls use and that the application
  // must synchronize before consuming their output.
  cudaStream_t managedStream = managedState.stream;
  pNppStreamContext->hStream = managedStream;
  pNppStreamContext->nCudaDeviceId = deviceId;
  pNppStreamContext->nMultiProcessorCount = prop.multiProcessorCount;
  pNppStreamContext->nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  pNppStreamContext->nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
  pNppStreamContext->nSharedMemPerBlock = prop.sharedMemPerBlock;

  // Get compute capability
  pNppStreamContext->nCudaDevAttrComputeCapabilityMajor = prop.major;
  pNppStreamContext->nCudaDevAttrComputeCapabilityMinor = prop.minor;

  // Do not call cudaStreamGetFlags() on the stored handle for the same reason
  // as nppSetStream(): it can fault on a bogus pointer.  The default flags are
  // sufficient for the managed-stream use cases.
  pNppStreamContext->nStreamFlags = cudaStreamDefault;

  // Reserved field
  pNppStreamContext->nReserved0 = 0;

  return NPP_NO_ERROR;
}

unsigned int nppGetStreamNumSMs(void) {
  int deviceId;
  if (nppGetManagedStreamDevice(&deviceId) != 0) {
    return 0;
  }

  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess) {
    return 0;
  }

  return static_cast<unsigned int>(prop.multiProcessorCount);
}

unsigned int nppGetStreamMaxThreadsPerSM(void) {
  int deviceId;
  if (nppGetManagedStreamDevice(&deviceId) != 0) {
    return 0;
  }

  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess) {
    return 0;
  }

  return static_cast<unsigned int>(prop.maxThreadsPerMultiProcessor);
}
