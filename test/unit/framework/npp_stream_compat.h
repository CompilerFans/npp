#pragma once

#include <cuda_runtime.h>
#include <npp.h>

// NPP 13 removed the managed-stream API from the public headers, but libnppc
// still exports the symbols.  Declare them so the shared test suite can
// exercise the reference implementation as well.
// Buffer-size getters use int* before NPP 12.8 and size_t* afterwards.  The
// in-tree compatibility header defines this for MPP builds; provide the same
// alias when the suite is built against the NVIDIA SDK.
#ifndef NPP_BUFFER_SIZE_TYPE
#if NPP_VERSION < 12300
#define NPP_BUFFER_SIZE_TYPE int
#else
#define NPP_BUFFER_SIZE_TYPE size_t
#endif
#endif

#if defined(USE_NVIDIA_NPP_TESTS) && defined(NPP_VERSION_MAJOR) && (NPP_VERSION_MAJOR >= 13)
extern "C" {
cudaStream_t nppGetStream(void);
NppStatus nppSetStream(cudaStream_t hStream);
NppStatus nppGetStreamContext(NppStreamContext *pNppStreamContext);
unsigned int nppGetStreamNumSMs(void);
unsigned int nppGetStreamMaxThreadsPerSM(void);
}
#endif
