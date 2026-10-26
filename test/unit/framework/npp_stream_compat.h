#pragma once

#include <cuda_runtime.h>
#include <npp.h>

// NPP 13 removed the managed-stream API from the public headers, but libnppc
// still exports the symbols.  Declare them so the shared test suite can
// exercise the reference implementation as well.
#if defined(USE_NVIDIA_NPP_TESTS) && defined(NPP_VERSION_MAJOR) && (NPP_VERSION_MAJOR >= 13)
extern "C" {
cudaStream_t nppGetStream(void);
NppStatus nppSetStream(cudaStream_t hStream);
NppStatus nppGetStreamContext(NppStreamContext *pNppStreamContext);
unsigned int nppGetStreamNumSMs(void);
unsigned int nppGetStreamMaxThreadsPerSM(void);
}
#endif
