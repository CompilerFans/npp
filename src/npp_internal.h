#ifndef NPP_INTERNAL_H
#define NPP_INTERNAL_H

#include "npp.h"
#include <cstring>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline NppStreamContext nppCreateDefaultStreamContext(void) {
  NppStreamContext ctx{};
  // Reuse the managed-stream context so the stream and the device whose
  // properties are reported always belong to the same device.
  if (nppGetStreamContext(&ctx) != NPP_NO_ERROR) {
    ctx.hStream = nppGetStream();
  }
  return ctx;
}

#ifdef __cplusplus
}
#endif

#endif // NPP_INTERNAL_H