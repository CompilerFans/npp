#ifndef NPP_VERSION_COMPAT_H
#define NPP_VERSION_COMPAT_H

#include "npp.h"

/**
 * NPP Version to CUDA SDK Version Mapping
 *
 * This header provides mapping between NPP version and CUDA SDK version
 * for handling API differences across different versions.
 */

#ifndef NPP_VERSION
#error "NPP_VERSION not defined - ensure npp.h is included"
#endif

// NPP Version to CUDA SDK Version Mapping Table
#if NPP_VERSION < 12000
// NPP 11.x -> CUDA SDK 11.4
#define CUDA_SDK_VERSION_MAJOR 11
#define CUDA_SDK_VERSION_MINOR 4

#elif NPP_VERSION < 12300
// NPP 12.0-12.2 -> CUDA SDK 12.2
#define CUDA_SDK_VERSION_MAJOR 12
#define CUDA_SDK_VERSION_MINOR 2

#else
// NPP 12.3+ -> CUDA SDK 12.8
#define CUDA_SDK_VERSION_MAJOR 12
#define CUDA_SDK_VERSION_MINOR 8
#endif

// Version comparison macros
#define CUDA_SDK_AT_LEAST(major, minor)                                                                                \
  ((CUDA_SDK_VERSION_MAJOR > (major)) || (CUDA_SDK_VERSION_MAJOR == (major) && CUDA_SDK_VERSION_MINOR >= (minor)))

#endif // NPP_VERSION_COMPAT_H