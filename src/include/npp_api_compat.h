#ifndef NPP_API_COMPAT_H
#define NPP_API_COMPAT_H

#include <npp.h>
#include <cuda_runtime.h>

/*
 * Compatibility layer for compiling the same sources against the NPP header
 * snapshots selected by switch_cuda_version.sh (API-11.4 ... API-13.4) and
 * against the unified contract.
 *
 * NPP_HEADER_MAJOR / NPP_HAS_13_4 are provided by the build system.
 */

#ifndef NPP_HEADER_MAJOR
#error "NPP_HEADER_MAJOR is not defined; configure through the project CMake build"
#endif

/* NPP 12.8 switched the buffer-size getters from int* to size_t*.  The
 * unified contract keeps the legacy int* C entry points and exposes size_t*
 * as C++ overloads, so it uses the int form here. */
/* Signal length type: NPP 12.8+ uses size_t, earlier versions use int. */
#if NPP_VERSION < 12300
using NppSignalLength = int;
#else
using NppSignalLength = size_t;
#endif

#if !NPP_HEADER_UNIFIED && (NPP_HEADER_MAJOR > 12 || (NPP_HEADER_MAJOR == 12 && NPP_HEADER_MINOR >= 3))
#define NPP_BUFFER_SIZE_IS_SIZE_T 1
#define NPP_BUFFER_SIZE_TYPE size_t
#ifdef __cplusplus
#define NPP_BUFFER_ABI_EXTERN_C extern "C"
#endif
#else
#define NPP_BUFFER_SIZE_IS_SIZE_T 0
#define NPP_BUFFER_SIZE_TYPE int
#ifdef __cplusplus
#define NPP_BUFFER_ABI_EXTERN_C
#endif
#endif

#if !NPP_HAS_13_4
/* NPP 13.4 introduced the color-space selector used by the NV12/UYVY/P010/
 * UYVP entry points.  Backfill it for older header sets so the shared
 * implementation still compiles. */
typedef enum {
  NPP_TRANSCODE_ONLY = 0,
  NPP_BT601JPEG = 1,
  NPP_BT709HDTV = 2,
  NPP_BT2020HDTV = 3
} NppiColorSpace;
#endif

#if !NPP_HEADER_UNIFIED && NPP_HEADER_MAJOR < 12
/* Types introduced with the NPP 12 headers, backfilled for 11.4. */
typedef enum {
  NPP_8U,
  NPP_8S,
  NPP_16U,
  NPP_16S,
  NPP_32U,
  NPP_32S,
  NPP_64U,
  NPP_64S,
  NPP_16F,
  NPP_32F,
  NPP_64F
} NppDataType;

typedef enum {
  NPP_CH_1,
  NPP_CH_2,
  NPP_CH_3,
  NPP_CH_4,
  NPP_CH_A4,
  NPP_CH_P2,
  NPP_CH_P3,
  NPP_CH_P4
} NppiChannels;

typedef struct {
  Npp32f x;
  Npp32f y;
} NppiPoint32f;

typedef struct {
  Npp64f x;
  Npp64f y;
} NppiPoint64f;

typedef struct {
  Npp32u nMarkerLabelID;
  Npp32u nContourPixelCount;
  Npp32u nContourStartingPixelOffset;
  Npp32u nSegmentNum;
} NppiContourBlockSegment;
#endif

#if !NPP_HEADER_UNIFIED && (NPP_HEADER_MAJOR < 12 || (NPP_HEADER_MAJOR == 12 && NPP_HEADER_MINOR < 3))
typedef struct {
  int nPixels;
  Npp32f nMeanIntensity;
  Npp32f nStdDevIntensity;
} NppiProfileData;
#endif

#if NPP_HEADER_MAJOR >= 13
/* NPP 13 headers are context-only, but libnppc still exports the managed
 * stream entry points used by the classic wrappers. */
#ifdef __cplusplus
extern "C" {
#endif
cudaStream_t nppGetStream(void);
NppStatus nppSetStream(cudaStream_t hStream);
NppStatus nppGetStreamContext(NppStreamContext *pNppStreamContext);
unsigned int nppGetStreamNumSMs(void);
unsigned int nppGetStreamMaxThreadsPerSM(void);
#ifdef __cplusplus
}
#endif
#endif

#endif  // NPP_API_COMPAT_H
