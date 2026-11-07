// NPP 13.4 P010 / P016 to RGB conversions.
//
// Both formats are read as 16-bit samples normalised into NPP's 10-bit
// conversion domain: x10 = x * (1023/65535).  P010 stores 10-bit data MSB
// aligned in the 16-bit container, P016 stores full 16-bit data; both are
// normalised by the same factor.  The decode matrix runs with chroma centred
// on 512 and the result is scaled to the destination type (x0.25 for 8-bit
// output, x65535/1023 for 16-bit output) before the +0.5f rounding.
#include "npp.h"
#include <cstdint>
#include <cuda_runtime.h>

namespace {

struct CsCoefs {
  float rv, gu, gv, bu;
};

__device__ __host__ inline CsCoefs cs_coefs(int cs) {
  if (cs == 1)
    return {1.1398299932479858398f, -0.39465001225471496582f, -0.58060002326965332031f,
            2.0321099758148193359f};
  if (cs == 2)
    return {1.5748000144958496094f, -0.18732400238513946533f, -0.46812400221824645996f,
            1.8555999994277954102f};
  return {1.4745999574661254883f, -0.16455310583114624023f, -0.57135307788848876953f,
          1.881399989128112793f};
}

template <typename T>
__device__ __forceinline__ float finish(float v) {
  if (v != v) return 0.0f;
  float t = __fadd_rn(v, 0.5f);
  const float mx = sizeof(T) == 1 ? 255.0f : 65535.0f;
  t = t < 0.0f ? 0.0f : (t > mx ? mx : t);
  return (float)(unsigned int)t;
}

template <bool C4, typename T>
__global__ void p010_to_rgb_kernel(const uint16_t *__restrict__ srcY, int nSrcStep,
                                   const uint16_t *__restrict__ srcUV, T *__restrict__ dst, int nDstStep,
                                   int width, int height, CsCoefs cf, T alpha) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  const int syStep = nSrcStep / 2;
  const float S = 0.015609979629516601562f;
  const float O = sizeof(T) == 1 ? 0.25f : 64.06158447265625f;
  const float yv = __fmul_rn((float)srcY[(size_t)y * syStep + x], S);
  const uint16_t *uv = srcUV + (size_t)(y >> 1) * syStep + (size_t)((x >> 1) << 1);
  const float du = __fmaf_rn((float)uv[0], S, -512.0f);
  const float dv = __fmaf_rn((float)uv[1], S, -512.0f);
  const float fr = __fmul_rn(__fmaf_rn(dv, cf.rv, yv), O);
  const float fg = __fmul_rn(__fmaf_rn(du, cf.gu, __fmaf_rn(dv, cf.gv, yv)), O);
  const float fb = __fmul_rn(__fmaf_rn(du, cf.bu, yv), O);
  const T R = (T)(unsigned int)finish<T>(fr);
  const T G = (T)(unsigned int)finish<T>(fg);
  const T B = (T)(unsigned int)finish<T>(fb);
  const int C = C4 ? 4 : 3;
  const size_t dstep = (size_t)(nDstStep / (int)sizeof(T));
  T *o = dst + (size_t)y * dstep + (size_t)x * C;
  o[0] = R;
  o[1] = G;
  o[2] = B;
  if (C4) o[3] = alpha;
}

}  // namespace

extern "C" {

cudaError_t nppiP010ToRgbLaunch(const Npp16u *srcY, int nSrcStep, const Npp16u *srcUV, void *dst, int nDstStep,
                                NppiSize roi, int cs, bool c4, bool wide, Npp16u alpha16, Npp8u alpha8,
                                cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((roi.width + 15) / 16, (roi.height + 15) / 16);
  const CsCoefs cf = cs_coefs(cs);
  if (wide) {
    if (c4)
      p010_to_rgb_kernel<true, Npp16u><<<grid, block, 0, stream>>>(srcY, nSrcStep, srcUV, (Npp16u *)dst, nDstStep,
                                                                   roi.width, roi.height, cf, alpha16);
    else
      p010_to_rgb_kernel<false, Npp16u><<<grid, block, 0, stream>>>(srcY, nSrcStep, srcUV, (Npp16u *)dst, nDstStep,
                                                                    roi.width, roi.height, cf, alpha16);
  } else {
    if (c4)
      p010_to_rgb_kernel<true, Npp8u><<<grid, block, 0, stream>>>(srcY, nSrcStep, srcUV, (Npp8u *)dst, nDstStep,
                                                                  roi.width, roi.height, cf, alpha8);
    else
      p010_to_rgb_kernel<false, Npp8u><<<grid, block, 0, stream>>>(srcY, nSrcStep, srcUV, (Npp8u *)dst, nDstStep,
                                                                   roi.width, roi.height, cf, alpha8);
  }
  return cudaGetLastError();
}

}  // extern "C"
