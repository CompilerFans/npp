#ifndef INTERPOLATOR_COMMON_CUH
#define INTERPOLATOR_COMMON_CUH

#include "npp.h"
#include <cuda_runtime.h>

// Helper function: linear interpolation
__device__ inline float lerp(float a, float b, float t) { return a + t * (b - a); }

// Cubic interpolation weight function (Catmull-Rom)
__device__ inline float cubicWeight(float x) {
  x = fabsf(x);
  if (x <= 1.0f) {
    return 1.5f * x * x * x - 2.5f * x * x + 1.0f;
  } else if (x < 2.0f) {
    return -0.5f * x * x * x + 2.5f * x * x - 4.0f * x + 2.0f;
  }
  return 0.0f;
}

// Lanczos-3 interpolation weight function
__device__ inline float lanczosWeight(float x) {
  x = fabsf(x);
  if (x < 3.0f) {
    if (x < 0.0001f)
      return 1.0f;
    constexpr float pi = 3.14159265358979323846f;
    float pi_x = pi * x;
    float pi_x_3 = pi_x / 3.0f;
    return (sinf(pi_x) / pi_x) * (sinf(pi_x_3) / pi_x_3);
  }
  return 0.0f;
}

#endif // INTERPOLATOR_COMMON_CUH
