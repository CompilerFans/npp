#include "npp.h"

#include <cmath>
#include <cuda_runtime.h>

namespace {

template <typename Destination> __device__ Destination distanceResult(double distance, int) {
  return static_cast<Destination>(distance);
}

template <> __device__ Npp8u distanceResult<Npp8u>(double distance, int scaleFactor) {
  double scaled;
  if (scaleFactor < -1024) {
    scaled = distance > 0.0 ? 256.0 : 0.0;
  } else if (scaleFactor > 1024) {
    scaled = 0.0;
  } else {
    scaled = ldexp(distance, -scaleFactor);
  }
  scaled = floor(scaled + 0.5);
  if (scaled <= 0.0) {
    return 0;
  }
  if (scaled >= 255.0) {
    return 255;
  }
  return static_cast<Npp8u>(scaled);
}

template <typename Source, typename Destination>
__global__ void sqrDistanceValidNormKernel(const Source *source, int sourceStep, const Source *tpl,
                                           int templateStep, int templateWidth,
                                           int templateHeight, Destination *destination, int destinationStep,
                                           int destinationWidth, int destinationHeight, int channels,
                                           int activeChannels, int scaleFactor) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= destinationWidth || y >= destinationHeight) {
    return;
  }

  Destination *destinationRow =
      reinterpret_cast<Destination *>(reinterpret_cast<char *>(destination) + y * destinationStep);
  for (int channel = 0; channel < activeChannels; ++channel) {
    double squaredDistance = 0.0;
    double sourceEnergy = 0.0;
    double templateEnergy = 0.0;
    for (int templateY = 0; templateY < templateHeight; ++templateY) {
      const Source *sourceRow = reinterpret_cast<const Source *>(
          reinterpret_cast<const char *>(source) + (y + templateY) * sourceStep);
      const Source *templateRow =
          reinterpret_cast<const Source *>(reinterpret_cast<const char *>(tpl) + templateY * templateStep);
      for (int templateX = 0; templateX < templateWidth; ++templateX) {
        const double sourceValue =
            static_cast<double>(sourceRow[(x + templateX) * channels + channel]);
        const double templateValue = static_cast<double>(templateRow[templateX * channels + channel]);
        const double difference = templateValue - sourceValue;
        squaredDistance += difference * difference;
        sourceEnergy += sourceValue * sourceValue;
        templateEnergy += templateValue * templateValue;
      }
    }

    double normalizedDistance = 0.0;
    if (sourceEnergy > 0.0 && templateEnergy > 0.0) {
      normalizedDistance = squaredDistance / (sqrt(sourceEnergy) * sqrt(templateEnergy));
    }
    destinationRow[x * channels + channel] =
        distanceResult<Destination>(normalizedDistance, scaleFactor);
  }
}

template <typename Source, typename Destination>
cudaError_t launchSqrDistanceValidNorm(const Source *source, int sourceStep, NppiSize sourceSize,
                                       const Source *tpl, int templateStep, NppiSize templateSize,
                                       Destination *destination, int destinationStep, int channels,
                                       int activeChannels, int scaleFactor, cudaStream_t stream) {
  const int destinationWidth = sourceSize.width - templateSize.width + 1;
  const int destinationHeight = sourceSize.height - templateSize.height + 1;
  const dim3 block(16, 16);
  const dim3 grid((destinationWidth + block.x - 1) / block.x,
                  (destinationHeight + block.y - 1) / block.y);
  sqrDistanceValidNormKernel<<<grid, block, 0, stream>>>(
      source, sourceStep, tpl, templateStep, templateSize.width, templateSize.height, destination,
      destinationStep, destinationWidth, destinationHeight, channels, activeChannels, scaleFactor);
  return cudaGetLastError();
}

} // namespace

extern "C" {

cudaError_t mppSqrDistanceValidNorm8u8uLaunch(const Npp8u *source, int sourceStep, NppiSize sourceSize,
                                               const Npp8u *tpl, int templateStep, NppiSize templateSize,
                                               Npp8u *destination, int destinationStep, int channels,
                                               int activeChannels, int scaleFactor, cudaStream_t stream) {
  return launchSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                    destination, destinationStep, channels, activeChannels, scaleFactor, stream);
}

cudaError_t mppSqrDistanceValidNorm8u32fLaunch(const Npp8u *source, int sourceStep, NppiSize sourceSize,
                                                const Npp8u *tpl, int templateStep, NppiSize templateSize,
                                                Npp32f *destination, int destinationStep, int channels,
                                                int activeChannels, int scaleFactor, cudaStream_t stream) {
  return launchSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                    destination, destinationStep, channels, activeChannels, scaleFactor, stream);
}

cudaError_t mppSqrDistanceValidNorm8s32fLaunch(const Npp8s *source, int sourceStep, NppiSize sourceSize,
                                                const Npp8s *tpl, int templateStep, NppiSize templateSize,
                                                Npp32f *destination, int destinationStep, int channels,
                                                int activeChannels, int scaleFactor, cudaStream_t stream) {
  return launchSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                    destination, destinationStep, channels, activeChannels, scaleFactor, stream);
}

cudaError_t mppSqrDistanceValidNorm16u32fLaunch(const Npp16u *source, int sourceStep, NppiSize sourceSize,
                                                 const Npp16u *tpl, int templateStep, NppiSize templateSize,
                                                 Npp32f *destination, int destinationStep, int channels,
                                                 int activeChannels, int scaleFactor, cudaStream_t stream) {
  return launchSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                    destination, destinationStep, channels, activeChannels, scaleFactor, stream);
}

cudaError_t mppSqrDistanceValidNorm32f32fLaunch(const Npp32f *source, int sourceStep, NppiSize sourceSize,
                                                 const Npp32f *tpl, int templateStep, NppiSize templateSize,
                                                 Npp32f *destination, int destinationStep, int channels,
                                                 int activeChannels, int scaleFactor, cudaStream_t stream) {
  return launchSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                    destination, destinationStep, channels, activeChannels, scaleFactor, stream);
}

} // extern "C"
