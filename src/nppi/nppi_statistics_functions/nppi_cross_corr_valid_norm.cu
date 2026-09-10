#include "npp.h"

#include <cuda_runtime.h>

#include <cmath>

namespace {

template <typename Destination> __device__ Destination correlationResult(double correlation, int) {
  return static_cast<Destination>(correlation);
}

template <> __device__ Npp8u correlationResult<Npp8u>(double correlation, int scaleFactor) {
  double scaled;
  if (scaleFactor < -1024) {
    scaled = correlation > 0.0 ? 256.0 : 0.0;
  } else if (scaleFactor > 1024) {
    scaled = 0.0;
  } else {
    scaled = ldexp(correlation, -scaleFactor);
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
__global__ void crossCorrValidNormKernel(const Source *source, int sourceStep, int sourceWidth, int sourceHeight,
                                         const Source *tpl, int templateStep, int templateWidth, int templateHeight,
                                         Destination *destination, int destinationStep, int channels,
                                         bool preserveAlpha, int scaleFactor) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int destinationWidth = sourceWidth - templateWidth + 1;
  const int destinationHeight = sourceHeight - templateHeight + 1;
  if (x >= destinationWidth || y >= destinationHeight) {
    return;
  }

  Destination *destinationRow =
      reinterpret_cast<Destination *>(reinterpret_cast<char *>(destination) + y * destinationStep);
  const int correlatedChannels = preserveAlpha ? 3 : channels;
  for (int channel = 0; channel < correlatedChannels; ++channel) {
    double dot = 0.0;
    double sourceSquare = 0.0;
    double templateSquare = 0.0;
    for (int templateY = 0; templateY < templateHeight; ++templateY) {
      const Source *sourceRow = reinterpret_cast<const Source *>(
          reinterpret_cast<const char *>(source) + (y + templateY) * sourceStep);
      const Source *templateRow =
          reinterpret_cast<const Source *>(reinterpret_cast<const char *>(tpl) + templateY * templateStep);
      for (int templateX = 0; templateX < templateWidth; ++templateX) {
        const double sourceValue = static_cast<double>(sourceRow[((x + templateX) * channels) + channel]);
        const double templateValue = static_cast<double>(templateRow[(templateX * channels) + channel]);
        dot += sourceValue * templateValue;
        sourceSquare += sourceValue * sourceValue;
        templateSquare += templateValue * templateValue;
      }
    }

    double correlation = 0.0;
    if (sourceSquare != 0.0 && templateSquare != 0.0) {
      correlation = dot / (sqrt(sourceSquare) * sqrt(templateSquare));
    }
    destinationRow[x * channels + channel] = correlationResult<Destination>(correlation, scaleFactor);
  }
}

template <typename Source, typename Destination>
cudaError_t launchCrossCorrValidNorm(const Source *source, int sourceStep, NppiSize sourceSize, const Source *tpl,
                                     int templateStep, NppiSize templateSize, Destination *destination,
                                     int destinationStep, int channels, bool preserveAlpha, int scaleFactor,
                                     cudaStream_t stream) {
  const int destinationWidth = sourceSize.width - templateSize.width + 1;
  const int destinationHeight = sourceSize.height - templateSize.height + 1;
  const dim3 block(16, 16);
  const dim3 grid((destinationWidth + block.x - 1) / block.x,
                  (destinationHeight + block.y - 1) / block.y);
  crossCorrValidNormKernel<<<grid, block, 0, stream>>>(
      source, sourceStep, sourceSize.width, sourceSize.height, tpl, templateStep, templateSize.width,
      templateSize.height, destination, destinationStep, channels, preserveAlpha, scaleFactor);
  return cudaGetLastError();
}

} // namespace

extern "C" {

cudaError_t mppCrossCorrValidNorm8u8uLaunch(const Npp8u *source, int sourceStep, NppiSize sourceSize,
                                            const Npp8u *tpl, int templateStep, NppiSize templateSize,
                                            Npp8u *destination, int destinationStep, int channels,
                                            bool preserveAlpha, int scaleFactor, cudaStream_t stream) {
  return launchCrossCorrValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize, destination,
                                  destinationStep, channels, preserveAlpha, scaleFactor, stream);
}

cudaError_t mppCrossCorrValidNorm8u32fLaunch(const Npp8u *source, int sourceStep, NppiSize sourceSize,
                                             const Npp8u *tpl, int templateStep, NppiSize templateSize,
                                             Npp32f *destination, int destinationStep, int channels,
                                             bool preserveAlpha, int scaleFactor, cudaStream_t stream) {
  return launchCrossCorrValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize, destination,
                                  destinationStep, channels, preserveAlpha, scaleFactor, stream);
}

cudaError_t mppCrossCorrValidNorm8s32fLaunch(const Npp8s *source, int sourceStep, NppiSize sourceSize,
                                             const Npp8s *tpl, int templateStep, NppiSize templateSize,
                                             Npp32f *destination, int destinationStep, int channels,
                                             bool preserveAlpha, int scaleFactor, cudaStream_t stream) {
  return launchCrossCorrValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize, destination,
                                  destinationStep, channels, preserveAlpha, scaleFactor, stream);
}

cudaError_t mppCrossCorrValidNorm16u32fLaunch(const Npp16u *source, int sourceStep, NppiSize sourceSize,
                                              const Npp16u *tpl, int templateStep, NppiSize templateSize,
                                              Npp32f *destination, int destinationStep, int channels,
                                              bool preserveAlpha, int scaleFactor, cudaStream_t stream) {
  return launchCrossCorrValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize, destination,
                                  destinationStep, channels, preserveAlpha, scaleFactor, stream);
}

cudaError_t mppCrossCorrValidNorm32f32fLaunch(const Npp32f *source, int sourceStep, NppiSize sourceSize,
                                              const Npp32f *tpl, int templateStep, NppiSize templateSize,
                                              Npp32f *destination, int destinationStep, int channels,
                                              bool preserveAlpha, int scaleFactor, cudaStream_t stream) {
  return launchCrossCorrValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize, destination,
                                  destinationStep, channels, preserveAlpha, scaleFactor, stream);
}

cudaError_t mppCrossCorrValidNorm64f64fLaunch(const Npp64f *source, int sourceStep, NppiSize sourceSize,
                                              const Npp64f *tpl, int templateStep, NppiSize templateSize,
                                              Npp64f *destination, int destinationStep, int channels,
                                              bool preserveAlpha, int scaleFactor, cudaStream_t stream) {
  return launchCrossCorrValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize, destination,
                                  destinationStep, channels, preserveAlpha, scaleFactor, stream);
}

} // extern "C"
