#include "npp.h"

#include <cuda_runtime.h>

extern "C" {
cudaError_t mppSqrDistanceValidNorm8u8uLaunch(const Npp8u *, int, NppiSize, const Npp8u *, int, NppiSize,
                                               Npp8u *, int, int, int, int, cudaStream_t);
cudaError_t mppSqrDistanceValidNorm8u32fLaunch(const Npp8u *, int, NppiSize, const Npp8u *, int, NppiSize,
                                                Npp32f *, int, int, int, int, cudaStream_t);
cudaError_t mppSqrDistanceValidNorm8s32fLaunch(const Npp8s *, int, NppiSize, const Npp8s *, int, NppiSize,
                                                Npp32f *, int, int, int, int, cudaStream_t);
cudaError_t mppSqrDistanceValidNorm16u32fLaunch(const Npp16u *, int, NppiSize, const Npp16u *, int, NppiSize,
                                                 Npp32f *, int, int, int, int, cudaStream_t);
cudaError_t mppSqrDistanceValidNorm32f32fLaunch(const Npp32f *, int, NppiSize, const Npp32f *, int, NppiSize,
                                                 Npp32f *, int, int, int, int, cudaStream_t);
}

namespace {

template <typename Source, typename Destination>
using LaunchFunction = cudaError_t (*)(const Source *, int, NppiSize, const Source *, int, NppiSize,
                                       Destination *, int, int, int, int, cudaStream_t);

template <typename Source, typename Destination>
NppStatus executeSqrDistanceValidNorm(const Source *source, int sourceStep, NppiSize sourceSize,
                                      const Source *tpl, int templateStep, NppiSize templateSize,
                                      Destination *destination, int destinationStep, int channels,
                                      int activeChannels, int scaleFactor, NppStreamContext context,
                                      LaunchFunction<Source, Destination> launch) {
  if (!source || !tpl || !destination) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (sourceSize.width <= 0 || sourceSize.height <= 0 || templateSize.width <= 0 ||
      templateSize.height <= 0 || templateSize.width > sourceSize.width ||
      templateSize.height > sourceSize.height) {
    return NPP_SIZE_ERROR;
  }

  const int destinationWidth = sourceSize.width - templateSize.width + 1;
  const long long minimumSourceStep =
      static_cast<long long>(sourceSize.width) * channels * sizeof(Source);
  const long long minimumTemplateStep =
      static_cast<long long>(templateSize.width) * channels * sizeof(Source);
  const long long minimumDestinationStep =
      static_cast<long long>(destinationWidth) * channels * sizeof(Destination);
  if (sourceStep < minimumSourceStep || templateStep < minimumTemplateStep ||
      destinationStep < minimumDestinationStep) {
    return NPP_STEP_ERROR;
  }

  const cudaError_t status =
      launch(source, sourceStep, sourceSize, tpl, templateStep, templateSize, destination, destinationStep,
             channels, activeChannels, scaleFactor, context.hStream);
  return status == cudaSuccess ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStreamContext currentStreamContext() {
  NppStreamContext context{};
  nppGetStreamContext(&context);
  return context;
}

} // namespace

NppStatus nppiSqrDistanceValid_Norm_8u_C1RSfs_Ctx(
    const Npp8u *source, int sourceStep, NppiSize sourceSize, const Npp8u *tpl, int templateStep,
    NppiSize templateSize, Npp8u *destination, int destinationStep, int scaleFactor,
    NppStreamContext context) {
  return executeSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                     destination, destinationStep, 1, 1, scaleFactor, context,
                                     mppSqrDistanceValidNorm8u8uLaunch);
}

NppStatus nppiSqrDistanceValid_Norm_8u_C1RSfs(const Npp8u *source, int sourceStep, NppiSize sourceSize,
                                               const Npp8u *tpl, int templateStep, NppiSize templateSize,
                                               Npp8u *destination, int destinationStep, int scaleFactor) {
  return nppiSqrDistanceValid_Norm_8u_C1RSfs_Ctx(source, sourceStep, sourceSize, tpl, templateStep,
                                                 templateSize, destination, destinationStep, scaleFactor,
                                                 currentStreamContext());
}

NppStatus nppiSqrDistanceValid_Norm_8u_C3RSfs_Ctx(
    const Npp8u *source, int sourceStep, NppiSize sourceSize, const Npp8u *tpl, int templateStep,
    NppiSize templateSize, Npp8u *destination, int destinationStep, int scaleFactor,
    NppStreamContext context) {
  return executeSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                     destination, destinationStep, 3, 3, scaleFactor, context,
                                     mppSqrDistanceValidNorm8u8uLaunch);
}

NppStatus nppiSqrDistanceValid_Norm_8u_C3RSfs(const Npp8u *source, int sourceStep, NppiSize sourceSize,
                                               const Npp8u *tpl, int templateStep, NppiSize templateSize,
                                               Npp8u *destination, int destinationStep, int scaleFactor) {
  return nppiSqrDistanceValid_Norm_8u_C3RSfs_Ctx(source, sourceStep, sourceSize, tpl, templateStep,
                                                 templateSize, destination, destinationStep, scaleFactor,
                                                 currentStreamContext());
}

NppStatus nppiSqrDistanceValid_Norm_8u_C4RSfs_Ctx(
    const Npp8u *source, int sourceStep, NppiSize sourceSize, const Npp8u *tpl, int templateStep,
    NppiSize templateSize, Npp8u *destination, int destinationStep, int scaleFactor,
    NppStreamContext context) {
  return executeSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                     destination, destinationStep, 4, 4, scaleFactor, context,
                                     mppSqrDistanceValidNorm8u8uLaunch);
}

NppStatus nppiSqrDistanceValid_Norm_8u_C4RSfs(const Npp8u *source, int sourceStep, NppiSize sourceSize,
                                               const Npp8u *tpl, int templateStep, NppiSize templateSize,
                                               Npp8u *destination, int destinationStep, int scaleFactor) {
  return nppiSqrDistanceValid_Norm_8u_C4RSfs_Ctx(source, sourceStep, sourceSize, tpl, templateStep,
                                                 templateSize, destination, destinationStep, scaleFactor,
                                                 currentStreamContext());
}

NppStatus nppiSqrDistanceValid_Norm_8u_AC4RSfs_Ctx(
    const Npp8u *source, int sourceStep, NppiSize sourceSize, const Npp8u *tpl, int templateStep,
    NppiSize templateSize, Npp8u *destination, int destinationStep, int scaleFactor,
    NppStreamContext context) {
  return executeSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                     destination, destinationStep, 4, 3, scaleFactor, context,
                                     mppSqrDistanceValidNorm8u8uLaunch);
}

NppStatus nppiSqrDistanceValid_Norm_8u_AC4RSfs(const Npp8u *source, int sourceStep, NppiSize sourceSize,
                                                const Npp8u *tpl, int templateStep, NppiSize templateSize,
                                                Npp8u *destination, int destinationStep, int scaleFactor) {
  return nppiSqrDistanceValid_Norm_8u_AC4RSfs_Ctx(source, sourceStep, sourceSize, tpl, templateStep,
                                                  templateSize, destination, destinationStep, scaleFactor,
                                                  currentStreamContext());
}

NppStatus nppiSqrDistanceValid_Norm_8u32f_C1R_Ctx(
    const Npp8u *source, int sourceStep, NppiSize sourceSize, const Npp8u *tpl, int templateStep,
    NppiSize templateSize, Npp32f *destination, int destinationStep, NppStreamContext context) {
  return executeSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                     destination, destinationStep, 1, 1, 0, context,
                                     mppSqrDistanceValidNorm8u32fLaunch);
}

NppStatus nppiSqrDistanceValid_Norm_8u32f_C1R(const Npp8u *source, int sourceStep, NppiSize sourceSize,
                                               const Npp8u *tpl, int templateStep, NppiSize templateSize,
                                               Npp32f *destination, int destinationStep) {
  return nppiSqrDistanceValid_Norm_8u32f_C1R_Ctx(source, sourceStep, sourceSize, tpl, templateStep,
                                                 templateSize, destination, destinationStep,
                                                 currentStreamContext());
}

NppStatus nppiSqrDistanceValid_Norm_8u32f_C3R_Ctx(
    const Npp8u *source, int sourceStep, NppiSize sourceSize, const Npp8u *tpl, int templateStep,
    NppiSize templateSize, Npp32f *destination, int destinationStep, NppStreamContext context) {
  return executeSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                     destination, destinationStep, 3, 3, 0, context,
                                     mppSqrDistanceValidNorm8u32fLaunch);
}

NppStatus nppiSqrDistanceValid_Norm_8u32f_C3R(const Npp8u *source, int sourceStep, NppiSize sourceSize,
                                               const Npp8u *tpl, int templateStep, NppiSize templateSize,
                                               Npp32f *destination, int destinationStep) {
  return nppiSqrDistanceValid_Norm_8u32f_C3R_Ctx(source, sourceStep, sourceSize, tpl, templateStep,
                                                 templateSize, destination, destinationStep,
                                                 currentStreamContext());
}

NppStatus nppiSqrDistanceValid_Norm_8u32f_C4R_Ctx(
    const Npp8u *source, int sourceStep, NppiSize sourceSize, const Npp8u *tpl, int templateStep,
    NppiSize templateSize, Npp32f *destination, int destinationStep, NppStreamContext context) {
  return executeSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                     destination, destinationStep, 4, 4, 0, context,
                                     mppSqrDistanceValidNorm8u32fLaunch);
}

NppStatus nppiSqrDistanceValid_Norm_8u32f_C4R(const Npp8u *source, int sourceStep, NppiSize sourceSize,
                                               const Npp8u *tpl, int templateStep, NppiSize templateSize,
                                               Npp32f *destination, int destinationStep) {
  return nppiSqrDistanceValid_Norm_8u32f_C4R_Ctx(source, sourceStep, sourceSize, tpl, templateStep,
                                                 templateSize, destination, destinationStep,
                                                 currentStreamContext());
}

NppStatus nppiSqrDistanceValid_Norm_8u32f_AC4R_Ctx(
    const Npp8u *source, int sourceStep, NppiSize sourceSize, const Npp8u *tpl, int templateStep,
    NppiSize templateSize, Npp32f *destination, int destinationStep, NppStreamContext context) {
  return executeSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                     destination, destinationStep, 4, 3, 0, context,
                                     mppSqrDistanceValidNorm8u32fLaunch);
}

NppStatus nppiSqrDistanceValid_Norm_8u32f_AC4R(const Npp8u *source, int sourceStep, NppiSize sourceSize,
                                                const Npp8u *tpl, int templateStep, NppiSize templateSize,
                                                Npp32f *destination, int destinationStep) {
  return nppiSqrDistanceValid_Norm_8u32f_AC4R_Ctx(source, sourceStep, sourceSize, tpl, templateStep,
                                                  templateSize, destination, destinationStep,
                                                  currentStreamContext());
}

NppStatus nppiSqrDistanceValid_Norm_8s32f_C1R_Ctx(
    const Npp8s *source, int sourceStep, NppiSize sourceSize, const Npp8s *tpl, int templateStep,
    NppiSize templateSize, Npp32f *destination, int destinationStep, NppStreamContext context) {
  return executeSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                     destination, destinationStep, 1, 1, 0, context,
                                     mppSqrDistanceValidNorm8s32fLaunch);
}

NppStatus nppiSqrDistanceValid_Norm_8s32f_C1R(const Npp8s *source, int sourceStep, NppiSize sourceSize,
                                               const Npp8s *tpl, int templateStep, NppiSize templateSize,
                                               Npp32f *destination, int destinationStep) {
  return nppiSqrDistanceValid_Norm_8s32f_C1R_Ctx(source, sourceStep, sourceSize, tpl, templateStep,
                                                 templateSize, destination, destinationStep,
                                                 currentStreamContext());
}

NppStatus nppiSqrDistanceValid_Norm_8s32f_C3R_Ctx(
    const Npp8s *source, int sourceStep, NppiSize sourceSize, const Npp8s *tpl, int templateStep,
    NppiSize templateSize, Npp32f *destination, int destinationStep, NppStreamContext context) {
  return executeSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                     destination, destinationStep, 3, 3, 0, context,
                                     mppSqrDistanceValidNorm8s32fLaunch);
}

NppStatus nppiSqrDistanceValid_Norm_8s32f_C3R(const Npp8s *source, int sourceStep, NppiSize sourceSize,
                                               const Npp8s *tpl, int templateStep, NppiSize templateSize,
                                               Npp32f *destination, int destinationStep) {
  return nppiSqrDistanceValid_Norm_8s32f_C3R_Ctx(source, sourceStep, sourceSize, tpl, templateStep,
                                                 templateSize, destination, destinationStep,
                                                 currentStreamContext());
}

NppStatus nppiSqrDistanceValid_Norm_8s32f_C4R_Ctx(
    const Npp8s *source, int sourceStep, NppiSize sourceSize, const Npp8s *tpl, int templateStep,
    NppiSize templateSize, Npp32f *destination, int destinationStep, NppStreamContext context) {
  return executeSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                     destination, destinationStep, 4, 4, 0, context,
                                     mppSqrDistanceValidNorm8s32fLaunch);
}

NppStatus nppiSqrDistanceValid_Norm_8s32f_C4R(const Npp8s *source, int sourceStep, NppiSize sourceSize,
                                               const Npp8s *tpl, int templateStep, NppiSize templateSize,
                                               Npp32f *destination, int destinationStep) {
  return nppiSqrDistanceValid_Norm_8s32f_C4R_Ctx(source, sourceStep, sourceSize, tpl, templateStep,
                                                 templateSize, destination, destinationStep,
                                                 currentStreamContext());
}

NppStatus nppiSqrDistanceValid_Norm_8s32f_AC4R_Ctx(
    const Npp8s *source, int sourceStep, NppiSize sourceSize, const Npp8s *tpl, int templateStep,
    NppiSize templateSize, Npp32f *destination, int destinationStep, NppStreamContext context) {
  return executeSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                     destination, destinationStep, 4, 3, 0, context,
                                     mppSqrDistanceValidNorm8s32fLaunch);
}

NppStatus nppiSqrDistanceValid_Norm_8s32f_AC4R(const Npp8s *source, int sourceStep, NppiSize sourceSize,
                                                const Npp8s *tpl, int templateStep, NppiSize templateSize,
                                                Npp32f *destination, int destinationStep) {
  return nppiSqrDistanceValid_Norm_8s32f_AC4R_Ctx(source, sourceStep, sourceSize, tpl, templateStep,
                                                  templateSize, destination, destinationStep,
                                                  currentStreamContext());
}

NppStatus nppiSqrDistanceValid_Norm_16u32f_C1R_Ctx(
    const Npp16u *source, int sourceStep, NppiSize sourceSize, const Npp16u *tpl, int templateStep,
    NppiSize templateSize, Npp32f *destination, int destinationStep, NppStreamContext context) {
  return executeSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                     destination, destinationStep, 1, 1, 0, context,
                                     mppSqrDistanceValidNorm16u32fLaunch);
}

NppStatus nppiSqrDistanceValid_Norm_16u32f_C1R(const Npp16u *source, int sourceStep, NppiSize sourceSize,
                                                const Npp16u *tpl, int templateStep, NppiSize templateSize,
                                                Npp32f *destination, int destinationStep) {
  return nppiSqrDistanceValid_Norm_16u32f_C1R_Ctx(source, sourceStep, sourceSize, tpl, templateStep,
                                                  templateSize, destination, destinationStep,
                                                  currentStreamContext());
}

NppStatus nppiSqrDistanceValid_Norm_16u32f_C3R_Ctx(
    const Npp16u *source, int sourceStep, NppiSize sourceSize, const Npp16u *tpl, int templateStep,
    NppiSize templateSize, Npp32f *destination, int destinationStep, NppStreamContext context) {
  return executeSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                     destination, destinationStep, 3, 3, 0, context,
                                     mppSqrDistanceValidNorm16u32fLaunch);
}

NppStatus nppiSqrDistanceValid_Norm_16u32f_C3R(const Npp16u *source, int sourceStep, NppiSize sourceSize,
                                                const Npp16u *tpl, int templateStep, NppiSize templateSize,
                                                Npp32f *destination, int destinationStep) {
  return nppiSqrDistanceValid_Norm_16u32f_C3R_Ctx(source, sourceStep, sourceSize, tpl, templateStep,
                                                  templateSize, destination, destinationStep,
                                                  currentStreamContext());
}

NppStatus nppiSqrDistanceValid_Norm_16u32f_C4R_Ctx(
    const Npp16u *source, int sourceStep, NppiSize sourceSize, const Npp16u *tpl, int templateStep,
    NppiSize templateSize, Npp32f *destination, int destinationStep, NppStreamContext context) {
  return executeSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                     destination, destinationStep, 4, 4, 0, context,
                                     mppSqrDistanceValidNorm16u32fLaunch);
}

NppStatus nppiSqrDistanceValid_Norm_16u32f_C4R(const Npp16u *source, int sourceStep, NppiSize sourceSize,
                                                const Npp16u *tpl, int templateStep, NppiSize templateSize,
                                                Npp32f *destination, int destinationStep) {
  return nppiSqrDistanceValid_Norm_16u32f_C4R_Ctx(source, sourceStep, sourceSize, tpl, templateStep,
                                                  templateSize, destination, destinationStep,
                                                  currentStreamContext());
}

NppStatus nppiSqrDistanceValid_Norm_16u32f_AC4R_Ctx(
    const Npp16u *source, int sourceStep, NppiSize sourceSize, const Npp16u *tpl, int templateStep,
    NppiSize templateSize, Npp32f *destination, int destinationStep, NppStreamContext context) {
  return executeSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                     destination, destinationStep, 4, 3, 0, context,
                                     mppSqrDistanceValidNorm16u32fLaunch);
}

NppStatus nppiSqrDistanceValid_Norm_16u32f_AC4R(const Npp16u *source, int sourceStep, NppiSize sourceSize,
                                                 const Npp16u *tpl, int templateStep, NppiSize templateSize,
                                                 Npp32f *destination, int destinationStep) {
  return nppiSqrDistanceValid_Norm_16u32f_AC4R_Ctx(source, sourceStep, sourceSize, tpl, templateStep,
                                                   templateSize, destination, destinationStep,
                                                   currentStreamContext());
}

NppStatus nppiSqrDistanceValid_Norm_32f_C1R_Ctx(
    const Npp32f *source, int sourceStep, NppiSize sourceSize, const Npp32f *tpl, int templateStep,
    NppiSize templateSize, Npp32f *destination, int destinationStep, NppStreamContext context) {
  return executeSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                     destination, destinationStep, 1, 1, 0, context,
                                     mppSqrDistanceValidNorm32f32fLaunch);
}

NppStatus nppiSqrDistanceValid_Norm_32f_C1R(const Npp32f *source, int sourceStep, NppiSize sourceSize,
                                             const Npp32f *tpl, int templateStep, NppiSize templateSize,
                                             Npp32f *destination, int destinationStep) {
  return nppiSqrDistanceValid_Norm_32f_C1R_Ctx(source, sourceStep, sourceSize, tpl, templateStep,
                                               templateSize, destination, destinationStep,
                                               currentStreamContext());
}

NppStatus nppiSqrDistanceValid_Norm_32f_C3R_Ctx(
    const Npp32f *source, int sourceStep, NppiSize sourceSize, const Npp32f *tpl, int templateStep,
    NppiSize templateSize, Npp32f *destination, int destinationStep, NppStreamContext context) {
  return executeSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                     destination, destinationStep, 3, 3, 0, context,
                                     mppSqrDistanceValidNorm32f32fLaunch);
}

NppStatus nppiSqrDistanceValid_Norm_32f_C3R(const Npp32f *source, int sourceStep, NppiSize sourceSize,
                                             const Npp32f *tpl, int templateStep, NppiSize templateSize,
                                             Npp32f *destination, int destinationStep) {
  return nppiSqrDistanceValid_Norm_32f_C3R_Ctx(source, sourceStep, sourceSize, tpl, templateStep,
                                               templateSize, destination, destinationStep,
                                               currentStreamContext());
}

NppStatus nppiSqrDistanceValid_Norm_32f_C4R_Ctx(
    const Npp32f *source, int sourceStep, NppiSize sourceSize, const Npp32f *tpl, int templateStep,
    NppiSize templateSize, Npp32f *destination, int destinationStep, NppStreamContext context) {
  return executeSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                     destination, destinationStep, 4, 4, 0, context,
                                     mppSqrDistanceValidNorm32f32fLaunch);
}

NppStatus nppiSqrDistanceValid_Norm_32f_C4R(const Npp32f *source, int sourceStep, NppiSize sourceSize,
                                             const Npp32f *tpl, int templateStep, NppiSize templateSize,
                                             Npp32f *destination, int destinationStep) {
  return nppiSqrDistanceValid_Norm_32f_C4R_Ctx(source, sourceStep, sourceSize, tpl, templateStep,
                                               templateSize, destination, destinationStep,
                                               currentStreamContext());
}

NppStatus nppiSqrDistanceValid_Norm_32f_AC4R_Ctx(
    const Npp32f *source, int sourceStep, NppiSize sourceSize, const Npp32f *tpl, int templateStep,
    NppiSize templateSize, Npp32f *destination, int destinationStep, NppStreamContext context) {
  return executeSqrDistanceValidNorm(source, sourceStep, sourceSize, tpl, templateStep, templateSize,
                                     destination, destinationStep, 4, 3, 0, context,
                                     mppSqrDistanceValidNorm32f32fLaunch);
}

NppStatus nppiSqrDistanceValid_Norm_32f_AC4R(const Npp32f *source, int sourceStep, NppiSize sourceSize,
                                              const Npp32f *tpl, int templateStep, NppiSize templateSize,
                                              Npp32f *destination, int destinationStep) {
  return nppiSqrDistanceValid_Norm_32f_AC4R_Ctx(source, sourceStep, sourceSize, tpl, templateStep,
                                                templateSize, destination, destinationStep,
                                                currentStreamContext());
}
