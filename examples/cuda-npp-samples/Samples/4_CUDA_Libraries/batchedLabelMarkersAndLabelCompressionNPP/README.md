# batchedLabelMarkersAndLabelCompressionNPP - Batched Label Markers and Label Compression with NPP

## Description

A NPP CUDA Sample that demonstrates how to use NPP's batched connected component labeling (CCL) functions to perform label markers and label compression on multiple images.

## Key Concepts

Performance Strategies, Image Processing, NPP Library, Connected Component Labeling, Union-Find Algorithm, Batched Processing

## Supported SM Architectures

[SM 5.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 5.3 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 6.1 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.2 ](https://developer.nvidia.com/cuda-gpus)  [SM 7.5 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.0 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.6 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.7 ](https://developer.nvidia.com/cuda-gpus)  [SM 8.9 ](https://developer.nvidia.com/cuda-gpus)  [SM 9.0 ](https://developer.nvidia.com/cuda-gpus)

## Supported OSes

Linux, Windows

## Supported CPU Architecture

x86_64, armv7l

## CUDA APIs involved

### [CUDA Runtime API](http://docs.nvidia.com/cuda/cuda-runtime-api/index.html)
cudaRuntimeGetVersion, cudaDriverGetVersion, cudaMalloc, cudaFree, cudaMemcpy

### [NPP Library](https://docs.nvidia.com/cuda/npp/index.html)
nppiLabelMarkersUF_8u32u_C1R, nppiLabelMarkersUFBatch_8u32u_C1R_Advanced, nppiCompressMarkerLabelsUF_32u_C1IR, nppiLabelMarkersUFGetBufferSize_32u_C1R, nppiCompressMarkerLabelsGetBufferSize_32u_C1R

## Dependencies needed to build/run
[FreeImage](../../../README.md#freeimage), [NPP](../../../README.md#npp)

## Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

## Build and Run

### Linux
```
$ cd Samples/4_CUDA_Libraries/batchedLabelMarkersAndLabelCompressionNPP
$ make
$ ./batchedLabelMarkersAndLabelCompressionNPP
```

### Windows
```
Open batchedLabelMarkersAndLabelCompressionNPP.sln with Visual Studio and build.
```

## References (for more details)

- [NPP Documentation](https://docs.nvidia.com/cuda/npp/index.html)
- [Connected Component Labeling](https://en.wikipedia.org/wiki/Connected-component_labeling)
