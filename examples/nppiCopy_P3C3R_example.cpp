#include <cuda_runtime.h>
#include <npp.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  // Image dimensions
  int width = 4;
  int height = 3;
  int channels = 3;

  // Calculate sizes and steps
  int planarStep = width * sizeof(Npp32f);
  int packedStep = width * channels * sizeof(Npp32f);
  int planarSize = width * height * sizeof(Npp32f);
  int packedSize = width * height * channels * sizeof(Npp32f);

  // Allocate host memory
  Npp32f *h_R = (Npp32f *)malloc(planarSize);
  Npp32f *h_G = (Npp32f *)malloc(planarSize);
  Npp32f *h_B = (Npp32f *)malloc(planarSize);
  Npp32f *h_packed = (Npp32f *)malloc(packedSize);

  // Initialize planar data (R, G, B planes)
  printf("Input planar data:\n");
  for (int i = 0; i < width * height; i++) {
    h_R[i] = (float)(i + 1) * 1.0f;       // R: 1.0, 2.0, 3.0, ...
    h_G[i] = (float)(i + 1) * 10.0f;      // G: 10.0, 20.0, 30.0, ...
    h_B[i] = (float)(i + 1) * 100.0f;     // B: 100.0, 200.0, 300.0, ...
  }

  printf("R plane: ");
  for (int i = 0; i < 12; i++) printf("%.0f ", h_R[i]);
  printf("\nG plane: ");
  for (int i = 0; i < 12; i++) printf("%.0f ", h_G[i]);
  printf("\nB plane: ");
  for (int i = 0; i < 12; i++) printf("%.0f ", h_B[i]);
  printf("\n\n");

  // Allocate device memory for planar images
  Npp32f *d_R, *d_G, *d_B;
  cudaMalloc(&d_R, planarSize);
  cudaMalloc(&d_G, planarSize);
  cudaMalloc(&d_B, planarSize);

  // Copy planar data to device
  cudaMemcpy(d_R, h_R, planarSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_G, h_G, planarSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, planarSize, cudaMemcpyHostToDevice);

  // Create planar pointer array on host (pointing to device memory)
  const Npp32f *pSrc[3] = {d_R, d_G, d_B};

  // Allocate device memory for packed output
  Npp32f *d_packed;
  cudaMalloc(&d_packed, packedSize);

  // Setup ROI
  NppiSize roiSize = {width, height};

  // Call nppiCopy_32f_P3C3R (planar to packed)
  NppStatus status = nppiCopy_32f_P3C3R(pSrc, planarStep, d_packed, packedStep, roiSize);

  if (status != NPP_SUCCESS) {
    printf("Error: nppiCopy_32f_P3C3R failed with status %d\n", status);
    return -1;
  }

  // Copy result back to host
  cudaMemcpy(h_packed, d_packed, packedSize, cudaMemcpyDeviceToHost);

  // Print packed result (interleaved RGB format)
  printf("Output packed data (RGB interleaved):\n");
  for (int y = 0; y < height; y++) {
    printf("Row %d: ", y);
    for (int x = 0; x < width; x++) {
      int idx = y * width * 3 + x * 3;
      printf("[%.0f,%.0f,%.0f] ", h_packed[idx], h_packed[idx + 1], h_packed[idx + 2]);
    }
    printf("\n");
  }

  // Verify correctness
  printf("\nVerification:\n");
  bool correct = true;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int planarIdx = y * width + x;
      int packedIdx = y * width * 3 + x * 3;

      if (h_packed[packedIdx] != h_R[planarIdx] || h_packed[packedIdx + 1] != h_G[planarIdx] ||
          h_packed[packedIdx + 2] != h_B[planarIdx]) {
        printf("Mismatch at (%d,%d)\n", x, y);
        correct = false;
      }
    }
  }
  printf("%s\n", correct ? "PASS: All values correct" : "FAIL: Data mismatch");

  // Cleanup
  free(h_R);
  free(h_G);
  free(h_B);
  free(h_packed);
  cudaFree(d_R);
  cudaFree(d_G);
  cudaFree(d_B);
  cudaFree(d_packed);

  return 0;
}
