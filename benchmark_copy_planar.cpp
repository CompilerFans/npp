#include "npp.h"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <iomanip>

#define CHECK_CUDA(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                << cudaGetErrorString(err) << std::endl; \
      exit(1); \
    } \
  } while(0)

#define CHECK_NPP(call) \
  do { \
    NppStatus status = call; \
    if (status != NPP_SUCCESS) { \
      std::cerr << "NPP error at " << __FILE__ << ":" << __LINE__ << " - " \
                << status << std::endl; \
      exit(1); \
    } \
  } while(0)

void benchmark_C3P3R(int width, int height, int iterations) {
  size_t packed_pitch = width * 3 * sizeof(Npp32f);
  size_t planar_pitch = width * sizeof(Npp32f);

  Npp32f *d_packed;
  Npp32f *d_planar[3];

  CHECK_CUDA(cudaMalloc(&d_packed, height * packed_pitch));
  CHECK_CUDA(cudaMalloc(&d_planar[0], height * planar_pitch));
  CHECK_CUDA(cudaMalloc(&d_planar[1], height * planar_pitch));
  CHECK_CUDA(cudaMalloc(&d_planar[2], height * planar_pitch));

  NppiSize roi = {width, height};

  CHECK_CUDA(cudaDeviceSynchronize());
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; i++) {
    CHECK_NPP(nppiCopy_32f_C3P3R(d_packed, packed_pitch, d_planar, planar_pitch, roi));
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();

  double elapsed = std::chrono::duration<double>(end - start).count();
  double total_bytes = (double)width * height * 3 * sizeof(Npp32f) * iterations;
  double throughput = total_bytes / elapsed / 1e9;
  double time_per_call = elapsed / iterations * 1000.0;

  std::cout << "nppiCopy_32f_C3P3R [" << width << "x" << height << "]:" << std::endl;
  std::cout << "  Time per call: " << std::fixed << std::setprecision(3)
            << time_per_call << " ms" << std::endl;
  std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
            << throughput << " GB/s" << std::endl;

  CHECK_CUDA(cudaFree(d_packed));
  CHECK_CUDA(cudaFree(d_planar[0]));
  CHECK_CUDA(cudaFree(d_planar[1]));
  CHECK_CUDA(cudaFree(d_planar[2]));
}

void benchmark_P3C3R(int width, int height, int iterations) {
  size_t packed_pitch = width * 3 * sizeof(Npp32f);
  size_t planar_pitch = width * sizeof(Npp32f);

  Npp32f *d_packed;
  Npp32f *d_planar[3];

  CHECK_CUDA(cudaMalloc(&d_packed, height * packed_pitch));
  CHECK_CUDA(cudaMalloc(&d_planar[0], height * planar_pitch));
  CHECK_CUDA(cudaMalloc(&d_planar[1], height * planar_pitch));
  CHECK_CUDA(cudaMalloc(&d_planar[2], height * planar_pitch));

  NppiSize roi = {width, height};

  CHECK_CUDA(cudaDeviceSynchronize());
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < iterations; i++) {
    CHECK_NPP(nppiCopy_32f_P3C3R((const Npp32f**)d_planar, planar_pitch, d_packed, packed_pitch, roi));
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();

  double elapsed = std::chrono::duration<double>(end - start).count();
  double total_bytes = (double)width * height * 3 * sizeof(Npp32f) * iterations;
  double throughput = total_bytes / elapsed / 1e9;
  double time_per_call = elapsed / iterations * 1000.0;

  std::cout << "nppiCopy_32f_P3C3R [" << width << "x" << height << "]:" << std::endl;
  std::cout << "  Time per call: " << std::fixed << std::setprecision(3)
            << time_per_call << " ms" << std::endl;
  std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
            << throughput << " GB/s" << std::endl;

  CHECK_CUDA(cudaFree(d_packed));
  CHECK_CUDA(cudaFree(d_planar[0]));
  CHECK_CUDA(cudaFree(d_planar[1]));
  CHECK_CUDA(cudaFree(d_planar[2]));
}

int main() {
  std::cout << "=== Copy Planar Performance Test ===" << std::endl << std::endl;

  int iterations = 1000;

  std::cout << "Testing very small images (320x240):" << std::endl;
  benchmark_C3P3R(320, 240, iterations);
  benchmark_P3C3R(320, 240, iterations);
  std::cout << std::endl;

  std::cout << "Testing small images (640x480):" << std::endl;
  benchmark_C3P3R(640, 480, iterations);
  benchmark_P3C3R(640, 480, iterations);
  std::cout << std::endl;

  std::cout << "Testing medium images (1920x1080):" << std::endl;
  benchmark_C3P3R(1920, 1080, iterations);
  benchmark_P3C3R(1920, 1080, iterations);
  std::cout << std::endl;

  std::cout << "Testing large images (3840x2160):" << std::endl;
  benchmark_C3P3R(3840, 2160, iterations);
  benchmark_P3C3R(3840, 2160, iterations);
  std::cout << std::endl;

  std::cout << "Testing very large images (7680x4320):" << std::endl;
  benchmark_C3P3R(7680, 4320, iterations / 10);
  benchmark_P3C3R(7680, 4320, iterations / 10);
  std::cout << std::endl;

  return 0;
}
