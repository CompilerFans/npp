#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <npp.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
  cudaError_t err = call; \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
    exit(1); \
  } \
} while(0)

#define CHECK_NPP(call) do { \
  NppStatus status = call; \
  if (status != NPP_SUCCESS) { \
    std::cerr << "NPP error: " << status << std::endl; \
    exit(1); \
  } \
} while(0)

// Reference CPU implementation
template<typename T>
void cpuSuperSampling(const T* pSrc, int nSrcStep, int srcW, int srcH,
                      T* pDst, int nDstStep, int dstW, int dstH, int channels = 1) {
  float scaleX = (float)srcW / dstW;
  float scaleY = (float)srcH / dstH;

  for (int dy = 0; dy < dstH; dy++) {
    for (int dx = 0; dx < dstW; dx++) {
      for (int c = 0; c < channels; c++) {
        float srcCenterX = (dx + 0.5f) * scaleX;
        float srcCenterY = (dy + 0.5f) * scaleY;
        float xMin = srcCenterX - scaleX * 0.5f;
        float xMax = srcCenterX + scaleX * 0.5f;
        float yMin = srcCenterY - scaleY * 0.5f;
        float yMax = srcCenterY + scaleY * 0.5f;

        int xMinInt = (int)std::ceil(xMin);
        int xMaxInt = (int)std::floor(xMax);
        int yMinInt = (int)std::ceil(yMin);
        int yMaxInt = (int)std::floor(yMax);

        xMinInt = std::max(0, xMinInt);
        xMaxInt = std::min(srcW, xMaxInt);
        yMinInt = std::max(0, yMinInt);
        yMaxInt = std::min(srcH, yMaxInt);

        float wxMin = std::ceil(xMin) - xMin;
        float wxMax = xMax - std::floor(xMax);
        float wyMin = std::ceil(yMin) - yMin;
        float wyMax = yMax - std::floor(yMax);

        wxMin = std::min(1.0f, std::max(0.0f, wxMin));
        wxMax = std::min(1.0f, std::max(0.0f, wxMax));
        wyMin = std::min(1.0f, std::max(0.0f, wyMin));
        wyMax = std::min(1.0f, std::max(0.0f, wyMax));

        float sum = 0.0f;

        // Top edge
        if (wyMin > 0.0f && yMinInt > 0) {
          int y = yMinInt - 1;
          const T* srcRow = (const T*)((const char*)pSrc + y * nSrcStep);
          if (wxMin > 0.0f && xMinInt > 0)
            sum += srcRow[(xMinInt-1)*channels + c] * wxMin * wyMin;
          for (int x = xMinInt; x < xMaxInt; x++)
            sum += srcRow[x*channels + c] * wyMin;
          if (wxMax > 0.0f && xMaxInt < srcW)
            sum += srcRow[xMaxInt*channels + c] * wxMax * wyMin;
        }

        // Middle
        for (int y = yMinInt; y < yMaxInt; y++) {
          const T* srcRow = (const T*)((const char*)pSrc + y * nSrcStep);
          if (wxMin > 0.0f && xMinInt > 0)
            sum += srcRow[(xMinInt-1)*channels + c] * wxMin;
          for (int x = xMinInt; x < xMaxInt; x++)
            sum += srcRow[x*channels + c];
          if (wxMax > 0.0f && xMaxInt < srcW)
            sum += srcRow[xMaxInt*channels + c] * wxMax;
        }

        // Bottom edge
        if (wyMax > 0.0f && yMaxInt < srcH) {
          int y = yMaxInt;
          const T* srcRow = (const T*)((const char*)pSrc + y * nSrcStep);
          if (wxMin > 0.0f && xMinInt > 0)
            sum += srcRow[(xMinInt-1)*channels + c] * wxMin * wyMax;
          for (int x = xMinInt; x < xMaxInt; x++)
            sum += srcRow[x*channels + c] * wyMax;
          if (wxMax > 0.0f && xMaxInt < srcW)
            sum += srcRow[xMaxInt*channels + c] * wxMax * wyMax;
        }

        float result = sum / (scaleX * scaleY);
        T* dstRow = (T*)((char*)pDst + dy * nDstStep);
        dstRow[dx*channels + c] = (T)(result + 0.5f);
      }
    }
  }
}

struct TestResult {
  int total, exact, within1, within2;
  int maxDiff;
  double avgAbsDiff;

  void compute(const unsigned char* cpu, const unsigned char* npp, int size) {
    total = size;
    exact = within1 = within2 = 0;
    maxDiff = 0;
    double sum = 0.0;

    for (int i = 0; i < size; i++) {
      int diff = std::abs((int)cpu[i] - (int)npp[i]);
      if (diff == 0) exact++;
      if (diff <= 1) within1++;
      if (diff <= 2) within2++;
      maxDiff = std::max(maxDiff, diff);
      sum += diff;
    }
    avgAbsDiff = sum / total;
  }

  const char* status() const {
    if (exact == total) return "PERFECT";
    if (within1 == total) return "EXCELLENT";
    if (within2 == total) return "GOOD";
    return "FAIR";
  }
};

void testShape(const char* name, int srcW, int srcH, int dstW, int dstH) {
  // Create gradient test pattern
  std::vector<unsigned char> srcData(srcW * srcH);
  for (int y = 0; y < srcH; y++) {
    for (int x = 0; x < srcW; x++) {
      int valX = (x * 255) / (srcW - 1);
      int valY = (y * 255) / (srcH - 1);
      srcData[y * srcW + x] = (valX + valY) / 2;
    }
  }

  // NPP
  unsigned char *d_src, *d_dst;
  CHECK_CUDA(cudaMalloc(&d_src, srcW * srcH));
  CHECK_CUDA(cudaMalloc(&d_dst, dstW * dstH));
  CHECK_CUDA(cudaMemcpy(d_src, srcData.data(), srcW * srcH, cudaMemcpyHostToDevice));

  NppiSize srcSize = {srcW, srcH};
  NppiRect srcROI = {0, 0, srcW, srcH};
  NppiSize dstSize = {dstW, dstH};
  NppiRect dstROI = {0, 0, dstW, dstH};

  CHECK_NPP(nppiResize_8u_C1R(d_src, srcW, srcSize, srcROI,
                              d_dst, dstW, dstSize, dstROI,
                              NPPI_INTER_SUPER));

  std::vector<unsigned char> nppResult(dstW * dstH);
  CHECK_CUDA(cudaMemcpy(nppResult.data(), d_dst, dstW * dstH, cudaMemcpyDeviceToHost));

  // CPU
  std::vector<unsigned char> cpuResult(dstW * dstH);
  cpuSuperSampling(srcData.data(), srcW, srcW, srcH,
                   cpuResult.data(), dstW, dstW, dstH, 1);

  // Compare
  TestResult res;
  res.compute(cpuResult.data(), nppResult.data(), dstW * dstH);

  float scaleX = (float)srcW / dstW;
  float scaleY = (float)srcH / dstH;

  printf("%-35s %5dx%-5d -> %4dx%-4d  scale(%4.2fx%4.2f)  exact:%5.1f%%  ±1:%5.1f%%  max:%d  %s\n",
         name, srcW, srcH, dstW, dstH, scaleX, scaleY,
         100.0*res.exact/res.total, 100.0*res.within1/res.total,
         res.maxDiff, res.status());

  CHECK_CUDA(cudaFree(d_src));
  CHECK_CUDA(cudaFree(d_dst));
}

int main() {
  std::cout << "=== Extensive Shape Combination Testing ===\n\n";
  printf("%-35s %13s    %11s  %-14s %-11s %s\n",
         "Test Name", "Source->Dest", "Scale(X,Y)", "Exact Match", "±1 Match", "Status");
  printf("%s\n", std::string(120, '-').c_str());

  // Category 1: Square images, integer scales
  std::cout << "\n[Category 1: Square, Integer Scales]\n";
  testShape("Square 2x", 128, 128, 64, 64);
  testShape("Square 3x", 180, 180, 60, 60);
  testShape("Square 4x", 256, 256, 64, 64);
  testShape("Square 8x", 512, 512, 64, 64);
  testShape("Square 16x", 512, 512, 32, 32);

  // Category 2: Square images, fractional scales
  std::cout << "\n[Category 2: Square, Fractional Scales]\n";
  testShape("Square 1.5x", 90, 90, 60, 60);
  testShape("Square 2.5x", 250, 250, 100, 100);
  testShape("Square 1.333x (128/96)", 128, 128, 96, 96);
  testShape("Square 1.667x (500/300)", 500, 500, 300, 300);
  testShape("Square 3.3x", 330, 330, 100, 100);
  testShape("Square 5.12x (512/100)", 512, 512, 100, 100);

  // Category 3: Landscape (width > height), integer scales
  std::cout << "\n[Category 3: Landscape, Integer Scales]\n";
  testShape("Landscape 2x (16:9)", 1920, 1080, 960, 540);
  testShape("Landscape 2x (4:3)", 640, 480, 320, 240);
  testShape("Landscape 4x (16:9)", 1280, 720, 320, 180);
  testShape("Landscape 2x (21:9)", 420, 180, 210, 90);

  // Category 4: Portrait (height > width), integer scales
  std::cout << "\n[Category 4: Portrait, Integer Scales]\n";
  testShape("Portrait 2x (9:16)", 540, 960, 270, 480);
  testShape("Portrait 3x (9:16)", 540, 960, 180, 320);
  testShape("Portrait 4x (3:4)", 480, 640, 120, 160);

  // Category 5: Landscape, fractional scales
  std::cout << "\n[Category 5: Landscape, Fractional Scales]\n";
  testShape("Landscape 1.5x (16:9)", 960, 540, 640, 360);
  testShape("Landscape 2.5x (4:3)", 500, 375, 200, 150);
  testShape("Landscape 1.333x (16:9)", 1280, 720, 960, 540);

  // Category 6: Portrait, fractional scales
  std::cout << "\n[Category 6: Portrait, Fractional Scales]\n";
  testShape("Portrait 1.5x (9:16)", 270, 480, 180, 320);
  testShape("Portrait 2.5x (3:4)", 300, 400, 120, 160);

  // Category 7: Different X and Y scales
  std::cout << "\n[Category 7: Different X/Y Scales]\n";
  testShape("Asymmetric 2x/3x", 256, 192, 128, 64);
  testShape("Asymmetric 4x/2x", 256, 128, 64, 64);
  testShape("Asymmetric 1.5x/2x", 300, 200, 200, 100);
  testShape("Asymmetric 2x/1.5x", 200, 150, 100, 100);

  // Category 8: Small images
  std::cout << "\n[Category 8: Small Images]\n";
  testShape("Small 16x16 -> 8x8", 16, 16, 8, 8);
  testShape("Small 32x24 -> 16x12", 32, 24, 16, 12);
  testShape("Small 64x48 -> 16x12", 64, 48, 16, 12);
  testShape("Small 100x75 -> 20x15", 100, 75, 20, 15);

  // Category 9: Very small dest
  std::cout << "\n[Category 9: Very Small Destination]\n";
  testShape("To 4x4", 128, 128, 4, 4);
  testShape("To 8x6", 256, 192, 8, 6);
  testShape("To 10x10", 500, 500, 10, 10);

  // Category 10: Extreme aspect ratios
  std::cout << "\n[Category 10: Extreme Aspect Ratios]\n";
  testShape("Ultra-wide 32:9 (2x)", 640, 90, 320, 45);
  testShape("Ultra-tall 9:32 (2x)", 90, 640, 45, 320);
  testShape("Panorama (4x)", 800, 100, 200, 25);

  // Category 11: Prime number dimensions
  std::cout << "\n[Category 11: Prime Number Dimensions]\n";
  testShape("Prime 127x127 -> 61x61", 127, 127, 61, 61);
  testShape("Prime 211x211 -> 101x101", 211, 211, 101, 101);
  testShape("Prime 97x67 -> 47x29", 97, 67, 47, 29);

  // Category 12: Powers of 2
  std::cout << "\n[Category 12: Powers of 2]\n";
  testShape("1024 -> 512", 1024, 1024, 512, 512);
  testShape("2048 -> 256", 2048, 2048, 256, 256);
  testShape("512x256 -> 256x128", 512, 256, 256, 128);

  // Category 13: Unusual fractional scales
  std::cout << "\n[Category 13: Unusual Fractional Scales]\n";
  testShape("Scale 1.111x (100/90)", 100, 100, 90, 90);
  testShape("Scale 1.234x (247/200)", 247, 247, 200, 200);
  testShape("Scale 7.5x (300/40)", 300, 300, 40, 40);

  std::cout << "\n" << std::string(120, '=') << "\n";
  std::cout << "Test complete. Check for:\n";
  std::cout << "  PERFECT: 100% exact match (ideal)\n";
  std::cout << "  EXCELLENT: 100% within ±1 (float precision limits)\n";
  std::cout << "  GOOD: 100% within ±2 (acceptable)\n";
  std::cout << "  FAIR: Some pixels >±2 (investigate)\n";
  std::cout << std::string(120, '=') << "\n";

  return 0;
}
