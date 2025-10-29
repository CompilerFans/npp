#include "npp_test_base.h"
#include <fstream>
#include <sys/stat.h>

using namespace npp_functional_test;

class ResizeSuperGoldenTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }
  void TearDown() override { NppTestBase::TearDown(); }

  // Check if golden file exists
  bool goldenFileExists(const std::string &filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
  }

  // Save golden reference data
  void saveGoldenData(const std::string &filename, const std::vector<Npp8u> &data) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
      FAIL() << "Failed to create golden file: " << filename;
    }
    ofs.write(reinterpret_cast<const char *>(data.data()), data.size());
    ofs.close();
  }

  // Load golden reference data
  bool loadGoldenData(const std::string &filename, size_t expectedSize, std::vector<Npp8u> &data) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
      ADD_FAILURE() << "Failed to open golden file: " << filename;
      return false;
    }

    ifs.seekg(0, std::ios::end);
    size_t fileSize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    if (fileSize != expectedSize) {
      ADD_FAILURE() << "Golden file size mismatch. Expected " << expectedSize << ", got " << fileSize;
      return false;
    }

    data.resize(expectedSize);
    ifs.read(reinterpret_cast<char *>(data.data()), expectedSize);
    ifs.close();

    return true;
  }

  // Compare with golden data (exact match)
  void compareWithGolden(const std::vector<Npp8u> &result, const std::vector<Npp8u> &golden,
                         const std::string &testName, int width, int height, int channels) {
    ASSERT_EQ(result.size(), golden.size()) << "Result size mismatch for " << testName;

    int totalPixels = width * height;
    int mismatchCount = 0;
    int maxMismatches = 10; // Report first 10 mismatches

    for (int i = 0; i < totalPixels * channels; i++) {
      if (result[i] != golden[i]) {
        mismatchCount++;
        if (mismatchCount <= maxMismatches) {
          int pixelIdx = i / channels;
          int channel = i % channels;
          int y = pixelIdx / width;
          int x = pixelIdx % width;
          EXPECT_EQ(result[i], golden[i]) << testName << " mismatch at pixel (" << x << "," << y << ") channel "
                                          << channel << ": got " << (int)result[i] << ", expected " << (int)golden[i];
        }
      }
    }

    if (mismatchCount > maxMismatches) {
      FAIL() << testName << ": Total " << mismatchCount << " pixel mismatches (showing first " << maxMismatches << ")";
    }

    EXPECT_EQ(mismatchCount, 0) << testName << ": Implementation differs from NVIDIA NPP golden reference";
  }

  // Helper to get full golden file path
  std::string getGoldenFilePath(const std::string &filename) {
#ifdef GOLDEN_DATA_DIR
    return std::string(GOLDEN_DATA_DIR) + "/" + filename;
#else
    return "test/golden/" + filename;
#endif
  }
};

// Test 1: Uniform block downsampling 4x
TEST_F(ResizeSuperGoldenTest, Super_8u_C3R_UniformBlock_4x_Golden) {
  const int srcWidth = 128, srcHeight = 128;
  const int dstWidth = 32, dstHeight = 32;
  const std::string goldenFile = getGoldenFilePath("resize_super_uniformblock_4x_c3r.bin");

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create 4x4 blocks with distinct values
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      int blockX = x / 32;
      int blockY = y / 32;
      int blockId = blockY * 4 + blockX;
      Npp8u value = (Npp8u)(blockId * 15 + 50);
      srcData[idx + 0] = value;
      srcData[idx + 1] = value + 10;
      srcData[idx + 2] = value + 20;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // When using MPP, compare against golden reference
  if (!goldenFileExists(goldenFile)) {
    FAIL() << "Golden file not found: " << goldenFile << ". Run with USE_NVIDIA_NPP=ON first to generate it.";
  }

  std::vector<Npp8u> goldenData;
  ASSERT_TRUE(loadGoldenData(goldenFile, resultData.size(), goldenData)) << "Failed to load golden data";
  compareWithGolden(resultData, goldenData, "UniformBlock_4x", dstWidth, dstHeight, 3);
}

// Test 2: Gradient pattern downsampling 2x
TEST_F(ResizeSuperGoldenTest, Super_8u_C3R_Gradient_2x_Golden) {
  const int srcWidth = 64, srcHeight = 64;
  const int dstWidth = 32, dstHeight = 32;
  const std::string goldenFile = getGoldenFilePath("resize_super_gradient_2x_c3r.bin");

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create gradient pattern
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = (Npp8u)(x * 255 / (srcWidth - 1));                   // R: horizontal gradient
      srcData[idx + 1] = (Npp8u)(y * 255 / (srcHeight - 1));                  // G: vertical gradient
      srcData[idx + 2] = (Npp8u)((x + y) * 255 / (srcWidth + srcHeight - 2)); // B: diagonal gradient
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  if (!goldenFileExists(goldenFile)) {
    FAIL() << "Golden file not found: " << goldenFile << ". Run with USE_NVIDIA_NPP=ON first to generate it.";
  }

  std::vector<Npp8u> goldenData;
  ASSERT_TRUE(loadGoldenData(goldenFile, resultData.size(), goldenData)) << "Failed to load golden data";
  compareWithGolden(resultData, goldenData, "Gradient_2x", dstWidth, dstHeight, 3);
}

// Test 3: Checkerboard pattern downsampling 8x
TEST_F(ResizeSuperGoldenTest, Super_8u_C3R_Checkerboard_8x_Golden) {
  const int srcWidth = 128, srcHeight = 128;
  const int dstWidth = 16, dstHeight = 16;
  const std::string goldenFile = getGoldenFilePath("resize_super_checkerboard_8x_c3r.bin");

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create fine checkerboard pattern
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      bool isWhite = ((x / 2) + (y / 2)) % 2 == 0;
      Npp8u r = isWhite ? 255 : 50;
      Npp8u g = isWhite ? 200 : 80;
      Npp8u b = isWhite ? 220 : 60;
      srcData[idx + 0] = r;
      srcData[idx + 1] = g;
      srcData[idx + 2] = b;
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  if (!goldenFileExists(goldenFile)) {
    FAIL() << "Golden file not found: " << goldenFile << ". Run with USE_NVIDIA_NPP=ON first to generate it.";
  }

  std::vector<Npp8u> goldenData;
  ASSERT_TRUE(loadGoldenData(goldenFile, resultData.size(), goldenData)) << "Failed to load golden data";
  compareWithGolden(resultData, goldenData, "Checkerboard_8x", dstWidth, dstHeight, 3);
}

// Test 4: Random pattern downsampling 4x
TEST_F(ResizeSuperGoldenTest, Super_8u_C3R_Random_4x_Golden) {
  const int srcWidth = 80, srcHeight = 80;
  const int dstWidth = 20, dstHeight = 20;
  const std::string goldenFile = getGoldenFilePath("resize_super_random_4x_c3r.bin");

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create deterministic pseudo-random pattern
  unsigned int seed = 12345;
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      seed = seed * 1103515245 + 12345; // Linear congruential generator
      srcData[idx + 0] = (Npp8u)((seed >> 16) & 0xFF);
      seed = seed * 1103515245 + 12345;
      srcData[idx + 1] = (Npp8u)((seed >> 16) & 0xFF);
      seed = seed * 1103515245 + 12345;
      srcData[idx + 2] = (Npp8u)((seed >> 16) & 0xFF);
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  if (!goldenFileExists(goldenFile)) {
    FAIL() << "Golden file not found: " << goldenFile << ". Run with USE_NVIDIA_NPP=ON first to generate it.";
  }

  std::vector<Npp8u> goldenData;
  ASSERT_TRUE(loadGoldenData(goldenFile, resultData.size(), goldenData)) << "Failed to load golden data";
  compareWithGolden(resultData, goldenData, "Random_4x", dstWidth, dstHeight, 3);
}

// Test 5: Complex multi-level pattern
TEST_F(ResizeSuperGoldenTest, Super_8u_C3R_MultiLevel_3x_Golden) {
  const int srcWidth = 96, srcHeight = 96;
  const int dstWidth = 32, dstHeight = 32;
  const std::string goldenFile = getGoldenFilePath("resize_super_multilevel_3x_c3r.bin");

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Create multi-level pattern with different frequencies
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      // R: high frequency
      srcData[idx + 0] = (Npp8u)((x % 4) * 60 + 40);
      // G: medium frequency
      srcData[idx + 1] = (Npp8u)((y % 8) * 30 + 50);
      // B: low frequency
      srcData[idx + 2] = (Npp8u)(((x / 16) + (y / 16)) * 40 + 80);
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  if (!goldenFileExists(goldenFile)) {
    FAIL() << "Golden file not found: " << goldenFile << ". Run with USE_NVIDIA_NPP=ON first to generate it.";
  }

  std::vector<Npp8u> goldenData;
  ASSERT_TRUE(loadGoldenData(goldenFile, resultData.size(), goldenData)) << "Failed to load golden data";
  compareWithGolden(resultData, goldenData, "MultiLevel_3x", dstWidth, dstHeight, 3);
}
