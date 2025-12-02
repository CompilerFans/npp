#include "npp_test_base.h"
#include <fstream>
#include <functional>
#include <memory>
#include <sys/stat.h>

using namespace npp_functional_test;

// Pattern generator interface for resize golden tests
class ResizePatternGenerator {
public:
  virtual ~ResizePatternGenerator() = default;
  virtual void generate(std::vector<Npp8u> &data, int width, int height, int channels) const = 0;
  virtual std::string getName() const = 0;
};

// Uniform block pattern
class UniformBlockGenerator : public ResizePatternGenerator {
  int blockSize_;

public:
  explicit UniformBlockGenerator(int blockSize = 32) : blockSize_(blockSize) {}

  void generate(std::vector<Npp8u> &data, int width, int height, int channels) const override {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (y * width + x) * channels;
        int blockX = x / blockSize_;
        int blockY = y / blockSize_;
        int blocksPerRow = (width + blockSize_ - 1) / blockSize_;
        int blockId = blockY * blocksPerRow + blockX;
        Npp8u value = (Npp8u)(blockId * 15 + 50);

        for (int c = 0; c < channels; c++) {
          data[idx + c] = value + c * 10;
        }
      }
    }
  }

  std::string getName() const override { return "UniformBlock"; }
};

// Gradient pattern
class GradientGenerator : public ResizePatternGenerator {
public:
  void generate(std::vector<Npp8u> &data, int width, int height, int channels) const override {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (y * width + x) * channels;
        if (channels >= 1)
          data[idx + 0] = (Npp8u)(x * 255 / std::max(1, width - 1));
        if (channels >= 2)
          data[idx + 1] = (Npp8u)(y * 255 / std::max(1, height - 1));
        if (channels >= 3)
          data[idx + 2] = (Npp8u)((x + y) * 255 / std::max(1, width + height - 2));
        for (int c = 3; c < channels; c++) {
          data[idx + c] = (Npp8u)(((x ^ y) * 255) / std::max(1, width * height));
        }
      }
    }
  }

  std::string getName() const override { return "Gradient"; }
};

// Checkerboard pattern
class CheckerboardGenerator : public ResizePatternGenerator {
  int checkSize_;
  Npp8u light_[3], dark_[3];

public:
  CheckerboardGenerator(int checkSize = 2) : checkSize_(checkSize) {
    light_[0] = 255;
    light_[1] = 200;
    light_[2] = 220;
    dark_[0] = 50;
    dark_[1] = 80;
    dark_[2] = 60;
  }

  void generate(std::vector<Npp8u> &data, int width, int height, int channels) const override {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (y * width + x) * channels;
        bool isLight = ((x / checkSize_) + (y / checkSize_)) % 2 == 0;
        const Npp8u *colors = isLight ? light_ : dark_;

        for (int c = 0; c < std::min(channels, 3); c++) {
          data[idx + c] = colors[c];
        }
        for (int c = 3; c < channels; c++) {
          data[idx + c] = 128;
        }
      }
    }
  }

  std::string getName() const override { return "Checkerboard"; }
};

// Random pattern
class RandomGenerator : public ResizePatternGenerator {
  unsigned int seed_;

public:
  explicit RandomGenerator(unsigned int seed = 12345) : seed_(seed) {}

  void generate(std::vector<Npp8u> &data, int width, int height, int channels) const override {
    unsigned int s = seed_;
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (y * width + x) * channels;
        for (int c = 0; c < channels; c++) {
          s = s * 1103515245 + 12345;
          data[idx + c] = (Npp8u)((s >> 16) & 0xFF);
        }
      }
    }
  }

  std::string getName() const override { return "Random"; }
};

// Multi-level pattern
class MultiLevelGenerator : public ResizePatternGenerator {
public:
  void generate(std::vector<Npp8u> &data, int width, int height, int channels) const override {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (y * width + x) * channels;
        if (channels >= 1)
          data[idx + 0] = (Npp8u)((x % 4) * 60 + 40);
        if (channels >= 2)
          data[idx + 1] = (Npp8u)((y % 8) * 30 + 50);
        if (channels >= 3)
          data[idx + 2] = (Npp8u)(((x / 16) + (y / 16)) * 40 + 80);
        for (int c = 3; c < channels; c++) {
          data[idx + c] = (Npp8u)(((x % 3) + (y % 5)) * 30 + 60);
        }
      }
    }
  }

  std::string getName() const override { return "MultiLevel"; }
};

class ResizeSuperGoldenTest : public NppTestBase {
protected:
  void SetUp() override { NppTestBase::SetUp(); }
  void TearDown() override { NppTestBase::TearDown(); }

  bool goldenFileExists(const std::string &filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
  }

  void saveGoldenData(const std::string &filename, const std::vector<Npp8u> &data) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
      ADD_FAILURE() << "Failed to create golden file: " << filename;
      return;
    }
    ofs.write(reinterpret_cast<const char *>(data.data()), data.size());
    ofs.close();
  }

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

  struct CompareResult {
    int totalPixels;
    int mismatchCount;
    int maxDiff;
    double avgDiff;
    bool passed;
  };

  CompareResult compareWithGolden(const std::vector<Npp8u> &result, const std::vector<Npp8u> &golden,
                                  const std::string &testName, int width, int height, int channels, int tolerance = 0) {
    CompareResult res = {};
    res.totalPixels = width * height * channels;
    res.mismatchCount = 0;
    res.maxDiff = 0;
    res.avgDiff = 0.0;
    res.passed = true;

    EXPECT_EQ(result.size(), golden.size()) << "Result size mismatch for " << testName;
    if (result.size() != golden.size()) {
      res.passed = false;
      return res;
    }

    int maxMismatchesToReport = 10;
    long long totalDiff = 0;

    for (int i = 0; i < res.totalPixels; i++) {
      int diff = std::abs((int)result[i] - (int)golden[i]);
      totalDiff += diff;

      if (diff > res.maxDiff) {
        res.maxDiff = diff;
      }

      if (diff > tolerance) {
        res.mismatchCount++;
        if (res.mismatchCount <= maxMismatchesToReport) {
          int pixelIdx = i / channels;
          int channel = i % channels;
          int y = pixelIdx / width;
          int x = pixelIdx % width;
          EXPECT_LE(diff, tolerance) << testName << " mismatch at pixel (" << x << "," << y << ") channel " << channel
                                     << ": got " << (int)result[i] << ", expected " << (int)golden[i]
                                     << ", diff=" << diff;
        }
      }
    }

    res.avgDiff = (double)totalDiff / res.totalPixels;
    res.passed = (res.mismatchCount == 0);

    if (res.mismatchCount > 0) {
      std::cout << testName << " comparison summary:\n"
                << "  Total pixels: " << res.totalPixels << "\n"
                << "  Mismatches: " << res.mismatchCount << " (" << (100.0 * res.mismatchCount / res.totalPixels)
                << "%)\n"
                << "  Max diff: " << res.maxDiff << "\n"
                << "  Avg diff: " << res.avgDiff << "\n"
                << "  Tolerance: " << tolerance << "\n";

      if (res.mismatchCount > maxMismatchesToReport) {
        ADD_FAILURE() << testName << ": Total " << res.mismatchCount << " mismatches (showing first "
                      << maxMismatchesToReport << ")";
      }
    }

    return res;
  }

  std::string getGoldenFilePath(const std::string &filename) {
#ifdef GOLDEN_DATA_DIR
    return std::string(GOLDEN_DATA_DIR) + "/" + filename;
#else
    return "test/golden/" + filename;
#endif
  }

  // Common test template
  void runSuperSamplingGoldenTest(const std::string &testName, int srcWidth, int srcHeight, int dstWidth, int dstHeight,
                                  int channels, const ResizePatternGenerator &generator, int tolerance = 0) {
    std::string goldenFile = getGoldenFilePath("resize_super_" + testName + "_c" + std::to_string(channels) + "r.bin");

    // Generate test data
    std::vector<Npp8u> srcData(srcWidth * srcHeight * channels);
    generator.generate(srcData, srcWidth, srcHeight, channels);

    // Allocate device memory
    NppImageMemory<Npp8u> src(srcWidth * channels, srcHeight);
    NppImageMemory<Npp8u> dst(dstWidth * channels, dstHeight);

    src.copyFromHost(srcData);

    // Setup NPP parameters
    NppiSize srcSize = {srcWidth, srcHeight};
    NppiSize dstSize = {dstWidth, dstHeight};
    NppiRect srcROI = {0, 0, srcWidth, srcHeight};
    NppiRect dstROI = {0, 0, dstWidth, dstHeight};

    // Run resize
    NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI, dst.get(), dst.step(), dstSize, dstROI,
                                         NPPI_INTER_SUPER);

    ASSERT_EQ(status, NPP_SUCCESS) << "nppiResize failed for " << testName;

    // Get result
    std::vector<Npp8u> resultData(dstWidth * dstHeight * channels);
    dst.copyToHost(resultData);

#ifdef USE_NVIDIA_NPP
    // Generate mode: save golden file
    if (!goldenFileExists(goldenFile)) {
      std::cout << "Generating golden file: " << goldenFile << "\n";
      saveGoldenData(goldenFile, resultData);
    }
#else
    // Validation mode: compare against golden
    if (!goldenFileExists(goldenFile)) {
      FAIL() << "Golden file not found: " << goldenFile << "\nRun with --use-nvidia-npp build first to generate it.";
    }

    std::vector<Npp8u> goldenData;
    ASSERT_TRUE(loadGoldenData(goldenFile, resultData.size(), goldenData))
        << "Failed to load golden data for " << testName;

    CompareResult res = compareWithGolden(resultData, goldenData, testName, dstWidth, dstHeight, channels, tolerance);
    EXPECT_TRUE(res.passed) << testName << ": Implementation differs from golden reference";

    if (res.passed) {
      std::cout << "  " << testName << ": Perfect match with golden reference ✓\n";
    }
#endif
  }
};

TEST_F(ResizeSuperGoldenTest, Super_8u_C3R_UniformBlock_4x_Golden) {
  UniformBlockGenerator gen(32);
  runSuperSamplingGoldenTest("uniformblock_4x", 128, 128, 32, 32, 3, gen);
}

TEST_F(ResizeSuperGoldenTest, Super_8u_C3R_Gradient_2x_Golden) {
  GradientGenerator gen;
  runSuperSamplingGoldenTest("gradient_2x", 64, 64, 32, 32, 3, gen);
}

TEST_F(ResizeSuperGoldenTest, Super_8u_C3R_Checkerboard_8x_Golden) {
  CheckerboardGenerator gen(2);
  runSuperSamplingGoldenTest("checkerboard_8x", 128, 128, 16, 16, 3, gen);
}

TEST_F(ResizeSuperGoldenTest, Super_8u_C3R_Random_4x_Golden) {
  RandomGenerator gen(12345);
  runSuperSamplingGoldenTest("random_4x", 80, 80, 20, 20, 3, gen);
}

TEST_F(ResizeSuperGoldenTest, Super_8u_C3R_MultiLevel_3x_Golden) {
  MultiLevelGenerator gen;
  runSuperSamplingGoldenTest("multilevel_3x", 96, 96, 32, 32, 3, gen);
}
