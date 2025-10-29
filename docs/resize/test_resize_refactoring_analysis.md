# Resize Test Code Refactoring Analysis

## Current Status

**File:** `test/unit/nppi/nppi_geometry_transforms/test_nppi_resize.cpp`
- **Lines of Code:** 2,139 lines
- **Number of Tests:** 43 tests
- **Average Test Length:** ~50 lines per test
- **Code Duplication:** Estimated 60-70%

## Key Issues

### 1. Repetitive Test Boilerplate

Every test follows the same pattern:

```cpp
// Pattern repeated 43 times:
const int srcWidth = X, srcHeight = Y;
const int dstWidth = A, dstHeight = B;

std::vector<Type> srcData(srcWidth * srcHeight * channels);
// ... fill data ...

NppImageMemory<Type> src(srcWidth * channels, srcHeight);
NppImageMemory<Type> dst(dstWidth * channels, dstHeight);
src.copyFromHost(srcData);

NppiSize srcSize = {srcWidth, srcHeight};
NppiSize dstSize = {dstWidth, dstHeight};
NppiRect srcROI = {0, 0, srcWidth, srcHeight};
NppiRect dstROI = {0, 0, dstWidth, dstHeight};

NppStatus status = nppiResize_XXX(...);
ASSERT_EQ(status, NPP_SUCCESS);

std::vector<Type> resultData(dstWidth * dstHeight * channels);
dst.copyToHost(resultData);
// ... validate ...
```

**Duplication Count:**
- Image memory setup: 43 times
- NppiSize/NppiRect setup: 43 times
- copyFromHost/copyToHost: 86 times
- Status assertion: 43 times

### 2. Common Test Data Patterns

Several test data patterns are used repeatedly:

1. **Checkerboard Pattern** - Used in 8+ tests
   ```cpp
   bool isWhite = ((x / size) + (y / size)) % 2 == 0;
   ```

2. **Gradient Pattern** - Used in 10+ tests
   ```cpp
   srcData[idx] = (Type)(x * 255 / (width - 1));  // Horizontal
   srcData[idx] = (Type)(y * 255 / (height - 1)); // Vertical
   ```

3. **Uniform Block** - Used in 6+ tests
   ```cpp
   if (x < threshold && y < threshold) value = A; else value = B;
   ```

4. **Random Pattern** - Used in 3+ tests
   ```cpp
   seed = seed * 1103515245 + 12345;
   ```

### 3. Common Validation Logic

Repeated validation patterns:

1. **Channel independence check** - Used in 4+ tests
2. **Gradient monotonicity check** - Used in 6+ tests
3. **Value range check** - Used in 10+ tests
4. **Averaging behavior check** - Used in 8+ tests

## Refactoring Strategy

### Phase 1: Extract Helper Functions (Conservative)

Create helper functions in the test fixture class without changing test structure.

#### 1.1 Test Setup Helpers

```cpp
class ResizeFunctionalTest : public NppTestBase {
protected:
  // Generic resize test runner
  template<typename T, int CHANNELS>
  void runResizeTest(
    int srcW, int srcH, int dstW, int dstH,
    int interpolation,
    std::function<void(std::vector<T>&)> fillData,
    std::function<void(const std::vector<T>&)> validate
  ) {
    std::vector<T> srcData(srcW * srcH * CHANNELS);
    fillData(srcData);

    NppImageMemory<T> src(srcW * CHANNELS, srcH);
    NppImageMemory<T> dst(dstW * CHANNELS, dstH);
    src.copyFromHost(srcData);

    NppiSize srcSize = {srcW, srcH};
    NppiSize dstSize = {dstW, dstH};
    NppiRect srcROI = {0, 0, srcW, srcH};
    NppiRect dstROI = {0, 0, dstW, dstH};

    NppStatus status = callResize<T, CHANNELS>(
      src, dst, srcSize, dstSize, srcROI, dstROI, interpolation);
    ASSERT_EQ(status, NPP_SUCCESS);

    std::vector<T> resultData(dstW * dstH * CHANNELS);
    dst.copyToHost(resultData);
    validate(resultData);
  }

private:
  template<typename T, int CHANNELS>
  NppStatus callResize(NppImageMemory<T>& src, NppImageMemory<T>& dst,
                       NppiSize srcSize, NppiSize dstSize,
                       NppiRect srcROI, NppiRect dstROI,
                       int interpolation);
};

// Specializations
template<>
NppStatus ResizeFunctionalTest::callResize<Npp8u, 1>(...) {
  return nppiResize_8u_C1R(...);
}

template<>
NppStatus ResizeFunctionalTest::callResize<Npp8u, 3>(...) {
  return nppiResize_8u_C3R(...);
}

// ... more specializations
```

#### 1.2 Data Generator Functions

```cpp
class ResizeFunctionalTest : public NppTestBase {
protected:
  // Generate checkerboard pattern
  template<typename T, int CHANNELS>
  void fillCheckerboard(std::vector<T>& data, int width, int height,
                        int blockSize, T white, T black) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (y * width + x) * CHANNELS;
        bool isWhite = ((x / blockSize) + (y / blockSize)) % 2 == 0;
        T value = isWhite ? white : black;
        for (int c = 0; c < CHANNELS; c++) {
          data[idx + c] = value;
        }
      }
    }
  }

  // Generate horizontal gradient
  template<typename T, int CHANNELS>
  void fillHorizontalGradient(std::vector<T>& data, int width, int height,
                               T minVal, T maxVal) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (y * width + x) * CHANNELS;
        T value = static_cast<T>(minVal + (maxVal - minVal) * x / (width - 1));
        for (int c = 0; c < CHANNELS; c++) {
          data[idx + c] = value;
        }
      }
    }
  }

  // Generate vertical gradient
  template<typename T, int CHANNELS>
  void fillVerticalGradient(std::vector<T>& data, int width, int height,
                            T minVal, T maxVal) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (y * width + x) * CHANNELS;
        T value = static_cast<T>(minVal + (maxVal - minVal) * y / (height - 1));
        for (int c = 0; c < CHANNELS; c++) {
          data[idx + c] = value;
        }
      }
    }
  }

  // Generate uniform block
  template<typename T, int CHANNELS>
  void fillUniformBlock(std::vector<T>& data, int width, int height,
                        int blockX, int blockY, int blockW, int blockH,
                        T insideVal, T outsideVal) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (y * width + x) * CHANNELS;
        bool inside = (x >= blockX && x < blockX + blockW &&
                      y >= blockY && y < blockY + blockH);
        T value = inside ? insideVal : outsideVal;
        for (int c = 0; c < CHANNELS; c++) {
          data[idx + c] = value;
        }
      }
    }
  }
};
```

#### 1.3 Validation Helper Functions

```cpp
class ResizeFunctionalTest : public NppTestBase {
protected:
  // Validate gradient monotonicity
  template<typename T, int CHANNELS>
  void validateHorizontalMonotonic(const std::vector<T>& data,
                                   int width, int height,
                                   int channel = 0) {
    for (int y = 0; y < height; y++) {
      for (int x = 1; x < width - 1; x++) {
        int idx = (y * width + x) * CHANNELS + channel;
        int prevIdx = (y * width + (x - 1)) * CHANNELS + channel;
        int nextIdx = (y * width + (x + 1)) * CHANNELS + channel;

        EXPECT_GE(data[idx], data[prevIdx])
          << "Not monotonic at (" << x << "," << y << ")";
        EXPECT_LE(data[idx], data[nextIdx])
          << "Not monotonic at (" << x << "," << y << ")";
      }
    }
  }

  // Validate channel independence
  template<typename T, int CHANNELS>
  void validateChannelValues(const std::vector<T>& data, int width, int height,
                             const T expected[CHANNELS], T tolerance) {
    for (int i = 0; i < width * height; i++) {
      for (int c = 0; c < CHANNELS; c++) {
        EXPECT_NEAR(data[i * CHANNELS + c], expected[c], tolerance)
          << "Channel " << c << " mismatch at pixel " << i;
      }
    }
  }

  // Validate value range
  template<typename T, int CHANNELS>
  void validateRange(const std::vector<T>& data, T minVal, T maxVal,
                     const std::string& message = "") {
    for (size_t i = 0; i < data.size(); i++) {
      EXPECT_GE(data[i], minVal) << message << " - too low at index " << i;
      EXPECT_LE(data[i], maxVal) << message << " - too high at index " << i;
    }
  }
};
```

### Phase 2: Refactor Existing Tests (Example)

**Before:**
```cpp
TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_ChannelIndependence) {
  const int srcWidth = 16, srcHeight = 16;
  const int dstWidth = 4, dstHeight = 4;

  std::vector<Npp8u> srcData(srcWidth * srcHeight * 3);

  // Fill each channel with different uniform values
  for (int y = 0; y < srcHeight; y++) {
    for (int x = 0; x < srcWidth; x++) {
      int idx = (y * srcWidth + x) * 3;
      srcData[idx + 0] = 80;  // R
      srcData[idx + 1] = 160; // G
      srcData[idx + 2] = 240; // B
    }
  }

  NppImageMemory<Npp8u> src(srcWidth * 3, srcHeight);
  NppImageMemory<Npp8u> dst(dstWidth * 3, dstHeight);

  src.copyFromHost(srcData);

  NppiSize srcSize = {srcWidth, srcHeight};
  NppiSize dstSize = {dstWidth, dstHeight};
  NppiRect srcROI = {0, 0, srcWidth, srcHeight};
  NppiRect dstROI = {0, 0, dstWidth, dstHeight};

  NppStatus status = nppiResize_8u_C3R(src.get(), src.step(), srcSize, srcROI,
                                       dst.get(), dst.step(), dstSize, dstROI,
                                       NPPI_INTER_SUPER);

  ASSERT_EQ(status, NPP_SUCCESS);

  std::vector<Npp8u> resultData(dstWidth * dstHeight * 3);
  dst.copyToHost(resultData);

  // Each channel should maintain its value independently
  for (int i = 0; i < dstWidth * dstHeight; i++) {
    EXPECT_NEAR(resultData[i * 3 + 0], 80, 1);
    EXPECT_NEAR(resultData[i * 3 + 1], 160, 1);
    EXPECT_NEAR(resultData[i * 3 + 2], 240, 1);
  }
}
```

**After (54 lines → 8 lines, 85% reduction):**
```cpp
TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_ChannelIndependence) {
  const Npp8u expected[3] = {80, 160, 240};

  runResizeTest<Npp8u, 3>(16, 16, 4, 4, NPPI_INTER_SUPER,
    [](auto& data) { fillUniform(data, {80, 160, 240}); },
    [&](const auto& result) { validateChannelValues(result, 4, 4, expected, 1); }
  );
}
```

### Phase 3: Parametrized Tests (Aggressive)

Use Google Test's `TEST_P` for similar tests:

```cpp
struct ResizeTestParams {
  int srcW, srcH, dstW, dstH;
  int interpolation;
  std::string testName;
};

class ResizeParametrizedTest : public ResizeFunctionalTest,
                                public ::testing::WithParamInterface<ResizeTestParams> {
};

TEST_P(ResizeParametrizedTest, Super_Checkerboard) {
  auto params = GetParam();

  runResizeTest<Npp8u, 3>(
    params.srcW, params.srcH, params.dstW, params.dstH,
    params.interpolation,
    [](auto& data) { fillCheckerboard(data, 4, 200, 50); },
    [](const auto& result) { validateRange(result, 50, 200); }
  );
}

INSTANTIATE_TEST_SUITE_P(
  SuperSampling,
  ResizeParametrizedTest,
  ::testing::Values(
    ResizeTestParams{64, 64, 32, 32, NPPI_INTER_SUPER, "2x"},
    ResizeTestParams{128, 128, 32, 32, NPPI_INTER_SUPER, "4x"},
    ResizeTestParams{100, 100, 30, 30, NPPI_INTER_SUPER, "NonInteger"}
  )
);
```

## Code Reduction Estimates

| Refactoring Phase | LOC Before | LOC After | Reduction |
|-------------------|------------|-----------|-----------|
| Phase 1: Helpers only | 2,139 | ~1,800 | 16% |
| Phase 2: Refactor tests | 2,139 | ~1,200 | 44% |
| Phase 3: Parametrize | 2,139 | ~800 | 63% |

### Detailed Savings Breakdown

**Phase 1 Helpers:**
- Helper functions: +300 lines
- Per-test savings: -8 lines × 43 tests = -344 lines
- Net: -344 + 300 = **-44 lines saved**

Wait, that doesn't match. Let me recalculate more carefully:

**Phase 2 Full Refactor:**
- Helper infrastructure: +400 lines
- Average test: 50 lines → 10 lines (80% reduction per test)
- 43 tests: 43 × 40 lines saved = 1,720 lines saved
- Net: **1,720 - 400 = 1,320 lines saved (62%)**

## Benefits

### Maintainability
- **Single source of truth** for test setup logic
- **Bug fixes apply to all tests** automatically
- **Easier to add new tests** - just define data and validation

### Readability
- **Test intent is clearer** - focus on what's being tested, not how
- **Less noise** - boilerplate hidden in helpers
- **Self-documenting** - helper names describe the pattern

### Extensibility
- **Easy to add new data patterns** - just add new generator function
- **Easy to add new validations** - just add new validator function
- **Easy to support new types** - template specializations

## Risks

### Complexity
- **Risk:** Helper functions add indirection
- **Mitigation:** Use clear naming, good documentation, lambda clarity

### Debuggability
- **Risk:** Stack traces go through helpers
- **Mitigation:** Keep helpers simple, add good error messages

### Test Independence
- **Risk:** Shared helpers could create dependencies
- **Mitigation:** Keep helpers pure functions, no state

## Migration Strategy

### Step 1: Add Helpers (No Changes to Tests)
1. Add helper functions to test fixture
2. Verify compilation
3. Run all tests to ensure no regression

### Step 2: Refactor One Test Group at a Time
1. Pick a group of similar tests (e.g., all Super_8u_C3R tests)
2. Refactor to use helpers
3. Run tests to verify behavior unchanged
4. Commit
5. Repeat for next group

### Step 3: Introduce Parametrized Tests (Optional)
1. Identify tests that differ only in parameters
2. Convert to TEST_P
3. Verify coverage is maintained
4. Remove redundant tests

## Recommendations

### Immediate Actions
1. ✅ **Create helper infrastructure** (Phase 1)
   - Low risk, immediate benefit
   - Can be done incrementally
   - Doesn't require changing existing tests initially

2. 🔄 **Refactor high-duplication groups** (Phase 2)
   - Target groups with 5+ similar tests
   - Start with Super sampling tests (most repetitive)
   - Gradual rollout

3. ⏸️ **Consider parametrization** (Phase 3)
   - Only for tests that are truly identical except parameters
   - May reduce test coverage visibility
   - Consider after Phase 2 is stable

### Priority Test Groups for Refactoring

1. **Super Sampling Tests** (12 tests) - Highest duplication
2. **LINEAR Interpolation Tests** (10 tests) - Similar structure
3. **NN Precision Tests** (8 tests) - Nearly identical
4. **Extended Type Tests** (3 tests) - Same pattern, different types

## Example Comparison

### Original Code (3 similar tests, 150 lines total)
```cpp
TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_2xDownscale) {
  // 50 lines of boilerplate + specific test logic
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_4xDownscale) {
  // 50 lines of boilerplate + specific test logic
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_NonIntegerScale) {
  // 50 lines of boilerplate + specific test logic
}
```

### Refactored Code (3 tests, 30 lines total + 50 lines shared helpers)
```cpp
// Shared helper (once for all tests): 50 lines
template<int CHANNELS>
void testSuperSampling(int srcW, int srcH, int dstW, int dstH,
                       DataGenerator gen, Validator val) { ... }

// Individual tests reduced to essence:
TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_2xDownscale) {
  testSuperSampling<3>(64, 64, 32, 32,
    checkerboard(4, 200, 50),
    rangeCheck(50, 200));
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_4xDownscale) {
  testSuperSampling<3>(128, 128, 32, 32,
    checkerboard(4, 200, 50),
    rangeCheck(50, 200));
}

TEST_F(ResizeFunctionalTest, Resize_8u_C3R_Super_NonIntegerScale) {
  testSuperSampling<3>(100, 100, 30, 30,
    gradientPattern(),
    monotonicity());
}
```

**Result:** 150 lines → 80 lines (47% reduction), with shared helpers reusable across all tests.

## Conclusion

The current test file has significant code duplication (~70%) that can be reduced through systematic refactoring. A phased approach starting with helper functions (Phase 1) provides immediate benefits with low risk, while more aggressive refactoring (Phases 2-3) can achieve 60%+ code reduction.

**Recommended Approach:** Implement Phase 1 helpers first, then selectively apply Phase 2 refactoring to the most repetitive test groups.
