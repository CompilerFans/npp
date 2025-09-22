#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>

#include "npp.h"

// Algorithm verification utilities based on reverse-engineered NPP behavior
class NPPFilterBoxAlgorithm {
public:
    // Calculate position weight based on NPP's boundary handling mechanism
    static float calculatePositionWeight(int x, int y, int imgW, int imgH, 
                                       int maskW, int maskH, int anchorX, int anchorY) {
        // Distance from boundaries
        int leftDist = std::min(x + 1, anchorX + 1);
        int rightDist = std::min(imgW - x, maskW - anchorX);
        int topDist = std::min(y + 1, anchorY + 1);
        int bottomDist = std::min(imgH - y, maskH - anchorY);
        
        // Coverage calculation
        int coveredPixels = leftDist * rightDist * topDist * bottomDist;
        int totalMaskPixels = maskW * maskH;
        
        // NPP's weight formula (reverse-engineered from test data)
        float baseWeight = (float)coveredPixels / totalMaskPixels;
        
        // Position-dependent correction factor
        float xCorrection = 1.0f;
        float yCorrection = 1.0f;
        
        // Edge position corrections based on empirical data
        if (x == 0 || x == imgW-1) xCorrection *= 0.89f;  // Corner correction
        if (y == 0 || y == imgH-1) yCorrection *= 0.89f;  // Corner correction
        
        return baseWeight * xCorrection * yCorrection;
    }
    
    // Predict NPP FilterBox output for a single pixel
    static float predictPixelValue(const std::vector<uint8_t>& input, int x, int y, 
                                 int imgW, int imgH, int maskW, int maskH, 
                                 int anchorX, int anchorY) {
        float sum = 0.0f;
        int validPixels = 0;
        
        // Calculate coverage area
        int startX = x - anchorX;
        int startY = y - anchorY;
        
        for (int my = 0; my < maskH; my++) {
            for (int mx = 0; mx < maskW; mx++) {
                int pixelX = startX + mx;
                int pixelY = startY + my;
                
                if (pixelX >= 0 && pixelX < imgW && pixelY >= 0 && pixelY < imgH) {
                    sum += input[pixelY * imgW + pixelX];
                    validPixels++;
                }
            }
        }
        
        if (validPixels == 0) return 0.0f;
        
        // NPP's algorithm: weighted average with position correction
        float baseResult = sum / validPixels;
        float positionWeight = calculatePositionWeight(x, y, imgW, imgH, maskW, maskH, anchorX, anchorY);
        
        return baseResult * positionWeight;
    }
    
    // Generate predicted output for entire image
    static std::vector<uint8_t> predictFullOutput(const std::vector<uint8_t>& input, 
                                                 int imgW, int imgH, int maskW, int maskH,
                                                 int anchorX, int anchorY) {
        std::vector<uint8_t> predicted(imgW * imgH);
        
        for (int y = 0; y < imgH; y++) {
            for (int x = 0; x < imgW; x++) {
                float value = predictPixelValue(input, x, y, imgW, imgH, maskW, maskH, anchorX, anchorY);
                predicted[y * imgW + x] = static_cast<uint8_t>(std::round(value));
            }
        }
        
        return predicted;
    }
};

// Test case generator for algorithm verification
class AlgorithmTestGenerator {
public:
    // Generate uniform pattern for weight verification
    static std::vector<uint8_t> generateUniformPattern(int width, int height, uint8_t value) {
        return std::vector<uint8_t>(width * height, value);
    }
    
    // Generate impulse pattern for response analysis
    static std::vector<uint8_t> generateImpulsePattern(int width, int height, 
                                                      int impulseX, int impulseY, uint8_t impulseValue) {
        std::vector<uint8_t> pattern(width * height, 0);
        if (impulseX >= 0 && impulseX < width && impulseY >= 0 && impulseY < height) {
            pattern[impulseY * width + impulseX] = impulseValue;
        }
        return pattern;
    }
    
    // Generate binary step pattern for edge analysis
    static std::vector<uint8_t> generateBinaryStepPattern(int width, int height, 
                                                         uint8_t leftValue, uint8_t rightValue) {
        std::vector<uint8_t> pattern(width * height);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                pattern[y * width + x] = (x < width / 2) ? leftValue : rightValue;
            }
        }
        return pattern;
    }
    
    // Generate checkerboard pattern for complex boundary testing
    static std::vector<uint8_t> generateCheckerboardPattern(int width, int height, 
                                                           uint8_t value1, uint8_t value2) {
        std::vector<uint8_t> pattern(width * height);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                pattern[y * width + x] = ((x + y) % 2 == 0) ? value1 : value2;
            }
        }
        return pattern;
    }
};

// Traditional border replication algorithm for comparison
class BorderReplicationAlgorithm {
public:
    // Standard border replication FilterBox implementation
    static std::vector<uint8_t> applyBorderReplication(const std::vector<uint8_t>& input,
                                                      int imgW, int imgH, int maskW, int maskH,
                                                      int anchorX, int anchorY) {
        std::vector<uint8_t> output(imgW * imgH);
        
        for (int y = 0; y < imgH; y++) {
            for (int x = 0; x < imgW; x++) {
                float sum = 0.0f;
                
                for (int my = 0; my < maskH; my++) {
                    for (int mx = 0; mx < maskW; mx++) {
                        int pixelX = x - anchorX + mx;
                        int pixelY = y - anchorY + my;
                        
                        // Border replication: clamp coordinates to image bounds
                        pixelX = std::max(0, std::min(imgW - 1, pixelX));
                        pixelY = std::max(0, std::min(imgH - 1, pixelY));
                        
                        sum += input[pixelY * imgW + pixelX];
                    }
                }
                
                output[y * imgW + x] = static_cast<uint8_t>(std::round(sum / (maskW * maskH)));
            }
        }
        
        return output;
    }
};

// Test verification utilities
class AlgorithmVerification {
public:
    // Compare predicted vs actual results
    static bool verifyPrediction(const std::vector<uint8_t>& predicted, 
                                const std::vector<uint8_t>& actual, 
                                int tolerance = 2) {
        if (predicted.size() != actual.size()) return false;
        
        int totalErrors = 0;
        int maxError = 0;
        
        for (size_t i = 0; i < predicted.size(); i++) {
            int error = std::abs(static_cast<int>(predicted[i]) - static_cast<int>(actual[i]));
            if (error > tolerance) {
                totalErrors++;
            }
            maxError = std::max(maxError, error);
        }
        
        float errorRate = (float)totalErrors / predicted.size();
        
        std::cout << "Verification Results:\n";
        std::cout << "  Total pixels: " << predicted.size() << "\n";
        std::cout << "  Errors (>" << tolerance << "): " << totalErrors << "\n";
        std::cout << "  Error rate: " << std::fixed << std::setprecision(2) << (errorRate * 100) << "%\n";
        std::cout << "  Max error: " << maxError << "\n";
        
        return errorRate < 0.05f; // Less than 5% error rate acceptable
    }
    
    // Calculate difference statistics between two results
    static void calculateDifferenceStats(const std::vector<uint8_t>& result1,
                                       const std::vector<uint8_t>& result2,
                                       const std::string& name1,
                                       const std::string& name2) {
        if (result1.size() != result2.size()) {
            std::cout << "Size mismatch in difference calculation\n";
            return;
        }
        
        int totalDiff = 0;
        int maxDiff = 0;
        int minDiff = 255;
        int zeroDiffs = 0;
        std::vector<int> diffCounts(256, 0);
        
        for (size_t i = 0; i < result1.size(); i++) {
            int diff = std::abs(static_cast<int>(result1[i]) - static_cast<int>(result2[i]));
            totalDiff += diff;
            maxDiff = std::max(maxDiff, diff);
            minDiff = std::min(minDiff, diff);
            
            if (diff == 0) zeroDiffs++;
            diffCounts[diff]++;
        }
        
        float avgDiff = (float)totalDiff / result1.size();
        float exactMatchRate = (float)zeroDiffs / result1.size() * 100;
        
        std::cout << "\n=== Difference Analysis: " << name1 << " vs " << name2 << " ===\n";
        std::cout << "Average difference: " << std::fixed << std::setprecision(2) << avgDiff << " pixels\n";
        std::cout << "Max difference: " << maxDiff << " pixels\n";
        std::cout << "Min difference: " << minDiff << " pixels\n";
        std::cout << "Exact matches: " << zeroDiffs << "/" << result1.size() 
                  << " (" << exactMatchRate << "%)\n";
        
        // Show difference distribution
        std::cout << "Difference distribution:\n";
        for (int d = 0; d <= maxDiff && d < 20; d++) {
            if (diffCounts[d] > 0) {
                float percent = (float)diffCounts[d] / result1.size() * 100;
                std::cout << "  Diff " << std::setw(2) << d << ": " 
                          << std::setw(4) << diffCounts[d] << " pixels (" 
                          << std::setw(5) << std::setprecision(1) << percent << "%)\n";
            }
        }
    }
    
    // Print detailed comparison matrix
    static void printComparison(const std::vector<uint8_t>& predicted, 
                               const std::vector<uint8_t>& actual, 
                               int width, int height) {
        std::cout << "\nPredicted vs Actual (format: P/A, diff):\n";
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                int pred = predicted[idx];
                int act = actual[idx];
                int diff = pred - act;
                
                std::cout << std::setw(2) << pred << "/" << std::setw(2) << act;
                if (diff != 0) {
                    std::cout << "(" << std::showpos << diff << std::noshowpos << ")";
                } else {
                    std::cout << "    ";
                }
                std::cout << " ";
            }
            std::cout << "\n";
        }
    }
};

// Test class for algorithm verification
class FilterBoxAlgorithmTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA context if needed
    }
    
    // Helper function to run NPP FilterBox and get result
    std::vector<uint8_t> runNPPFilterBox(const std::vector<uint8_t>& input, 
                                        int width, int height, int maskW, int maskH,
                                        int anchorX, int anchorY) {
        // Allocate device memory
        Npp8u* d_src = (Npp8u*)nppsMalloc_8u(width * height);
        Npp8u* d_dst = (Npp8u*)nppsMalloc_8u(width * height);
        EXPECT_NE(d_src, nullptr);
        EXPECT_NE(d_dst, nullptr);
        
        // Copy to device
        cudaMemcpy(d_src, input.data(), input.size() * sizeof(Npp8u), cudaMemcpyHostToDevice);
        
        // Apply filter
        NppStatus status = nppiFilterBox_8u_C1R(
            d_src, width * sizeof(Npp8u),
            d_dst, width * sizeof(Npp8u),
            {width, height},
            {maskW, maskH},
            {anchorX, anchorY}
        );
        EXPECT_EQ(status, NPP_SUCCESS);
        
        // Copy result back
        std::vector<uint8_t> output(width * height);
        cudaMemcpy(output.data(), d_dst, output.size() * sizeof(Npp8u), cudaMemcpyDeviceToHost);
        
        nppiFree(d_src);
        nppiFree(d_dst);
        
        return output;
    }
};

// Algorithm verification tests
TEST_F(FilterBoxAlgorithmTest, UniformPatternWeightVerification) {
    std::cout << "\n=== Uniform Pattern Weight Verification ===\n";
    
    const int width = 6, height = 6;
    const int maskW = 3, maskH = 3;
    const int anchorX = 1, anchorY = 1;
    
    auto input = AlgorithmTestGenerator::generateUniformPattern(width, height, 128);
    auto actual = runNPPFilterBox(input, width, height, maskW, maskH, anchorX, anchorY);
    auto predicted = NPPFilterBoxAlgorithm::predictFullOutput(input, width, height, maskW, maskH, anchorX, anchorY);
    
    std::cout << "Test configuration: " << width << "x" << height << " image, " 
              << maskW << "x" << maskH << " mask, anchor(" << anchorX << "," << anchorY << ")\n";
    
    AlgorithmVerification::printComparison(predicted, actual, width, height);
    EXPECT_TRUE(AlgorithmVerification::verifyPrediction(predicted, actual, 3));
}

TEST_F(FilterBoxAlgorithmTest, ImpulseResponseVerification) {
    std::cout << "\n=== Impulse Response Verification ===\n";
    
    const int width = 5, height = 5;
    const int maskW = 3, maskH = 3;
    const int anchorX = 1, anchorY = 1;
    
    auto input = AlgorithmTestGenerator::generateImpulsePattern(width, height, 0, 0, 255);
    auto actual = runNPPFilterBox(input, width, height, maskW, maskH, anchorX, anchorY);
    auto predicted = NPPFilterBoxAlgorithm::predictFullOutput(input, width, height, maskW, maskH, anchorX, anchorY);
    
    std::cout << "Corner impulse (255 at [0,0]) response:\n";
    AlgorithmVerification::printComparison(predicted, actual, width, height);
    EXPECT_TRUE(AlgorithmVerification::verifyPrediction(predicted, actual, 3));
}

TEST_F(FilterBoxAlgorithmTest, BinaryStepResponseVerification) {
    std::cout << "\n=== Binary Step Response Verification ===\n";
    
    const int width = 6, height = 4;
    const int maskW = 3, maskH = 3;
    const int anchorX = 1, anchorY = 1;
    
    auto input = AlgorithmTestGenerator::generateBinaryStepPattern(width, height, 0, 255);
    auto actual = runNPPFilterBox(input, width, height, maskW, maskH, anchorX, anchorY);
    auto predicted = NPPFilterBoxAlgorithm::predictFullOutput(input, width, height, maskW, maskH, anchorX, anchorY);
    
    std::cout << "Binary step (0|255) edge processing:\n";
    AlgorithmVerification::printComparison(predicted, actual, width, height);
    EXPECT_TRUE(AlgorithmVerification::verifyPrediction(predicted, actual, 5));
}

TEST_F(FilterBoxAlgorithmTest, DifferentAnchorPositionsVerification) {
    std::cout << "\n=== Different Anchor Positions Verification ===\n";
    
    const int width = 4, height = 4;
    const int maskW = 3, maskH = 3;
    
    auto input = AlgorithmTestGenerator::generateUniformPattern(width, height, 100);
    
    // Test different anchor positions
    std::vector<std::pair<int, int>> anchors = {{0,0}, {1,1}, {2,2}};
    
    for (auto [anchorX, anchorY] : anchors) {
        std::cout << "\nAnchor position (" << anchorX << "," << anchorY << "):\n";
        
        auto actual = runNPPFilterBox(input, width, height, maskW, maskH, anchorX, anchorY);
        auto predicted = NPPFilterBoxAlgorithm::predictFullOutput(input, width, height, maskW, maskH, anchorX, anchorY);
        
        AlgorithmVerification::printComparison(predicted, actual, width, height);
        EXPECT_TRUE(AlgorithmVerification::verifyPrediction(predicted, actual, 3));
    }
}

TEST_F(FilterBoxAlgorithmTest, DifferentMaskSizesVerification) {
    std::cout << "\n=== Different Mask Sizes Verification ===\n";
    
    const int width = 6, height = 6;
    auto input = AlgorithmTestGenerator::generateCheckerboardPattern(width, height, 50, 200);
    
    // Test different mask sizes
    std::vector<std::tuple<int, int, int, int>> masks = {
        {3, 3, 1, 1},  // 3x3 center
        {5, 5, 2, 2},  // 5x5 center
        {3, 1, 1, 0},  // 3x1 horizontal
        {1, 3, 0, 1}   // 1x3 vertical
    };
    
    for (auto [maskW, maskH, anchorX, anchorY] : masks) {
        std::cout << "\nMask " << maskW << "x" << maskH << ", anchor(" << anchorX << "," << anchorY << "):\n";
        
        auto actual = runNPPFilterBox(input, width, height, maskW, maskH, anchorX, anchorY);
        auto predicted = NPPFilterBoxAlgorithm::predictFullOutput(input, width, height, maskW, maskH, anchorX, anchorY);
        
        // Allow higher tolerance for complex patterns
        EXPECT_TRUE(AlgorithmVerification::verifyPrediction(predicted, actual, 5));
    }
}

TEST_F(FilterBoxAlgorithmTest, EdgeCaseSizeVerification) {
    std::cout << "\n=== Edge Case Size Verification ===\n";
    
    // Test image size equals mask size
    const int width = 3, height = 3;
    const int maskW = 3, maskH = 3;
    const int anchorX = 1, anchorY = 1;
    
    auto input = AlgorithmTestGenerator::generateBinaryStepPattern(width, height, 64, 192);
    auto actual = runNPPFilterBox(input, width, height, maskW, maskH, anchorX, anchorY);
    auto predicted = NPPFilterBoxAlgorithm::predictFullOutput(input, width, height, maskW, maskH, anchorX, anchorY);
    
    std::cout << "3x3 image with 3x3 mask (exact coverage):\n";
    AlgorithmVerification::printComparison(predicted, actual, width, height);
    EXPECT_TRUE(AlgorithmVerification::verifyPrediction(predicted, actual, 3));
}

// Border replication vs NPP comparison tests
TEST_F(FilterBoxAlgorithmTest, BorderReplicationVsNPPComparison) {
    std::cout << "\n=== Border Replication vs NPP Comparison ===\n";
    
    struct TestConfig {
        int width, height, maskW, maskH, anchorX, anchorY;
        std::string description;
    };
    
    std::vector<TestConfig> configs = {
        {4, 4, 3, 3, 1, 1, "4x4_3x3_center"},
        {6, 6, 3, 3, 1, 1, "6x6_3x3_center"},
        {8, 8, 5, 5, 2, 2, "8x8_5x5_center"},
        {5, 5, 5, 5, 2, 2, "5x5_5x5_exact"}
    };
    
    for (const auto& config : configs) {
        std::cout << "\n--- Configuration: " << config.description << " ---\n";
        
        // Test uniform input
        auto uniformInput = AlgorithmTestGenerator::generateUniformPattern(
            config.width, config.height, 128);
        
        auto nppResult = runNPPFilterBox(uniformInput, config.width, config.height, 
                                        config.maskW, config.maskH, config.anchorX, config.anchorY);
        auto borderRepResult = BorderReplicationAlgorithm::applyBorderReplication(
            uniformInput, config.width, config.height, 
            config.maskW, config.maskH, config.anchorX, config.anchorY);
        
        std::cout << "Uniform input (value=128):\n";
        AlgorithmVerification::calculateDifferenceStats(nppResult, borderRepResult, "NPP", "BorderReplication");
        
        // Test binary step input
        auto stepInput = AlgorithmTestGenerator::generateBinaryStepPattern(
            config.width, config.height, 0, 255);
        
        nppResult = runNPPFilterBox(stepInput, config.width, config.height, 
                                   config.maskW, config.maskH, config.anchorX, config.anchorY);
        borderRepResult = BorderReplicationAlgorithm::applyBorderReplication(
            stepInput, config.width, config.height, 
            config.maskW, config.maskH, config.anchorX, config.anchorY);
        
        std::cout << "\nBinary step input (0|255):\n";
        AlgorithmVerification::calculateDifferenceStats(nppResult, borderRepResult, "NPP", "BorderReplication");
        
        // Test impulse input
        auto impulseInput = AlgorithmTestGenerator::generateImpulsePattern(
            config.width, config.height, 0, 0, 255);
        
        nppResult = runNPPFilterBox(impulseInput, config.width, config.height, 
                                   config.maskW, config.maskH, config.anchorX, config.anchorY);
        borderRepResult = BorderReplicationAlgorithm::applyBorderReplication(
            impulseInput, config.width, config.height, 
            config.maskW, config.maskH, config.anchorX, config.anchorY);
        
        std::cout << "\nImpulse input (255 at [0,0]):\n";
        AlgorithmVerification::calculateDifferenceStats(nppResult, borderRepResult, "NPP", "BorderReplication");
    }
}

TEST_F(FilterBoxAlgorithmTest, ComprehensiveBorderReplicationAnalysis) {
    std::cout << "\n=== Comprehensive Border Replication Analysis ===\n";
    
    // Large scale analysis
    const int width = 20, height = 15;
    const int maskW = 5, maskH = 5;
    const int anchorX = 2, anchorY = 2;
    
    // Generate complex test pattern
    std::vector<uint8_t> complexInput(width * height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Create a pattern with gradients and edges
            int value = (x * 255 / width + y * 255 / height) / 2;
            if ((x + y) % 3 == 0) value = 255 - value; // Add some noise
            complexInput[y * width + x] = static_cast<uint8_t>(value);
        }
    }
    
    auto nppResult = runNPPFilterBox(complexInput, width, height, maskW, maskH, anchorX, anchorY);
    auto borderRepResult = BorderReplicationAlgorithm::applyBorderReplication(
        complexInput, width, height, maskW, maskH, anchorX, anchorY);
    
    std::cout << "Complex pattern analysis (" << width << "x" << height << "):\n";
    AlgorithmVerification::calculateDifferenceStats(nppResult, borderRepResult, "NPP", "BorderReplication");
    
    // Analyze boundary regions specifically
    int boundaryPixels = 0;
    int interiorPixels = 0;
    int boundaryTotalDiff = 0;
    int interiorTotalDiff = 0;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            int diff = std::abs(static_cast<int>(nppResult[idx]) - static_cast<int>(borderRepResult[idx]));
            
            // Determine if pixel is on boundary (affected by border handling)
            bool isBoundary = (x < anchorX) || (x >= width - (maskW - anchorX - 1)) ||
                             (y < anchorY) || (y >= height - (maskH - anchorY - 1));
            
            if (isBoundary) {
                boundaryPixels++;
                boundaryTotalDiff += diff;
            } else {
                interiorPixels++;
                interiorTotalDiff += diff;
            }
        }
    }
    
    float avgBoundaryDiff = boundaryPixels > 0 ? (float)boundaryTotalDiff / boundaryPixels : 0.0f;
    float avgInteriorDiff = interiorPixels > 0 ? (float)interiorTotalDiff / interiorPixels : 0.0f;
    
    std::cout << "\nBoundary vs Interior analysis:\n";
    std::cout << "Boundary pixels: " << boundaryPixels << ", avg diff: " << avgBoundaryDiff << "\n";
    std::cout << "Interior pixels: " << interiorPixels << ", avg diff: " << avgInteriorDiff << "\n";
    std::cout << "Boundary impact ratio: " << (avgBoundaryDiff / (avgInteriorDiff + 0.001f)) << "x\n";
}
