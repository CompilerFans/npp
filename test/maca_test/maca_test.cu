/*
 * NPP User Test Program for MACA Platform
 *
 * This is a typical user program that uses NPP API.
 * It demonstrates common image processing operations using:
 * - NPP API (npp.h, nppi.h, npps.h)
 * - CUDA Runtime API (cuda_runtime.h)
 *
 * On MACA platform, these APIs are transparently mapped to:
 * - NPP -> MPP (via cu-bridge wrapper headers)
 * - CUDA Runtime -> MACA Runtime (via cu-bridge)
 *
 * Compile with cucc (cu-bridge compiler wrapper):
 *   cucc test_npp_user.cu -o test_npp_user -lnppial
 *
 * Run:
 *   ./test_npp_user
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <npps.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            return; \
        } \
    } while(0)

static int total_tests = 0;
static int passed_tests = 0;

static void test_assert(int condition, const char* test_name) {
    total_tests++;
    if (condition) {
        printf("  [PASS] %s\n", test_name);
        passed_tests++;
    } else {
        printf("  [FAIL] %s\n", test_name);
    }
}

/*
 * Test 1: Get library version
 */
void test_get_version(void) {
    const NppLibraryVersion* ver;

    printf("\n[Test 1] nppGetLibVersion\n");

    ver = nppGetLibVersion();
    test_assert(ver != NULL, "Version pointer is valid");

    if (ver) {
        printf("  Library version: %d.%d.%d\n",
               ver->major, ver->minor, ver->build);
    }
}

/*
 * Test 2: Image memory allocation and deallocation
 */
void test_image_malloc(void) {
    Npp8u* pImage;
    int nStep;
    int width = 1920;
    int height = 1080;

    printf("\n[Test 2] nppiMalloc / nppiFree\n");

    pImage = nppiMalloc_8u_C1(width, height, &nStep);
    test_assert(pImage != NULL, "nppiMalloc_8u_C1 returns valid pointer");
    test_assert(nStep >= width, "Step is at least image width");

    if (pImage) {
        printf("  Allocated %dx%d image, step=%d bytes\n", width, height, nStep);
        nppiFree(pImage);
        printf("  Memory freed successfully\n");
    }
}

/*
 * Test 3: Copy image
 */
void test_image_copy(void) {
    Npp8u* pSrc;
    Npp8u* pDst;
    int nSrcStep, nDstStep;
    NppiSize oSizeROI;
    NppStatus status;
    int width = 32;
    int height = 32;
    int i, correct;

    printf("\n[Test 3] nppiCopy_8u_C1R\n");

    pSrc = nppiMalloc_8u_C1(width, height, &nSrcStep);
    pDst = nppiMalloc_8u_C1(width, height, &nDstStep);
    test_assert(pSrc != NULL && pDst != NULL, "Memory allocation");

    if (!pSrc || !pDst) {
        if (pSrc) nppiFree(pSrc);
        if (pDst) nppiFree(pDst);
        return;
    }

    oSizeROI.width = width;
    oSizeROI.height = height;

    /* Initialize source with pattern using cudaMemcpy */
    size_t bufSize = (size_t)nSrcStep * height;
    Npp8u* hSrc = (Npp8u*)malloc(bufSize);
    Npp8u* hResult = (Npp8u*)malloc(bufSize);
    if (!hSrc || !hResult) {
        free(hSrc); free(hResult);
        nppiFree(pSrc); nppiFree(pDst);
        return;
    }

    memset(hSrc, 200, bufSize);
    CUDA_CHECK(cudaMemcpy(pSrc, hSrc, bufSize, cudaMemcpyHostToDevice));

    /* Copy source to destination */
    status = nppiCopy_8u_C1R(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
    test_assert(status == NPP_SUCCESS, "nppiCopy_8u_C1R returns NPP_SUCCESS");

    cudaDeviceSynchronize();

    /* Verify */
    CUDA_CHECK(cudaMemcpy(hResult, pDst, bufSize, cudaMemcpyDeviceToHost));

    correct = 1;
    for (i = 0; i < width * height && correct; i++) {
        if (hResult[(i / width) * nDstStep + (i % width)] != 200) {
            correct = 0;
        }
    }
    test_assert(correct, "Copied data matches source");

    free(hSrc);
    free(hResult);
    nppiFree(pSrc);
    nppiFree(pDst);
}

/*
 * Test 4: Add constant to image
 */
void test_image_addc(void) {
    Npp8u* pSrc;
    Npp8u* pDst;
    int nSrcStep, nDstStep;
    NppiSize oSizeROI;
    NppStatus status;
    int width = 32;
    int height = 32;
    Npp8u initValue = 100;
    Npp8u addValue = 50;
    int nScaleFactor = 0;
    int i, correct;

    printf("\n[Test 4] nppiAddC_8u_C1RSfs\n");

    pSrc = nppiMalloc_8u_C1(width, height, &nSrcStep);
    pDst = nppiMalloc_8u_C1(width, height, &nDstStep);

    if (!pSrc || !pDst) {
        printf("  [SKIP] Memory allocation failed\n");
        if (pSrc) nppiFree(pSrc);
        if (pDst) nppiFree(pDst);
        return;
    }

    oSizeROI.width = width;
    oSizeROI.height = height;

    /* Initialize source using cudaMemcpy */
    size_t bufSize = (size_t)nSrcStep * height;
    Npp8u* hBuf = (Npp8u*)malloc(bufSize);
    if (!hBuf) {
        nppiFree(pSrc); nppiFree(pDst);
        return;
    }

    memset(hBuf, initValue, bufSize);
    CUDA_CHECK(cudaMemcpy(pSrc, hBuf, bufSize, cudaMemcpyHostToDevice));

    /* Add constant */
    status = nppiAddC_8u_C1RSfs(pSrc, nSrcStep, addValue, pDst, nDstStep,
                                 oSizeROI, nScaleFactor);
    test_assert(status == NPP_SUCCESS, "nppiAddC_8u_C1RSfs returns NPP_SUCCESS");

    cudaDeviceSynchronize();

    /* Verify: result should be initValue + addValue = 150 */
    CUDA_CHECK(cudaMemcpy(hBuf, pDst, bufSize, cudaMemcpyDeviceToHost));

    correct = 1;
    for (i = 0; i < width * height && correct; i++) {
        Npp8u expected = initValue + addValue;
        Npp8u actual = hBuf[(i / width) * nDstStep + (i % width)];
        if (actual != expected) {
            correct = 0;
            printf("  Mismatch at %d: expected %d, got %d\n", i, expected, actual);
        }
    }
    test_assert(correct, "Result = src + constant");

    free(hBuf);
    nppiFree(pSrc);
    nppiFree(pDst);
}

/*
 * Test 5: Image multiplication
 */
void test_image_mul(void) {
    Npp8u* pSrc1;
    Npp8u* pSrc2;
    Npp8u* pDst;
    int nStep;
    NppiSize oSizeROI;
    NppStatus status;
    int width = 16;
    int height = 16;
    int nScaleFactor = 8;
    int i, correct;

    printf("\n[Test 5] nppiMul_8u_C1RSfs\n");

    pSrc1 = nppiMalloc_8u_C1(width, height, &nStep);
    pSrc2 = nppiMalloc_8u_C1(width, height, &nStep);
    pDst = nppiMalloc_8u_C1(width, height, &nStep);

    if (!pSrc1 || !pSrc2 || !pDst) {
        printf("  [SKIP] Memory allocation failed\n");
        if (pSrc1) nppiFree(pSrc1);
        if (pSrc2) nppiFree(pSrc2);
        if (pDst) nppiFree(pDst);
        return;
    }

    oSizeROI.width = width;
    oSizeROI.height = height;

    /* Initialize using cudaMemcpy: 16 * 16 = 256, with scale factor 8: 256 >> 8 = 1 */
    size_t bufSize = (size_t)nStep * height;
    Npp8u* hBuf = (Npp8u*)malloc(bufSize);
    if (!hBuf) {
        nppiFree(pSrc1); nppiFree(pSrc2); nppiFree(pDst);
        return;
    }

    memset(hBuf, 16, bufSize);
    CUDA_CHECK(cudaMemcpy(pSrc1, hBuf, bufSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pSrc2, hBuf, bufSize, cudaMemcpyHostToDevice));

    status = nppiMul_8u_C1RSfs(pSrc1, nStep, pSrc2, nStep, pDst, nStep,
                                oSizeROI, nScaleFactor);
    test_assert(status == NPP_SUCCESS, "nppiMul_8u_C1RSfs returns NPP_SUCCESS");

    cudaDeviceSynchronize();

    /* Verify */
    CUDA_CHECK(cudaMemcpy(hBuf, pDst, bufSize, cudaMemcpyDeviceToHost));

    correct = 1;
    for (i = 0; i < width * height && correct; i++) {
        Npp8u actual = hBuf[(i / width) * nStep + (i % width)];
        /* 16 * 16 = 256, >> 8 = 1 */
        if (actual != 1) {
            correct = 0;
            printf("  Mismatch at %d: expected 1, got %d\n", i, actual);
        }
    }
    test_assert(correct, "Result = (src1 * src2) >> scaleFactor");

    free(hBuf);
    nppiFree(pSrc1);
    nppiFree(pSrc2);
    nppiFree(pDst);
}

/*
 * Test 6: Type definitions and constants
 */
void test_type_definitions(void) {
    NppiSize size;
    NppiRect rect;
    NppiPoint point;

    printf("\n[Test 6] NPP Type Definitions\n");

    /* NppiSize */
    size.width = 1920;
    size.height = 1080;
    test_assert(size.width == 1920 && size.height == 1080, "NppiSize");

    /* NppiRect */
    rect.x = 100;
    rect.y = 200;
    rect.width = 640;
    rect.height = 480;
    test_assert(rect.x == 100 && rect.y == 200 &&
                rect.width == 640 && rect.height == 480, "NppiRect");

    /* NppiPoint */
    point.x = 50;
    point.y = 60;
    test_assert(point.x == 50 && point.y == 60, "NppiPoint");

    /* Status codes */
    test_assert(NPP_SUCCESS == 0, "NPP_SUCCESS == 0");
    test_assert(NPP_NO_ERROR == 0, "NPP_NO_ERROR == 0");
    test_assert(NPP_ERROR < 0, "NPP_ERROR < 0");

    /* Interpolation modes */
    test_assert(NPPI_INTER_NN == 1, "NPPI_INTER_NN");
    test_assert(NPPI_INTER_LINEAR == 2, "NPPI_INTER_LINEAR");

    /* Axis */
    test_assert(NPP_HORIZONTAL_AXIS == 0, "NPP_HORIZONTAL_AXIS");
    test_assert(NPP_VERTICAL_AXIS == 1, "NPP_VERTICAL_AXIS");
}

/*
 * Test 7: 3-channel image operations
 */
void test_image_3channel(void) {
    Npp8u* pDst;
    int nDstStep;
    NppiSize oSizeROI;
    NppStatus status;
    int width = 32;
    int height = 32;
    int i, correct;

    printf("\n[Test 7] nppiCopy_8u_C3R (3-channel)\n");

    Npp8u* pSrc = nppiMalloc_8u_C3(width, height, &nDstStep);
    pDst = nppiMalloc_8u_C3(width, height, &nDstStep);
    test_assert(pSrc != NULL && pDst != NULL, "3-channel image allocation");

    if (!pSrc || !pDst) {
        if (pSrc) nppiFree(pSrc);
        if (pDst) nppiFree(pDst);
        return;
    }

    oSizeROI.width = width;
    oSizeROI.height = height;

    /* Initialize source with RGB pattern using cudaMemcpy */
    size_t bufSize = (size_t)nDstStep * height;
    Npp8u* hBuf = (Npp8u*)malloc(bufSize);
    if (!hBuf) {
        nppiFree(pSrc); nppiFree(pDst);
        return;
    }

    /* Set RGB values: R=255, G=128, B=64 */
    for (i = 0; i < width * height; i++) {
        int row = i / width;
        int col = i % width;
        Npp8u* pixel = &hBuf[row * nDstStep + col * 3];
        pixel[0] = 255;  /* R */
        pixel[1] = 128;  /* G */
        pixel[2] = 64;   /* B */
    }

    CUDA_CHECK(cudaMemcpy(pSrc, hBuf, bufSize, cudaMemcpyHostToDevice));

    /* Copy using NPP */
    status = nppiCopy_8u_C3R(pSrc, nDstStep, pDst, nDstStep, oSizeROI);
    test_assert(status == NPP_SUCCESS, "nppiCopy_8u_C3R returns NPP_SUCCESS");

    cudaDeviceSynchronize();

    /* Verify */
    CUDA_CHECK(cudaMemcpy(hBuf, pDst, bufSize, cudaMemcpyDeviceToHost));

    correct = 1;
    for (i = 0; i < width * height && correct; i++) {
        int row = i / width;
        int col = i % width;
        Npp8u* pixel = &hBuf[row * nDstStep + col * 3];
        if (pixel[0] != 255 || pixel[1] != 128 || pixel[2] != 64) {
            correct = 0;
            printf("  Mismatch at (%d,%d): expected (255,128,64), got (%d,%d,%d)\n",
                   col, row, pixel[0], pixel[1], pixel[2]);
        }
    }
    test_assert(correct, "All pixels have correct RGB values");

    free(hBuf);
    nppiFree(pSrc);
    nppiFree(pDst);
}

/*
 * Test 8: Logical operations with constants
 */
void test_logical_ops(void) {
    Npp8u* pSrc;
    Npp8u* pDst;
    int nStep;
    NppiSize oSizeROI;
    NppStatus status;
    int width = 16;
    int height = 16;
    int i, correct;

    printf("\n[Test 8] Logical Operations (with constants)\n");

    pSrc = nppiMalloc_8u_C1(width, height, &nStep);
    pDst = nppiMalloc_8u_C1(width, height, &nStep);

    if (!pSrc || !pDst) {
        printf("  [SKIP] Memory allocation failed\n");
        if (pSrc) nppiFree(pSrc);
        if (pDst) nppiFree(pDst);
        return;
    }

    oSizeROI.width = width;
    oSizeROI.height = height;

    size_t bufSize = (size_t)nStep * height;
    Npp8u* hBuf = (Npp8u*)malloc(bufSize);
    if (!hBuf) {
        nppiFree(pSrc); nppiFree(pDst);
        return;
    }

    /* Initialize: src = 0xAA */
    memset(hBuf, 0xAA, bufSize);
    CUDA_CHECK(cudaMemcpy(pSrc, hBuf, bufSize, cudaMemcpyHostToDevice));

    /* Test AndC: 0xAA & 0x0F = 0x0A */
    status = nppiAndC_8u_C1R(pSrc, nStep, 0x0F, pDst, nStep, oSizeROI);
    test_assert(status == NPP_SUCCESS, "nppiAndC_8u_C1R returns NPP_SUCCESS");

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaMemcpy(hBuf, pDst, bufSize, cudaMemcpyDeviceToHost));

    correct = 1;
    for (i = 0; i < width * height && correct; i++) {
        if (hBuf[(i / width) * nStep + (i % width)] != 0x0A) {
            correct = 0;
        }
    }
    test_assert(correct, "AndC result: 0xAA & 0x0F = 0x0A");

    /* Test OrC: 0xAA | 0x55 = 0xFF */
    status = nppiOrC_8u_C1R(pSrc, nStep, 0x55, pDst, nStep, oSizeROI);
    test_assert(status == NPP_SUCCESS, "nppiOrC_8u_C1R returns NPP_SUCCESS");

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaMemcpy(hBuf, pDst, bufSize, cudaMemcpyDeviceToHost));

    correct = 1;
    for (i = 0; i < width * height && correct; i++) {
        if (hBuf[(i / width) * nStep + (i % width)] != 0xFF) {
            correct = 0;
        }
    }
    test_assert(correct, "OrC result: 0xAA | 0x55 = 0xFF");

    /* Test LShiftC: 0x0A << 2 = 0x28 */
    memset(hBuf, 0x0A, bufSize);
    CUDA_CHECK(cudaMemcpy(pSrc, hBuf, bufSize, cudaMemcpyHostToDevice));

    status = nppiLShiftC_8u_C1R(pSrc, nStep, 2, pDst, nStep, oSizeROI);
    test_assert(status == NPP_SUCCESS, "nppiLShiftC_8u_C1R returns NPP_SUCCESS");

    cudaDeviceSynchronize();
    CUDA_CHECK(cudaMemcpy(hBuf, pDst, bufSize, cudaMemcpyDeviceToHost));

    correct = 1;
    for (i = 0; i < width * height && correct; i++) {
        if (hBuf[(i / width) * nStep + (i % width)] != 0x28) {
            correct = 0;
        }
    }
    test_assert(correct, "LShiftC result: 0x0A << 2 = 0x28");

    free(hBuf);
    nppiFree(pSrc);
    nppiFree(pDst);
}

/*
 * Test 9: CUDA Runtime API basic operations
 */
void test_cuda_runtime(void) {
    int deviceCount = 0;
    int device = 0;
    cudaDeviceProp prop;
    cudaError_t err;

    printf("\n[Test 9] CUDA Runtime API\n");

    /* Get device count */
    err = cudaGetDeviceCount(&deviceCount);
    test_assert(err == cudaSuccess, "cudaGetDeviceCount succeeds");
    test_assert(deviceCount > 0, "At least one GPU device found");
    printf("  Device count: %d\n", deviceCount);

    /* Get current device */
    err = cudaGetDevice(&device);
    test_assert(err == cudaSuccess, "cudaGetDevice succeeds");
    printf("  Current device: %d\n", device);

    /* Get device properties */
    err = cudaGetDeviceProperties(&prop, device);
    test_assert(err == cudaSuccess, "cudaGetDeviceProperties succeeds");
    printf("  Device name: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Multiprocessors: %d\n", prop.multiProcessorCount);

    /* Test memory allocation */
    void* devPtr = NULL;
    size_t allocSize = 1024 * 1024;  /* 1 MB */
    err = cudaMalloc(&devPtr, allocSize);
    test_assert(err == cudaSuccess, "cudaMalloc succeeds");
    test_assert(devPtr != NULL, "cudaMalloc returns valid pointer");

    if (devPtr) {
        err = cudaFree(devPtr);
        test_assert(err == cudaSuccess, "cudaFree succeeds");
    }

    /* Test synchronize */
    err = cudaDeviceSynchronize();
    test_assert(err == cudaSuccess, "cudaDeviceSynchronize succeeds");
}

/*
 * Main function
 */
int main(int argc, char* argv[]) {
    printf("========================================\n");
    printf("  NPP User Test Program (MACA Platform)\n");
    printf("========================================\n");
    printf("\n");
    printf("This test uses NPP and CUDA Runtime APIs.\n");
    printf("On MACA platform, they are mapped to MPP and MACA Runtime.\n");
    printf("\n");

    /* Run all tests */
    test_get_version();
    test_image_malloc();
    test_image_copy();
    test_image_addc();
    test_image_mul();
    test_type_definitions();
    test_image_3channel();
    test_logical_ops();
    test_cuda_runtime();

    /* Summary */
    printf("\n========================================\n");
    printf("  Results: %d / %d tests passed\n", passed_tests, total_tests);
    printf("========================================\n");

    if (passed_tests == total_tests) {
        printf("\nSUCCESS: All tests passed!\n");
        return 0;
    } else {
        printf("\nFAILURE: Some tests failed.\n");
        return 1;
    }
}
