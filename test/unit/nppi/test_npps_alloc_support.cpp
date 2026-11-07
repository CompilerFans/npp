// Signal allocator coverage for the nppsMalloc variants that are not
// exercised indirectly by the functional suites.
//
// The legacy test/unit/npps tree is not part of the build (its sources
// reference APIs that are still being brought up), so these checks live in
// the built nppi support tree instead.
#include "npp.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace {

// Allocates n elements, verifies the first and last byte of the requested
// range are writable device memory, and releases the signal.
template <typename T, typename Fn> void checkSignalAlloc(const char *name, Fn fn) {
  const size_t n = 97;
  T *p = fn(n);
  ASSERT_NE(p, nullptr) << name;
  ASSERT_EQ(cudaMemset(p, 0x5A, n * sizeof(T)), cudaSuccess) << name;
  unsigned char first = 0, last = 0;
  ASSERT_EQ(cudaMemcpy(&first, p, 1, cudaMemcpyDeviceToHost), cudaSuccess) << name;
  ASSERT_EQ(cudaMemcpy(&last, (const char *)p + n * sizeof(T) - 1, 1, cudaMemcpyDeviceToHost), cudaSuccess) << name;
  EXPECT_EQ(first, 0x5A) << name;
  EXPECT_EQ(last, 0x5A) << name;
  nppsFree(p);
}

}  // namespace

TEST(NppsAllocSupport, Malloc_RemainingTypes) {
  checkSignalAlloc<Npp8s>("nppsMalloc_8s", [](size_t n) { return nppsMalloc_8s(n); });
  checkSignalAlloc<Npp16u>("nppsMalloc_16u", [](size_t n) { return nppsMalloc_16u(n); });
  checkSignalAlloc<Npp16sc>("nppsMalloc_16sc", [](size_t n) { return nppsMalloc_16sc(n); });
  checkSignalAlloc<Npp32u>("nppsMalloc_32u", [](size_t n) { return nppsMalloc_32u(n); });
  checkSignalAlloc<Npp32sc>("nppsMalloc_32sc", [](size_t n) { return nppsMalloc_32sc(n); });
  checkSignalAlloc<Npp64s>("nppsMalloc_64s", [](size_t n) { return nppsMalloc_64s(n); });
  checkSignalAlloc<Npp64sc>("nppsMalloc_64sc", [](size_t n) { return nppsMalloc_64sc(n); });
  checkSignalAlloc<Npp64fc>("nppsMalloc_64fc", [](size_t n) { return nppsMalloc_64fc(n); });
}
