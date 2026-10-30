// Shared helpers for the NPPI ColorTwist differential suites.
//
// The runners below build padded-stride buffers, capture the whole output
// buffer (ROI + row-end padding) and record a golden case.  Inputs are chosen
// so the recorded goldens also answer semantic questions offline:
//   - twist3()  : dyadic coefficients (exact in float32) -> rounding probes
//   - twist3b() : big coefficients -> saturation/clamp probes
//   - boundary fills include NaN / +-inf / +-1e30 and exact .5 values
#pragma once

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;

namespace npp_color_twist_diff {

// Allocate a device buffer that stays alive for the process lifetime (see the
// note in test_nppi_convert_generated.cpp about the maca allocator).
template <typename T> DeviceMemory<T> &arena(std::size_t n) {
  static std::vector<std::unique_ptr<DeviceMemory<T>>> v;
  v.push_back(std::make_unique<DeviceMemory<T>>(n));
  return *v.back();
}

inline NppStreamContext current_context() {
  NppStreamContext ctx{};
  NppStatus status = nppGetStreamContext(&ctx);
  EXPECT_EQ(status, NPP_NO_ERROR) << "nppGetStreamContext failed";
  return ctx;
}

template <typename D> ValueKind out_kind() {
  if constexpr (std::is_same_v<D, Npp8s>)
    return ValueKind::I8;
  else if constexpr (std::is_same_v<D, Npp8u>)
    return ValueKind::U8;
  else if constexpr (std::is_same_v<D, Npp16s>)
    return ValueKind::I16;
  else if constexpr (std::is_same_v<D, Npp16u>)
    return ValueKind::U16;
  else if constexpr (std::is_same_v<D, Npp32s>)
    return ValueKind::I32;
  else if constexpr (std::is_same_v<D, Npp32u>)
    return ValueKind::U32;
  else if constexpr (std::is_same_v<D, Npp32f>)
    return ValueKind::F32;
  else if constexpr (std::is_same_v<D, Npp16f>)
    return ValueKind::F16;
  else
    return ValueKind::Bytes;
}

inline int align_step(int step, int esize) { return ((step + esize - 1) / esize) * esize; }

// Dyadic 3x4 twist (every coefficient exact in float32).
inline const Npp32f(&twist3())[3][4] {
  static const Npp32f t[3][4] = {
      {0.5f, 0.25f, -0.125f, 8.0f}, {0.75f, -0.25f, 0.5f, -4.0f}, {0.125f, 0.375f, 0.25f, 2.0f}};
  return t;
}

// Big-coefficient 3x4 twist (saturation probe).
inline const Npp32f(&twist3b())[3][4] {
  static const Npp32f t[3][4] = {{64.0f, 64.0f, 64.0f, 8192.0f},
                                 {-96.0f, 32.0f, -8.0f, -4096.0f},
                                 {8.0f, -16.0f, 128.0f, 1024.0f}};
  return t;
}

// 4x4 twist + constants for the "C" variants (dyadic; the 4th row/column and
// the constants make alpha handling observable).
inline void twist4c(Npp32f m[4][4], Npp32f c[4]) {
  const Npp32f src[4][4] = {{0.5f, 0.25f, -0.125f, 0.0625f},
                            {0.75f, -0.25f, 0.5f, -0.0625f},
                            {0.125f, 0.375f, 0.25f, 0.03125f},
                            {-0.5f, 0.125f, 0.25f, 0.375f}};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      m[i][j] = src[i][j];
    }
  }
  c[0] = 8.0f;
  c[1] = -4.0f;
  c[2] = 2.0f;
  c[3] = 1.5f;
}

// Boundary pixel values for float images (NaN / inf / huge / exact .5 steps).
inline std::vector<Npp32f> boundary_floats() {
  return {-1e30f,
          -std::numeric_limits<Npp32f>::infinity(),
          -3.75f, -1.5f, -0.5f, -0.0f, 0.0f, 0.5f, 1.5f, 3.75f, 2.5f, 4.0f, 64.0f, 127.5f, 255.5f, 32767.5f, 65535.5f,
          std::numeric_limits<Npp32f>::infinity(),
          std::numeric_limits<Npp32f>::quiet_NaN(),
          1e30f};
}

// Boundary pixel values for a given pixel type: floats keep the full set
// (NaN/inf via the fp16 host conversion), integers get type-range limits and
// small odd/even values (no float->int UB; every value in range).
template <typename T> std::vector<T> boundary_pixels() {
  if constexpr (std::is_same_v<T, Npp16f>) {
    const std::vector<Npp32f> f = boundary_floats();
    std::vector<T> out;
    out.reserve(f.size());
    for (Npp32f v : f) {
      out.push_back(float_to_npp16f_host(v));
    }
    return out;
  } else if constexpr (std::is_same_v<T, Npp32f>) {
    return boundary_floats();
  } else {
    static_assert(std::is_integral_v<T>, "unsupported pixel type");
    const long long lo = (long long)std::numeric_limits<T>::min();
    const long long hi = (long long)std::numeric_limits<T>::max();
    const long long v[] = {lo, lo + 1, lo / 2, -3, -2, -1, 0, 1, 2, 3, 5, 17, hi / 2, hi - 2, hi - 1, hi};
    std::vector<T> out;
    for (long long x : v) {
      out.push_back(static_cast<T>(x));
    }
    return out;
  }
}

// Deterministic sentinel for out-of-ROI bytes.
template <typename T> inline T sentinel_pixel(Rng &rng) {
  if constexpr (std::is_same_v<T, Npp16f>) {
    return float_to_npp16f_host(-1234.5f);
  } else if constexpr (std::is_same_v<T, Npp32f>) {
    return static_cast<T>((double)(rng.next() % 4096u) / 64.0 - 32.0);
  } else {
    return static_cast<T>(rng.next());
  }
}

// Fill `out` (whole buffer, elems) with sentinels, then overwrite the first
// widthElems elements of each row with `pattern` (cycling) so ROI pixels carry
// the interesting values and the row padding carries the sentinel.
template <typename T>
void fill_roi_pattern(std::vector<T> &out, int height, int stepElems, int widthElems, const std::vector<T> &pattern) {
  Rng rng(0xC0FFEEu + (uint64_t)widthElems * 17u + (uint64_t)height * 31u);
  for (auto &e : out) {
    e = sentinel_pixel<T>(rng);
  }
  for (int y = 0; y < height; ++y) {
    T *row = out.data() + (size_t)y * stepElems;
    for (int x = 0; x < widthElems; ++x) {
      row[x] = pattern[(size_t)(y * widthElems + x) % pattern.size()];
    }
  }
}

// ---- packed 3x4 twist, out-of-place -----------------------------------------
// `call` receives (src, srcStep, dst, dstStep, roi) and invokes the entry point
// under test (Ctx or non-Ctx adapter).
template <typename T, typename Call>
void packedR_case(const char *fname, const std::string &id, int width, int height, int chan, int pad,
                  const Npp32f (&twist)[3][4], Call &&call, uint64_t seed, bool boundary = false) {
  const int se = (int)sizeof(T);
  const int srcStep = align_step(width * chan * se + pad, se);
  const int dstStep = align_step(width * chan * se + pad, se);
  const size_t srcElems = (size_t)height * srcStep / se;
  const size_t dstElems = (size_t)height * dstStep / se;
  std::vector<T> hs = make_uniform<T>(srcElems, seed + 11);
  std::vector<T> hd = make_uniform<T>(dstElems, seed + 29);
  if (boundary) {
    fill_roi_pattern<T>(hs, height, srcStep / se, width * chan, boundary_pixels<T>());
  }
  DeviceMemory<T> &s = arena<T>(srcElems);
  DeviceMemory<T> &d = arena<T>(dstElems);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  NppiSize roi{width, height};
  clear_cuda_error();
  const NppStatus st = call(s.get(), srcStep, d.get(), dstStep, roi);
  Case c(id, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.input(hd.data(), hd.size() * sizeof(T));
  c.input(twist, sizeof(twist));
  c.status(st);
  c.device_output<T>(d.get(), dstElems, out_kind<T>(), 1e-5, 1e-6);
  c.finish();
}

// ---- packed 3x4 twist, in-place ----------------------------------------------
template <typename T, typename Call>
void packedIR_case(const char *fname, const std::string &id, int width, int height, int chan, int pad,
                   const Npp32f (&twist)[3][4], Call &&call, uint64_t seed, bool boundary = false) {
  const int se = (int)sizeof(T);
  const int step = align_step(width * chan * se + pad, se);
  const size_t elems = (size_t)height * step / se;
  std::vector<T> hb = make_uniform<T>(elems, seed + 41);
  if (boundary) {
    fill_roi_pattern<T>(hb, height, step / se, width * chan, boundary_pixels<T>());
  }
  DeviceMemory<T> &b = arena<T>(elems);
  b.copyFromHost(hb);
  NppiSize roi{width, height};
  clear_cuda_error();
  const NppStatus st = call(b.get(), step, roi);
  Case c(id, fname);
  c.input(hb.data(), hb.size() * sizeof(T));
  c.input(twist, sizeof(twist));
  c.status(st);
  c.device_output<T>(b.get(), elems, out_kind<T>(), 1e-5, 1e-6);
  c.finish();
}

// ---- planar 3x4 twist ----------------------------------------------------------
template <typename T, typename Call>
void planarP3_case(const char *fname, const std::string &id, int width, int height, int pad,
                   const Npp32f (&twist)[3][4], Call &&call, uint64_t seed, bool boundary = false) {
  const int se = (int)sizeof(T);
  const int srcStep = align_step(width * se + pad, se);
  const int dstStep = align_step(width * se + pad, se);
  const size_t elems = (size_t)height * srcStep / se;
  std::vector<T> hs[3], hd[3];
  DeviceMemory<T> *s[3];
  DeviceMemory<T> *d[3];
  for (int p = 0; p < 3; ++p) {
    hs[p] = make_uniform<T>(elems, seed + 11 + p);
    hd[p] = make_uniform<T>(elems, seed + 29 + p);
    if (boundary) {
      fill_roi_pattern<T>(hs[p], height, srcStep / se, width, boundary_pixels<T>());
    }
    s[p] = &arena<T>(elems);
    d[p] = &arena<T>(elems);
    s[p]->copyFromHost(hs[p]);
    d[p]->copyFromHost(hd[p]);
  }
  const T *srcs[3] = {s[0]->get(), s[1]->get(), s[2]->get()};
  T *dsts[3] = {d[0]->get(), d[1]->get(), d[2]->get()};
  NppiSize roi{width, height};
  clear_cuda_error();
  const NppStatus st = call(srcs, srcStep, dsts, dstStep, roi);
  Case c(id, fname);
  for (int p = 0; p < 3; ++p) {
    c.input(hs[p].data(), hs[p].size() * sizeof(T));
    c.input(hd[p].data(), hd[p].size() * sizeof(T));
  }
  c.input(twist, sizeof(twist));
  c.status(st);
  for (int p = 0; p < 3; ++p) {
    c.device_output<T>(dsts[p], elems, out_kind<T>(), 1e-5, 1e-6);
  }
  c.finish();
}

template <typename T, typename Call>
void planarIP3_case(const char *fname, const std::string &id, int width, int height, int pad,
                    const Npp32f (&twist)[3][4], Call &&call, uint64_t seed, bool boundary = false) {
  const int se = (int)sizeof(T);
  const int step = align_step(width * se + pad, se);
  const size_t elems = (size_t)height * step / se;
  std::vector<T> hb[3];
  DeviceMemory<T> *b[3];
  for (int p = 0; p < 3; ++p) {
    hb[p] = make_uniform<T>(elems, seed + 41 + p);
    if (boundary) {
      fill_roi_pattern<T>(hb[p], height, step / se, width, boundary_pixels<T>());
    }
    b[p] = &arena<T>(elems);
    b[p]->copyFromHost(hb[p]);
  }
  T *bufs[3] = {b[0]->get(), b[1]->get(), b[2]->get()};
  NppiSize roi{width, height};
  clear_cuda_error();
  const NppStatus st = call(bufs, step, roi);
  Case c(id, fname);
  for (int p = 0; p < 3; ++p) {
    c.input(hb[p].data(), hb[p].size() * sizeof(T));
  }
  c.input(twist, sizeof(twist));
  c.status(st);
  for (int p = 0; p < 3; ++p) {
    c.device_output<T>(bufs[p], elems, out_kind<T>(), 1e-5, 1e-6);
  }
  c.finish();
}

// ---- 4x4 twist + constants ("C" variants) --------------------------------------
template <typename T, typename Call>
void packedC_case(const char *fname, const std::string &id, int width, int height, int pad, Call &&call, uint64_t seed,
                  bool boundary = false) {
  Npp32f twist[4][4];
  Npp32f consts[4];
  twist4c(twist, consts);
  const int se = (int)sizeof(T);
  const int srcStep = align_step(width * 4 * se + pad, se);
  const int dstStep = align_step(width * 4 * se + pad, se);
  const size_t srcElems = (size_t)height * srcStep / se;
  const size_t dstElems = (size_t)height * dstStep / se;
  std::vector<T> hs = make_uniform<T>(srcElems, seed + 11);
  std::vector<T> hd = make_uniform<T>(dstElems, seed + 29);
  if (boundary) {
    fill_roi_pattern<T>(hs, height, srcStep / se, width * 4, boundary_pixels<T>());
  }
  DeviceMemory<T> &s = arena<T>(srcElems);
  DeviceMemory<T> &d = arena<T>(dstElems);
  s.copyFromHost(hs);
  d.copyFromHost(hd);
  NppiSize roi{width, height};
  clear_cuda_error();
  const NppStatus st = call(s.get(), srcStep, d.get(), dstStep, roi, twist, consts);
  Case c(id, fname);
  c.input(hs.data(), hs.size() * sizeof(T));
  c.input(hd.data(), hd.size() * sizeof(T));
  c.input(twist, sizeof(twist));
  c.input(consts, sizeof(consts));
  c.status(st);
  c.device_output<T>(d.get(), dstElems, out_kind<T>(), 1e-5, 1e-6);
  c.finish();
}

template <typename T, typename Call>
void packedCIR_case(const char *fname, const std::string &id, int width, int height, int pad, Call &&call, uint64_t seed,
                    bool boundary = false) {
  Npp32f twist[4][4];
  Npp32f consts[4];
  twist4c(twist, consts);
  const int se = (int)sizeof(T);
  const int step = align_step(width * 4 * se + pad, se);
  const size_t elems = (size_t)height * step / se;
  std::vector<T> hb = make_uniform<T>(elems, seed + 41);
  if (boundary) {
    fill_roi_pattern<T>(hb, height, step / se, width * 4, boundary_pixels<T>());
  }
  DeviceMemory<T> &b = arena<T>(elems);
  b.copyFromHost(hb);
  NppiSize roi{width, height};
  clear_cuda_error();
  const NppStatus st = call(b.get(), step, roi, twist, consts);
  Case c(id, fname);
  c.input(hb.data(), hb.size() * sizeof(T));
  c.input(twist, sizeof(twist));
  c.input(consts, sizeof(consts));
  c.status(st);
  c.device_output<T>(b.get(), elems, out_kind<T>(), 1e-5, 1e-6);
  c.finish();
}

}  // namespace npp_color_twist_diff
