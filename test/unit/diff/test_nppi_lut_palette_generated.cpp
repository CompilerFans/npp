// Differential tests for package c1: nppiLUTPalette and nppiLUTPaletteSwap.
//
// Probes pin (a) low-nBitSize-bit masking of the source value, (b) the memory
// layout of the palette tables (in particular the packed 3-byte entries used
// by the 24u variants), (c) untouched alpha channels for the AC4 variants,
// and (d) the C3 -> C4 "alpha first, channels reversed" order of
// nppiLUTPaletteSwap.  dst buffers are pre-filled with a sentinel so untouched
// bytes are visible in the golden.
//
// NOTE (see commit 005b2c1): for the 24u variants the destination is Npp8u*
// with 3-byte pixels; the destination step must be >= width*3 bytes or NVIDIA
// NPP 13.4 writes overlapping rows (data race).  All cases below use valid
// steps.

#include "npp_diff_harness.h"
#include "npp_test_base.h"

#include <memory>
#include <vector>

using namespace npp_diff;
using namespace npp_functional_test;
namespace {

template <typename T> DeviceMemory<T> &dev_buf(const std::vector<T> &h) {
  static std::vector<std::unique_ptr<DeviceMemory<T>>> arena;
  arena.push_back(std::make_unique<DeviceMemory<T>>(h.size()));
  DeviceMemory<T> &d = *arena.back();
  if (!h.empty()) d.copyFromHost(h);
  return d;
}

NppStreamContext current_ctx() {
  NppStreamContext c{};
  NppStatus st = nppGetStreamContext(&c);
  EXPECT_EQ(st, NPP_NO_ERROR);
  return c;
}

int align_step(int step, int es) { return ((step + es - 1) / es) * es; }

std::vector<Npp8u> ramp8u(int count) {
  std::vector<Npp8u> v(count);
  for (int i = 0; i < count; ++i) v[i] = static_cast<Npp8u>(i);
  return v;
}

// Single-table C1 palette (8u_C1R, 8u24u, 8u32u, 16u*, 16u8u, 16u24u, 16u32u).
// srcCh/srcEs: source channel count and element size; dstPx: destination
// bytes per pixel; table Es: element size of table entries.
template <typename SrcT, typename DstT, typename TabT, typename F>
void pal_c1(const char *fname, const std::string &key, int width, int height, int srcPad, int dstPad,
            int dstPx, const std::vector<SrcT> &hs, const std::vector<TabT> &table, int bitSize,
            ValueKind kind, F &&f) {
  const int sEs = (int)sizeof(SrcT);
  const int sStep = align_step(width * sEs + srcPad, sEs);
  const int dEs = (int)sizeof(DstT);
  const int dStep = align_step(width * dstPx + dstPad, dEs < dstPx ? dstPx : dEs);
  const size_t sn = (size_t)height * sStep / sEs;
  const size_t dn = (size_t)height * dStep / dEs;
  std::vector<SrcT> hs_full(hs);
  hs_full.resize(sn);
  std::vector<DstT> hd = make_uniform<DstT>(dn, DstT(0xA5));
  DeviceMemory<SrcT> &s = dev_buf(hs_full);
  DeviceMemory<DstT> &d = dev_buf(hd);
  DeviceMemory<TabT> &t = dev_buf(table);
  clear_cuda_error();
  NppStatus st = f(s.get(), sStep, d.get(), dStep, NppiSize{width, height}, t.get(), bitSize, current_ctx());
  Case c("nppiLUTPalette/" + key, fname);
  c.input(hs_full.data(), hs_full.size() * sizeof(SrcT));
  c.status(st);
  c.device_output<DstT>(d.get(), dn, kind);
  c.finish();
}

// Multi-table palette (C3R / C4R / AC4R).
template <typename SrcT, typename DstT, typename TabT, int NTab, typename F>
void pal_multi(const char *fname, const std::string &key, int width, int height, int srcPad, int dstPad,
               int srcCh, int dstCh, const std::vector<SrcT> &hs, const std::vector<std::vector<TabT>> &tables,
               int bitSize, ValueKind kind, F &&f) {
  const int sEs = (int)sizeof(SrcT);
  const int dEs = (int)sizeof(DstT);
  const int sPx = srcCh * sEs, dPx = dstCh * dEs;
  const int sStep = align_step(width * sPx + srcPad, sPx);
  const int dStep = align_step(width * dPx + dstPad, dPx);
  const size_t sn = (size_t)height * sStep / sEs;
  const size_t dn = (size_t)height * dStep / dEs;
  std::vector<SrcT> hs_full(hs);
  hs_full.resize(sn);
  std::vector<DstT> hd = make_uniform<DstT>(dn, DstT(0x5A));
  DeviceMemory<SrcT> &s = dev_buf(hs_full);
  DeviceMemory<DstT> &d = dev_buf(hd);
  const TabT *pt[NTab];
  for (int i = 0; i < NTab; ++i) pt[i] = dev_buf(tables[i]).get();
  clear_cuda_error();
  NppStatus st = f(s.get(), sStep, d.get(), dStep, NppiSize{width, height}, pt, bitSize, current_ctx());
  Case c("nppiLUTPalette/" + key, fname);
  c.input(hs_full.data(), hs_full.size() * sizeof(SrcT));
  c.status(st);
  c.device_output<DstT>(d.get(), dn, kind);
  c.finish();
}

}  // namespace

class NppiLutPaletteDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

// ===== nppiLUTPalette: 8-bit sources =====================================
TEST_F(NppiLutPaletteDiff, Palette_8u_C1) {
  // nBitSize = 8: full identity-ish table over 0..255 ramp.
  std::vector<Npp8u> t256(256);
  for (int i = 0; i < 256; ++i) t256[i] = static_cast<Npp8u>((i * 7 + 3) & 0xFF);
  pal_c1<Npp8u, Npp8u, Npp8u>("nppiLUTPalette_8u_C1R", "8u/C1/bits8", 256, 1, 3, 5, 1, ramp8u(256), t256, 8,
                              ValueKind::U8,
                              [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp8u *t, int n,
                                 NppStreamContext c) {
                                return nppiLUTPalette_8u_C1R_Ctx(s, ss, d, ds, o, t, n, c);
                              });
  // nBitSize = 4: index = src & 15.
  std::vector<Npp8u> t16(16);
  for (int i = 0; i < 16; ++i) t16[i] = static_cast<Npp8u>(i * 15 + 7);
  pal_c1<Npp8u, Npp8u, Npp8u>("nppiLUTPalette_8u_C1R", "8u/C1/bits4", 256, 1, 3, 3, 1, ramp8u(256), t16, 4,
                              ValueKind::U8,
                              [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp8u *t, int n,
                                 NppStreamContext c) {
                                return nppiLUTPalette_8u_C1R_Ctx(s, ss, d, ds, o, t, n, c);
                              });
  // nBitSize = 1: two entries; random source.
  std::vector<Npp8u> t2 = {201, 2};
  pal_c1<Npp8u, Npp8u, Npp8u>("nppiLUTPalette_8u_C1R", "8u/C1/bits1", 17, 13, 3, 5, 1,
                              make_uniform<Npp8u>(17 * 13, 800), t2, 1, ValueKind::U8,
                              [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp8u *t, int n,
                                 NppStreamContext c) {
                                return nppiLUTPalette_8u_C1R_Ctx(s, ss, d, ds, o, t, n, c);
                              });
}

TEST_F(NppiLutPaletteDiff, Palette_8u_Multi) {
  auto rnd = make_uniform<Npp8u>(17 * 13 * 4, 801);
  std::vector<std::vector<Npp8u>> t3 = {{11, 21, 31, 41}, {12, 22, 32, 42}, {13, 23, 33, 43}};
  std::vector<std::vector<Npp8u>> t4 = {{11, 21, 31, 41}, {12, 22, 32, 42}, {13, 23, 33, 43}, {14, 24, 34, 44}};
  pal_multi<Npp8u, Npp8u, Npp8u, 3>("nppiLUTPalette_8u_C3R", "8u/C3/bits2", 17, 13, 5, 3, 1, 3, rnd, t3, 2,
                                    ValueKind::U8,
                                    [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp8u *t[],
                                       int n, NppStreamContext c) {
                                      return nppiLUTPalette_8u_C3R_Ctx(s, ss, d, ds, o, t, n, c);
                                    });
  pal_multi<Npp8u, Npp8u, Npp8u, 4>("nppiLUTPalette_8u_C4R", "8u/C4/bits2", 17, 13, 5, 3, 1, 4, rnd, t4, 2,
                                    ValueKind::U8,
                                    [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp8u *t[],
                                       int n, NppStreamContext c) {
                                      return nppiLUTPalette_8u_C4R_Ctx(s, ss, d, ds, o, t, n, c);
                                    });
  pal_multi<Npp8u, Npp8u, Npp8u, 3>("nppiLUTPalette_8u_AC4R", "8u/AC4/bits2", 17, 13, 5, 3, 1, 4, rnd, t3, 2,
                                    ValueKind::U8,
                                    [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp8u *t[],
                                       int n, NppStreamContext c) {
                                      return nppiLUTPalette_8u_AC4R_Ctx(s, ss, d, ds, o, t, n, c);
                                    });
}

TEST_F(NppiLutPaletteDiff, Palette_8u_Wide) {
  // 8u24u: table is 4 packed 3-byte entries with byte values equal to their
  // offset (0..11) - the golden reveals the entry layout and byte order.
  std::vector<Npp8u> t24(12);
  for (int j = 0; j < 12; ++j) t24[j] = static_cast<Npp8u>(j);
  pal_c1<Npp8u, Npp8u, Npp8u>("nppiLUTPalette_8u24u_C1R", "8u24u/bits2", 256, 1, 3, 3, 3, ramp8u(256), t24, 2,
                              ValueKind::U8,
                              [](const Npp8u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp8u *t, int n,
                                 NppStreamContext c) {
                                return nppiLUTPalette_8u24u_C1R_Ctx(s, ss, d, ds, o, t, n, c);
                              });
  // 8u32u: single 32-bit entries with distinguishable bytes.
  std::vector<Npp32u> t32(8);
  for (int i = 0; i < 8; ++i) t32[i] = 0x01000000u | (i * 3 + 1) | ((i * 3 + 2) << 8) | ((i * 3 + 3) << 16);
  pal_c1<Npp8u, Npp32u, Npp32u>("nppiLUTPalette_8u32u_C1R", "8u32u/bits3", 256, 1, 3, 5, 4, ramp8u(256), t32, 3,
                                ValueKind::U32,
                                [](const Npp8u *s, int ss, Npp32u *d, int ds, NppiSize o, const Npp32u *t, int n,
                                   NppStreamContext c) {
                                  return nppiLUTPalette_8u32u_C1R_Ctx(s, ss, d, ds, o, t, n, c);
                                });
}

// ===== nppiLUTPalette: 16-bit sources ====================================
TEST_F(NppiLutPaletteDiff, Palette_16u_C1) {
  // nBitSize = 16: table index = full source value.  Use a 512-pixel ramp.
  std::vector<Npp16u> src(512);
  for (int i = 0; i < 512; ++i) src[i] = static_cast<Npp16u>(i * 137);
  std::vector<Npp16u> t(65536);
  for (int i = 0; i < 65536; ++i) t[i] = static_cast<Npp16u>((i * 13 + 5) & 0xFFFF);
  pal_c1<Npp16u, Npp16u, Npp16u>("nppiLUTPalette_16u_C1R", "16u/C1/bits16", 512, 1, 6, 5, 2, src, t, 16,
                                 ValueKind::U16,
                                 [](const Npp16u *s, int ss, Npp16u *d, int ds, NppiSize o, const Npp16u *tb, int n,
                                    NppStreamContext c) {
                                   return nppiLUTPalette_16u_C1R_Ctx(s, ss, d, ds, o, tb, n, c);
                                 });
  // nBitSize = 2: masking probe on random 16-bit values.
  std::vector<Npp16u> t4 = {1000, 2000, 3000, 4000};
  pal_c1<Npp16u, Npp16u, Npp16u>("nppiLUTPalette_16u_C1R", "16u/C1/bits2", 17, 13, 6, 5, 2,
                                 make_uniform<Npp16u>(17 * 13, 810), t4, 2, ValueKind::U16,
                                 [](const Npp16u *s, int ss, Npp16u *d, int ds, NppiSize o, const Npp16u *tb, int n,
                                    NppStreamContext c) {
                                   return nppiLUTPalette_16u_C1R_Ctx(s, ss, d, ds, o, tb, n, c);
                                 });
}

TEST_F(NppiLutPaletteDiff, Palette_16u_Multi) {
  auto rnd = make_uniform<Npp16u>(17 * 13 * 4, 811);
  std::vector<std::vector<Npp16u>> t3 = {{1111, 2222, 3333, 4444}, {1212, 2323, 3434, 4545}, {1313, 2424, 3535, 4646}};
  std::vector<std::vector<Npp16u>> t4 = {{1111, 2222, 3333, 4444}, {1212, 2323, 3434, 4545},
                                         {1313, 2424, 3535, 4646}, {1414, 2525, 3636, 4747}};
  pal_multi<Npp16u, Npp16u, Npp16u, 3>("nppiLUTPalette_16u_C3R", "16u/C3/bits2", 17, 13, 6, 5, 1, 3, rnd, t3, 2,
                                       ValueKind::U16,
                                       [](const Npp16u *s, int ss, Npp16u *d, int ds, NppiSize o, const Npp16u *t[],
                                          int n, NppStreamContext c) {
                                         return nppiLUTPalette_16u_C3R_Ctx(s, ss, d, ds, o, t, n, c);
                                       });
  pal_multi<Npp16u, Npp16u, Npp16u, 4>("nppiLUTPalette_16u_C4R", "16u/C4/bits2", 17, 13, 6, 5, 1, 4, rnd, t4, 2,
                                       ValueKind::U16,
                                       [](const Npp16u *s, int ss, Npp16u *d, int ds, NppiSize o, const Npp16u *t[],
                                          int n, NppStreamContext c) {
                                         return nppiLUTPalette_16u_C4R_Ctx(s, ss, d, ds, o, t, n, c);
                                       });
  pal_multi<Npp16u, Npp16u, Npp16u, 3>("nppiLUTPalette_16u_AC4R", "16u/AC4/bits2", 17, 13, 6, 5, 1, 4, rnd, t3, 2,
                                       ValueKind::U16,
                                       [](const Npp16u *s, int ss, Npp16u *d, int ds, NppiSize o, const Npp16u *t[],
                                          int n, NppStreamContext c) {
                                         return nppiLUTPalette_16u_AC4R_Ctx(s, ss, d, ds, o, t, n, c);
                                       });
}

TEST_F(NppiLutPaletteDiff, Palette_16u_Narrow) {
  // 16u8u: 16-bit source, 8-bit output.
  std::vector<Npp8u> t8(16);
  for (int i = 0; i < 16; ++i) t8[i] = static_cast<Npp8u>(i * 16 + 3);
  pal_c1<Npp16u, Npp8u, Npp8u>("nppiLUTPalette_16u8u_C1R", "16u8u/bits4", 256, 1, 6, 5, 1,
                               make_uniform<Npp16u>(256, 812), t8, 4, ValueKind::U8,
                               [](const Npp16u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp8u *t, int n,
                                  NppStreamContext c) {
                                 return nppiLUTPalette_16u8u_C1R_Ctx(s, ss, d, ds, o, t, n, c);
                               });
  // 16u24u: packed 3-byte entries, offsets as values.
  std::vector<Npp8u> t24(24);
  for (int j = 0; j < 24; ++j) t24[j] = static_cast<Npp8u>(j);
  pal_c1<Npp16u, Npp8u, Npp8u>("nppiLUTPalette_16u24u_C1R", "16u24u/bits3", 256, 1, 6, 3, 3,
                               make_uniform<Npp16u>(256, 813), t24, 3, ValueKind::U8,
                               [](const Npp16u *s, int ss, Npp8u *d, int ds, NppiSize o, const Npp8u *t, int n,
                                  NppStreamContext c) {
                                 return nppiLUTPalette_16u24u_C1R_Ctx(s, ss, d, ds, o, t, n, c);
                               });
  // 16u32u: 32-bit entries.
  std::vector<Npp32u> t32(8);
  for (int i = 0; i < 8; ++i) t32[i] = 0x02000000u | (i * 3 + 1) | ((i * 3 + 2) << 8) | ((i * 3 + 3) << 16);
  pal_c1<Npp16u, Npp32u, Npp32u>("nppiLUTPalette_16u32u_C1R", "16u32u/bits3", 256, 1, 6, 5, 4,
                                 make_uniform<Npp16u>(256, 814), t32, 3, ValueKind::U32,
                                 [](const Npp16u *s, int ss, Npp32u *d, int ds, NppiSize o, const Npp32u *t, int n,
                                    NppStreamContext c) {
                                   return nppiLUTPalette_16u32u_C1R_Ctx(s, ss, d, ds, o, t, n, c);
                                 });
}

// ===== nppiLUTPaletteSwap =================================================
class NppiLutPaletteSwapDiff : public ::testing::Test {
protected:
  void TearDown() override { (void)cudaDeviceSynchronize(); }
};

namespace swap_tests {
template <typename T>
void swap_run(const char *fname, const std::string &key, int alpha, const std::vector<T> &hs, int bitSize,
              ValueKind kind) {
  const int es = (int)sizeof(T);
  const int sPx = 3 * es, dPx = 4 * es;
  const int width = 17, height = 13;
  const int sStep = align_step(width * sPx + 5, sPx);
  const int dStep = align_step(width * dPx + 3, dPx);
  const size_t sn = (size_t)height * sStep / es;
  const size_t dn = (size_t)height * dStep / es;
  std::vector<T> hs_full(hs);
  hs_full.resize(sn);
  std::vector<T> hd = make_uniform<T>(dn, T(0xC3));
  DeviceMemory<T> &s = dev_buf(hs_full);
  DeviceMemory<T> &d = dev_buf(hd);
  // table c entry i = i*10+c (c in 0..2): the output bytes identify the source
  // channel each destination slot came from.
  std::vector<std::vector<T>> tables(3);
  for (int c = 0; c < 3; ++c) {
    tables[c].resize(1 << bitSize);
    for (int i = 0; i < (1 << bitSize); ++i) tables[c][i] = static_cast<T>(i * 10 + c);
  }
  const T *pt[3] = {dev_buf(tables[0]).get(), dev_buf(tables[1]).get(), dev_buf(tables[2]).get()};
  NppStatus st;
  if constexpr (sizeof(T) == 1) {
    clear_cuda_error();
    st = nppiLUTPaletteSwap_8u_C3A0C4R_Ctx(s.get(), sStep, alpha, d.get(), dStep, NppiSize{width, height}, pt,
                                           bitSize, current_ctx());
  } else {
    clear_cuda_error();
    st = nppiLUTPaletteSwap_16u_C3A0C4R_Ctx(s.get(), sStep, alpha, d.get(), dStep, NppiSize{width, height}, pt,
                                            bitSize, current_ctx());
  }
  Case c("nppiLUTPaletteSwap/" + key, fname);
  c.input(hs_full.data(), hs_full.size() * sizeof(T));
  c.status(st);
  c.device_output<T>(d.get(), dn, kind);
  c.finish();
}
}  // namespace swap_tests

TEST_F(NppiLutPaletteSwapDiff, Swap_8u) {
  using namespace swap_tests;
  auto rnd = make_uniform<Npp8u>(17 * 13 * 3, 820);
  swap_run<Npp8u>("nppiLUTPaletteSwap_8u_C3A0C4R", "8u/alpha0", 0, rnd, 4, ValueKind::U8);
  swap_run<Npp8u>("nppiLUTPaletteSwap_8u_C3A0C4R", "8u/alpha128", 128, rnd, 4, ValueKind::U8);
  swap_run<Npp8u>("nppiLUTPaletteSwap_8u_C3A0C4R", "8u/alpha255", 255, rnd, 4, ValueKind::U8);
  swap_run<Npp8u>("nppiLUTPaletteSwap_8u_C3A0C4R", "8u/alphaNeg", -1, rnd, 4, ValueKind::U8);
  swap_run<Npp8u>("nppiLUTPaletteSwap_8u_C3A0C4R", "8u/alpha256", 256, rnd, 4, ValueKind::U8);
}

TEST_F(NppiLutPaletteSwapDiff, Swap_16u) {
  using namespace swap_tests;
  auto rnd = make_uniform<Npp16u>(17 * 13 * 3, 821);
  swap_run<Npp16u>("nppiLUTPaletteSwap_16u_C3A0C4R", "16u/alpha0", 0, rnd, 4, ValueKind::U16);
  swap_run<Npp16u>("nppiLUTPaletteSwap_16u_C3A0C4R", "16u/alpha32768", 32768, rnd, 4, ValueKind::U16);
  swap_run<Npp16u>("nppiLUTPaletteSwap_16u_C3A0C4R", "16u/alpha65535", 65535, rnd, 4, ValueKind::U16);
  swap_run<Npp16u>("nppiLUTPaletteSwap_16u_C3A0C4R", "16u/alphaNeg", -1, rnd, 4, ValueKind::U16);
  swap_run<Npp16u>("nppiLUTPaletteSwap_16u_C3A0C4R", "16u/alpha65536", 65536, rnd, 4, ValueKind::U16);
}

// ===== nppiLUTPalette / Swap: contract ====================================
TEST_F(NppiLutPaletteDiff, Palette_Contract) {
  auto run8 = [&](const std::string &key, int bitSize, NppiSize roi, bool nullSrc, bool nullDst) {
    std::vector<Npp8u> hs = make_uniform<Npp8u>(4096, 830);
    std::vector<Npp8u> hd = make_uniform<Npp8u>(4096, 831);
    DeviceMemory<Npp8u> &s = dev_buf(hs);
    DeviceMemory<Npp8u> &d = dev_buf(hd);
    std::vector<Npp8u> t(256);
    for (int i = 0; i < 256; ++i) t[i] = static_cast<Npp8u>(i);
    DeviceMemory<Npp8u> &td = dev_buf(t);
    clear_cuda_error();
    NppStatus st = nppiLUTPalette_8u_C1R_Ctx(nullSrc ? nullptr : s.get(), 16, nullDst ? nullptr : d.get(), 16, roi,
                                             td.get(), bitSize, current_ctx());
    Case c("nppiLUTPalette/contract/" + key, "nppiLUTPalette_8u_C1R");
    c.status(st);
    c.finish();
  };
  run8("bits0", 0, NppiSize{16, 4}, false, false);
  run8("bitsNeg", -1, NppiSize{16, 4}, false, false);
  run8("bits9", 9, NppiSize{16, 4}, false, false);
  run8("emptyRoi", 4, NppiSize{0, 4}, false, false);
  run8("negRoi", 4, NppiSize{-4, 4}, false, false);
  run8("nullSrc", 4, NppiSize{16, 4}, true, false);
  run8("nullDst", 4, NppiSize{16, 4}, false, true);
  {
    // 16-bit source: bit size 17 fails.
    std::vector<Npp16u> hs = make_uniform<Npp16u>(4096, 832);
    std::vector<Npp16u> hd = make_uniform<Npp16u>(4096, 833);
    DeviceMemory<Npp16u> &s = dev_buf(hs);
    DeviceMemory<Npp16u> &d = dev_buf(hd);
    std::vector<Npp16u> t(65536);
    DeviceMemory<Npp16u> &td = dev_buf(t);
    clear_cuda_error();
    NppStatus st = nppiLUTPalette_16u_C1R_Ctx(s.get(), 32, d.get(), 32, NppiSize{16, 4}, td.get(), 17,
                                              current_ctx());
    Case c("nppiLUTPalette/contract/16uBits17", "nppiLUTPalette_16u_C1R");
    c.status(st);
    c.finish();
    clear_cuda_error();
    st = nppiLUTPalette_16u_C1R_Ctx(s.get(), 32, d.get(), 32, NppiSize{16, 4}, td.get(), 16, current_ctx());
    Case c2("nppiLUTPalette/contract/16uBits16", "nppiLUTPalette_16u_C1R");
    c2.status(st);
    c2.finish();
    clear_cuda_error();
    st = nppiLUTPalette_16u_C1R_Ctx(s.get(), 32, d.get(), 32, NppiSize{16, 4}, td.get(), 0, current_ctx());
    Case c3("nppiLUTPalette/contract/16uBits0", "nppiLUTPalette_16u_C1R");
    c3.status(st);
    c3.finish();
    clear_cuda_error();
    st = nppiLUTPalette_16u_C1R_Ctx(s.get(), 32, d.get(), 32, NppiSize{16, 4}, td.get(), -1, current_ctx());
    Case c4("nppiLUTPalette/contract/16uBitsNeg", "nppiLUTPalette_16u_C1R");
    c4.status(st);
    c4.finish();
    clear_cuda_error();
    st = nppiLUTPalette_16u_C1R_Ctx(s.get(), 32, d.get(), 32, NppiSize{16, 0}, td.get(), 8, current_ctx());
    Case c5("nppiLUTPalette/contract/16uH0", "nppiLUTPalette_16u_C1R");
    c5.status(st);
    c5.finish();
  }
  {
    // Swap with bad bit size / bad ROI; alpha in range so rows would be
    // modified if the call succeeded.
    std::vector<Npp8u> hs = make_uniform<Npp8u>(4096, 834);
    std::vector<Npp8u> hd = make_uniform<Npp8u>(4096, 835);
    DeviceMemory<Npp8u> &s = dev_buf(hs);
    DeviceMemory<Npp8u> &d = dev_buf(hd);
    std::vector<std::vector<Npp8u>> tables(3, std::vector<Npp8u>(256));
    for (int c = 0; c < 3; ++c)
      for (int i = 0; i < 256; ++i) tables[c][i] = static_cast<Npp8u>(i * 10 + c);
    const Npp8u *pt[3] = {dev_buf(tables[0]).get(), dev_buf(tables[1]).get(), dev_buf(tables[2]).get()};
    clear_cuda_error();
    NppStatus st = nppiLUTPaletteSwap_8u_C3A0C4R_Ctx(s.get(), 64, 7, d.get(), 68, NppiSize{16, 4}, pt, 0,
                                                     current_ctx());
    Case c1("nppiLUTPaletteSwap/contract/bits0", "nppiLUTPaletteSwap_8u_C3A0C4R");
    c1.status(st);
    c1.finish();
    clear_cuda_error();
    st = nppiLUTPaletteSwap_8u_C3A0C4R_Ctx(s.get(), 64, 7, d.get(), 68, NppiSize{-4, 4}, pt, 4, current_ctx());
    Case c2("nppiLUTPaletteSwap/contract/negRoi", "nppiLUTPaletteSwap_8u_C3A0C4R");
    c2.status(st);
    c2.finish();
  }
}
