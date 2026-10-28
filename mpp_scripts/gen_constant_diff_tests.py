#!/usr/bin/env python3
"""Generate representative differential tests for the npps constant arithmetic
families (p1.1b) into test/unit/diff/test_npps_constant_generated.cpp.

Each curated case emits a _Ctx TEST_F plus a plain-entry TEST_F guarded by
NPP_DIFF_HAS_NONCTX that share one case id, so the 5090 recorded golden is
validated against both MPP entry points.

Usage: python3 mpp_scripts/gen_constant_diff_tests.py --emit
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "api_analysis" / "packages" / "p1.1b.csv"
OUT = ROOT / "test" / "unit" / "diff" / "test_npps_constant_generated.cpp"

# dtype -> (C type, host constructor, ValueKind, is_float)
D = {
    "8u":  ("uint8_t",  "make_vec<uint8_t>",   "U8",   False),
    "16u": ("uint16_t", "make_vec<uint16_t>",  "U16",  False),
    "16s": ("int16_t",  "make_vec<int16_t>",   "I16",  False),
    "32u": ("uint32_t", "make_vec<uint32_t>",  "U32",  False),
    "32s": ("int32_t",  "make_vec<int32_t>",   "I32",  False),
    "64s": ("long long","make_vec<long long>", "I64",  False),
    "32f": ("float",    "make_vec<float>",     "F32",  True),
    "64f": ("double",   "make_vec<double>",    "F64",  True),
    "16sc":("Npp16sc",  "v16sc",               "I16",  False),
    "32sc":("Npp32sc",  "v32",                 "I32",  False),
    "32fc":("Npp32fc",  "vfc",                 "CF32", True),
    "64fc":("Npp64fc",  "vfc64",               "CF64", True),
}

def seedstr(base, tag):
    return str((abs(hash(base + tag)) + 1) & 0x0FFFFFFFFFFFFFFF)

def var_gen(dtype, name, n, seed):
    dm = D[dtype]
    if dtype in ("16sc", "32sc", "32fc", "64fc"):
        return f"std::vector<{dm[0]}> {name} = {dm[1]}(n, {seed});"
    return f"std::vector<{dm[0]}> {name} = make_vec<{dm[0]}>(n, {seed});"

FAMILIES = ("nppsAddProductC", "nppsSubCRev", "nppsMulC_Low", "nppsDivCRev",
            "nppsAddC", "nppsSubC", "nppsMulC", "nppsDivC", "nppsAndC",
            "nppsOrC", "nppsXorC", "nppsNormalize")

def family_of(base):
    for p in FAMILIES:
        if base.startswith(p): return p
    return "?"

def dst_dtype(base, inp):
    if "32f16s" in base or "Low_32f16s" in base: return "16s"
    if "64f64s" in base: return "64s"
    return inp

# (base, n, dtype, mode, scale, value_decl)
OTHERS = [
    ("nppsAddC_8u_ISfs", 17, "8u",  "I", 8,  "Npp8u nv = 3;"),
    ("nppsAddC_8u_Sfs",   257, "8u", "O", 4,  "Npp8u nv = 5;"),
    ("nppsAddC_16u_Sfs",  17, "16u", "O", 4,  "Npp16u nv = 11;"),
    ("nppsAddC_16u_ISfs", 257, "16u","I", 8,  "Npp16u nv = 9;"),
    ("nppsAddC_16s_Sfs",  1024, "16s","O", 1, "Npp16s nv = 7;"),
    ("nppsAddC_16s_ISfs", 17, "16s", "I", 1,  "Npp16s nv = 5;"),
    ("nppsAddC_16sc_Sfs", 17, "16sc","O", 1,  "Npp16sc nv; nv.re=3; nv.im=2;"),
    ("nppsAddC_32f",      257, "32f","O", None, "float nv = 0.7f;"),
    ("nppsAddC_32f_I",    17, "32f", "I", None, "float nv = 1.5f;"),
    ("nppsAddC_32fc",     17, "32fc","O", None, "Npp32fc nv; nv.re=0.5f; nv.im=1.0f;"),
    ("nppsAddC_32s_Sfs",  257, "32s","O", 8,  "Npp32s nv = 13;"),
    ("nppsAddC_32sc_Sfs", 17, "32sc","O", 1,  "Npp32sc nv; nv.re=3; nv.im=4;"),
    ("nppsAddC_64f",      1,  "64f", "O", None, "double nv = 0.25;"),
    ("nppsAddC_64fc",     1,  "64fc","O", None, "Npp64fc nv; nv.re=0.5; nv.im=1.5;"),
    ("nppsSubC_8u_Sfs",   17, "8u",  "O", 1,  "Npp8u nv = 7;"),
    ("nppsSubC_16u_ISfs", 17, "16u", "I", 8,  "Npp16u nv = 9;"),
    ("nppsSubC_16s_Sfs",  1024,"16s","O", 1,  "Npp16s nv = 5;"),
    ("nppsSubC_16s_ISfs", 17, "16s", "I", 1,  "Npp16s nv = 3;"),
    ("nppsSubC_16sc_Sfs", 17, "16sc","O", 1,  "Npp16sc nv; nv.re=4; nv.im=2;"),
    ("nppsSubC_32s_ISfs", 17, "32s", "I", 4,  "Npp32s nv = 7;"),
    ("nppsSubC_32sc_Sfs", 17, "32sc","O", 1,  "Npp32sc nv; nv.re=5; nv.im=3;"),
    ("nppsSubC_32f",      257,"32f","O", None, "float nv = 1.0f;"),
    ("nppsSubC_32f_I",    17, "32f", "I", None, "float nv = 0.5f;"),
    ("nppsSubC_32fc",     17, "32fc","O", None, "Npp32fc nv; nv.re=1.0f; nv.im=0.5f;"),
    ("nppsSubC_64f",      17, "64f", "O", None, "double nv = 2.0;"),
    ("nppsSubC_64fc_I",   17, "64fc","I", None, "Npp64fc nv; nv.re=0.5; nv.im=1.0;"),
    ("nppsSubCRev_8u_Sfs", 17, "8u",  "O", 1,  "Npp8u nv = 9;"),
    ("nppsSubCRev_16u_ISfs",17,"16u","I", 8,  "Npp16u nv = 3;"),
    ("nppsSubCRev_16s_ISfs",17,"16s","I", 1,  "Npp16s nv = 7;"),
    ("nppsSubCRev_16sc_Sfs",17,"16sc","O", 1, "Npp16sc nv; nv.re=5; nv.im=2;"),
    ("nppsSubCRev_32s_Sfs", 17, "32s","O", 1,  "Npp32s nv = 11;"),
    ("nppsSubCRev_32sc_Sfs",17,"32sc","O", 1,  "Npp32sc nv; nv.re=6; nv.im=3;"),
    ("nppsSubCRev_32f",  17, "32f", "O", None, "float nv = 4.0f;"),
    ("nppsSubCRev_32f_I",17, "32f", "I", None, "float nv = 0.75f;"),
    ("nppsSubCRev_32fc", 17, "32fc","O", None, "Npp32fc nv; nv.re=2.0f; nv.im=1.0f;"),
    ("nppsSubCRev_64f_I",17, "64f", "I", None, "double nv = 3.0;"),
    ("nppsSubCRev_64fc", 17, "64fc","O", None, "Npp64fc nv; nv.re=1.0; nv.im=2.0;"),
    ("nppsMulC_8u_ISfs",   17, "8u",  "I", 1,  "Npp8u nv = 3;"),
    ("nppsMulC_16u_Sfs",   17, "16u", "O", 1,  "Npp16u nv = 5;"),
    ("nppsMulC_16s_Sfs",   257,"16s", "O", 1,  "Npp16s nv = 7;"),
    ("nppsMulC_16s_ISfs",  17, "16s", "I", 4,  "Npp16s nv = 3;"),
    ("nppsMulC_16sc_Sfs",  17, "16sc","O", 1,  "Npp16sc nv; nv.re=3; nv.im=2;"),
    ("nppsMulC_32s_Sfs",   257,"32s", "O", 4,  "Npp32s nv = 5;"),
    ("nppsMulC_32sc_Sfs",  17, "32sc","O", 1,  "Npp32sc nv; nv.re=4; nv.im=2;"),
    ("nppsMulC_32f",       257,"32f", "O", None, "float nv = 0.7f;"),
    ("nppsMulC_32f_I",     17, "32f", "I", None, "float nv = 1.5f;"),
    ("nppsMulC_32fc",      17, "32fc","O", None, "Npp32fc nv; nv.re=0.5f; nv.im=1.0f;"),
    ("nppsMulC_64f",       17, "64f", "O", None, "double nv = 0.5;"),
    ("nppsMulC_64f_I",     17, "64f", "I", None, "double nv = 1.5;"),
    ("nppsMulC_64fc_I",    17, "64fc","I", None, "Npp64fc nv; nv.re=0.5; nv.im=1.0;"),
    ("nppsMulC_32f16s_Sfs", 257,"32f","O", 2,   "float nv = 0.6f;"),
    ("nppsMulC_64f64s_ISfs",17, "64s","I", 1,   "double nv = 0.75;"),
    ("nppsMulC_Low_32f16s", 257,"32f","O", None, "float nv = 0.5f;"),
    ("nppsDivC_8u_Sfs",   17, "8u",  "O", 1,  "Npp8u nv = 3;"),
    ("nppsDivC_16u_Sfs",  17, "16u", "O", 1,  "Npp16u nv = 5;"),
    ("nppsDivC_16u_ISfs", 17, "16u", "I", 1,  "Npp16u nv = 3;"),
    ("nppsDivC_16s_Sfs",  1024,"16s","O", 1,  "Npp16s nv = 3;"),
    ("nppsDivC_16s_ISfs", 17, "16s", "I", 1,  "Npp16s nv = 5;"),
    ("nppsDivC_16sc_Sfs", 17, "16sc","O", 1,  "Npp16sc nv; nv.re=4; nv.im=2;"),
    ("nppsDivC_32f",      257,"32f", "O", None, "float nv = 0.7f;"),
    ("nppsDivC_32f_I",    17, "32f", "I", None, "float nv = 2.0f;"),
    ("nppsDivC_32fc",     17, "32fc","O", None, "Npp32fc nv; nv.re=1.0f; nv.im=0.5f;"),
    ("nppsDivC_64f",      17, "64f", "O", None, "double nv = 0.5;"),
    ("nppsDivC_64fc",     17, "64fc","O", None, "Npp64fc nv; nv.re=2.0; nv.im=1.0;"),
    ("nppsDivCRev_16u",   17, "16u", "O", None, "Npp16u nv = 9;"),
    ("nppsDivCRev_16u_I", 17, "16u", "I", None, "Npp16u nv = 12;"),
    ("nppsDivCRev_32f",   257,"32f","O", None, "float nv = 1.0f;"),
    ("nppsDivCRev_32f_I", 17, "32f", "I", None, "float nv = 4.0f;"),
    ("nppsAndC_8u",   17, "8u",  "O", None, "Npp8u nv = 0x3A;"),
    ("nppsAndC_16u_I",17, "16u", "I", None, "Npp16u nv = 0x0F0F;"),
    ("nppsAndC_32u",  17, "32u", "O", None, "Npp32u nv = 0xF00F0000u;"),
    ("nppsOrC_8u_I",  17, "8u",  "I", None, "Npp8u nv = 0x81;"),
    ("nppsOrC_16u",   17, "16u", "O", None, "Npp16u nv = 0x0303;"),
    ("nppsOrC_32u_I", 17, "32u", "I", None, "Npp32u nv = 0x0000000Fu;"),
    ("nppsXorC_8u",   17, "8u",  "O", None, "Npp8u nv = 0x55;"),
    ("nppsXorC_16u_I",17, "16u", "I", None, "Npp16u nv = 0xFFFF;"),
    ("nppsXorC_32u",  17, "32u", "O", None, "Npp32u nv = 0x0F0F0F0Fu;"),
    ("nppsAddProductC_32f", 17, "32f", "O", None, "float nv = 0.6f;"),
]

NORM_BASES = [
    ("nppsNormalize_16s_Sfs", 257, "16s", 1,  "Npp16s vs = 5;", "8"),
    ("nppsNormalize_16sc_Sfs",17, "16sc", 1,  "Npp16sc vs; vs.re=5; vs.im=2;", "8"),
    ("nppsNormalize_32f",      257,"32f", None, "float vs = 1.0f;", "2.0f"),
    ("nppsNormalize_32fc",     17, "32fc", None,"Npp32fc vs; vs.re=1.0f; vs.im=0.5f;", "4.0f"),
    ("nppsNormalize_64f",      17, "64f", None, "double vs = 1.0;", "2.0"),
    ("nppsNormalize_64fc",     17, "64fc", None,"Npp64fc vs; vs.re=1.0; vs.im=0.5;", "4.0"),
]

HDR = [
    "// Differential tests for the npps constant arithmetic families.",
    "// GENERATED by mpp_scripts/gen_constant_diff_tests.py -- DO NOT EDIT.",
    "// A _Ctx case and its plain entry point share one case id so the 5090",
    "// recorded golden is validated against both MPP entry points.",
    "",
    '#include "npp_diff_harness.h"',
    '#include "npp_test_base.h"',
    "#include <cstdint>",
    "#include <cstdio>",
    "#include <cstdlib>",
    "#include <cstring>",
    "#include <string>",
    "#include <vector>",
    "",
    "using namespace npp_diff;",
    "using namespace npp_functional_test;",
    "",
    "namespace {",
    "",
    "NppStreamContext current_context() {",
    "  NppStreamContext ctx{};",
    "  NppStatus st = nppGetStreamContext(&ctx);",
    "  EXPECT_EQ((int)st, (int)NPP_NO_ERROR) << \"nppGetStreamContext failed\";",
    "  return ctx;",
    "}",
    "",
    "void out2(Case &c, const void *dst, size_t count, ValueKind kind, double rtol = 1e-5, double atol = 1e-5) {",
    "  if (count > 0 && dst != nullptr) c.device_output(static_cast<const Npp8u *>(dst), count, kind, rtol, atol);",
    "}",
    "",
    "template <typename T> std::vector<T> make_vec(size_t n, uint64_t s) {",
    "  std::vector<T> v(n); Rng r(s);",
    "  for (auto &x : v) {",
    "    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) x = static_cast<T>(r.unit_double() * 8.0 - 4.0);",
    "    else x = static_cast<T>(r.integer<uint64_t>(0, ~0ULL));",
    "  }",
    "  return v;",
    "}",
    "std::vector<Npp16sc> v16sc(size_t n, uint64_t s) { std::vector<Npp16sc> v(n); Rng r(s); for (auto &x : v) { x.re = r.integer<Npp16s>(-3000, 3000); x.im = r.integer<Npp16s>(-3000, 3000); } return v; }",
    "std::vector<Npp32sc> v32(size_t n, uint64_t s) { std::vector<Npp32sc> v(n); Rng r(s); for (auto &x : v) { x.re = r.integer<Npp32s>(-20000, 20000); x.im = r.integer<Npp32s>(-20000, 20000); } return v; }",
    "std::vector<Npp32fc> vfc(size_t n, uint64_t s) { std::vector<Npp32fc> v(n); Rng r(s); for (auto &x : v) { x.re = r.unit_float() * 8.f - 4.f; x.im = r.unit_float() * 8.f - 4.f; } return v; }",
    "std::vector<Npp64fc> vfc64(size_t n, uint64_t s) { std::vector<Npp64fc> v(n); Rng r(s); for (auto &x : v) { x.re = r.unit_double() * 8.0 - 4.0; x.im = r.unit_double() * 8.0 - 4.0; } return v; }",
    "",
    "void finish_case(const char *id, const char *fn, NppStatus st, const void *dst, size_t count,",
    "                 ValueKind kind, double rtol = 1e-5, double atol = 1e-5) {",
    "  Case c(id, fn);",
    "  c.status(st);",
    "  out2(c, dst, count, kind, rtol, atol);",
    "  c.finish();",
    "}",
    "",
    "} // namespace",
    "",
    "class NppsConstantArithmetic : public ::testing::Test {",
    "protected:",
    "  void TearDown() override { (void)cudaDeviceSynchronize(); }",
    "};",
    "",
]

def emit_other(base, n, dtype, mode, scale, vdecl):
    dd = dst_dtype(base, dtype)
    dm = D[dd]
    count = 2 * n if dd in ("16sc", "32sc") else n
    tol = "1e-5, 1e-5" if dm[3] else "0, 0"
    id_ = f"nppsConstant/{family_of(base)}/{base}"
    mid = (", " + str(scale)) if scale is not None else ""
    src_ct, dst_ct = D[dtype][0], D[dd][0]
    if mode == "I":
        ctx_call = f"{base}_Ctx(nv, da.get(), n{mid}, current_context())"
        plain_call = f"{base}(nv, da.get(), n{mid})"
        dst_ptr = "da.get()"
    else:
        ctx_call = f"{base}_Ctx(da.get(), nv, dc.get(), n{mid}, current_context())"
        plain_call = f"{base}(da.get(), nv, dc.get(), n{mid})"
        dst_ptr = "dc.get()"
    L = []
    for flavour, call, func in (("Ctx", ctx_call, base + "_Ctx"), ("NonCtx", plain_call, base)):
        L.append(f"TEST_F(NppsConstantArithmetic, {base}_{flavour}) {{")
        L.append(f"  const NppSignalLength n = {n};")
        L.append("  " + var_gen(dtype, "a", n, seedstr(base, "a")))
        if mode != "I":
            L.append("  " + var_gen(dd, "d", n, seedstr(base, "d")))
        L.append("  " + vdecl)
        if mode == "I":
            L.append(f"  DeviceMemory<{src_ct}> da(n); da.copyFromHost(a);")
        else:
            L.append(f"  DeviceMemory<{src_ct}> da(n); da.copyFromHost(a);")
            L.append(f"  DeviceMemory<{dst_ct}> dc(n); dc.copyFromHost(d);")
        L.append("  clear_cuda_error();")
        L.append(f"  NppStatus st = {call};")
        L.append(f"  finish_case(\"{id_}\", \"{func}\", st, {dst_ptr}, {count}, ValueKind::{dm[2]}, {tol});")
        L.append("}")
        if flavour == "Ctx":
            L.append("#if NPP_DIFF_HAS_NONCTX")
        else:
            L.append("#endif")
        L.append("")
    return L

DIVT = {"16s":"int", "16sc":"int", "32f":"float", "32fc":"float",
        "64f":"double", "64fc":"double"}

def emit_normalize(base, n, dtype, scale, vsub_decl, vdiv_expr):
    dm = D[dtype]
    count = 2 * n if dtype == "16sc" else n
    tol = "1e-5, 1e-5" if dm[3] else "0, 0"
    id_ = f"nppsConstant/nppsNormalize/{base}"
    mid = (", " + str(scale)) if scale is not None else ""
    vdt = DIVT[dtype]
    L = []
    for flavour, func in (("Ctx", base + "_Ctx"), ("NonCtx", base)):
        L.append(f"TEST_F(NppsConstantArithmetic, {base}_{flavour}) {{")
        L.append(f"  const NppSignalLength n = {n};")
        L.append("  " + var_gen(dtype, "a", n, seedstr(base, "a")))
        L.append("  " + var_gen(dtype, "d", n, seedstr(base, "d")))
        L.append("  " + vsub_decl)
        L.append(f"  {vdt} vd = {vdiv_expr};")
        L.append(f"  DeviceMemory<{D[dtype][0]}> da(n); da.copyFromHost(a);")
        L.append(f"  DeviceMemory<{D[dtype][0]}> dc(n); dc.copyFromHost(d);")
        L.append("  clear_cuda_error();")
        if flavour == "Ctx":
            L.append(f"  NppStatus st = {base}_Ctx(da.get(), dc.get(), n, vs, vd{mid}, current_context());")
        else:
            L.append(f"  NppStatus st = {base}(da.get(), dc.get(), n, vs, vd{mid});")
        L.append(f"  finish_case(\"{id_}\", \"{func}\", st, dc.get(), {count}, ValueKind::{dm[2]}, {tol});")
        L.append("}")
        if flavour == "Ctx":
            L.append("#if NPP_DIFF_HAS_NONCTX")
        else:
            L.append("#endif")
        L.append("")
    return L

def emit_contract():
    L = ["// ===== contract cases ====="]
    L.append("TEST_F(NppsConstantArithmetic, ContractNullPtr_Ctx) {")
    L.append("  const NppSignalLength n = 16;")
    L.append("  std::vector<int16_t> a = make_vec<int16_t>(n, 999);")
    L.append("  Npp16s nv = 3;")
    L.append("  DeviceMemory<int16_t> da(n); da.copyFromHost(a);")
    L.append("  clear_cuda_error();")
    L.append("  NppStatus st = nppsAddC_16s_Sfs_Ctx(da.get(), nv, nullptr, n, 1, current_context());")
    L.append('  finish_case("nppsConstant/contract/addc_null", "nppsAddC_16s_Sfs_Ctx", st, nullptr, 0, ValueKind::I16, 0, 0);')
    L.append("}")
    L.append("#if NPP_DIFF_HAS_NONCTX")
    L.append("TEST_F(NppsConstantArithmetic, ContractNullPtr_NonCtx) {")
    L.append("  const NppSignalLength n = 16;")
    L.append("  std::vector<int16_t> a = make_vec<int16_t>(n, 999);")
    L.append("  Npp16s nv = 3;")
    L.append("  DeviceMemory<int16_t> da(n); da.copyFromHost(a);")
    L.append("  clear_cuda_error();")
    L.append("  NppStatus st = nppsAddC_16s_Sfs(da.get(), nv, nullptr, n, 1);")
    L.append('  finish_case("nppsConstant/contract/addc_null", "nppsAddC_16s_Sfs", st, nullptr, 0, ValueKind::I16, 0, 0);')
    L.append("}")
    L.append("#endif")
    L.append("")
    L.append("TEST_F(NppsConstantArithmetic, ContractZeroLength_Ctx) {")
    L.append("  const NppSignalLength n = 0;")
    L.append("  Npp16s nv = 3; Npp16s dummy = 0;")
    L.append("  clear_cuda_error();")
    L.append("  NppStatus st = nppsAddC_16s_Sfs_Ctx(&dummy, nv, &dummy, n, 1, current_context());")
    L.append('  finish_case("nppsConstant/contract/zeroLen", "nppsAddC_16s_Sfs_Ctx", st, nullptr, 0, ValueKind::I16, 0, 0);')
    L.append("}")
    L.append("#if NPP_DIFF_HAS_NONCTX")
    L.append("TEST_F(NppsConstantArithmetic, ContractZeroLength_NonCtx) {")
    L.append("  const NppSignalLength n = 0;")
    L.append("  Npp16s nv = 3; Npp16s dummy = 0;")
    L.append("  clear_cuda_error();")
    L.append("  NppStatus st = nppsAddC_16s_Sfs(&dummy, nv, &dummy, n, 1);")
    L.append('  finish_case("nppsConstant/contract/zeroLen", "nppsAddC_16s_Sfs", st, nullptr, 0, ValueKind::I16, 0, 0);')
    L.append("}")
    L.append("#endif")
    L.append("")
    L.append("TEST_F(NppsConstantArithmetic, ContractNegativeLen_Ctx) {")
    L.append("  const NppSignalLength n = -4;")
    L.append("  std::vector<int16_t> a = make_vec<int16_t>(8, 42);")
    L.append("  Npp16s nv = 3;")
    L.append("  DeviceMemory<int16_t> da(8); da.copyFromHost(a);")
    L.append("  DeviceMemory<int16_t> dc(8); dc.copyFromHost(a);")
    L.append("  clear_cuda_error();")
    L.append("  NppStatus st = nppsAddC_16s_Sfs_Ctx(da.get(), nv, dc.get(), n, 1, current_context());")
    L.append('  finish_case("nppsConstant/contract/negLen", "nppsAddC_16s_Sfs_Ctx", st, nullptr, 0, ValueKind::I16, 0, 0);')
    L.append("}")
    L.append("#if NPP_DIFF_HAS_NONCTX")
    L.append("TEST_F(NppsConstantArithmetic, ContractNegativeLen_NonCtx) {")
    L.append("  const NppSignalLength n = -4;")
    L.append("  std::vector<int16_t> a = make_vec<int16_t>(8, 42);")
    L.append("  Npp16s nv = 3;")
    L.append("  DeviceMemory<int16_t> da(8); da.copyFromHost(a);")
    L.append("  DeviceMemory<int16_t> dc(8); dc.copyFromHost(a);")
    L.append("  clear_cuda_error();")
    L.append("  NppStatus st = nppsAddC_16s_Sfs(da.get(), nv, dc.get(), n, 1);")
    L.append('  finish_case("nppsConstant/contract/negLen", "nppsAddC_16s_Sfs", st, nullptr, 0, ValueKind::I16, 0, 0);')
    L.append("}")
    L.append("#endif")
    return L

def main():
    emit = "--emit" in sys.argv
    W = list(HDR) + ["// MARKER", ""]
    last = None
    for t in OTHERS:
        fam = family_of(t[0])
        if fam != last:
            W.append(f"// ===== {fam} =====")
            last = fam
        W += emit_other(*t)
    W.append("// ===== nppsNormalize =====")
    for t in NORM_BASES:
        W += emit_normalize(*t)
    W += emit_contract()
    text = "\n".join(W) + "\n"
    if emit:
        OUT.write_text(text)
        print("wrote %d cases -> %s" % (len(OTHERS) + len(NORM_BASES), OUT))
    else:
        print(text)

if __name__ == "__main__":
    for t in OTHERS + NORM_BASES:
        fn = t[0]
        valid = False
        for r in csv.DictReader(open(CSV)):
            rr = r["function_name"]
            if rr == fn or rr == fn + "_Ctx":
                valid = True
                break
        if not valid:
            print("WARN base not in csv:", fn)
    main()
