#!/usr/bin/env python3
"""Generate the npps binary 1D executor backend (.cu with explicit
instantiations) and the six per-family JSON specs from the p1.1a package CSV.

Outputs:
  * src/npps/npps_arithmetic_operations/npps_binary_ops.cu
  * mpp_scripts/families/npps_binary_<family>.json
"""
import csv, re, collections, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "api_analysis" / "packages" / "p1.1a.csv"
OUT_CU = ROOT / "src" / "npps" / "npps_arithmetic_operations" / "npps_binary_ops.cu"
FAM_DIR = ROOT / "mpp_scripts" / "families"

def split_params(params):
    out=[]; depth=0; start=0
    for i,ch in enumerate(params):
        if ch in "([<": depth+=1
        elif ch in ")]>": depth-=1
        elif ch=="," and depth==0:
            out.append(params[start:i].strip()); start=i+1
    out.append(params[start:].strip())
    return out

def type_of(params, name):
    for part in split_params(params):
        p = part.strip()
        m = re.match(r"^(?:const\s+)?([A-Za-z_]\w*)\s*\*+\s*(?:const\s+)?([A-Za-z_]\w*)$", p)
        if m and m.group(2) == name:
            return m.group(1)
    raise KeyError(f"param {name} not found in: {params}")

rows = list(csv.DictReader(open(CSV)))

def classify(fam, params):
    has_sf = "nScaleFactor" in params
    inpl   = "pSrcDst" in params
    if fam == "nppsAddProduct":
        return "AddProductOp", "binaryAccum", "pSrc1", "pSrc2", "pDst", ("nScaleFactor" if has_sf else "0")
    if fam == "nppsDiv_Round":
        entry = "binaryRoundInplace" if inpl else "binaryRound"
        if inpl: return "DivRoundOp", entry, "pSrc", "pSrcDst", "pSrcDst", "nScaleFactor"
        return "DivRoundOp", entry, "pSrc2", "pSrc1", "pDst", "nScaleFactor"
    if fam == "nppsDiv":
        entry = "binaryInplace" if inpl else "binary"
        if inpl: return "DivOp", entry, "pSrc", "pSrcDst", "pSrcDst", ("nScaleFactor" if has_sf else "0")
        return "DivOp", entry, "pSrc2", "pSrc1", "pDst", ("nScaleFactor" if has_sf else "0")
    op = {"nppsAdd":"AddOp","nppsSub":"SubOp","nppsMul":"MulOp"}[fam]
    entry = "binaryInplace" if inpl else "binary"
    if inpl: return op, entry, "pSrc", "pSrcDst", "pSrcDst", ("nScaleFactor" if has_sf else "0")
    return op, entry, "pSrc1", "pSrc2", "pDst", ("nScaleFactor" if has_sf else "0")

inst = collections.defaultdict(set)
for r in rows:
    fam=r["family"]; params=r["params"]
    F, entry, Aa, Ba, Ca, scale = classify(fam, params)
    inst[(F, entry)].add((type_of(params,Aa), type_of(params,Ba), type_of(params,Ca)))

# ---------------- executor .cu ----------------
ENTRY_NPARAMS = {
    "binary":         2,   # (A*, B*)
    "binaryInplace":  1,
    "binaryAccum":    2,
    "binaryRound":    2,
    "binaryRoundInplace": 1,
}
ENTRY_SIG = {
    "binary": ("template <typename F, typename A, typename B, typename D>",
               "NppStatus binary(const A* a, const B* b, D* d, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream)",
               "binary_kernel<A, B, D, F>"),
    "binaryInplace": ("template <typename F, typename A, typename D>",
               "NppStatus binaryInplace(const A* a, D* d, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream)",
               "binary_inplace_kernel<A, D, F>"),
    "binaryAccum": ("template <typename F, typename A, typename B, typename D>",
               "NppStatus binaryAccum(const A* a, const B* b, D* d, NppSignalLength nLength, int nScaleFactor, cudaStream_t stream)",
               "binary_accum_kernel<A, B, D, F>"),
    "binaryRound": ("template <typename F, typename A, typename B, typename D>",
               "NppStatus binaryRound(const A* a, const B* b, D* d, NppSignalLength nLength, NppRoundMode nRndMode, int nScaleFactor, cudaStream_t stream)",
               "binary_round_kernel<A, B, D, F>"),
    "binaryRoundInplace": ("template <typename F, typename A, typename D>",
               "NppStatus binaryRoundInplace(const A* a, D* d, NppSignalLength nLength, NppRoundMode nRndMode, int nScaleFactor, cudaStream_t stream)",
               "binary_inplace_round_kernel<A, D, F>"),
}

def launch_args(entry):
    if entry == "binaryInplace":       return "a, d, nLength, op, nScaleFactor"
    if entry == "binaryAccum":         return "a, b, d, nLength, op, nScaleFactor"
    if entry == "binaryRound":         return "a, b, d, nLength, nRndMode, op, nScaleFactor"
    if entry == "binaryRoundInplace":  return "a, d, nLength, nRndMode, op, nScaleFactor"
    return "a, b, d, nLength, op, nScaleFactor"

# 128-bit packet dispatch helpers defined in npps_elementwise_executor.h
# (package opt_elem). Each returns true when the vectorized kernel launched;
# the original scalar kernel runs otherwise.
VEC_LAUNCH = {
    "binary":             "detail::launchVecBinary(a, b, d, n, op, nScaleFactor, stream)",
    "binaryInplace":      "detail::launchVecInplace(a, d, n, op, nScaleFactor, stream)",
    "binaryAccum":        "detail::launchVecAccum(a, b, d, n, op, nScaleFactor, stream)",
    "binaryRound":        "detail::launchVecRound(a, b, d, n, nRndMode, op, nScaleFactor, stream)",
    "binaryRoundInplace": "detail::launchVecRoundInplace(a, d, n, nRndMode, op, nScaleFactor, stream)",
}

def write_cu():
    defn = []
    defn.append("// GENERATED by mpp_scripts/gen_binary_ops.py -- DO NOT EDIT.")
    defn.append("// Shared 1D element-wise kernels + executor for the npps binary families.")
    defn.append('#include "npp.h"')
    defn.append('#include "npp_version_compat.h"')
    defn.append('#include "npps_elementwise_executor.h"')
    defn.append("")
    defn.append("namespace npps { namespace elementwise {")
    defn.append("")
    for entry in ["binary","binaryInplace","binaryAccum","binaryRound","binaryRoundInplace"]:
        tmpl, sig, kernel = ENTRY_SIG[entry]
        defn.append(tmpl)
        defn.append(sig + " {")
        defn.append("  if (!a" + ("" if entry.startswith("binaryRoundInplace") or entry=="binaryInplace" else " || !b") + " || !d) return NPP_NULL_POINTER_ERROR;")
        defn.append("  if (nLength < 0) return NPP_SIZE_ERROR;")
        defn.append("  if (nLength == 0) return NPP_NO_ERROR;")
        defn.append("  const size_t n = static_cast<size_t>(nLength);")
        defn.append("  const size_t grid = (n + detail::kBlock - 1) / detail::kBlock;")
        defn.append("  F op;")
        defn.append("  (void)cudaGetLastError();  // clear stale error before launch")
        defn.append("  if (!" + VEC_LAUNCH[entry] + ")")
        defn.append("    {k}<<<(unsigned)grid, detail::kBlock, 0, stream>>>({args});".format(k=kernel,args=launch_args(entry)))
        defn.append("  return detail::checkCudaKernelError();")
        defn.append("}")
        defn.append("")
    # explicit instantiations
    inst_lines=[]
    for (F, entry), combos in inst.items():
        for (A,B,C) in sorted(combos):
            tmpl, sig, kern = ENTRY_SIG[entry]
            # build explicit instantiation: template NppStatus entry<F, A, B, C>(...)
            if entry.endswith("Inplace"):
                full = f"{entry}<{F}, {A}, {C}>"
            else:
                full = f"{entry}<{F}, {A}, {B}, {C}>"

            # deduce signature parameter list
            if entry=="binaryRoundInplace":
                args=f"const {A}* a, {C}* d, NppSignalLength, NppRoundMode, int, cudaStream_t"
            elif entry=="binaryInplace":
                args=f"const {A}* a, {C}* d, NppSignalLength, int, cudaStream_t"
            else:
                args=f"const {A}* a, const {B}* b, {C}* d, NppSignalLength, int, cudaStream_t"
            if entry in ("binaryRound",):
                args=f"const {A}* a, const {B}* b, {C}* d, NppSignalLength, NppRoundMode, int, cudaStream_t"
            inst_lines.append(f"template NppStatus {full}({args});")
    defn.append("\n".join(inst_lines))
    defn.append("")
    defn.append("} }  // namespace npps::elementwise")
    return "\n".join(defn) + "\n"

OUT_CU.write_text(write_cu())
print("wrote", OUT_CU)

# ---------------- per-family JSON specs ----------------
def build_spec(family, op):
    bin_b = f"npps::elementwise::binary<npps::elementwise::{op}>"
    bin_i = f"npps::elementwise::binaryInplace<npps::elementwise::{op}>"
    binop = bin_b
    sp = {
        "family": family,
        "module": "npps_arithmetic_and_logical_operations.h",
        "source": f"src/npps/npps_arithmetic_operations/{family}_generated.cpp",
        "includes": ["npps_elementwise_executor.h"],
        "call_variants": [],
        "call": "",
    }
    if family == "nppsAddProduct":
        acc = f"npps::elementwise::binaryAccum<npps::elementwise::{op}>"
        sp["call_variants"].append({"contains": "nScaleFactor",
            "call": f"{acc}(pSrc1, pSrc2, pDst, nLength, nScaleFactor, nppStreamCtx.hStream)"})
        sp["call"] = f"{acc}(pSrc1, pSrc2, pDst, nLength, 0, nppStreamCtx.hStream)"
    elif family == "nppsDiv_Round":
        rin = "npps::elementwise::binaryRoundInplace<npps::elementwise::DivRoundOp>"
        rout = "npps::elementwise::binaryRound<npps::elementwise::DivRoundOp>"
        sp["call_variants"].append({"contains": "pSrcDst",
            "call": f"{rin}(pSrc, pSrcDst, nLength, nRndMode, nScaleFactor, nppStreamCtx.hStream)"})
        sp["call"] = f"{rout}(pSrc2, pSrc1, pDst, nLength, nRndMode, nScaleFactor, nppStreamCtx.hStream)"
    elif family == "nppsDiv":
        sp["call_variants"] = [
            {"contains": "pSrcDst, int nLength, int nScaleFactor",
             "call": f"{bin_i}(pSrc, pSrcDst, nLength, nScaleFactor, nppStreamCtx.hStream)"},
            {"contains": "pSrcDst",
             "call": f"{bin_i}(pSrc, pSrcDst, nLength, 0, nppStreamCtx.hStream)"},
            {"contains": "int nLength, int nScaleFactor",
             "call": f"{binop}(pSrc2, pSrc1, pDst, nLength, nScaleFactor, nppStreamCtx.hStream)"},
        ]
        sp["call"] = f"{binop}(pSrc2, pSrc1, pDst, nLength, 0, nppStreamCtx.hStream)"
    else:  # Add / Sub / Mul
        # NPP subtraction is "reversed": nppsSub computes pDst = pSrc2 - pSrc1
        # (see the SubOp comment in npps_elementwise_executor.h), so the
        # out-of-place calls swap the source operands for nppsSub.
        s1, s2 = ("pSrc2", "pSrc1") if family == "nppsSub" else ("pSrc1", "pSrc2")
        sp["call_variants"] = [
            {"contains": "pSrcDst, int nLength, int nScaleFactor",
             "call": f"{bin_i}(pSrc, pSrcDst, nLength, nScaleFactor, nppStreamCtx.hStream)"},
            {"contains": "pSrcDst",
             "call": f"{bin_i}(pSrc, pSrcDst, nLength, 0, nppStreamCtx.hStream)"},
            {"contains": "int nLength, int nScaleFactor",
             "call": f"{binop}({s1}, {s2}, pDst, nLength, nScaleFactor, nppStreamCtx.hStream)"},
        ]
        sp["call"] = f"{binop}({s1}, {s2}, pDst, nLength, 0, nppStreamCtx.hStream)"
    return sp

FAM_OP = {
    "nppsAdd": "AddOp",
    "nppsSub": "SubOp",
    "nppsMul": "MulOp",
    "nppsDiv": "DivOp",
    "nppsAddProduct": "AddProductOp",
    "nppsDiv_Round": "DivRoundOp",
}

FAM_DIR.mkdir(exist_ok=True)
for fam, op in FAM_OP.items():
    spec = build_spec(fam, op)
    out = FAM_DIR / f"npps_binary_{fam}.json"
    out.write_text(json.dumps(spec, indent=2) + "\n")
    print("wrote", out)
