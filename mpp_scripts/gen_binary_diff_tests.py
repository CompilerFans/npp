#!/usr/bin/env python3
"""Generate the body (TEST_F cases) for  src/../test/unit/diff/test_npps_binary_generated.cpp.

Reads api_analysis/packages/p1.1a.csv, and for each case in CASES emits a _Ctx
TEST_F plus a plain-entry TEST_F (guarded by NPP_DIFF_HAS_NONCTX) that share a
case id.  Shape (out-of-place / in-place / accumulate / round) and element C
types come from the exact CSV signature.

Usage: python3 mpp_scripts/gen_binary_diff_tests.py   # rewrites the cpp file
"""
import csv, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "api_analysis" / "packages" / "p1.1a.csv"
OUT = ROOT / "test" / "unit" / "diff" / "test_npps_binary_generated.cpp"

CT = {"Npp8u":"uint8_t","Npp8s":"int8_t","Npp16u":"uint16_t","Npp16s":"int16_t","Npp32u":"uint32_t","Npp32s":"int32_t","Npp64s":"long long","Npp32f":"float","Npp64f":"double","Npp32fc":"Npp32fc","Npp64fc":"Npp64fc","Npp16sc":"Npp16sc","Npp32sc":"Npp32sc"}
KIND = {"Npp8u":"U8","Npp8s":"I8","Npp16u":"U16","Npp16s":"I16","Npp32u":"U32","Npp32s":"I32","Npp64s":"I64","Npp32f":"F32","Npp64f":"F64","Npp32fc":"CF32","Npp64fc":"CF64","Npp16sc":"I16","Npp32sc":"I32"}
FLOAT = {"Npp32f","Npp64f","Npp32fc","Npp64fc"}
SCINT = {"Npp16sc","Npp32sc"}

# (base, n, scale, mode).  mode == "-" for the non-round families.
CASES = [
    # nppsAdd (53 missing; representative)
    ("nppsAdd_8u_Sfs",257,4,"-"),("nppsAdd_16s",257,0,"-"),("nppsAdd_16s32f",17,0,"-"),
    ("nppsAdd_16s_Sfs",1024,8,"-"),("nppsAdd_16s_I",257,0,"-"),("nppsAdd_16s_ISfs",17,1,"-"),
    ("nppsAdd_16u",1,0,"-"),("nppsAdd_16u_Sfs",17,4,"-"),("nppsAdd_16u_ISfs",257,8,"-"),
    ("nppsAdd_32f",257,0,"-"),("nppsAdd_32f_I",17,0,"-"),("nppsAdd_32fc",17,0,"-"),
    ("nppsAdd_32s_Sfs",257,1,"-"),("nppsAdd_32sc_Sfs",17,4,"-"),("nppsAdd_32u",17,0,"-"),
    ("nppsAdd_64f",17,0,"-"),("nppsAdd_64fc",17,0,"-"),("nppsAdd_64s_Sfs",17,1,"-"),
    ("nppsAdd_8u16u",17,0,"-"),("nppsAdd_16sc_Sfs",17,4,"-"),
    # nppsSub
    ("nppsSub_8u_Sfs",17,4,"-"),("nppsSub_16s",257,0,"-"),("nppsSub_16s32f",17,0,"-"),
    ("nppsSub_16s_I",257,0,"-"),("nppsSub_16s_ISfs",17,1,"-"),("nppsSub_32f",257,0,"-"),
    ("nppsSub_32fc",17,0,"-"),("nppsSub_32s_Sfs",17,1,"-"),("nppsSub_64f",17,0,"-"),
    ("nppsSub_64fc_I",17,0,"-"),("nppsSub_16u_Sfs",17,4,"-"),
    # nppsMul
    ("nppsMul_8u_Sfs",17,4,"-"),("nppsMul_8u16u",17,0,"-"),("nppsMul_16s",257,0,"-"),
    ("nppsMul_16s32f",17,0,"-"),("nppsMul_16s32s_Sfs",17,4,"-"),("nppsMul_16s_ISfs",17,8,"-"),
    ("nppsMul_16u16s_Sfs",17,1,"-"),("nppsMul_16u_Sfs",17,4,"-"),("nppsMul_32f",257,0,"-"),
    ("nppsMul_32f32fc",17,0,"-"),("nppsMul_32fc",17,0,"-"),("nppsMul_32s32sc_Sfs",17,4,"-"),
    ("nppsMul_32s_Sfs",257,1,"-"),("nppsMul_32sc_Sfs",17,4,"-"),("nppsMul_64f",17,0,"-"),
    ("nppsMul_64fc_I",17,0,"-"),
    # nppsDiv  (note : pSrc1 divisor, pSrc2 dividend -> pDst = pSrc2/pSrc1)
    ("nppsDiv_8u_Sfs",17,4,"-"),("nppsDiv_16s_Sfs",1024,4,"-"),("nppsDiv_16u_Sfs",17,4,"-"),
    ("nppsDiv_32f",257,0,"-"),("nppsDiv_32f_I",17,0,"-"),("nppsDiv_32fc",17,0,"-"),
    ("nppsDiv_32s16s_Sfs",17,4,"-"),("nppsDiv_32s_Sfs",17,1,"-"),("nppsDiv_64f",17,0,"-"),
    # nppsDiv_Round (12)
    ("nppsDiv_Round_8u_Sfs",17,4,"NPP_RND_FINANCIAL"),("nppsDiv_Round_8u_ISfs",17,4,"NPP_RND_ZERO"),
    ("nppsDiv_Round_16s_ISfs",17,4,"NPP_RND_NEAR"),("nppsDiv_Round_16s_Sfs",257,8,"NPP_RND_FINANCIAL"),
    ("nppsDiv_Round_16u_Sfs",17,4,"NPP_RND_ZERO"),("nppsDiv_Round_16u_ISfs",17,8,"NPP_RND_NEAR"),
    # nppsAddProduct
    ("nppsAddProduct_32f",17,0,"-"),("nppsAddProduct_32fc",17,0,"-"),("nppsAddProduct_64f",17,0,"-"),
    ("nppsAddProduct_64fc",17,0,"-"),("nppsAddProduct_32s_Sfs",17,4,"-"),("nppsAddProduct_16s32s_Sfs",17,4,"-"),
]

def split_params(p):
    out=[]; d=0; s=0
    for i,ch in enumerate(p):
        if ch in "([<": d+=1
        elif ch in ")]>": d-=1
        elif ch=="," and d==0: out.append(p[s:i].strip()); s=i+1
    out.append(p[s:].strip()); return out

def ptype(params, name):
    for x in split_params(params):
        if "*" in x and re.search(r"\b"+name+r"\b", x):
            return re.sub(r"\bconst\b","",x).replace("*","").strip()
    return None

def var(host_, name, seed):
    if host_=="Npp16sc": return f"std::vector<Npp16sc> {name} = v16(n, {seed});"
    if host_=="Npp32sc": return f"std::vector<Npp32sc> {name} = v32(n, {seed});"
    if host_=="Npp32fc": return f"std::vector<Npp32fc> {name} = vfc(n, {seed});"
    if host_=="Npp64fc": return f"std::vector<Npp64fc> {name} = vfc64(n, {seed});"
    return f"std::vector<{CT[host_]}> {name} = make_vec<{CT[host_]}>(n, {seed});"

def seed(base, tag):
    return (abs(hash(base+tag)) + 1) & 0x0FFFFFFFFFFFFFFF

def gen_body(rows):
    W = []
    A = W.append
    A("")
    A("// ========================= nppsAdd =================================")
    for (base, n, scale, mode) in CASES:
        r = rows.get(base)
        if not r: 
            print("  ! missing", base); continue
        p = r["params"]; family = r["family"]
        has_src2 = ptype(p,"pSrc2") is not None
        inplace = ptype(p,"pSrcDst") is not None
        has_scale = "nScaleFactor" in p
        roundf = "nRndMode" in p
        acc = "AddProduct" in base
        a_ct = ptype(p,"pSrc1") or ptype(p,"pSrc") or ptype(p,"pSrcDst")
        b_ct = ptype(p,"pSrc2")
        d_ct = ptype(p,"pSrcDst") or ptype(p,"pDst")
        kind = KIND[d_ct]
        count = 2*n if d_ct in SCINT else n
        tol = "0, 0" if d_ct not in FLOAT else "1e-5, 1e-5"
        id = f"nppsBinary/{family}/{base}"
        sf = f", {scale}" if has_scale else ""
        if roundr: sf = f", {mode}" + (f", {scale}" if has_scale else "")
        # calls
        if inplace: frag = f"da.get(), dc.get(), n{sf}"
        elif acc:   frag = f"da.get(), db.get(), dc.get(), n{sf}"
        else:       frag = f"da.get(), db.get(), dc.get(), n{sf}"
        A(f"TEST_F(NppsBinaryArithmetic, {base}_Ctx) {{")
        A(f"  const NppSignalLength n = {n};")
        A(f"  {var(a_ct,'a',seed(base,'a'))}")
        if has_src2:
            A(f"  {var(b_ct,'b',seed(base,'b'))}")
        A(f"  {var(d_ct,'d',seed(base,'d'))}")
        A(f"  DeviceMemory<{CT[a_ct]}> da(n); da.copyFromHost(a);")
        if has_src2:
            A(f"  DeviceMemory<{CT[b_ct]}> db(n); db.copyFromHost(b);")
        A(f"  DeviceMemory<{CT[d_ct]}> dc(n); dc.copyFromHost(d);")
        A("  clear_cuda_error();")
        A(f"  NppStatus st = {base}_Ctx({frag}, current_context());")
        A(f"  finish_case(\"{id_}\", \"{base}_Ctx\", st, dc.get(), {count}, ValueKind::{kind}, {tol});")
        A("}")
        A("#if NPP_DIFF_HAS_NONCTX")
        A(f"TEST_F(NppsBinaryArithmetic, {base}_NonCtx) {{")
        A(f"  const NppSignalLength n = {n};")
        A(f"  {var(ty(a_ct),'a',seed(base,'PA'))}")  # #!/usr/bin/env python3
"""Generate representative differential tests for the npps binary arithmetic
families from api_analysis/packages/p1.1a.csv and rewrite
test/unit/diff/test_npps_binary_generated.cpp.

For each curated case it emits a _Ctx TEST_F plus a plain-entry TEST_F guarded
by NPP_DIFF_HAS_NONCTX.  Both share one case id so the 5090 recorded golden is
validated against both MPP entry points.  Invocation shape and element C types
are derived from the exact CSV signature.

Usage: python3 mpp_scripts/gen_binary_diff_tests.py
"""
import csv, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "test" / "unit" / "diff" / "test_npps_binary_generated.cpp"
CSV = ROOT / "api_analysis" / "packages" / "p1.1a.csv"

CT = {"Npp8u":"uint8_t","Npp8s":"int8_t","Npp16u":"uint16_t","Npp16s":"int16_t","Npp32u":"uint32_t","Npp32s":"int32_t","Npp64s":"long long","Npp32f":"float","Npp64f":"double","Npp32fc":"Npp32fc","Npp64fc":"Npp64fc","Npp16sc":"Npp16sc","Npp32sc":"Npp32sc"}
KIND = {"Npp8u":"U8","Npp8s":"I8","Npp16u":"U16","Npp16s":"I16","Npp32u":"U32","Npp32s":"I32","Npp64s":"I64","Npp32f":"F32","Npp64f":"F64","Npp32fc":"CF32","Npp64fc":"CF64","Npp16sc":"I16","Npp32sc":"I32"}
FLOAT = {"Npp32f","Npp64f","Npp32fc","Npp64fc"}
SCINT = {"Npp16sc","Npp32sc"}

CASES = [
    ("nppsAdd_8u_Sfs",257,4,"-"),("nppsAdd_16s",257,0,"-"),("nppsAdd_16s32f",17,0,"-"),
    ("nppsAdd_16s_Sfs",1024,8,"-"),("nppsAdd_16s_I",257,0,"-"),("nppsAdd_16s_ISfs",17,1,"-"),
    ("nppsAdd_16u",1,0,"-"),("nppsAdd_16u_Sfs",17,4,"-"),("nppsAdd_16u_ISfs",257,8,"-"),
    ("nppsAdd_32f",257,0,"-"),("nppsAdd_32f_I",17,0,"-"),("nppsAdd_32fc",17,0,"-"),
    ("nppsAdd_32s_Sfs",257,1,"-"),("nppsAdd_32sc_Sfs",17,4,"-"),("nppsAdd_32u",17,0,"-"),
    ("nppsAdd_64f",17,0,"-"),("nppsAdd_64fc",17,0,"-"),("nppsAdd_64s_Sfs",17,1,"-"),
    ("nppsAdd_8u16u",17,0,"-"),("nppsAdd_16sc_Sfs",17,4,"-"),
    ("nppsSub_8u_Sfs",17,4,"-"),("nppsSub_16s",257,0,"-"),("nppsSub_16s32f",17,0,"-"),
    ("nppsSub_16s_I",257,0,"-"),("nppsSub_16s_ISfs",17,1,"-"),("nppsSub_32f",257,0,"-"),
    ("nppsSub_32fc",17,0,"-"),("nppsSub_32s_Sfs",17,1,"-"),("nppsSub_64f",17,0,"-"),
    ("nppsSub_64fc_I",17,0,"-"),("nppsSub_16u_Sfs",17,4,"-"),
    ("nppsMul_8u_Sfs",17,4,"-"),("nppsMul_8u16u",17,0,"-"),("nppsMul_16s",257,0,"-"),
    ("nppsMul_16s32f",17,0,"-"),("nppsMul_16s32s_Sfs",17,4,"-"),("nppsMul_16s_ISfs",17,8,"-"),
    ("nppsMul_16u16s_Sfs",17,1,"-"),("nppsMul_16u_Sfs",17,4,"-"),("nppsMul_32f",257,0,"-"),
    ("nppsMul_32f32fc",17,0,"-"),("nppsMul_32fc",17,0,"-"),("nppsMul_32s32sc_Sfs",17,4,"-"),
    ("nppsMul_32s_Sfs",257,1,"-"),("nppsMul_32sc_Sfs",17,4,"-"),("nppsMul_64f",17,0,"-"),
    ("nppsMul_64fc_I",17,0,"-"),
    ("nppsDiv_8u_Sfs",17,4,"-"),("nppsDiv_16s_Sfs",1024,4,"-"),("nppsDiv_16u_Sfs",17,4,"-"),
    ("nppsDiv_32f",257,0,"-"),("nppsDiv_32f_I",17,0,"-"),("nppsDiv_32fc",17,0,"-"),
    ("nppsDiv_32s16s_Sfs",17,4,"-"),("nppsDiv_32s_Sfs",17,1,"-"),("nppsDiv_64f",17,0,"-"),
    ("nppsDiv_Round_8u_Sfs",17,4,"NPP_RND_FINANCIAL"),("nppsDiv_Round_8u_ISfs",17,4,"NPP_RND_ZERO"),
    ("nppsDiv_Round_16s_ISfs",17,4,"NPP_RND_NEAR"),("nppsDiv_Round_16s_Sfs",257,8,"NPP_RND_FINANCIAL"),
    ("nppsDiv_Round_16u_Sfs",17,4,"NPP_RND_ZERO"),("nppsDiv_Round_16u_ISfs",17,8,"NPP_RND_NEAR"),
    ("nppsAddProduct_32f",17,0,"-"),("nppsAddProduct_32fc",17,0,"-"),("nppsAddProduct_64f",17,0,"-"),
    ("nppsAddProduct_64fc",17,0,"-"),("nppsAddProduct_32s_Sfs",17,4,"-"),("nppsAddProduct_16s32s_Sfs",17,4,"-"),
]

def split_params(p):
    out=[];d=0;s=0
    for i,ch in enumerate(p):
        if ch in "([<": d+=1
        elif ch in ")]>": d-=1
        elif ch=="," and d==0: out.append(p[s:i].strip()); s=i+1
    out.append(p[s:].strip()); return out

def ptype(params,name):
    for x in split_params(params):
        if "*" in x and re.search(r"\b"+name+r"\b", x):
            before = x.split("*")[0]
            return re.sub(r"\bconst\b","",before).replace("[","").strip()
    return None

def var(ct, name, n, seed):
    h = CT[ct]
    if h == "Npp16sc": return f"std::vector<Npp16sc> {name} = v16(n,{seed});"
    if h == "Npp32sc": return f"std::vector<Npp32sc> {name} = v32(n,{seed});"
    if h == "Npp32fc": return f"std::vector<Npp32fc> {name} = vfc(n,{seed});"
    if h == "Npp64fc": return f"std::vector<Npp64fc> {name} = vfc64(n,{seed});"
    return f"std::vector<{h}> {name} = make_vec<{h}>(n,{seed});"

def seed(base, tag):
    return (abs(hash(base + tag)) + 1) & 0xFFFFFFFFFFFFFF

def build():
    rows = {r["function_name"]: r for r in csv.DictReader(open(CSV))}
    body = []
    for (base, n, scale, mode) in CASES:
        r = rows.get(base)
        if not r:
            print("MISSING", base)
            continue
        p = r["params"]; family = r["family"]
        has_src2 = ptype(p, "pSrc2") is not None
        inplace = ptype(p, "pSrcDst") is not None
        has_scale = "nScaleFactor" in p
        roundf = "nRndMode" in p
        a_ct = ptype(p, "pSrc1") or ptype(p, "pSrc") or ptype(p, "pSrcDst")
        b_ct = ptype(p, "pSrc2")
        d_ct = ptype(p, "pSrcDst") or ptype(p, "pDst")
        kind = KIND[d_ct]
        count = 2 * n if d_ct in SCINT else n
        tol = "0, 0" if d_ct not in FLOAT else "1e-5, 1e-5"
        id_ = f"nppsBinary/{family}/{base}"
        sf = ", " + str(scale) if has_scale else ""
        if roundf:
            sf = ", " + mode + (", " + str(scale) if has_scale else "")
        frag = f"da.get(), db.get(), dc.get(), n{sf}"
        if inplace:
            frag = f"da.get(), dc.get(), n{sf}"
        body.append(f"TEST_F(NppsBinaryArithmetic, {base}_Ctx) {{")
        body.append(f"  const NppSignalLength n = {n};")
        body.append("  " + var(a_ct, "a", n, seed(base, "a")))
        if has_src2:
            body.append("  " + var(b_ct, "b", n, seed(base, "b")))
        body.append("  " + var(d_ct, "d", n, seed(base, "d")))
        body.append(f"  DeviceMemory<{CT[a_ct]}> da(n); da.copyFromHost(a);")
        if has_src2:
            body.append(f"  DeviceMemory<{CT[b_ct]}> db(n); db.copyFromHost(b);")
        body.append(f"  DeviceMemory<{CT[d_ct]}> dc(n); dc.copyFromHost(d);")
        body.append("  clear_cuda_error();")
        body.append(f"  NppStatus st = {base}_Ctx({frag}, current_context());")
        body.append(f"  finish_case(\"{id_}\", \"{base}_Ctx\", st, dc.get(), {count}, ValueKind::{kind}, {tol});")
        body.append("}")
        body.append("#if NPP_DIFF_HAS_NONCTX")
        body.append(f"TEST_F(NppsBinaryArithmetic, {base}_NonCtx) {{")
        body.append(f"  const NppSignalLength n = {n};")
        body.append("  " + var(a_ct, "a", n, seed(base, "a")))
        if has_src2:
            body.append("  " + var(b_ct, "b", n, seed(base, "b")))
        body.append("  " + var(d_ct, "d", n, seed(base, "d")))
        body.append("  " + f"DeviceMemory<{CT[a_ct]}> da(n); da.copyFromHost(a);".replace("  ", " "))
        if has_src2:
            body.append("  " + f"DeviceMemory<{CT[b_ct]}> db(n); db.copyFromHost(b);".replace("  ", " "))
        body.append("  " + f"DeviceMemory<{CT[d_ct]}> dc(n); dc.copyFromHost(d);".replace("  ", " "))
        body.append("  clear_cuda_error();")
        body.append(f"  NppStatus st = {base}({frag});")
        body.append(f"  finish_case(\"{id_}\", \"{base}\", st, dc.get(), {count}, ValueKind::{kind}, {tol});")
        body.append("}")
        body.append("#endif")
        body.append("")
    base_text = open(IN).read().split("// MARKER")[0].rstrip() + "\n\n// MARKER\n\n"
    open(IN, "w").write(base_text + "\n".join(body) + "\n")

if __name__ == "__main__":
    build()
