#!/usr/bin/env python3
"""Generate NPP API wrappers from a declarative family spec.

The generator covers the mechanical part of a family: every dtype/channel
variant with the same executor call.  The family executor (kernels, validation,
workspace management) and its differential tests stay hand-written.

Spec (JSON):
{
  "family": "nppsSum",
  "module": "npps_statistics_functions.h",
  "source": "src/npps/npps_statistics_functions/npps_sum_generated.cpp",
  "includes": ["npps_sum_executor.h"],
  "call": "nppi::statistics::nppsSum::run<{dtype}>(pSrc, nLength, pSum, pDeviceBuffer, nppStreamCtx.hStream)",
  "ctx_only": false,
  "plain_self": false,      # families without _Ctx twins: emit the plain
                            # function from its own signature + call template
  "call_by_name": {"nppsSum_8u_Sfs_Ctx": "..."},  # optional per-Ctx-function call
  "exclude": ["nppsSum_32f"],              # optional exact names
  "functions": ["nppsSum_8u_Sfs"]           # optional exact-name mode
}

Placeholders: {name} {family} {dtype} {channels} {layout}

Usage:
  python3 mpp_scripts/gen_family.py --spec mpp_scripts/families/npps_sum.json
  python3 mpp_scripts/gen_family.py --spec ... --emit [--force] [--prune-stubs]

Without --emit the tool only prints what would be generated.  A function that
already has a real implementation is always skipped; a stub is generated and
reported so it can be removed from src/nppi/nppi_not_support.cpp.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "api_analysis" / "api_catalog.csv"
NOT_SUPPORT = ROOT / "src" / "nppi" / "nppi_not_support.cpp"

DTYPE_TO_CTYPE = {
    "8u": "Npp8u",
    "8s": "Npp8s",
    "16u": "Npp16u",
    "16s": "Npp16s",
    "16f": "Npp16f",
    "32u": "Npp32u",
    "32s": "Npp32s",
    "32f": "Npp32f",
    "64f": "Npp64f",
    "64u": "Npp64u",
    "64s": "Npp64s",
    "8uc": "Npp8uc",
    "8sc": "Npp8sc",
    "16uc": "Npp16uc",
    "16sc": "Npp16sc",
    "32uc": "Npp32uc",
    "32sc": "Npp32sc",
    "32fc": "Npp32fc",
    "64fc": "Npp64fc",
    "64sc": "Npp64sc",
}


@dataclass
class Func:
    name: str
    module: str
    family: str
    dtype: str
    channels: str
    layout: str
    ctx: bool
    status: str
    return_type: str
    params: str


def load_catalog(path: Path) -> Dict[str, Func]:
    funcs: Dict[str, Func] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            funcs[row["function_name"]] = Func(
                name=row["function_name"],
                module=row["module"],
                family=row["family"],
                dtype=row["dtype"],
                channels=row["channels"],
                layout=row["layout"],
                ctx=row["ctx"] == "1",
                status=row["status"],
                return_type=row["return_type"],
                params=row["params"],
            )
    return funcs


def split_params(params: str) -> List[str]:
    """Split a C parameter list on top-level commas."""
    out: List[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(params):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(params[start:i].strip())
            start = i + 1
    tail = params[start:].strip()
    if tail:
        out.append(tail)
    return out


PARAM_NAME_RE = re.compile(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\]\s*)*$")
FUNC_PTR_NAME_RE = re.compile(r"\(\s*\*\s*([A-Za-z_]\w*)\s*\)")


def param_name(param: str) -> str:
    match = FUNC_PTR_NAME_RE.search(param)
    if match:
        return match.group(1)
    match = PARAM_NAME_RE.search(param)
    if not match:
        raise ValueError(f"cannot parse parameter name from: {param!r}")
    return match.group(1)


def param_names(params: str) -> List[str]:
    return [param_name(p) for p in split_params(params)]


def choose_call(spec: dict, func: Func) -> str:
    by_name = spec.get("call_by_name", {})
    if func.name in by_name:
        return by_name[func.name]
    by_family = spec.get("call_by_family", {})
    if func.family in by_family:
        return by_family[func.family]
    for variant in spec.get("call_variants", []):
        # "name_contains" matches the function name (layout suffixes such as
        # _C3C2R / _AC4C2R are not visible in the parameter list).
        name_needle = variant.get("name_contains", "")
        if name_needle:
            if name_needle in func.name:
                return variant["call"]
            continue
        # "contains_all" requires every needle in the parameter list; the
        # single "contains" form is kept for compatibility.
        needles = variant.get("contains_all")
        if needles is None:
            needle = variant.get("contains", "")
            needles = [needle] if needle else []
        if needles and all(needle in func.params for needle in needles):
            return variant["call"]
    return spec["call"]


NPPI_LAYOUT_RE = re.compile(r"_AC4")
NPPI_CHANNEL_RE = re.compile(r"_C([1234])(?:R|IR)(?=_|$)")


def nppi_channel_count(name: str) -> int:
    """NPPI channel count from the layout suffix (C1..C4 / AC4 / P3 / P4)."""
    if NPPI_LAYOUT_RE.search(name):
        return 4
    # GradientVector C3C1R: 3 source channels, 1-channel outputs.
    if "_C3C1R" in name:
        return 3
    m = re.search(r"_[PC]([1234])R", name)
    return int(m.group(1)) if m else 1


def nppi_is_ac4(name: str) -> str:
    return "1" if NPPI_LAYOUT_RE.search(name) else "0"


def split_dtype_pair(code: str) -> Tuple[str, str]:
    """Split an nppiConvert src+dst dtype code (e.g. '16f32f') into two tokens."""
    for k in DTYPE_TO_CTYPE:
        if code.startswith(k) and code[len(k):] in DTYPE_TO_CTYPE:
            return k, code[len(k):]
    return "", ""


def render(spec: dict, func: Func, ctx_twin: Optional[Func]) -> str:
    dtype = DTYPE_TO_CTYPE.get(func.dtype, "")
    src_token, dst_token = split_dtype_pair(func.name.split("_")[1] if "_" in func.name else func.name)
    # Families whose name carries a single dtype token (e.g. nppiLUT_8u_C1R)
    # reuse the source type as destination; paired names (nppiConvert_8u32f)
    # still resolve the second token.
    dst_dtype = DTYPE_TO_CTYPE.get(dst_token, dtype)
    values = {
        "name": func.name,
        "family": spec["family"],
        "dtype": dtype,
        "dst_dtype": dst_dtype,
        "chan": str(nppi_channel_count(func.name)),
        "is_ac4": nppi_is_ac4(func.name),
        "channels": func.channels or "1",
        "layout": func.layout,
    }
    if func.ctx or ctx_twin is None:
        # A non-Ctx function without a _Ctx twin (host-only helpers such as
        # nppiMorphGetBufferSize_* or the unsharp GetBufferSize family, which
        # sets plain_self) is implemented directly, exactly like a Ctx
        # function but without any stream parameter.
        call_template = choose_call(spec, func)
        if "{dtype}" in call_template and not dtype:
            raise ValueError(f"{func.name}: spec uses {{dtype}} but the function has no dtype token")
        if "nppStreamCtx" in call_template and "nppStreamCtx" not in param_names(func.params):
            raise ValueError(f"{func.name}: spec references nppStreamCtx but the signature does not use that name")
        call = call_template.format(**values)
        unused = (
            "  (void)nppStreamCtx;\n"
            if "nppStreamCtx" not in call and "nppStreamCtx" in param_names(func.params)
            else ""
        )
        return (
            f"NppStatus {func.name}({func.params}) {{\n"
            f"{unused}"
            f"  return {call};\n"
            f"}}\n"
        )

    names = param_names(func.params)
    ctx_name = param_names(ctx_twin.params)[-1]
    ctx_type = split_params(ctx_twin.params)[-1]
    if "NppStreamContext" not in ctx_type:
        raise ValueError(f"{ctx_twin.name}: last parameter is not a NppStreamContext")
    args = ", ".join(names)
    forwarded = f"{args}, nppStreamCtx" if args else "nppStreamCtx"
    return (
        f"NppStatus {func.name}({func.params}) {{\n"
        f"  NppStreamContext nppStreamCtx;\n"
        f"  NppStatus status = nppGetStreamContext(&nppStreamCtx);\n"
        f"  if (status != NPP_NO_ERROR) {{\n"
        f"    return status;\n"
        f"  }}\n"
        f"  return {ctx_twin.name}({forwarded});\n"
        f"}}\n"
    )


def select(spec: dict, funcs: Dict[str, Func]) -> List[Func]:
    exact = set(spec.get("functions", []))
    exclude = set(spec.get("exclude", []))
    families = set(spec.get("families", [spec["family"]]))
    selected = []
    for f in funcs.values():
        if f.module != spec["module"]:
            continue
        if exact:
            if f.name not in exact:
                continue
        elif f.family not in families:
            continue
        if f.name in exclude:
            continue
        selected.append(f)
    selected.sort(key=lambda f: (not f.ctx, f.name))
    return selected


def ctx_twin_of(func: Func, funcs: Dict[str, Func]) -> Optional[Func]:
    if func.ctx:
        return func
    return funcs.get(func.name + "_Ctx")



def render_size_t_overload(func: Func, ctx_twin: Optional[Func]) -> Optional[str]:
    """C++ overload taking `size_t *hpBufferSize` (the NPP 13 buffer-size ABI).

    mcMPP headers keep the legacy int* entry point (C linkage) and add a C++
    size_t* overload, matching NVIDIA NPP >= 12.8.  The overload delegates to
    the int* implementation and widens the value.
    """
    names = param_names(func.params)
    if "hpBufferSize" not in names:
        return None
    params = split_params(func.params)
    out_params = []
    for p in params:
        if param_name(p) == "hpBufferSize":
            out_params.append("size_t * hpBufferSize")
        else:
            out_params.append(p)
    signature = ", ".join(out_params)
    args = []
    for n in names:
        args.append("hpBufferSize" if n == "hpBufferSize" else n)
    if func.ctx:
        call = f"{func.name}({', '.join(args)})"
        return (
            f"NppStatus {func.name}({signature}) {{\n"
            f"  int hpBufferSizeInt = 0;\n"
            f"  NppStatus status = {func.name}({', '.join(['&hpBufferSizeInt' if n == 'hpBufferSize' else n for n in names])});\n"
            f"  if (status == NPP_NO_ERROR && hpBufferSize) *hpBufferSize = (size_t)hpBufferSizeInt;\n"
            f"  return status;\n"
            f"}}\n"
        )
    if ctx_twin is None:
        return None
    ctx_name = param_names(ctx_twin.params)[-1]
    return (
        f"NppStatus {func.name}({signature}) {{\n"
        f"  NppStreamContext {ctx_name};\n"
        f"  NppStatus status = nppGetStreamContext(&{ctx_name});\n"
        f"  if (status != NPP_NO_ERROR) return status;\n"
        f"  return {ctx_twin.name}({', '.join(args + [ctx_name])});\n"
        f"}}\n"
    )


def generate(spec: dict, funcs: Dict[str, Func], force: bool) -> Tuple[str, List[str], List[str], List[str]]:
    selected = select(spec, funcs)
    if not selected:
        raise SystemExit("no functions matched the spec")

    emitted: List[str] = []
    skipped: List[str] = []
    stubs: List[str] = []
    chunks: List[str] = []

    for f in selected:
        if f.status == "real" and not force:
            skipped.append(f.name)
            continue
        twin = ctx_twin_of(f, funcs)
        if twin is None and not f.ctx:
            # Host-only helper without a _Ctx twin (e.g. the nppiMorphGetBufferSize_*
            # family): the spec's call is emitted directly, render() validates it.
            pass
        chunks.append(render(spec, f, twin))
        emitted.append(f.name)
        if f.status == "stub":
            stubs.append(f.name)

    header = [
        "// GENERATED by mpp_scripts/gen_family.py -- DO NOT EDIT.",
        f"// family: {spec['family']}",
        f"// spec:   {spec.get('_spec_path', '<inline>')}",
        "",
        '#include "npp.h"',
        '#include "npp_api_compat.h"',
    ]
    for inc in spec.get("includes", []):
        header.append(f"#include {inc}" if inc.startswith("<") else f'#include "{inc}"')
    header.append("")
    if spec.get("size_t_overloads"):
        extras = []
        for f in selected:
            twin = ctx_twin_of(f, funcs)
            extra = render_size_t_overload(f, twin)
            if extra is not None:
                extras.append(extra)
        if extras:
            chunks.append("#if !NPP_BUFFER_SIZE_IS_SIZE_T")
            chunks.extend(extras)
            chunks.append("#endif")
    body = "\n".join(chunks)
    body = re.sub(r"\bint(\s*)\*(\s*)(hpBufferSize|hpDeviceMemoryBufferSize)\b",
                  r"NPP_BUFFER_SIZE_TYPE\1*\2\3", body)
    body = re.sub(r"\bint(\s+)nLength\b", r"NppSignalLength\1nLength", body)
    return "\n".join(header) + "\n" + body, emitted, skipped, stubs


# ---------------------------------------------------------------------------
# Stub pruning
# ---------------------------------------------------------------------------

DEF_AT = re.compile(r"\bNppStatus\s+([A-Za-z_]\w*)\s*\(")


def find_matching(text: str, pos: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    for i in range(pos, len(text)):
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def remove_function(text: str, name: str) -> Tuple[str, bool]:
    match = re.search(rf"\bNppStatus\s+{re.escape(name)}\s*\(", text)
    if not match:
        return text, False
    open_paren = match.end() - 1
    close_paren = find_matching(text, open_paren, "(", ")")
    if close_paren < 0:
        return text, False
    brace = text.find("{", close_paren)
    if brace < 0:
        return text, False
    end = find_matching(text, brace, "{", "}")
    if end < 0:
        return text, False
    start = text.rfind("\n", 0, match.start()) + 1
    # Remove the doc comment immediately above the definition, if any.  Only
    # whitespace may sit between the comment and the definition; scanning
    # backwards from the definition keeps the span local (an anchored
    # "/\\*\\*.*?\\*/$" regex would happily start at the first comment in the
    # file and swallow everything in between).
    above = text[:start].rstrip()
    if above.endswith("*/"):
        cstart = above.rfind("/**")
        if cstart >= 0:
            start = cstart
    end = text.find("\n", end)
    if end < 0:
        end = len(text)
    else:
        end += 1
    return text[:start] + text[end:], True


def prune_stubs(names: Sequence[str]) -> int:
    text = NOT_SUPPORT.read_text()
    removed = 0
    for name in names:
        text, changed = remove_function(text, name)
        removed += 1 if changed else 0
    if removed:
        NOT_SUPPORT.write_text(text)
    return removed


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, default=CATALOG)
    ap.add_argument("--emit", action="store_true", help="write the generated source file")
    ap.add_argument("--force", action="store_true", help="regenerate functions that already exist")
    ap.add_argument("--prune-stubs", action="store_true", help="remove matching stubs from nppi_not_support.cpp")
    ap.add_argument("--list", action="store_true", help="list selected functions and exit")
    args = ap.parse_args()

    spec = json.loads(args.spec.read_text())
    spec["_spec_path"] = str(args.spec)
    funcs = load_catalog(args.catalog)

    if args.list:
        for f in select(spec, funcs):
            twin = ctx_twin_of(f, funcs)
            print(f"{f.status:8s} ctx={int(f.ctx)} {f.name}  twin={twin.name if twin else '-'}")
        return 0

    text, emitted, skipped, stubs = generate(spec, funcs, args.force)
    print(f"selected : {len(emitted) + len(skipped)}")
    print(f"emitted  : {len(emitted)}")
    print(f"skipped  : {len(skipped)} (already real)")
    if emitted:
        print("functions:")
        for name in emitted:
            marker = " (replaces stub)" if name in stubs else ""
            print(f"  + {name}{marker}")
    if skipped:
        print("already implemented (skipped):")
        for name in skipped:
            print(f"  = {name}")

    if not args.emit:
        print("\n(dry run; pass --emit to write the source file)")
        return 0

    out = ROOT / spec["source"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"\nwrote {out}")

    if args.prune_stubs and emitted:
        removed = prune_stubs(emitted)
        print(f"pruned {removed} stub definition(s) from {NOT_SUPPORT.relative_to(ROOT)}")
    elif stubs:
        print(
            f"warning: {len(stubs)} generated function(s) still have stubs in "
            f"{NOT_SUPPORT.relative_to(ROOT)}; rerun with --prune-stubs or delete them manually"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
