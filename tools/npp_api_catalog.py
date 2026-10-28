#!/usr/bin/env python3
"""Build an accurate NPP API catalog for mcMPP.

Difference from tools/check_coverage.py:
  * the public contract is parsed from API/ (API-unified), not a version CSV;
  * a definition whose body is exactly ``return NPP_NOT_IMPLEMENTED_ERROR;``
    is classified as ``stub`` instead of ``implemented``;
  * each function is classified by family, dtype, channels, layout, in-place
    and buffer-helper axes so work can be planned and generated per family.

Outputs (with --update):
  api_analysis/api_catalog.csv    one row per public function
  api_analysis/coverage.csv       legacy-compatible columns, corrected status
  api_analysis/backlog.csv        family-level backlog used by work packages
  api_analysis/coverage_summary.md
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
API_DIR = ROOT / "API"
SRC_DIR = ROOT / "src"
TEST_DIR = ROOT / "test"
OUT_DIR = ROOT / "api_analysis"

sys.path.insert(0, str(ROOT / "tools"))
from analyze_api import APIAnalyzer  # noqa: E402

# Data-type tokens that separate <family> from the variant suffix.
DTYPE_RE = re.compile(
    r"_(?:8u|8s|16u|16s|16f|32u|32s|32f|64f|8uc|8sc|16uc|16sc|32uc|32sc|32fc|64fc)(?=_|$)"
)
CHANNEL_RE = re.compile(r"^(?P<prefix>AC|P)?(?P<c>C(?P<cm>\d+)|(?P<pm>\d+))(?P<rest>.*)$")

BUFFER_HELPER_MARKERS = (
    "GetBufferHostSize",
    "GetBufferSize",
    "GetBufferSizeHost",
    "GetGeometryListsSize",
    "GetGeometryListsHostSize",
)


@dataclass
class ApiFunc:
    name: str
    module: str
    return_type: str
    params: str
    signature: str
    family: str = ""
    group: str = ""
    dtype: str = ""
    channels: str = ""
    layout: str = ""
    inplace: bool = False
    ctx: bool = False
    buffer_helper: bool = False
    status: str = "missing"  # real | stub | missing
    impl_file: str = ""
    tested: bool = False
    test_files: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


def _dedupe_overloads(funcs: Iterable[ApiFunc]) -> List[ApiFunc]:
    """Keep one declaration per function name.

    API-unified declares some npps entry points twice in C++ mode: the
    extern "C" ``int`` signature and a ``size_t`` overload.  Prefer the
    int/C-ABI declaration because that is the symbol the library defines.
    """
    by_name: Dict[str, List[ApiFunc]] = defaultdict(list)
    for f in funcs:
        by_name[f.name].append(f)

    picked: List[ApiFunc] = []
    for name, variants in by_name.items():
        if len(variants) == 1:
            picked.append(variants[0])
            continue

        def score(v: ApiFunc) -> Tuple[int, int]:
            has_size_t = "size_t" in v.params
            return (0 if not has_size_t else 1, len(v.params))

        variants.sort(key=score)
        picked.append(variants[0])
    return picked


def parse_api_headers(api_dir: Path) -> List[ApiFunc]:
    analyzer = APIAnalyzer(api_dir)
    raw = analyzer.analyze()
    funcs = []
    for sig in raw.values():
        funcs.append(
            ApiFunc(
                name=sig.name,
                module=sig.file,
                return_type=sig.return_type,
                params=sig.params,
                signature=sig.signature(),
            )
        )
    funcs = _dedupe_overloads(funcs)
    for f in funcs:
        classify(f)
    return sorted(funcs, key=lambda f: (f.module, f.name))


def classify(f: ApiFunc) -> None:
    base = f.name
    f.ctx = base.endswith("_Ctx")
    if f.ctx:
        base = base[:-4]

    f.buffer_helper = any(m in base for m in BUFFER_HELPER_MARKERS)
    f.inplace = bool(re.search(r"_I(?:_|$)|(?:^|_)I[0-9A-Z]", base)) or bool(
        re.search(r"C[0-9]+IR", base)
    )

    m = DTYPE_RE.search(base)
    if m:
        f.family = base[: m.start()]
        rest = base[m.end() - 1 :]
    else:
        f.family = base
        rest = ""

    f.group = f.family
    for marker in BUFFER_HELPER_MARKERS:
        if f.group.endswith(marker):
            f.group = f.group[: -len(marker)]
            break

    # dtype is the first token after the family
    if m:
        dtype_match = re.match(r"_(8u|8s|16u|16s|16f|32u|32s|32f|64f|8uc|8sc|16uc|16sc|32uc|32sc|32fc|64fc)(_|$)", rest)
        f.dtype = dtype_match.group(1) if dtype_match else ""

    # channel/layout token: first token after dtype, e.g. C1R, AC4R, C1IR,
    # C3CMR, P3R, C1C3R.  Only the first token is used for the generic axis;
    # irregular layouts stay available through the full signature.
    tail = rest
    if f.dtype:
        tail = tail[len(f.dtype) + 1 :]
    tok = tail.split("_", 1)[0] if tail else ""
    cm = re.match(r"^(AC|P)?(C\d+|\d+)(I?)(M?)(.*)$", tok)
    if cm:
        prefix, ch, ir, mm, extra = cm.groups()
        if ch.startswith("C"):
            f.channels = ch[1:]
        else:
            f.channels = ch
        if prefix == "AC":
            f.layout = "ac4"
        elif prefix == "P":
            f.layout = "planar"
        elif mm:
            f.layout = "masked"
        else:
            f.layout = "packed"
        if ir:
            f.inplace = True
    elif "AC4" in tok:
        f.layout = "ac4"
        f.channels = "4"


# ---------------------------------------------------------------------------
# Source scanning
# ---------------------------------------------------------------------------

DEF_RE = re.compile(r"\bNppStatus\s+(nppi\w+|npps\w+)\s*\(")
MALLOC_RE = re.compile(r"\bNpp\w+\s*\*\s*(nppiMalloc_\w+|nppsMalloc_\w+|nppiMalloc\w*|nppsMalloc\w*)\s*\(")
FREE_RE = re.compile(r"\bvoid\s+(nppiFree|nppsFree)\s*\(")
STUB_BODY_RE = re.compile(r"^\s*return\s+NPP_NOT_IMPLEMENTED_ERROR\s*;\s*$")


def _find_matching(text: str, pos: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    i = pos
    n = len(text)
    while i < n:
        c = text[i]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _function_body(text: str, open_paren: int) -> Optional[str]:
    """Return the body of a function starting at ``open_paren`` or None."""
    close_paren = _find_matching(text, open_paren, "(", ")")
    if close_paren < 0:
        return None
    # skip constructor-initializer/attributes/whitespace up to the body
    i = close_paren + 1
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    if i >= len(text) or text[i] != "{":
        return None
    end = _find_matching(text, i, "{", "}")
    if end < 0:
        return None
    return text[i + 1 : end]


def _public_names_for_impl(name: str) -> List[str]:
    """Map a *_impl helper to the public entry points it backs.

    ``nppiCopy_8u_C1R_Ctx_impl`` backs both ``nppiCopy_8u_C1R_Ctx`` and (via
    the macro-generated adapter) ``nppiCopy_8u_C1R``.  The public wrappers are
    emitted by token-pasting macros and are invisible to a text scan.
    """
    if name.endswith("_Ctx_impl"):
        base = name[: -len("_Ctx_impl")]
        return [base + "_Ctx", base]
    if name.endswith("_impl"):
        base = name[: -len("_impl")]
        return [base, base + "_Ctx"]
    return []


def scan_sources(src_dir: Path) -> Dict[str, Tuple[str, str]]:
    """Return {function_name: (status, relative_file)} for definitions.

    Only definitions with a body are recorded; a body that is exactly
    ``return NPP_NOT_IMPLEMENTED_ERROR;`` is classified as a stub.  Real
    ``*_impl`` helpers mark their public entry points as real because this
    repository emits the public wrappers from macros.
    """
    found: Dict[str, Tuple[str, str]] = {}
    for path in sorted(list(src_dir.rglob("*.cpp")) + list(src_dir.rglob("*.cu"))):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = str(path.relative_to(src_dir.parent))
        for regex in (DEF_RE, MALLOC_RE, FREE_RE):
            for m in regex.finditer(text):
                name = m.group(1)
                if regex is FREE_RE:
                    found.setdefault(name, ("real", rel))
                    continue
                body = _function_body(text, m.end() - 1)
                if body is None:
                    continue
                status = "stub" if STUB_BODY_RE.match(body) else "real"
                if status == "real" or name not in found:
                    found[name] = (status, rel)

                impl_public = _public_names_for_impl(name)
                if impl_public and status == "real":
                    for public in impl_public:
                        # Do not overwrite an explicit stub with an impl mark.
                        if found.get(public, ("", ""))[0] != "stub":
                            found.setdefault(public, ("real", rel))
    return found


# ---------------------------------------------------------------------------
# Test scanning
# ---------------------------------------------------------------------------

TEST_RE = re.compile(r"\b(nppi\w+|npps\w+)\s*\(")
TEST_PTR_RE = re.compile(r"&\s*(nppi\w+|npps\w+)\b")


def scan_tests(test_dir: Path) -> Dict[str, List[str]]:
    found: Dict[str, List[str]] = defaultdict(list)
    for path in sorted(test_dir.rglob("test_*.cpp")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = str(path.relative_to(test_dir.parent))
        for regex in (TEST_RE, TEST_PTR_RE):
            for m in regex.finditer(text):
                found[m.group(1)].append(rel)
    return found


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def build(api_dir: Path, src_dir: Path, test_dir: Path) -> List[ApiFunc]:
    funcs = parse_api_headers(api_dir)
    impls = scan_sources(src_dir)
    tests = scan_tests(test_dir)

    # Malloc/free are not in the nppi/npps headers inventory; add them from
    # the csv surface only if present.  They are reported separately.

    for f in funcs:
        if f.name in impls:
            f.status, f.impl_file = impls[f.name]
        else:
            f.status = "missing"
        files = tests.get(f.name, [])
        if not files and f.ctx:
            files = tests.get(f.name[:-4], [])
        f.tested = bool(files)
        f.test_files = sorted(set(files))
    return funcs


def write_catalog(funcs: List[ApiFunc], out_dir: Path) -> None:
    fields = [
        "function_name",
        "module",
        "family",
        "group",
        "dtype",
        "channels",
        "layout",
        "inplace",
        "ctx",
        "buffer_helper",
        "status",
        "impl_file",
        "tested",
        "test_file",
        "return_type",
        "params",
        "signature",
    ]
    with open(out_dir / "api_catalog.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for f in funcs:
            w.writerow(
                {
                    "function_name": f.name,
                    "module": f.module,
                    "family": f.family,
                    "group": f.group,
                    "dtype": f.dtype,
                    "channels": f.channels,
                    "layout": f.layout,
                    "inplace": int(f.inplace),
                    "ctx": int(f.ctx),
                    "buffer_helper": int(f.buffer_helper),
                    "status": f.status,
                    "impl_file": f.impl_file,
                    "tested": int(f.tested),
                    "test_file": ";".join(f.test_files),
                    "return_type": f.return_type,
                    "params": f.params,
                    "signature": f.signature,
                }
            )

    # Legacy-compatible coverage.csv: implemented=yes only for real code.
    with open(out_dir / "coverage.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["function_name", "module", "implemented", "impl_file", "tested", "test_file"])
        for f in sorted(funcs, key=lambda x: (x.module, x.name)):
            w.writerow(
                [
                    f.name,
                    f.module,
                    "yes" if f.status == "real" else "no",
                    f.impl_file,
                    "yes" if f.tested else "no",
                    ";".join(f.test_files),
                ]
            )


def write_backlog(funcs: List[ApiFunc], out_dir: Path) -> None:
    rows = defaultdict(
        lambda: {
            "total": 0,
            "real": 0,
            "stub": 0,
            "missing": 0,
            "missing_ctx": 0,
            "missing_nonctx": 0,
            "tested": 0,
        }
    )
    for f in funcs:
        r = rows[(f.module, f.group)]
        r["total"] += 1
        r[f.status] += 1
        if f.status != "real":
            r["missing_ctx" if f.ctx else "missing_nonctx"] += 1
        if f.tested:
            r["tested"] += 1

    with open(out_dir / "backlog.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "module",
                "group",
                "total",
                "real",
                "stub",
                "missing",
                "missing_ctx",
                "missing_nonctx",
                "tested",
                "remaining",
            ]
        )
        for (module, group), r in sorted(
            rows.items(), key=lambda kv: (-(kv[1]["stub"] + kv[1]["missing"]), kv[0])
        ):
            w.writerow(
                [
                    module,
                    group,
                    r["total"],
                    r["real"],
                    r["stub"],
                    r["missing"],
                    r["missing_ctx"],
                    r["missing_nonctx"],
                    r["tested"],
                    r["stub"] + r["missing"],
                ]
            )


def write_summary(funcs: List[ApiFunc], out_dir: Path) -> None:
    total = len(funcs)
    real = sum(1 for f in funcs if f.status == "real")
    stub = sum(1 for f in funcs if f.status == "stub")
    missing = sum(1 for f in funcs if f.status == "missing")
    tested = sum(1 for f in funcs if f.tested)
    real_tested = sum(1 for f in funcs if f.status == "real" and f.tested)

    by_module = defaultdict(lambda: Counter())
    for f in funcs:
        m = by_module[f.module]
        m["total"] += 1
        m[f.status] += 1
        if f.tested:
            m["tested"] += 1

    lines = ["# NPP API Catalog Summary", ""]
    lines.append(f"- Public functions (API/): {total}")
    lines.append(f"- Real implementations: {real} ({real / total * 100:.1f}%)")
    lines.append(f"- Stubs (NPP_NOT_IMPLEMENTED_ERROR): {stub}")
    lines.append(f"- Missing: {missing}")
    lines.append(f"- Tested: {tested} ({tested / total * 100:.1f}%)")
    lines.append(f"- Real and tested: {real_tested}")
    lines.append("")
    lines.append("## By module")
    lines.append("")
    lines.append("| module | total | real | stub | missing | tested |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for module in sorted(by_module):
        m = by_module[module]
        lines.append(
            f"| {module} | {m['total']} | {m['real']} | {m['stub']} | {m['missing']} | {m['tested']} |"
        )
    lines.append("")
    lines.append("## Largest remaining groups")
    lines.append("")
    groups = defaultdict(lambda: Counter())
    for f in funcs:
        if f.status != "real":
            groups[(f.module, f.group)]["remaining"] += 1
            groups[(f.module, f.group)]["ctx"] += 1 if f.ctx else 0
    lines.append("| module | group | remaining | ctx |")
    lines.append("|---|---|---:|---:|")
    for (module, group), c in sorted(groups.items(), key=lambda kv: -kv[1]["remaining"])[:40]:
        lines.append(f"| {module} | {group} | {c['remaining']} | {c['ctx']} |")
    lines.append("")

    (out_dir / "coverage_summary.md").write_text("\n".join(lines))


def print_summary(funcs: List[ApiFunc]) -> None:
    total = len(funcs)
    status = Counter(f.status for f in funcs)
    print(f"public functions : {total}")
    print(f"real             : {status['real']}")
    print(f"stub             : {status['stub']}")
    print(f"missing          : {status['missing']}")
    print(f"tested           : {sum(1 for f in funcs if f.tested)}")
    print(f"remaining        : {status['stub'] + status['missing']}")
    print(f"remaining _Ctx   : {sum(1 for f in funcs if f.status != 'real' and f.ctx)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api-dir", type=Path, default=API_DIR)
    ap.add_argument("--src-dir", type=Path, default=SRC_DIR)
    ap.add_argument("--test-dir", type=Path, default=TEST_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--update", action="store_true", help="write catalog/backlog/summary files")
    ap.add_argument("--module", help="filter by module header (substring)")
    ap.add_argument("--group", help="filter by family/group (substring)")
    ap.add_argument("--status", choices=["real", "stub", "missing"], help="filter by status")
    ap.add_argument("--list", action="store_true", help="list matching function names")
    args = ap.parse_args()

    funcs = build(args.api_dir, args.src_dir, args.test_dir)
    if args.module:
        funcs = [f for f in funcs if args.module in f.module]
    if args.group:
        funcs = [f for f in funcs if args.group in f.group or args.group in f.family]
    if args.status:
        funcs = [f for f in funcs if f.status == args.status]

    if args.list:
        for f in funcs:
            print(f"{f.status:8s} ctx={int(f.ctx)} {f.name}")
        return 0

    if args.update:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_catalog(funcs, args.out_dir)
        write_backlog(funcs, args.out_dir)
        write_summary(funcs, args.out_dir)
        print(f"wrote {args.out_dir}/api_catalog.csv")
        print(f"wrote {args.out_dir}/coverage.csv")
        print(f"wrote {args.out_dir}/backlog.csv")
        print(f"wrote {args.out_dir}/coverage_summary.md")
        print()
    print_summary(funcs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
