#!/usr/bin/env python3
"""Generate the superset declaration header for the NPP 13.x header sets.

NPP 13 removed the stream-context-free entry points from the public headers.
The library still implements the full legacy surface, so the shared test suite
and the library itself pre-include a header that declares it again.  The
declarations are derived from the 13.x ``_Ctx`` prototypes (drop the trailing
``NppStreamContext`` parameter and the ``_Ctx`` suffix) so parameter types and
the buffer-size ABI match the selected header set.  A handful of legacy C++
overloads that 13.x no longer declares are appended outside ``extern "C"``.

Usage:
  python3 tools/gen_npp_api_extra_13_4.py
  python3 tools/gen_npp_api_extra_13_4.py --unified API-unified \
      --api-13-4 API-13.4 --output test/unit/framework/npp_api_extra_13_4.h
"""

from __future__ import annotations

import argparse
import glob
import os
import re

STREAM_COMPAT = {"nppGetStreamContext", "nppSetStream"}


def comment_mask(text: str) -> bytearray:
    mask = bytearray(len(text))
    i, n = 0, len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            mask[i:j] = b"\x01" * (j - i)
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            mask[i:j] = b"\x01" * (j - i)
            i = j
        elif text[i] == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            i = j + 1
        else:
            i += 1
    return mask


def parse_declarations(api_dir: str) -> dict[str, list[tuple[str, str]]]:
    out: dict[str, list[tuple[str, str]]] = {}
    for path in sorted(glob.glob(os.path.join(api_dir, "*.h"))):
        text = open(path, encoding="utf-8", errors="surrogateescape").read()
        mask = comment_mask(text)
        for match in re.finditer(r"\bNppStatus\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text):
            name = match.group(1)
            if mask[match.start()]:
                continue
            i, depth = match.end() - 1, 0
            while i < len(text):
                if text[i] == "(":
                    depth += 1
                elif text[i] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            end = text.find(";", i)
            if end < 0:
                continue
            out.setdefault(name, []).append((text[match.end():i], text[match.start():end + 1]))
    return out


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def normalize(params: str) -> str:
    text = re.sub(r"\s+", " ", strip_comments(params)).strip()
    return re.sub(r"\s+([,;)])", r"\1", text)


def signature_key(params: str) -> str:
    """Normalize the 12.x int ABI and the 13.x size_t ABI to one key."""
    key = normalize(params)
    key = re.sub(r"\b(int|size_t|NppSignalLength)\s+(nLength)\b", r"T \2", key)
    key = re.sub(r"\b(int|size_t)\s*\*\s*(hp\w*(?:Buffer)?Size)\b", r"T * \2", key)
    key = re.sub(r"\b(int|size_t)\s*\*\s*(pBufferSize|hpDeviceMemoryBufferSize)\b", r"T * \2", key)
    return key


def split_params(params: str) -> list[str]:
    parts: list[str] = []
    depth, current = 0, ""
    for char in params:
        if char in "([":
            depth += 1
        if char in ")]":
            depth -= 1
        if char == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)
    return parts


def strip_stream_context(params: str) -> str | None:
    parts = split_params(params)
    if "NppStreamContext" not in normalize(parts[-1]):
        return None
    return normalize(",".join(parts[:-1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--unified", default="API-unified", help="superset API header directory")
    parser.add_argument("--api-13-4", default="API-13.4", help="NPP 13.x API header directory")
    parser.add_argument(
        "--output",
        default="test/unit/framework/npp_api_extra_13_4.h",
        help="generated header path",
    )
    args = parser.parse_args()

    unified = parse_declarations(args.unified)
    api13 = parse_declarations(args.api_13_4)

    c_decls: list[str] = []
    cpp_decls: list[str] = []
    seen: set[str] = set()

    for name in sorted(unified):
        if name in STREAM_COMPAT:
            continue
        udecls = unified[name]
        target_keys = {signature_key(params) for params, _ in api13.get(name, [])}
        covered: set[str] = set()

        if name not in api13:
            # Reuse the 13.x prototype with the stream-context parameter removed
            # so the declaration follows the selected header's ABI.
            chosen = None
            for twin_params, _ in api13.get(name + "_Ctx", []):
                stripped = strip_stream_context(twin_params)
                if stripped is None:
                    continue
                key = signature_key(stripped)
                if any(signature_key(p) == key for p, _ in udecls):
                    chosen = ("NppStatus %s(%s);" % (name, stripped), key)
                    break
            if chosen is not None:
                if chosen[0] not in seen:
                    seen.add(chosen[0])
                    c_decls.append(chosen[0])
                covered.add(chosen[1])
            elif udecls:
                decl = normalize(udecls[0][1])
                if decl not in seen:
                    seen.add(decl)
                    c_decls.append(decl)
                covered.add(signature_key(udecls[0][0]))

        for params, decl in udecls:
            key = signature_key(params)
            if key in target_keys or key in covered:
                continue
            # Stream-context-free _Ctx forwarders are a unified-header
            # extension, not part of the NPP 13.x contract.
            if name.endswith("_Ctx") and "NppStreamContext" not in key:
                continue
            text = normalize(decl)
            if text not in seen:
                seen.add(text)
                cpp_decls.append(text)

    lines = [
        "#ifndef NPP_API_EXTRA_13_4_H",
        "#define NPP_API_EXTRA_13_4_H",
        "",
        "/* Superset declarations for the API-13.4 header set.  NPP 13 dropped the",
        " * stream-context-free entry points from its public headers; declare the",
        " * legacy surface here so the shared implementation and test suite keep",
        " * compiling against the 13.x headers. */",
        "#include <npp.h>",
        '#include "npp_api_compat.h"',
        "",
        "#if !NPP_HEADER_UNIFIED && NPP_HEADER_MAJOR >= 13",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        *c_decls,
        "",
        "#ifdef __cplusplus",
        '} /* extern "C" */',
        "",
        "/* Legacy C++ overloads that the 13.x headers no longer declare. */",
        *cpp_decls,
        "",
        "#endif",
        "",
        "#endif /* 13.x header set */",
        "",
        "#endif  // NPP_API_EXTRA_13_4_H",
    ]
    with open(args.output, "w", encoding="utf-8", errors="surrogateescape") as out:
        out.write("\n".join(lines) + "\n")
    print(f"wrote {args.output}: {len(c_decls)} C declarations, {len(cpp_decls)} C++ overloads")


if __name__ == "__main__":
    main()
