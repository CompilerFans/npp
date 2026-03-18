#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(dirname "$SCRIPT_DIR")"

show_help() {
  cat <<'EOF'
Usage:
  ./benchmark/scripts/new_benchmark.sh <type> <category> <header> --functions f1,f2[,f3...] [--variant-tags tag1,tag2] [--force]

Arguments:
  type           nppi or npps
  category       output category name, e.g. addc, morphology
  header         header file to include, e.g. nppi_arithmetic_and_logical_operations.h

Options:
  --functions    Comma-separated function list to scaffold
  --variant-tags Optional comma-separated scenario tags, e.g. baseline,linear,cubic
  --force        Overwrite existing output file
  --help, -h     Show this help

Example:
  ./benchmark/scripts/new_benchmark.sh nppi addc nppi_arithmetic_and_logical_operations.h \
    --functions nppiAddC_8u_C1RSfs,nppiAddC_32f_C1R \
    --variant-tags baseline,stress
EOF
  exit 0
}

require_value() {
  local opt="$1"
  local value="${2:-}"
  if [ -z "$value" ] || [[ "$value" == -* ]]; then
    echo "Option $opt requires a value" >&2
    exit 1
  fi
}

if [ $# -lt 3 ]; then
  show_help
fi

TYPE="$1"
CATEGORY="$2"
HEADER="$3"
shift 3

FUNCTIONS=""
VARIANT_TAGS=""
FORCE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --functions)
      require_value "$1" "${2:-}"
      FUNCTIONS="${2:-}"
      shift 2
      ;;
    --variant-tags)
      require_value "$1" "${2:-}"
      VARIANT_TAGS="${2:-}"
      shift 2
      ;;
    --force)
      FORCE=true
      shift
      ;;
    --help|-h)
      show_help
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [ "$TYPE" != "nppi" ] && [ "$TYPE" != "npps" ]; then
  echo "type must be nppi or npps" >&2
  exit 1
fi

if [ -z "$FUNCTIONS" ]; then
  echo "--functions is required" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found" >&2
  exit 1
fi

OUTPUT_DIR="$BENCHMARK_DIR/$TYPE"
OUTPUT_FILE="$OUTPUT_DIR/bench_${CATEGORY}.cpp"
mkdir -p "$OUTPUT_DIR"

if [ -f "$OUTPUT_FILE" ] && [ "$FORCE" != true ]; then
  echo "File already exists: $OUTPUT_FILE" >&2
  echo "Use --force to overwrite." >&2
  exit 1
fi

python3 - "$TYPE" "$CATEGORY" "$HEADER" "$FUNCTIONS" "$VARIANT_TAGS" "$OUTPUT_FILE" <<'PY'
import re
import sys
from pathlib import Path

type_name, category, header, functions_csv, variant_tags_csv, output_file = sys.argv[1:7]

functions = [item.strip() for item in functions_csv.split(",") if item.strip()]
variant_tags = [item.strip() for item in variant_tags_csv.split(",") if item.strip()]
if not functions:
    raise SystemExit("No functions provided")


def sanitize_identifier(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", value)
    if cleaned and cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned


def infer_data_type(function_name: str) -> str:
    match = re.search(r"_([0-9A-Za-z]+)_(?:C\d|AC4|P\d|R|I|RSfs|Ctx|A)", function_name)
    if match:
        return match.group(1)
    parts = function_name.split("_")
    return parts[1] if len(parts) > 1 else "unknown"


def infer_channels(function_name: str) -> int:
    match = re.search(r"_(?:C|P)(\d)", function_name)
    if match:
        return int(match.group(1))
    if "_AC4" in function_name:
        return 4
    return 1


def cpp_string_list(items):
    return ", ".join(f'"{item}"' for item in items)


variant_literal = cpp_string_list(variant_tags)
module_sizes = "getStandardImageSizes()" if type_name == "nppi" else "getStandardSignalSizes()"
size_type = "ImageSize" if type_name == "nppi" else "SignalSize"

if type_name == "nppi":
    append_body = """\
    void appendResult(std::vector<BenchmarkResult>& results,
                      const ImageSize& size,
                      const std::string& tag,
                      bool symbolFound,
                      const std::string& symbolError) const {
        BenchmarkResult result;
        result.functionName = name_;
        result.size = size.toString();
        result.variantTags = tag;
        result.dataType = kDataType;
        result.channels = kChannels;
        result.success = false;
        result.errorMessage = symbolFound
            ? "TODO: replace generated stub with real benchmark body"
            : std::string("Missing symbol ") + name_ + ": " + symbolError;
        results.push_back(result);
    }
"""
else:
    append_body = """\
    void appendResult(std::vector<BenchmarkResult>& results,
                      const SignalSize& size,
                      const std::string& tag,
                      bool symbolFound,
                      const std::string& symbolError) const {
        BenchmarkResult result;
        result.functionName = name_;
        result.size = size.toString();
        result.variantTags = tag;
        result.dataType = kDataType;
        result.channels = kChannels;
        result.success = false;
        result.errorMessage = symbolFound
            ? "TODO: replace generated stub with real benchmark body"
            : std::string("Missing symbol ") + name_ + ": " + symbolError;
        results.push_back(result);
    }
"""

blocks = []
for function_name in functions:
    class_name = f"BenchGenerated_{sanitize_identifier(function_name)}"
    data_type = infer_data_type(function_name)
    channels = infer_channels(function_name)
    block = f"""\
class {class_name} : public BenchmarkBase {{
public:
    using RawFn = void (*)();

    {class_name}() : BenchmarkBase("{function_name}") {{}}

    std::vector<BenchmarkResult> run(int warmupIterations, int measureIterations) override {{
        (void)warmupIterations;
        std::vector<BenchmarkResult> results;
        const auto sizes = benchmarkCases();
        const std::vector<std::string> variantTags = kVariantTags.empty() ? std::vector<std::string>{{""}} : kVariantTags;

        DynamicSymbol<RawFn> symbol(name_.c_str());
        std::string symbolError;
        const bool symbolFound = (symbol.get(&symbolError) != nullptr);

        for (const auto& size : sizes) {{
            for (const auto& tag : variantTags) {{
                appendResult(results, size, tag, symbolFound, symbolError);
            }}
        }}
        return results;
    }}

private:
    static constexpr const char* kDataType = "{data_type}";
    static constexpr int kChannels = {channels};
    static auto benchmarkCases() {{
        return {module_sizes};
    }}
{append_body}}};

REGISTER_BENCHMARK({class_name})
"""
    blocks.append(block)

content = f"""\
/**
 * @file bench_{category}.cpp
 * @brief Generated benchmark stubs for {type_name} {category}
 *
 * Generated by new_benchmark.sh with:
 *   header       = {header}
 *   functions    = {", ".join(functions)}
 *   variant_tags = {", ".join(variant_tags) if variant_tags else "(none)"}
 *
 * Notes:
 * - This file is intentionally compilable immediately after generation.
 * - Each generated benchmark is registered and returns success=false until the
 *   real benchmark body is implemented.
 * - Variant tags are exported in the dedicated `variant_tags` field.
 * - Override `benchmarkCases()` in this file when this API needs custom sizes.
 */

#include "../benchmark_base.h"
#include <{header}>

namespace npp_benchmark {{

namespace {{
const std::vector<std::string> kVariantTags = {{{variant_literal}}};
}} // namespace

{chr(10).join(blocks)}

}} // namespace npp_benchmark
"""

Path(output_file).write_text(content, encoding="utf-8")
PY

echo "Created: $OUTPUT_FILE"
echo "Next:"
echo "  1. build: ./benchmark/scripts/build_benchmark.sh"
echo "  2. list:  ./benchmark/scripts/run_benchmark.sh -l --no-build"
echo "  3. run:   ./benchmark/scripts/run_benchmark.sh -f ${CATEGORY} --no-build"
