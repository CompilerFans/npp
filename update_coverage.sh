#!/bin/bash
# Update coverage analysis

set -e

cd "$(dirname "$0")"
API_DIR="api_analysis"

echo "Updating coverage analysis..."

# Run coverage analysis
python3 tools/check_coverage.py

# Generate untested functions list for arithmetic module
echo "Generating untested arithmetic APIs list..."
grep "nppi_arithmetic_and_logical_operations.h" "$API_DIR/coverage.csv" | \
    awk -F',' '$5=="no" {print $1}' | sort > "$API_DIR/untested_arithmetic.txt"

UNTESTED_COUNT=$(wc -l < "$API_DIR/untested_arithmetic.txt")
TOTAL_COUNT=$(grep -c "nppi_arithmetic_and_logical_operations.h" "$API_DIR/coverage.csv")
TESTED_COUNT=$((TOTAL_COUNT - UNTESTED_COUNT))

PERCENT=$(awk "BEGIN {printf \"%.1f\", ${TESTED_COUNT}*100/${TOTAL_COUNT}}")
cat << EOF > "$API_DIR/untested_arithmetic_summary.md"
# Untested Arithmetic APIs

**Status**: ${TESTED_COUNT}/${TOTAL_COUNT} tested (${PERCENT}%)
**Remaining**: ${UNTESTED_COUNT} functions

## By Category
EOF

# Add category counts
for op in Add Sub Mul Div AddC SubC MulC DivC And Or Xor Not AndC OrC XorC \
          LShiftC RShiftC Sqrt Sqr Exp Ln MulScale AlphaComp AlphaPremul \
          AddDeviceC SubDeviceC MulDeviceC DivDeviceC; do
    count=$(grep -c "^nppi${op}" "$API_DIR/untested_arithmetic.txt" 2>/dev/null || true)
    count=${count:-0}
    count=$(echo "$count" | tr -d '[:space:]')
    if [ -n "$count" ] && [ "$count" -gt 0 ] 2>/dev/null; then
        echo "- nppi${op}: ${count}" >> "$API_DIR/untested_arithmetic_summary.md"
    fi
done

echo "" >> "$API_DIR/untested_arithmetic_summary.md"
echo "## Full List" >> "$API_DIR/untested_arithmetic_summary.md"
echo "See: untested_arithmetic.txt" >> "$API_DIR/untested_arithmetic_summary.md"

echo "Coverage analysis updated:"
echo "  - api_analysis/coverage.csv"
echo "  - api_analysis/coverage_summary.md"
echo "  - api_analysis/untested_arithmetic.txt ($UNTESTED_COUNT functions)"
echo "  - api_analysis/untested_arithmetic_summary.md"
