#!/bin/bash
# Update coverage analysis

set -e

echo "Updating coverage analysis..."

# Run coverage analysis
python3 tools/check_coverage.py

echo "Coverage analysis updated:"
echo "  - api_analysis/coverage.csv"
echo "  - api_analysis/coverage_summary.md"
