#!/bin/bash
# Extract and deduplicate NPPI API functions from opencv usage file

INPUT="nppi_api_in_opencv.txt"
OUTPUT="nppi_apis_unique.txt"

# Extract nppi function names (must contain underscore to filter out library names)
grep -o 'nppi[A-Za-z0-9_]*' "$INPUT" | grep '_' | sort -u > "$OUTPUT"

# Print statistics
TOTAL=$(grep -o 'nppi[A-Za-z0-9_]*' "$INPUT" | grep '_' | wc -l)
UNIQUE=$(wc -l < "$OUTPUT")

echo "Extracted NPPI APIs from $INPUT"
echo "Total occurrences: $TOTAL"
echo "Unique APIs: $UNIQUE"
echo "Output saved to: $OUTPUT"
