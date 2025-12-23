#!/bin/bash
#
# Test script for data_file.py
#
# Usage: ./tests/test_data_file.sh <directory>
#
# Scans the directory for seismic files (.sgy, .segy, .rd3, .rd7) and attempts
# to load each one using data_file.py, reporting success or failure.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="${PROJECT_DIR}/venv_geo_sense/bin/python"
DATA_FILE="${PROJECT_DIR}/data_file.py"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

DIRECTORY="$1"

if [ ! -d "$DIRECTORY" ]; then
    echo "Error: '$DIRECTORY' is not a valid directory"
    exit 1
fi

# Find seismic files
FILES=$(find "$DIRECTORY" -maxdepth 1 -type f \( -iname "*.sgy" -o -iname "*.segy" -o -iname "*.rd3" -o -iname "*.rd7" \) | sort)

if [ -z "$FILES" ]; then
    echo "No seismic files found in '$DIRECTORY'"
    echo "Supported extensions: .sgy, .segy, .rd3, .rd7"
    exit 0
fi

FILE_COUNT=$(echo "$FILES" | wc -l)
echo "Testing $FILE_COUNT seismic file(s) in '$DIRECTORY'"
echo "============================================================"

PASSED=0
FAILED=0

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

for filepath in $FILES; do
    filename=$(basename "$filepath")

    if output=$("$PYTHON" "$DATA_FILE" "$filepath" 2>&1); then
        echo -e "${GREEN}PASS${NC}: $filename"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}FAIL${NC}: $filename"
        echo "      $output"
        FAILED=$((FAILED + 1))
    fi
done

echo "============================================================"
echo "Results: $PASSED passed, $FAILED failed, $FILE_COUNT total"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
exit 0
