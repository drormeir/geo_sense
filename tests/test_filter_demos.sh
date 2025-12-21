#!/bin/bash
# Test all filter demos
# Runs each filter module and checks that it starts successfully

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="$PROJECT_DIR/venv_geo_sense/bin/python"
TIMEOUT_SEC=3

cd "$PROJECT_DIR"

passed=0
failed=0

for subdir in filters/*/; do
    # Skip __pycache__ directories
    [[ "$subdir" == *"__pycache__"* ]] && continue

    for pyfile in "$subdir"*.py; do
        # Skip if no .py files found
        [[ ! -f "$pyfile" ]] && continue

        # Skip files starting with _
        basename=$(basename "$pyfile")
        [[ "$basename" == _* ]] && continue

        # Convert path to module name (filters/frequency/fir.py -> filters.frequency.fir)
        module="${pyfile%.py}"
        module="${module//\//.}"

        echo -n "Testing $module ... "

        # Run with timeout, capture stderr
        output=$(timeout "$TIMEOUT_SEC" "$PYTHON" -m "$module" 2>&1)
        exit_code=$?

        # Exit code 124 = timeout (window opened, waiting for user) = success
        # Exit code 0 = completed normally = success
        if [[ $exit_code -eq 124 ]] || [[ $exit_code -eq 0 ]]; then
            echo "OK"
            ((passed++))
        else
            echo "FAILED (exit code: $exit_code)"
            echo "$output" | head -5
            ((failed++))
        fi
    done
done

echo ""
echo "=============================="
echo "Passed: $passed"
echo "Failed: $failed"
echo "=============================="

[[ $failed -eq 0 ]] && exit 0 || exit 1
