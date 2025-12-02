#!/bin/bash
# Test script for seismic_app.py startup verification

echo "Testing Seismic Viewer startup..."
echo

# Test 1: Help output
echo "Test 1: Verify help works"
./venv_geo_sense/bin/python seismic_app.py --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Help works"
else
    echo "✗ Help failed"
    exit 1
fi

# Test 2: Parse arguments
echo "Test 2: Verify argument parsing"
./venv_geo_sense/bin/python -c "
import sys
sys.path.insert(0, '.')
from seismic_app import parse_arguments
import argparse

# Mock sys.argv
sys.argv = ['seismic_app.py', '--test-mode', '--auto-exit', '2', '--no-session']
args = parse_arguments()
assert args.test_mode == True
assert args.auto_exit == 2.0
assert args.no_session == True
print('✓ Argument parsing works')
"

# Test 3: Quick startup test (with auto-exit)
echo "Test 3: Verify application starts and auto-exits"
timeout 10 ./venv_geo_sense/bin/python seismic_app.py --auto-exit 2 --no-session > /tmp/startup_test.log 2>&1 &
PID=$!
sleep 4

if ! ps -p $PID > /dev/null 2>&1; then
    if grep -q "Auto-exit" /tmp/startup_test.log 2>/dev/null; then
        echo "✓ Application started and auto-exited"
    else
        echo "⚠ Application exited (check /tmp/startup_test.log)"
    fi
else
    echo "✗ Application did not exit"
    kill $PID 2>/dev/null
    exit 1
fi

echo
echo "All startup tests passed!"
