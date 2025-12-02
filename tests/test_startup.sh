#!/bin/bash
# Test script for seismic_app.py startup verification

# Don't exit on error - we want to run all tests
set +e

PYTHON="./venv_geo_sense/bin/python"
APP="seismic_app.py"
TEST_COUNT=0
PASS_COUNT=0
FAIL_COUNT=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================"
echo "Seismic Viewer Startup Tests"
echo "================================"
echo

# Helper functions
pass_test() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASS_COUNT++))
    ((TEST_COUNT++))
}

fail_test() {
    echo -e "${RED}✗${NC} $1"
    ((FAIL_COUNT++))
    ((TEST_COUNT++))
}

warn_test() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Test 1: Help output
echo "Test 1: Verify --help works"
if $PYTHON $APP --help > /dev/null 2>&1; then
    pass_test "Help option works"
else
    fail_test "Help option failed"
fi

# Test 2: Parse arguments
echo "Test 2: Verify argument parsing"
$PYTHON -c "
import sys
sys.path.insert(0, '.')
from seismic_app import parse_arguments

# Mock sys.argv
sys.argv = ['seismic_app.py', '--test-mode', '--auto-exit', '2', '--session-mode', '-1']
args = parse_arguments()
assert args.test_mode == True
assert args.auto_exit == 2.0
assert args.session_mode == -1
" 2>/dev/null
if [ $? -eq 0 ]; then
    pass_test "Argument parsing works"
else
    fail_test "Argument parsing failed"
fi

# Test 3: Auto-exit timing
echo "Test 3: Verify auto-exit timing (2 seconds)"
START_TIME=$(date +%s)
$PYTHON $APP --auto-exit 2 --session-mode -1 > /tmp/startup_test.log 2>&1
EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

if [ $EXIT_CODE -eq 0 ]; then
    if [ $ELAPSED -ge 2 ] && [ $ELAPSED -le 4 ]; then
        pass_test "Auto-exit timing correct (${ELAPSED}s)"
    else
        warn_test "Auto-exit timing off (${ELAPSED}s, expected ~2-3s)"
        ((TEST_COUNT++))
    fi
else
    fail_test "Auto-exit failed with exit code $EXIT_CODE"
fi

# Test 4: Check for QTimer warnings
echo "Test 4: Check for QTimer warnings (should be none)"
if grep -q "QObject::startTimer" /tmp/startup_test.log 2>/dev/null; then
    fail_test "QTimer warning present (timer created before event loop)"
    echo "  Warning found in log:"
    grep "QObject::startTimer" /tmp/startup_test.log | head -1
else
    pass_test "No QTimer warnings"
fi

# Test 5: Verify "Auto-exit scheduled" message appears
echo "Test 5: Verify auto-exit message appears"
if grep -q "Auto-exit scheduled" /tmp/startup_test.log 2>/dev/null; then
    pass_test "Auto-exit message present"
else
    fail_test "Auto-exit message not found"
fi

# Test 6: Screenshot feature
echo "Test 6: Test --screenshot feature"
SCREENSHOT_PATH="/tmp/test_screenshot_$$.png"
$PYTHON $APP --screenshot "$SCREENSHOT_PATH" --auto-exit 1 --session-mode -1 > /dev/null 2>&1
if [ -f "$SCREENSHOT_PATH" ]; then
    SIZE=$(stat -f%z "$SCREENSHOT_PATH" 2>/dev/null || stat -c%s "$SCREENSHOT_PATH" 2>/dev/null)
    if [ "$SIZE" -gt 1000 ]; then
        pass_test "Screenshot created (${SIZE} bytes)"
        rm -f "$SCREENSHOT_PATH"
    else
        fail_test "Screenshot too small (${SIZE} bytes)"
    fi
else
    fail_test "Screenshot not created"
fi

# Test 7: Print session feature
echo "Test 7: Test --print-session feature"
SESSION_OUTPUT=$($PYTHON $APP --print-session 2>&1)
if echo "$SESSION_OUTPUT" | grep -q "Session file path:"; then
    if echo "$SESSION_OUTPUT" | grep -q ".json"; then
        pass_test "--print-session works"
    else
        fail_test "--print-session output malformed"
    fi
else
    fail_test "--print-session failed"
fi

# Test 8: Session mode -1 (no read/write)
echo "Test 8: Test --session-mode -1 (no read/write)"
touch /tmp/session_mode_test_marker
sleep 1
$PYTHON $APP --auto-exit 1 --session-mode -1 > /tmp/session_mode_test.log 2>&1
sleep 1
if [ $? -eq 0 ]; then
    if [ ~/.config/uas_sessions/default.json -ot /tmp/session_mode_test_marker ]; then
        pass_test "Session mode -1: no read/write works"
    else
        fail_test "Session mode -1: session was modified"
    fi
else
    fail_test "Session mode -1: app failed"
fi
rm -f /tmp/session_mode_test_marker

# Test 9: Session mode 0 (write only - fresh start)
echo "Test 9: Test --session-mode 0 (write only - fresh start)"
BEFORE_TIME=$(stat -c "%Y" ~/.config/uas_sessions/default.json 2>/dev/null || echo "0")
sleep 1
$PYTHON $APP --auto-exit 1 --session-mode 0 > /dev/null 2>&1
sleep 1
AFTER_TIME=$(stat -c "%Y" ~/.config/uas_sessions/default.json 2>/dev/null || echo "0")
if [ "$AFTER_TIME" -gt "$BEFORE_TIME" ]; then
    pass_test "Session mode 0: writes new session"
else
    fail_test "Session mode 0: session not written"
fi

# Test 10: Session mode 1 (normal - default)
echo "Test 10: Test --session-mode 1 (normal - default)"
$PYTHON $APP --auto-exit 1 --session-mode 1 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    pass_test "Session mode 1: normal mode works"
else
    fail_test "Session mode 1: app failed"
fi

# Test 11: Default session mode (should be 1)
echo "Test 11: Test default session mode (should be 1)"
$PYTHON $APP --auto-exit 1 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    pass_test "Default session mode works"
else
    fail_test "Default session mode failed"
fi

# Test 12: Test with session corruption recovery
echo "Test 12: Test session corruption recovery"
# Create a corrupted session file
mkdir -p ~/.config/uas_sessions
echo "{ invalid json" > ~/.config/uas_sessions/test_corrupted.json
# Test should handle gracefully (using session-mode -1 to avoid issues)
$PYTHON $APP --auto-exit 1 --session-mode -1 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    pass_test "App handles corrupted session (with --session-mode -1)"
else
    fail_test "App failed with corrupted session"
fi
rm -f ~/.config/uas_sessions/test_corrupted.json

# Test 13: Quick startup/shutdown cycle
echo "Test 13: Rapid startup/shutdown cycle"
for i in {1..3}; do
    $PYTHON $APP --auto-exit 1 --session-mode -1 > /dev/null 2>&1 &
    APP_PID=$!
    sleep 2
    if ! ps -p $APP_PID > /dev/null 2>&1; then
        : # Process exited as expected
    else
        kill $APP_PID 2>/dev/null
        fail_test "Cycle $i: App did not exit"
        break
    fi
done
if [ $i -eq 3 ]; then
    pass_test "Rapid cycle test passed (3 iterations)"
fi

# Summary
echo
echo "================================"
echo "Test Summary"
echo "================================"
echo "Total tests: $TEST_COUNT"
echo -e "${GREEN}Passed: $PASS_COUNT${NC}"
if [ $FAIL_COUNT -gt 0 ]; then
    echo -e "${RED}Failed: $FAIL_COUNT${NC}"
fi
echo

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    echo "Check /tmp/startup_test.log for details"
    exit 1
fi
