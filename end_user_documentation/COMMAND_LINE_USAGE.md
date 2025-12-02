# Seismic Viewer - Command Line Usage

## Getting Help

### Linux/Mac
```bash
python seismic_app.py --help
python seismic_app.py -h
```

### Windows (Command Prompt)
```cmd
python seismic_app.py --help
python seismic_app.py -h
```

### Windows (PowerShell)
```powershell
python seismic_app.py --help
python seismic_app.py -h
```

**Both `-h` and `--help` work identically on all platforms.**

---

## Command-Line Options

### Normal Usage
```bash
# Start normally
python seismic_app.py

# Start without loading previous session
python seismic_app.py --no-session
```

### Debugging & Testing Options
```bash
# Auto-exit after 3 seconds (useful for automated tests)
python seismic_app.py --auto-exit 3

# Quick startup test (no session, auto-exit)
python seismic_app.py --auto-exit 2 --no-session

# Take a screenshot after startup for debugging
python seismic_app.py --screenshot /tmp/screenshot.png --auto-exit 2

# Print session file location and contents
python seismic_app.py --print-session

# Test mode (reserved for future use)
python seismic_app.py --test-mode
```

---

## Available Options

| Option | Short | Description |
|--------|-------|-------------|
| `--help` | `-h` | Show help message and exit |
| `--test-mode` | - | Run in test mode (minimal GUI) |
| `--auto-exit SECONDS` | - | Exit automatically after N seconds |
| `--no-session` | - | Don't load or save session state |
| `--screenshot PATH` | - | Save screenshot to PATH after startup |
| `--print-session` | - | Print session file path and contents, then exit |

---

## Examples

### Development
```bash
# Start fresh every time (good for development)
python seismic_app.py --no-session
```

### Automated Testing
```bash
# Verify application starts and exits cleanly
python seismic_app.py --auto-exit 2 --no-session
echo $?  # Check exit code (0 = success)

# Take a screenshot for visual verification
python seismic_app.py --screenshot /tmp/app_test.png --auto-exit 2 --no-session

# Check session file contents
python seismic_app.py --print-session
```

### Windows Batch File
```batch
@echo off
python seismic_app.py --no-session
if errorlevel 1 (
    echo Application failed to start
    exit /b 1
)
```

### Linux Shell Script
```bash
#!/bin/bash
python seismic_app.py --no-session
if [ $? -ne 0 ]; then
    echo "Application failed to start"
    exit 1
fi
```

---

## Platform-Specific Notes

### Linux/Mac with Virtual Environment
```bash
./venv_geo_sense/bin/python seismic_app.py --help
```

### Windows with Virtual Environment
```cmd
venv_geo_sense\Scripts\python seismic_app.py --help
```

### Cross-Platform Virtual Environment Activation

**Linux/Mac:**
```bash
source venv_geo_sense/bin/activate
python seismic_app.py --help
```

**Windows (cmd.exe):**
```cmd
venv_geo_sense\Scripts\activate.bat
python seismic_app.py --help
```

**Windows (PowerShell):**
```powershell
venv_geo_sense\Scripts\Activate.ps1
python seismic_app.py --help
```

---

## Quick Reference

```
# Show help (works everywhere)
python seismic_app.py -h

# Normal start
python seismic_app.py

# Fresh start
python seismic_app.py --no-session

# Test startup (exits after 3 seconds)
python seismic_app.py --auto-exit 3
```
