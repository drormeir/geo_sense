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

### Opening Files
```bash
# Open a seismic file directly
python seismic_app.py data.segy
python seismic_app.py /path/to/file.rd3

# Open ONLY this file (no session restore)
python seismic_app.py --session-mode -1 data.rd3

# Open file, save as new session on exit
python seismic_app.py --session-mode 0 data.segy
```

### Normal Usage
```bash
# Start normally (loads and saves session)
python seismic_app.py

# Start without loading or saving session
python seismic_app.py --session-mode -1

# Start fresh (don't load, but save new session on exit)
python seismic_app.py --session-mode 0
```

### Debugging & Testing Options
```bash
# Auto-exit after 3 seconds (useful for automated tests)
python seismic_app.py --auto-exit 3

# Quick startup test (no session, auto-exit)
python seismic_app.py --auto-exit 2 --session-mode -1

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
| `FILE` | - | Seismic file to open (SEGY .sgy/.segy, MALA .rd3/.rd7). Opens in addition to session windows by default. |
| `--help` | `-h` | Show help message and exit |
| `--test-mode` | - | Run in test mode (minimal GUI) |
| `--auto-exit SECONDS` | - | Exit automatically after N seconds |
| `--session-mode MODE` | - | Session mode: `-1`=no read/write, `0`=write only (fresh start), `1`=normal (default) |
| `--screenshot PATH` | - | Save screenshot to PATH after startup |
| `--print-session` | - | Print session file path and contents, then exit |

---

## Session Modes Explained

The `--session-mode` argument controls how the application handles session persistence:

### Mode -1: No Read/Write
```bash
python seismic_app.py --session-mode -1
```
- Does NOT load previous session state
- Does NOT save session on exit
- Perfect for automated testing and CI/CD
- Equivalent to the old `--no-session` flag

### Mode 0: Write Only (Fresh Start)
```bash
python seismic_app.py --session-mode 0
```
- Does NOT load previous session state (starts fresh)
- DOES save session on exit
- Good for development when you want to start clean but preserve changes
- Useful for creating a new baseline session

### Mode 1: Normal (Default)
```bash
python seismic_app.py
# or explicitly:
python seismic_app.py --session-mode 1
```
- DOES load previous session state (restores windows, settings, etc.)
- DOES save session on exit
- Standard behavior for end users
- This is the default mode

---

## Opening Files from Command Line

You can open a seismic file directly by passing it as an argument:

```bash
python seismic_app.py mydata.segy
python seismic_app.py /path/to/survey.rd3
```

### Supported File Formats
- **SEGY files**: `.sgy`, `.segy`
- **MALA GPR files**: `.rd3`, `.rd7` (requires companion `.rad` header file)

### How FILE Interacts with Sessions

By default (session-mode 1), the FILE argument opens **in addition to** any windows restored from the previous session:

```bash
# If session has 2 windows, this results in 3 windows (2 from session + 1 file)
python seismic_app.py data.rd3
```

To open **only** the specified file without restoring session windows:

```bash
# Open only this file, no session read/write
python seismic_app.py --session-mode -1 data.rd3

# Open only this file, save session on exit
python seismic_app.py --session-mode 0 data.rd3
```

### Summary Table

| Command | Session Windows | File Window | Save on Exit |
|---------|-----------------|-------------|--------------|
| `seismic_app.py file.rd3` | Restored | Yes | Yes |
| `seismic_app.py --session-mode 0 file.rd3` | No | Yes | Yes |
| `seismic_app.py --session-mode -1 file.rd3` | No | Yes | No |

---

## Examples

### Opening Files
```bash
# Open a SEGY file (adds to any existing session windows)
python seismic_app.py survey_data.segy

# Open a MALA GPR file without session (clean start)
python seismic_app.py --session-mode -1 GPR_line01.rd3

# Open file and save as new session on exit
python seismic_app.py --session-mode 0 /path/to/data.rd7
```

### Development
```bash
# Start without loading or saving session (good for testing)
python seismic_app.py --session-mode -1

# Start fresh each time, but save session on exit (good for development)
python seismic_app.py --session-mode 0
```

### Automated Testing
```bash
# Verify application starts and exits cleanly (no session read/write)
python seismic_app.py --auto-exit 2 --session-mode -1
echo $?  # Check exit code (0 = success)

# Take a screenshot for visual verification
python seismic_app.py --screenshot /tmp/app_test.png --auto-exit 2 --session-mode -1

# Check session file contents
python seismic_app.py --print-session
```

### Windows Batch File
```batch
@echo off
python seismic_app.py --session-mode -1
if errorlevel 1 (
    echo Application failed to start
    exit /b 1
)
```

### Linux Shell Script
```bash
#!/bin/bash
python seismic_app.py --session-mode -1
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

# Open a file directly (adds to session windows)
python seismic_app.py data.rd3

# Open ONLY this file (no session)
python seismic_app.py --session-mode -1 data.rd3

# Normal start (default: load and save session)
python seismic_app.py

# No session read/write (good for testing)
python seismic_app.py --session-mode -1

# Fresh start, save on exit (good for development)
python seismic_app.py --session-mode 0

# Test startup (exits after 3 seconds)
python seismic_app.py --auto-exit 3
```
