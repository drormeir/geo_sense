#!/bin/bash
# Run the geo_sense visualizer application
# Handles venv activation/deactivation automatically

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$SCRIPT_DIR/venv_geo_sense"
PROGRAM="$SCRIPT_DIR/seismic_app.py"

# Check if already in the project's venv
if [[ "$VIRTUAL_ENV" == "$VENV_PATH" ]]; then
    # Already in the correct venv, just run
    python "$PROGRAM" "$@"
else
    # Not in the correct venv
    if [[ -n "$VIRTUAL_ENV" ]]; then
        # In a different venv - save it to restore later
        ORIGINAL_VENV="$VIRTUAL_ENV"
    fi

    # Activate project venv and run
    source "$VENV_PATH/bin/activate"
    python "$PROGRAM" "$@"
    deactivate

    # Restore original venv if there was one
    if [[ -n "$ORIGINAL_VENV" ]]; then
        source "$ORIGINAL_VENV/bin/activate"
    fi
fi
