@echo off
setlocal

:: Check if venv exists
if not exist "venv_geo_sense" (
    echo ERROR: Virtual environment 'venv_geo_sense' not found.
    echo Please run install.bat first.
    goto :end
)

:: Activate virtual environment
call venv_geo_sense\Scripts\activate.bat

python seismic_app.py

:end
endlocal
pause

