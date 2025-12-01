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

:: Run the application and exit immediately
start "" pythonw seismic_app.py

:: Exit the batch file (command prompt will close)
exit

:end
endlocal
