@echo off
setlocal

echo ============================================
echo  GeoSense Installation Script for Windows
echo ============================================
echo.

:: Pull latest changes from main branch
echo Pulling latest changes from git...
git pull origin main
if errorlevel 1 (
    echo WARNING: Failed to pull from git. Continuing with installation...
)
echo.

:: Check if venv already exists
if exist "venv_geo_sense" (
    echo Virtual environment 'venv_geo_sense' already exists.
    echo Exiting installation...
    goto :end
)

:: Find Python and print its path
echo Looking for Python...
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH.
    echo Please install Python 3.11+ from https://www.python.org/downloads/
    echo Installation aborted.
    goto :end
)

for /f "delims=" %%i in ('where python') do (
    echo Found Python at: %%i
    goto :found_python
)

:found_python
:: Print Python version
python --version
echo.

:: Create virtual environment
echo Creating virtual environment 'venv_geo_sense'...
python -m venv venv_geo_sense
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    goto :end
)
echo Virtual environment created successfully.
echo.

:: Activate virtual environment
echo Activating virtual environment...
call venv_geo_sense\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    goto :end
)
echo Virtual environment activated.
echo.

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip, continuing anyway...
)
echo.

:: Install requirements
echo Installing requirements from requirements.txt...
if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found in current directory.
    goto :deactivate
)

pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements.
    goto :deactivate
)

echo.
echo ============================================
echo Installing GPRpy...
echo ============================================
echo.
git clone https://github.com/drormeir/GPRPy.git
if errorlevel 1 (
    echo ERROR: Failed to clone GPRpy.
    goto :deactivate
)
cd GPRPy
pip install .
if errorlevel 1 (
    echo ERROR: Failed to install GPRpy.
    cd ..
    goto :deactivate
)
cd ..

rmdir /s /q GPRPy
if errorlevel 1 (
    echo WARNING: Failed to remove GPRpy directory.
)
echo.
echo GPRpy installed successfully.
echo.

echo.
echo Requirements installed successfully.



:deactivate
:: Deactivate virtual environment
echo.
echo Deactivating virtual environment...
call deactivate


echo.
echo ============================================
echo  Installation complete!
echo  Type: run.bat from the command line to start the application.
echo ============================================

:end
endlocal
pause
