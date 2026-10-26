@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   CHICAGO CRIME INTELLIGENCE SYSTEM - SETUP & INSTALLER
echo ============================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not detected in your system PATH.
    echo Please install Python 3.10+ from https://www.python.org/
    echo and ensure "Add Python to PATH" is checked during installation.
    echo.
    pause
    exit /b 1
)

echo [INFO] Python detected:
python --version
echo.

:: Navigate to script directory
cd /d "%~dp0"

:: Create virtual environment if it does not exist
if not exist ".venv" (
    echo [INFO] Creating Python virtual environment (.venv)...
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [INFO] Virtual environment created successfully.
) else (
    echo [INFO] Existing virtual environment found (.venv).
)
echo.

:: Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

:: Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
echo [INFO] Installing required dependencies from requirements.txt...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [WARNING] Some dependencies could not be installed automatically.
    echo Please review requirements.txt or install them manually.
) else (
    echo.
    echo ============================================================
    echo   [SUCCESS] ALL DEPENDENCIES SUCCESSFULLY INSTALLED!
    echo ============================================================
)

echo.
echo Next Step:
echo Run "START_HERE.bat" to launch the Chicago Crime Intelligence Dashboard!
echo.
pause
